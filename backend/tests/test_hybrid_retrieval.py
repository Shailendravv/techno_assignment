"""Phase 5: chunking, embeddings, RRF fusion, and the hybrid gate.

Split by what each test needs:

- Chunking, fusion and normalisation are pure arithmetic and run everywhere.
- Anything needing real vectors is marked `embeddings` and skips when
  `fastembed` is absent, because it is a dev dependency and the deployed
  profile uses Gemini instead.

The test that matters most here is `test_hybrid_rescues_the_vocabulary_mismatch_question`.
Adding dense retrieval was justified in advance by one predicted failure, and
that test is the claim being checked rather than asserted.
"""

from __future__ import annotations

import math

import pytest

from agent.config import Retrieval, load_settings
from agent.core.chunk import chunk_corpus, split_document
from agent.core.corpus import load_corpus
from agent.core.models import Candidate, Doc, QuerySpec
from agent.core.query import analyze_query
from agent.core.retrieve import (
    best_cosine,
    bm25_search,
    build_dense_index,
    build_index,
    dense_search,
    metadata_filter,
    passes_hybrid_gate,
    rrf_fuse,
)
from agent.embed import (
    EmbedderUnavailable,
    NullEmbedder,
    cosine,
    l2_normalise,
)

def _has_fastembed() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


embeddings = pytest.mark.skipif(
    not _has_fastembed(),
    reason="fastembed is a dev-only dependency; the deployed profile uses Gemini",
)


@pytest.fixture(scope="module")
def docs():
    return load_corpus("runbooks")


@pytest.fixture(scope="module")
def lexical_index(docs):
    return build_index(docs)


@pytest.fixture(scope="module")
def dense_index(docs):
    from agent.embed import get_embedder

    return build_dense_index(docs, get_embedder(load_settings("local")))


def _doc(doc_id: str, text: str, service: str | None = "checkout-api") -> Doc:
    return Doc(
        doc_id=doc_id,
        title=f"{doc_id} title",
        service=service,
        failure_mode="cpu",
        doc_type="runbook",
        date=None,
        text=text,
    )


# --------------------------------------------------------------------------
# Chunking.
# --------------------------------------------------------------------------

def test_a_document_splits_on_its_level_two_headings():
    doc = _doc("RB-900", "## Symptoms\n\n" + "a" * 60 + "\n\n## Mitigation\n\n" + "b" * 60)
    chunks = split_document(doc)

    assert [c.section for c in chunks] == ["Symptoms", "Mitigation"]


def test_every_chunk_knows_its_parent_document():
    """The brief's contract is document IDs, so chunks must collapse back."""
    doc = _doc("RB-900", "## One\n\n" + "a" * 60 + "\n\n## Two\n\n" + "b" * 60)

    assert {c.doc_id for c in split_document(doc)} == {"RB-900"}


def test_each_chunk_carries_the_title_and_heading():
    """Without this prefix, generic sections embed identically across documents.

    An "Escalation" section is boilerplate in every runbook; the only thing
    making RB-001's differ from RB-003's is the service name in the title.
    """
    doc = _doc("RB-900", "## Escalation\n\n" + "page the on-call engineer " * 4)
    chunk = split_document(doc)[0]

    assert chunk.text.startswith("RB-900 title\nEscalation")


def test_a_document_with_no_headings_still_produces_one_chunk():
    doc = _doc("RB-900", "Just prose, no headings at all. " * 4)
    chunks = split_document(doc)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "RB-900"


def test_a_very_short_document_is_never_dropped():
    """A document absent from the dense index is invisible to the dense arm."""
    doc = _doc("RB-900", "## H\n\ntiny")
    chunks = split_document(doc)

    assert len(chunks) == 1
    assert "tiny" in chunks[0].text


def test_repeated_headings_get_distinct_chunk_ids():
    doc = _doc("RB-900", "## Verifying\n\n" + "a" * 60 + "\n\n## Verifying\n\n" + "b" * 60)
    ids = [c.chunk_id for c in split_document(doc)]

    assert len(set(ids)) == len(ids) == 2


def test_sub_headings_stay_with_their_parent_section():
    doc = _doc("RB-900", "## Symptoms\n\n" + "a" * 60 + "\n\n### Detail\n\n" + "b" * 60)
    chunks = split_document(doc)

    assert len(chunks) == 1
    assert "Detail" in chunks[0].text


def test_every_document_in_the_corpus_is_represented(docs):
    chunks = chunk_corpus(docs)

    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}


# --------------------------------------------------------------------------
# Vector arithmetic.
#
# `l2_normalise` has its own test because getting it wrong is silent. Gemini
# normalises at 3072 dimensions, so a vector truncated to 768 is not unit
# length, and skipping the re-normalisation degrades every similarity score in
# the system without raising anything.
# --------------------------------------------------------------------------

def test_normalising_produces_a_unit_vector():
    assert math.isclose(sum(x * x for x in l2_normalise([3.0, 4.0])), 1.0, abs_tol=1e-9)


def test_a_truncated_vector_is_renormalised_to_unit_length():
    """The exact Gemini failure: truncate a unit vector and it is no longer one."""
    full = l2_normalise([0.5] * 16)
    truncated = full[:8]

    assert not math.isclose(math.sqrt(sum(x * x for x in truncated)), 1.0, abs_tol=1e-6)
    assert math.isclose(
        sum(x * x for x in l2_normalise(truncated)), 1.0, abs_tol=1e-9
    )


def test_normalising_a_zero_vector_does_not_divide_by_zero():
    assert l2_normalise([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_of_a_vector_with_itself_is_one():
    assert math.isclose(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, abs_tol=1e-9)


def test_mixing_two_embedding_spaces_raises_rather_than_returning_nonsense():
    """384-dim vectors and 768-dim vectors are not comparable, and saying so
    loudly is the only way that failure gets noticed."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine([1.0] * 384, [1.0] * 768)


def test_disabling_embeddings_refuses_rather_than_returning_zeros():
    """Zero vectors would silently turn hybrid into lexical, and the comparison
    between the two arms would report a difference of nothing."""
    with pytest.raises(EmbedderUnavailable):
        NullEmbedder().embed_query("anything")


# --------------------------------------------------------------------------
# RRF fusion.
# --------------------------------------------------------------------------

def _candidates(*ids: str) -> list[Candidate]:
    return [Candidate(doc=_doc(i, "text")) for i in ids]


def test_a_document_both_arms_rank_beats_one_only_a_single_arm_ranks():
    """The property that makes fusion worth doing: agreement is evidence."""
    lexical = _candidates("RB-001", "RB-002")
    dense = _candidates("RB-003", "RB-001")

    fused = rrf_fuse(lexical, dense)

    assert fused[0].doc_id == "RB-001"


def test_fusion_keeps_documents_only_one_arm_found():
    """This is the recall the dense arm was added for."""
    fused = rrf_fuse(_candidates("RB-001"), _candidates("RB-002"))

    assert {c.doc_id for c in fused} == {"RB-001", "RB-002"}


def test_fusion_combines_ranks_not_scores():
    """A huge BM25 score must not swamp the dense arm, because the two numbers
    are not on the same scale and no conversion between them is principled."""
    lexical = [Candidate(doc=_doc("RB-001", "t"), lexical_score=9999.0)]
    dense = [
        Candidate(doc=_doc("RB-002", "t"), dense_score=0.9),
        Candidate(doc=_doc("RB-001", "t"), dense_score=0.1),
    ]

    fused = rrf_fuse(lexical, dense)
    top_two = {c.doc_id for c in fused[:2]}

    assert top_two == {"RB-001", "RB-002"}
    # Rank 1 + rank 2 beats rank 1 alone, whatever the raw scores were.
    assert fused[0].doc_id == "RB-001"
    assert fused[0].fused_score > fused[1].fused_score


def test_fusion_preserves_the_original_scores_for_the_gate():
    lexical = [Candidate(doc=_doc("RB-001", "t"), lexical_score=4.2)]
    dense = [Candidate(doc=_doc("RB-001", "t"), dense_score=0.83)]

    fused = rrf_fuse(lexical, dense)

    assert fused[0].lexical_score == pytest.approx(4.2)
    assert fused[0].dense_score == pytest.approx(0.83)


def test_fusing_two_empty_lists_is_empty():
    assert rrf_fuse([], []) == []


# --------------------------------------------------------------------------
# The hybrid gate.
#
# Dense retrieval makes `no_match` *harder*: a vector search always returns its
# nearest neighbours, so there is no such thing as "nothing matched". These
# tests pin the mitigation.
# --------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return Retrieval()


def test_corpus_coverage_binds_even_when_the_dense_score_is_high(lexical_index, cfg, docs):
    """The load-bearing rule of the hybrid gate.

    A question whose words are largely absent from the corpus is rejected
    however confident the vector space looks, because coverage measures
    something the embedder cannot see - whether this corpus is about this
    subject at all.
    """
    spec = analyze_query("How many vacation days do engineers get?", docs)
    candidate = Candidate(doc=docs[0], dense_score=0.99, lexical_score=50.0)

    passed, why = passes_hybrid_gate(spec, [candidate], lexical_index, cfg)

    assert passed is False
    assert "content words" in why


def test_a_strong_dense_score_can_admit_a_question_lexical_would_reject(
    lexical_index, cfg, docs
):
    """The vocabulary-mismatch path this phase exists for."""
    spec = analyze_query("checkout-api deploy rollback previous version", docs)
    candidate = Candidate(doc=docs[0], lexical_score=0.0, dense_score=0.95)

    passed, why = passes_hybrid_gate(spec, [candidate], lexical_index, cfg)

    assert passed is True
    assert "vocabulary mismatch" in why


def test_neither_signal_clearing_its_floor_is_rejected(lexical_index, cfg, docs):
    spec = analyze_query("checkout-api deploy rollback previous version", docs)
    candidate = Candidate(doc=docs[0], lexical_score=0.0, dense_score=0.1)

    passed, why = passes_hybrid_gate(spec, [candidate], lexical_index, cfg)

    assert passed is False
    assert "neither signal" in why


def test_an_unknown_service_still_short_circuits_under_hybrid(lexical_index, cfg, docs):
    """Dense retrieval must not resurrect a question about a service we have no
    documents for - it will happily find the nearest one."""
    spec = analyze_query("How do I restart the recommendation-engine service?", docs)
    candidate = Candidate(doc=docs[0], dense_score=0.99)

    passed, _ = passes_hybrid_gate(spec, [candidate], lexical_index, cfg)

    assert passed is False


def test_an_empty_candidate_list_is_rejected(lexical_index, cfg, docs):
    spec = analyze_query("checkout-api is running hot on CPU", docs)

    passed, why = passes_hybrid_gate(spec, [], lexical_index, cfg)

    assert passed is False
    assert "metadata filter" in why


def test_best_cosine_of_nothing_is_zero():
    assert best_cosine([]) == 0.0


# --------------------------------------------------------------------------
# With real vectors.
# --------------------------------------------------------------------------

@embeddings
def test_the_dense_index_covers_every_document(dense_index, docs):
    assert set(dense_index.by_doc_id) == {d.doc_id for d in docs}


@embeddings
def test_a_document_is_scored_by_its_best_section_not_its_average(dense_index):
    """A question about escalation should be answered by the document whose
    escalation section matches, not penalised for its four other sections."""
    from agent.embed import get_embedder

    vector = get_embedder(load_settings("local")).embed_query(
        "who do I page when checkout-api is still broken after mitigation"
    )
    best = dense_index.best_by_document(vector)

    assert all(0.0 <= score <= 1.0 for score, _ in best.values())
    assert max(best.values())[0] > 0.5


@embeddings
def test_hybrid_rescues_the_vocabulary_mismatch_question(docs, lexical_index, dense_index):
    """The measurement that justifies this entire phase.

    Question 14 - "The checkout service is dragging its feet and the boxes are
    working too hard" - is a question about CPU containing no CPU vocabulary.
    It was written in Phase 1, before any retrieval code existed, precisely to
    fail under lexical-only retrieval.

    Lexical-only never puts RB-001 in the pack the model is shown, so the right
    document cannot be cited. Adding the dense arm puts it there. That is the
    whole claim, and this is it being checked rather than asserted.
    """
    from agent.embed import get_embedder

    cfg = load_settings("local")
    question = (
        "The checkout service is dragging its feet and the boxes are working too hard."
    )
    spec = analyze_query(question, docs)
    lexical = bm25_search(lexical_index, spec, cfg.retrieval.bm25_top_k)

    lexical_pack = metadata_filter(spec, lexical, cfg.retrieval.final_top_k)
    assert "RB-001" not in [c.doc_id for c in lexical_pack], (
        "If lexical-only now finds RB-001, this question no longer tests "
        "vocabulary mismatch and the dense arm needs re-justifying."
    )

    vector = get_embedder(cfg).embed_query(question)
    dense = dense_search(dense_index, vector, cfg.retrieval.dense_top_k)
    fused = rrf_fuse(lexical, dense, cfg.retrieval.rrf_k)
    hybrid_pack = metadata_filter(spec, fused, cfg.retrieval.final_top_k)

    assert "RB-001" in [c.doc_id for c in hybrid_pack]


@embeddings
def test_the_dense_arm_alone_ranks_the_right_document_first_for_q14(
    docs, dense_index
):
    """Stronger than the above, and worth recording separately: on this
    question the dense arm is not merely contributing, it is correct on its own
    while BM25 is not."""
    from agent.embed import get_embedder

    vector = get_embedder(load_settings("local")).embed_query(
        "The checkout service is dragging its feet and the boxes are working too hard."
    )
    ranked = dense_search(dense_index, vector, 8)

    assert ranked[0].doc_id == "RB-001"


@embeddings
def test_dense_scores_do_not_separate_answerable_from_unanswerable(
    docs, lexical_index, dense_index
):
    """A negative result, pinned so it is not quietly forgotten.

    The plan proposed an absolute cosine floor calibrated against known
    negatives. Measured on this corpus, the two distributions overlap:
    answerable questions reach 0.62-0.89 and unanswerable ones reach 0.58-0.73.
    No threshold separates them, which is the concrete form of "dense retrieval
    makes no_match harder, not easier".

    This is why the coverage check binds regardless of the dense score, and why
    the cosine floor is set above the highest negative rather than at a knee.
    If this ever starts failing, the floor can be re-derived - but silently
    trusting a threshold that never separated anything would be worse.
    """
    from agent.embed import get_embedder
    from eval.questions import ALL_QUESTIONS

    embedder = get_embedder(load_settings("local"))

    def top_cosine(question: str) -> float:
        return dense_search(dense_index, embedder.embed_query(question), 1)[0].dense_score

    answerable = [top_cosine(q.question) for q in ALL_QUESTIONS if not q.is_no_match]
    unanswerable = [top_cosine(q.question) for q in ALL_QUESTIONS if q.is_no_match]

    assert min(answerable) < max(unanswerable), (
        "The distributions no longer overlap. That would be good news, and it "
        "means the cosine floor can become a real discriminator - but the "
        "write-up currently reports the opposite, so update both together."
    )
