"""Phase 2: the metadata filter, which is the component that wins the exercise.

No network, no LLM, no graph. Every assertion here runs in milliseconds and
would still pass on a machine with no API keys and no internet, which is the
point: the part of the system that decides between two near-identical documents
should not need a model to do it.

If exactly one test file in this repository deserves attention, it is this one.
"""

from __future__ import annotations

import pytest

from agent.config import Retrieval
from agent.core.corpus import by_id, load_corpus
from agent.core.models import Candidate, QuerySpec
from agent.core.query import analyze_query
from agent.core.retrieve import (
    bm25_search,
    build_index,
    coverage,
    metadata_filter,
    passes_gate,
    tokenize,
)


@pytest.fixture(scope="module")
def docs():
    return load_corpus("runbooks")


@pytest.fixture(scope="module")
def index(docs):
    return build_index(docs)


@pytest.fixture(scope="module")
def lookup(docs):
    return by_id(docs)


@pytest.fixture
def cfg():
    return Retrieval()


def candidates_for(lookup, *doc_ids) -> list[Candidate]:
    """Build a candidate list directly, so filter tests do not depend on BM25."""
    return [Candidate(doc=lookup[d], lexical_score=1.0) for d in doc_ids]


# --------------------------------------------------------------------------
# The near-duplicate trap. This is the whole exercise in four tests.
# --------------------------------------------------------------------------

def test_wrong_service_is_dropped(lookup):
    """RB-003 is a 69% token match for RB-001. Only the service differs, and
    that difference has to be decisive rather than merely influential."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    kept = metadata_filter(spec, candidates_for(lookup, "RB-001", "RB-003"), 4)

    assert [c.doc_id for c in kept] == ["RB-001"]


def test_wrong_failure_mode_is_dropped(lookup):
    """RB-004 is the same service as RB-001. Only the failure mode differs."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    kept = metadata_filter(spec, candidates_for(lookup, "RB-001", "RB-004"), 4)

    assert [c.doc_id for c in kept] == ["RB-001"]


def test_the_q1_trap_in_full(lookup):
    """Both traps at once, which is the actual shape of question 1."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    kept = metadata_filter(
        spec, candidates_for(lookup, "RB-001", "RB-003", "RB-004"), 4
    )

    assert [c.doc_id for c in kept] == ["RB-001"]


def test_a_dropped_candidate_records_which_field_contradicted(lookup):
    """When a question fails we need to see which stage discarded the right
    document, and why, rather than reconstructing it."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    cands = candidates_for(lookup, "RB-003", "RB-004")
    metadata_filter(spec, cands, 4)

    by_doc = {c.doc_id: c for c in cands}
    assert by_doc["RB-003"].verdict == "dropped"
    assert "service mismatch" in by_doc["RB-003"].reason
    assert "payments-api" in by_doc["RB-003"].reason
    assert "failure mode mismatch" in by_doc["RB-004"].reason


# --------------------------------------------------------------------------
# The rule that is easiest to get wrong: general documents.
# --------------------------------------------------------------------------

def test_general_policy_docs_survive_a_service_specific_question(lookup):
    """A policy doc has service=None. Dropping it for 'mismatching' a named
    service would break every question a policy answers - including question 5."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    kept = metadata_filter(
        spec, candidates_for(lookup, "RB-001", "RB-009", "RB-011"), 4
    )

    assert [c.doc_id for c in kept] == ["RB-001", "RB-009", "RB-011"]


def test_docs_without_a_failure_mode_survive_a_failure_specific_question(lookup):
    """RB-005 is about checkout-api but about no particular failure. The same
    reasoning as general policy docs applies."""
    spec = QuerySpec("checkout-api high cpu", "checkout-api", "cpu", "diagnose", None)
    kept = metadata_filter(spec, candidates_for(lookup, "RB-001", "RB-005"), 4)

    assert "RB-005" in [c.doc_id for c in kept]


def test_service_specific_docs_outrank_general_ones(lookup):
    """Being kept is not the same as being the best answer. 'How do I roll back
    checkout-api' is answered by the checkout-api runbook, not by the
    company-wide deploy process - even when the latter shares more words."""
    spec = QuerySpec("roll back checkout-api", "checkout-api", None, "rollback", None)
    # RB-010 deliberately supplied first, with the higher lexical score.
    cands = [
        Candidate(doc=lookup["RB-010"], lexical_score=9.0),
        Candidate(doc=lookup["RB-005"], lexical_score=5.0),
    ]
    kept = metadata_filter(spec, cands, 4)

    assert [c.doc_id for c in kept] == ["RB-005", "RB-010"]


def test_a_question_naming_nothing_specific_drops_nothing(lookup):
    spec = QuerySpec("what is our policy", None, None, "policy", None)
    kept = metadata_filter(
        spec, candidates_for(lookup, "RB-001", "RB-003", "RB-011"), 8
    )

    assert len(kept) == 3


# --------------------------------------------------------------------------
# Dates: what separates the postmortem from the runbook it shares a service
# and a failure mode with.
# --------------------------------------------------------------------------

def test_a_postmortem_with_the_wrong_date_is_dropped(lookup):
    spec = QuerySpec(
        "checkout-api incident on 2025-01-01", "checkout-api", None,
        "postmortem", "2025-01-01",
    )
    kept = metadata_filter(spec, candidates_for(lookup, "RB-012"), 4)

    assert kept == []


def test_the_matching_postmortem_survives(lookup):
    spec = QuerySpec(
        "checkout-api incident on 2026-08-10", "checkout-api", None,
        "postmortem", "2026-08-10",
    )
    kept = metadata_filter(spec, candidates_for(lookup, "RB-012", "RB-002"), 4)

    assert "RB-012" in [c.doc_id for c in kept]


# --------------------------------------------------------------------------
# The gate. no_match is a designed outcome, not a fallback.
# --------------------------------------------------------------------------

def test_an_unknown_service_short_circuits_immediately(docs, index, cfg):
    """'search-api' is service-shaped and we document no such service. That is
    positive evidence we cannot answer, and it is different from the question
    simply not naming a service."""
    spec = analyze_query("What is the runbook for search-api high latency?", docs)
    assert spec.unknown_service == "search-api"

    passed, why = passes_gate(spec, [], index, cfg)
    assert not passed
    assert "search-api" in why


def test_the_gate_rejects_an_off_topic_question(docs, index, cfg):
    """The canonical example from the brief: nothing in the corpus covers this,
    so we must not cite the closest-sounding document."""
    question = "How many vacation days do engineers get?"
    spec = analyze_query(question, docs)
    kept = metadata_filter(spec, bm25_search(index, spec, 8), 4)

    passed, why = passes_gate(spec, kept, index, cfg)
    assert not passed, f"expected the gate to reject this, but it passed: {why}"


def test_the_gate_rejects_an_empty_candidate_list(docs, index, cfg):
    spec = analyze_query("checkout-api high cpu", docs)
    passed, why = passes_gate(spec, [], index, cfg)

    assert not passed
    assert "filter" in why


def test_the_gate_admits_a_question_the_corpus_answers(docs, index, cfg):
    spec = analyze_query("checkout-api is running hot on CPU", docs)
    kept = metadata_filter(spec, bm25_search(index, spec, 8), 4)

    passed, _ = passes_gate(spec, kept, index, cfg)
    assert passed


def test_coverage_is_lower_for_off_topic_questions(docs, index):
    on_topic = coverage("checkout-api is running hot on CPU", index)
    off_topic = coverage("How many vacation days do engineers get?", index)

    assert on_topic > off_topic


def test_the_gate_records_a_reason_either_way(docs, index, cfg):
    """Both outcomes are explained, because a silent pass is as hard to debug
    as a silent rejection."""
    for question in ("checkout-api high cpu", "how many vacation days"):
        spec = analyze_query(question, docs)
        kept = metadata_filter(spec, bm25_search(index, spec, 8), 4)
        _, why = passes_gate(spec, kept, index, cfg)
        assert why


# --------------------------------------------------------------------------
# Tokenization, which is what gives BM25 its discriminating power here.
# --------------------------------------------------------------------------

def test_hyphenated_names_are_kept_whole_and_split():
    """The whole form is the rarest and most discriminating token in the
    corpus; the split form is what lets a bare 'checkout' still match."""
    terms = tokenize("checkout-api is hot")

    assert "checkout-api" in terms
    assert "checkout" in terms
    assert "api" in terms


def test_bm25_ranks_the_right_document_first_for_the_given_questions(docs, index):
    """End to end through retrieval only, on the five questions from the brief.

    Q14 and Q16 are deliberately excluded: Q14 is the vocabulary-mismatch case
    that lexical retrieval is expected to fail, and Q16 is expected to reach
    the model, which is the second gate.
    """
    from eval.questions import GIVEN_QUESTIONS

    cfg = Retrieval()
    for q in GIVEN_QUESTIONS:
        spec = analyze_query(q.question, docs)
        kept = metadata_filter(spec, bm25_search(index, spec, cfg.bm25_top_k), cfg.final_top_k)
        passed, why = passes_gate(spec, kept, index, cfg)

        assert passed, f"Q{q.id} was gated: {why}"
        assert kept[0].doc_id in q.expected_doc_ids, (
            f"Q{q.id} ranked {kept[0].doc_id} first, expected one of "
            f"{q.expected_doc_ids}. Order was {[c.doc_id for c in kept]}"
        )
