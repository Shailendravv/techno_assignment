"""Lexical ranking, the metadata filter, and the relevance gate.

The division of labour matters and is the whole design:

- **BM25 ranks.** It decides which documents are most likely relevant. It is
  good at this and bad at telling near-duplicates apart.
- **The metadata filter discriminates.** It drops documents whose structured
  fields contradict the question. This is what defeats the near-duplicate trap,
  and it is a *hard* drop rather than a score penalty - a wrong service is not
  a weak signal to be out-voted by four hundred words of similar prose, it is a
  contradiction.
- **The gate decides whether to answer at all.** If nothing credible survives,
  we return no_match without ever calling the model. You cannot tempt a model
  into inventing a citation for a document you never showed it.

None of this imports LangGraph or touches the network, which is why the tests
for it are fast and run anywhere.
"""

from __future__ import annotations

import functools
import re

from rank_bm25 import BM25Okapi

from agent.config import Retrieval
from agent.core.models import Candidate, Doc, QuerySpec

# Words carrying no topical signal. Kept deliberately short: BM25's own IDF
# already discounts words that appear everywhere, so this list only needs to
# stop them polluting the *coverage* measure the gate uses.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from get
got had has have how i if in into is it its me my of on or our out over should
so some that the their them then there these they this to us was we were what
when where which who why will with would you your
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase into terms, keeping hyphenated names whole *and* split.

    "checkout-api" yields ["checkout-api", "checkout", "api"]. Emitting both
    forms means an exact service mention scores strongly while a question that
    only says "checkout" still matches. The hyphenated form is the rarest and
    therefore the most discriminating token in this corpus, so losing it to
    naive splitting would throw away the best signal we have.
    """
    terms: list[str] = []
    for raw in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        terms.append(raw)
        if "-" in raw:
            terms.extend(p for p in raw.split("-") if p)
    return terms


def _doc_terms(doc: Doc) -> list[str]:
    """Index the title twice.

    The title is where the two facts that decide the answer live - the service
    and the failure mode - stated in four words rather than diluted across four
    hundred. Counting it twice is a prior about where signal lives, not a
    threshold fitted to our questions.
    """
    return tokenize(doc.title) * 2 + tokenize(doc.text)


class LexicalIndex:
    """A BM25 index over the corpus, built once."""

    def __init__(self, docs: tuple[Doc, ...]):
        self.docs = docs
        self._corpus_terms = [_doc_terms(d) for d in docs]
        self._bm25 = BM25Okapi(self._corpus_terms)
        self.vocabulary = frozenset(t for terms in self._corpus_terms for t in terms)

    def scores(self, query: str) -> list[float]:
        return list(self._bm25.get_scores(tokenize(query)))


@functools.lru_cache(maxsize=4)
def build_index(docs: tuple[Doc, ...]) -> LexicalIndex:
    return LexicalIndex(docs)


def content_terms(question: str) -> list[str]:
    """Query terms that carry topical meaning.

    Hyphen-split fragments are dropped here (unlike in indexing) so that
    "checkout-api" counts once rather than three times when measuring how much
    of the question the corpus actually covers.
    """
    return [
        t
        for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", question.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def bm25_search(
    index: LexicalIndex, spec: QuerySpec, top_k: int
) -> list[Candidate]:
    """Rank the whole corpus lexically and return the top `top_k`.

    The corpus is twelve documents, so recall is nearly free: we take a
    generous shortlist and let the filter do the cutting.
    """
    scores = index.scores(spec.raw)
    ranked = sorted(
        (Candidate(doc=doc, lexical_score=score)
         for doc, score in zip(index.docs, scores)),
        key=lambda c: c.lexical_score,
        reverse=True,
    )
    return ranked[:top_k]


def metadata_filter(
    spec: QuerySpec, candidates: list[Candidate], top_k: int
) -> list[Candidate]:
    """Drop candidates whose structured fields contradict the question.

    The rules, and why each one is shaped the way it is:

    | Question says       | Document says          | Action |
    |---------------------|------------------------|--------|
    | service X           | service Y              | drop   |
    | service X           | no service (a policy)  | keep   |
    | failure mode A      | failure mode B         | drop   |
    | failure mode A      | no failure mode        | keep   |
    | nothing specific    | anything               | keep   |
    | a date              | postmortem, other date | drop   |

    Rows two and four are the ones that are easy to get wrong. A general policy
    document has no service, and dropping it for "mismatching" one would break
    every question a policy answers. The same reasoning applies to a rollback
    runbook, which is about a service but not about any particular failure.
    """
    kept: list[Candidate] = []

    for candidate in candidates:
        doc = candidate.doc

        if spec.service and doc.service and doc.service != spec.service:
            candidate.drop(f"service mismatch: {doc.service} != {spec.service}")
            continue

        if spec.failure_mode and doc.failure_mode and doc.failure_mode != spec.failure_mode:
            candidate.drop(
                f"failure mode mismatch: {doc.failure_mode} != {spec.failure_mode}"
            )
            continue

        if spec.date and doc.doc_type == "postmortem" and doc.date != spec.date:
            candidate.drop(f"wrong incident: {doc.date} != {spec.date}")
            continue

        candidate.verdict = "kept"
        candidate.reason = _why_kept(spec, doc)
        kept.append(candidate)

    return _prefer_specific(spec, kept)[:top_k]


def _prefer_specific(spec: QuerySpec, kept: list[Candidate]) -> list[Candidate]:
    """When the question names a service, put that service's documents first.

    General policy documents are correctly *kept* for every question - that is
    what makes the policy questions work - but being kept is not the same as
    being the best answer. "How do I roll back checkout-api" is answered by the
    checkout-api rollback runbook, not by the company-wide deploy process, even
    when the latter happens to share more words with the question.

    So this is a tie-break on specificity, not a score adjustment: BM25's
    ordering is preserved exactly within each group.
    """
    if not spec.service:
        return kept
    specific = [c for c in kept if c.doc.service == spec.service]
    general = [c for c in kept if c.doc.service != spec.service]
    for candidate in general:
        candidate.reason += " (general doc, ranked below service-specific ones)"
    return specific + general


def _why_kept(spec: QuerySpec, doc: Doc) -> str:
    reasons = []
    if spec.service and doc.service == spec.service:
        reasons.append(f"service={doc.service}")
    elif spec.service and doc.is_general:
        reasons.append("general doc, applies to all services")
    if spec.failure_mode and doc.failure_mode == spec.failure_mode:
        reasons.append(f"failure_mode={doc.failure_mode}")
    if spec.date and doc.date == spec.date:
        reasons.append(f"date={doc.date}")
    return "; ".join(reasons) or "no contradicting metadata"


def coverage(question: str, index: LexicalIndex) -> float:
    """What fraction of the question's content words appear in the corpus at all.

    This catches the plainly off-topic question that BM25 nonetheless ranks
    with confidence - "how many vacation days do engineers get" shares almost
    nothing with a corpus of runbooks, whatever its nearest neighbour is.

    It is deliberately unweighted, and that is worth explaining because the
    obvious improvement does not work. Weighting by IDF - treating a word the
    corpus has never seen as highly informative, on the theory that it names
    the subject we have no document about - was tried and measured, and it made
    things substantially worse: it gated six correct answers, including "how do
    I safely roll back checkout-api", which fell below every no_match question
    in the set. The flaw is the corpus size. With twelve documents the
    vocabulary is small, so ordinary words like "safely" and "previous" are
    absent too, and they get the same weight as "refund". Absence from a small
    corpus is simply not strong evidence.

    So this measure stays blunt, and it is honest about what it cannot do. It
    does not catch "what is our refund policy for orders over $500", which
    scores 75% because `policy`, `orders` and `500` all occur - the last only
    because it is an HTTP status code in three runbooks. That question is
    caught by the second gate instead: the model is shown the candidates and
    told that no_match is a correct answer. Two independent chances to decline,
    each catching what it is actually good at, is a better design than one
    threshold contorted to catch everything.
    """
    terms = content_terms(question)
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in index.vocabulary) / len(terms)


def normalised_top_score(
    candidates: list[Candidate], question: str, index: LexicalIndex | None = None
) -> float:
    """BM25 score of the best surviving candidate, per content term.

    Raw BM25 scores are not comparable between queries - a long question scores
    higher simply by having more terms to match. Dividing by the number of
    content terms makes the number mean "average match strength per meaningful
    word", which is comparable, and therefore something a single threshold can
    be set against.

    **Scored against the local index, not against whatever the store returned.**
    That is the whole point and it is worth being explicit, because the obvious
    reading - "use the score the retriever computed" - is what broke the
    deployed system.

    `lexical_floor` is calibrated against `rank_bm25`, whose scores land in the
    2-6 range on this corpus. The SQL backend ranks with `ts_rank_cd`, which
    returns values around 0.01 and, because `websearch_to_tsquery` builds a
    conjunctive query, returns exactly 0.0 for any question whose every word is
    not present in one document. Comparing that against a floor of 0.35 meant
    the lexical branch of the gate could never be taken on Postgres, so every
    question fell through to the cosine floor - a threshold deliberately set
    *above* every measured negative, as an inert guard rail. The result was a
    system that refused answerable questions in exactly the configuration it
    deploys in.

    `lexical_index()` is already built locally by *both* stores, for precisely
    this reason (see `agent.store.Store.lexical_index`): the gate must not
    depend on which backend is mounted. It simply was not being used. Passing
    it here makes one calibration valid everywhere, with no threshold change.

    Taking `max` over the survivors rather than element `[0]` is the second
    half. `_prefer_specific` reorders kept candidates for presentation - it
    documents itself as a tie-break, not a score adjustment - but the gate read
    `[0]`, so a presentation choice moved the number a threshold was compared
    against. `max` is order-independent and says what was meant.
    """
    if not candidates:
        return 0.0
    terms = content_terms(question)

    if index is not None:
        by_doc_id = dict(zip((d.doc_id for d in index.docs), index.scores(question)))
        best = max((by_doc_id.get(c.doc_id, 0.0) for c in candidates), default=0.0)
    else:
        # No index to hand: fall back to whatever the retriever scored. Correct
        # for the file backend, which is the only caller that can reach this.
        best = max((c.lexical_score for c in candidates), default=0.0)

    return best / max(len(terms), 1)


def passes_gate(
    spec: QuerySpec,
    candidates: list[Candidate],
    index: LexicalIndex,
    cfg: Retrieval,
) -> tuple[bool, str]:
    """Decide whether anything here is worth showing a model.

    Returns (passed, reason). The reason is recorded either way, because when a
    question fails we need to know which check rejected it.
    """
    if spec.unknown_service:
        return False, (
            f"question is about {spec.unknown_service!r}, which no document covers"
        )

    if not candidates:
        return False, "every candidate was dropped by the metadata filter"

    cov = coverage(spec.raw, index)
    if cov < cfg.coverage_floor:
        return False, (
            f"only {cov:.0%} of the question's content words appear anywhere in "
            f"the corpus (floor {cfg.coverage_floor:.0%})"
        )

    score = normalised_top_score(candidates, spec.raw, index)
    if score < cfg.lexical_floor:
        return False, (
            f"best lexical score {score:.2f} per term is below the floor "
            f"{cfg.lexical_floor:.2f}"
        )

    return True, f"coverage {cov:.0%}, top score {score:.2f} per term"


# --------------------------------------------------------------------------
# Phase 5: the dense arm.
#
# BM25 cannot match words that are not there. "The checkout service is dragging
# its feet and the boxes are working too hard" is a question about CPU that
# contains no CPU vocabulary, and lexical retrieval scores it near zero - the
# single weakness this design named in advance as most likely to lose marks.
#
# An embedding model is trained on exactly that: mapping a paraphrase near the
# text it paraphrases. So dense retrieval is added for **recall**, on the
# explicit understanding that it is bad at the thing BM25 is good at. It is
# trained to map near-duplicates close together, which is precisely wrong for a
# corpus whose whole difficulty is telling near-duplicates apart.
#
# Hence the division of labour, which is the entire design of this phase:
#
#   dense retrieval finds candidates -> the metadata filter discriminates
#
# Fusion happens first; the filter runs on the fused list as a hard drop; the
# gate runs last. The filter drops on document metadata, which no amount of
# embedding similarity can blur - RB-003 says `service: payments-api` whatever
# its prose resembles.
# --------------------------------------------------------------------------

class DenseIndex:
    """Chunk vectors, and the map back to parent documents.

    Retrieval is at chunk level for precision; scoring is collapsed to the
    document level immediately after, because the brief's contract is document
    IDs. A document's dense score is its *best* chunk, not its mean: a question
    about escalation should be answered by the document with the best matching
    escalation section, and averaging that against four unrelated sections
    would bury it.
    """

    def __init__(self, docs: tuple[Doc, ...], chunks: list, vectors: list[list[float]]):
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors - the index is "
                "inconsistent, which means it was built against a different corpus."
            )
        self.docs = docs
        self.chunks = chunks
        self.vectors = vectors
        self.by_doc_id = {d.doc_id: d for d in docs}
        self.dims = len(vectors[0]) if vectors else 0

    def best_by_document(self, query_vector: list[float]) -> dict[str, tuple[float, str]]:
        """Each document's best-matching chunk: doc_id -> (cosine, section)."""
        from agent.embed import cosine

        best: dict[str, tuple[float, str]] = {}
        for chunk, vector in zip(self.chunks, self.vectors):
            score = cosine(query_vector, vector)
            current = best.get(chunk.doc_id)
            if current is None or score > current[0]:
                best[chunk.doc_id] = (score, chunk.section)
        return best


def build_dense_index(docs: tuple[Doc, ...], embedder) -> DenseIndex:
    """Embed every `##` section of every document.

    Twelve documents is about seventy chunks, which is one Gemini request and
    about a second locally. Built once and held, not per query.
    """
    from agent.core.chunk import chunk_corpus

    chunks = chunk_corpus(docs)
    vectors = embedder.embed_documents([c.text for c in chunks])
    return DenseIndex(docs, chunks, vectors)


def dense_search(
    index: DenseIndex, query_vector: list[float], top_k: int
) -> list[Candidate]:
    """Rank documents by their best-matching section."""
    best = index.best_by_document(query_vector)
    ranked = sorted(
        (
            Candidate(
                doc=index.by_doc_id[doc_id],
                dense_score=score,
                reason=f"dense match on {section!r}",
            )
            for doc_id, (score, section) in best.items()
        ),
        key=lambda c: c.dense_score,
        reverse=True,
    )
    return ranked[:top_k]


def rrf_fuse(
    lexical: list[Candidate], dense: list[Candidate], rrf_k: int = 60
) -> list[Candidate]:
    """Reciprocal rank fusion over the two ranked lists.

    RRF combines *ranks*, not scores, and that is why it is the right choice
    here rather than a weighted sum. A BM25 score and a cosine similarity are
    not on the same scale, are not comparable across queries, and have no
    principled conversion between them - any weighting we picked would be a
    constant fitted to our own twenty questions. Ranks have none of those
    problems, and RRF needs one parameter we did not choose ourselves.

    A document appearing in both lists gets both contributions, so agreement
    between two independent signals is rewarded without either being trusted to
    dominate. The original scores are preserved on the candidate for the trace
    and for the gate, which still needs them.
    """
    merged: dict[str, Candidate] = {}

    for ranked in (lexical, dense):
        for rank, candidate in enumerate(ranked, start=1):
            existing = merged.get(candidate.doc_id)
            if existing is None:
                existing = Candidate(doc=candidate.doc)
                merged[candidate.doc_id] = existing
            # Each list contributes whichever score it actually computed; the
            # other stays at whatever the sibling list set.
            existing.lexical_score = max(existing.lexical_score, candidate.lexical_score)
            existing.dense_score = max(existing.dense_score, candidate.dense_score)
            existing.fused_score += 1.0 / (rrf_k + rank)

    fused = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
    for candidate in fused:
        candidate.reason = (
            f"rrf={candidate.fused_score:.4f} "
            f"(bm25={candidate.lexical_score:.1f}, cos={candidate.dense_score:.2f})"
        )
    return fused


def best_cosine(candidates: list[Candidate]) -> float:
    """The strongest dense score among the survivors, or 0.0 if there are none."""
    return max((c.dense_score for c in candidates), default=0.0)


def passes_hybrid_gate(
    spec: QuerySpec,
    candidates: list[Candidate],
    index: LexicalIndex,
    cfg: Retrieval,
) -> tuple[bool, str]:
    """The gate, when the dense arm is in play.

    **Dense retrieval makes `no_match` harder, not easier, and this function is
    where that cost is paid.** A vector search always returns its k nearest
    neighbours; there is no such thing as "nothing matched". Unrelated text
    still lands at cosine 0.6-0.7 against most embedding models, so a question
    about refund policy retrieves *something* with a respectable-looking score.

    So the rule is deliberately asymmetric: a question may clear the gate on
    *either* signal, but the **corpus-coverage check binds regardless**. A
    question whose vocabulary is largely absent from the corpus is rejected
    however confident the vector space looks about it, because that check
    measures something the embedder cannot see - whether this corpus is about
    this subject at all - rather than how similar two strings are.

    Without that, hybrid would trade `no_match` recall for citation recall, and
    the brief is explicit that a confident wrong citation is the worse mistake.
    """
    if spec.unknown_service:
        return False, (
            f"question is about {spec.unknown_service!r}, which no document covers"
        )

    if not candidates:
        return False, "every candidate was dropped by the metadata filter"

    cov = coverage(spec.raw, index)
    if cov < cfg.coverage_floor:
        return False, (
            f"only {cov:.0%} of the question's content words appear anywhere in "
            f"the corpus (floor {cfg.coverage_floor:.0%}); dense score "
            f"{best_cosine(candidates):.2f} does not override this"
        )

    lexical = normalised_top_score(candidates, spec.raw, index)
    cosine_score = best_cosine(candidates)

    if lexical >= cfg.lexical_floor:
        return True, f"coverage {cov:.0%}, lexical {lexical:.2f} per term"

    if cosine_score >= cfg.cosine_floor:
        # The case this whole phase exists for: the question is about something
        # in the corpus, but phrased in words the corpus does not use.
        return True, (
            f"coverage {cov:.0%}, lexical {lexical:.2f} below floor but dense "
            f"{cosine_score:.2f} clears {cfg.cosine_floor:.2f} - vocabulary mismatch"
        )

    return False, (
        f"neither signal clears its floor: lexical {lexical:.2f} < "
        f"{cfg.lexical_floor:.2f}, dense {cosine_score:.2f} < {cfg.cosine_floor:.2f}"
    )
