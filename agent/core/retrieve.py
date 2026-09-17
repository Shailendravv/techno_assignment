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


def normalised_top_score(candidates: list[Candidate], question: str) -> float:
    """BM25 score of the best candidate, per content term.

    Raw BM25 scores are not comparable between queries - a long question scores
    higher simply by having more terms to match. Dividing by the number of
    content terms makes the number mean "average match strength per meaningful
    word", which is comparable, and therefore something a single threshold can
    be set against.
    """
    if not candidates:
        return 0.0
    terms = content_terms(question)
    return candidates[0].lexical_score / max(len(terms), 1)


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

    score = normalised_top_score(candidates, spec.raw)
    if score < cfg.lexical_floor:
        return False, (
            f"best lexical score {score:.2f} per term is below the floor "
            f"{cfg.lexical_floor:.2f}"
        )

    return True, f"coverage {cov:.0%}, top score {score:.2f} per term"
