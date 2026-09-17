"""The contracts every pipeline stage talks through.

Each stage takes one of these types and returns another, which is what lets
every stage be tested on its own. Nothing here imports LangGraph, a network
client, or anything else heavy - see the rule in IMPLEMENTATION-PLAN.md about
what `agent/core/` is allowed to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The four confidence labels the brief requires. `no_match` lives in the same
# channel as the confidence score, which is why it is listed alongside them.
Confidence = str  # "high" | "medium" | "low" | "no_match"


@dataclass(frozen=True)
class Doc:
    """One runbook, as parsed from `runbooks/*.md` front-matter."""

    doc_id: str  # "RB-001"
    title: str
    service: str | None  # None for general/policy docs - see metadata_filter
    failure_mode: str | None
    doc_type: str  # "runbook" | "policy" | "postmortem"
    date: str | None  # postmortems only, YYYY-MM-DD
    text: str

    @property
    def is_general(self) -> bool:
        """True for docs that apply to every service.

        These must never be dropped for "mismatching" a service the question
        names - that rule is what makes the policy questions work.
        """
        return self.service is None


@dataclass(frozen=True)
class Chunk:
    """A `##` section of a Doc, embedded independently.

    Retrieval happens at chunk level for precision, but citations are always
    reported as the parent `doc_id` - the brief's contract is document IDs.
    """

    chunk_id: str  # "RB-001#symptoms"
    doc_id: str  # parent document
    section: str  # the `##` heading this came from
    text: str


@dataclass(frozen=True)
class QuerySpec:
    """The same structured fields the loader pulled off the documents, but
    extracted from the question - so the two can be compared like with like."""

    raw: str
    service: str | None
    failure_mode: str | None
    intent: str  # "diagnose" | "rollback" | "policy" | "postmortem"
    date: str | None

    # A service-shaped name the question mentions that the corpus has no
    # documents for - "search-api", say. This is positive evidence that we
    # cannot answer, as opposed to `service=None`, which merely means the
    # question did not name one. The two need to be distinguishable: the first
    # should short-circuit to no_match, the second should not.
    unknown_service: str | None = None


@dataclass
class Candidate:
    """A document under consideration, and the record of why.

    `verdict` and `reason` are not decoration. When a question fails we need to
    see *which stage* discarded the right document, rather than guessing.
    """

    doc: Doc
    lexical_score: float = 0.0
    dense_score: float = 0.0  # cosine similarity, 0.0 when running lexical-only
    fused_score: float = 0.0  # RRF output, 0.0 when running lexical-only
    verdict: str = "kept"  # "kept" | "dropped"
    reason: str = ""  # "service mismatch: payments-api != checkout-api"

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id

    def drop(self, reason: str) -> "Candidate":
        self.verdict = "dropped"
        self.reason = reason
        return self


@dataclass
class Answer:
    """What `answer_question()` serialises to a dict."""

    answer: str
    cited_doc_ids: list[str] = field(default_factory=list)
    confidence: Confidence = "no_match"

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "cited_doc_ids": list(self.cited_doc_ids),
            "confidence": self.confidence,
        }
