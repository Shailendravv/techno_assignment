"""The evaluation set. Written in Phase 1, before any retrieval code existed.

That ordering is deliberate. The corpus is self-authored, so there is a real
risk of unconsciously writing documents that our own retriever happens to be
good at. Freezing both the corpus and the questions before writing the retriever
is the only honest mitigation available.

The five GIVEN questions come from the brief. We will also be graded on
questions we have not seen, so the other fifteen exist to stop us tuning to
those five.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tags let the report break results down by *kind* of failure rather than
# reporting one undifferentiated accuracy number.
GIVEN = "given"                 # supplied in the brief
TRAP = "near-duplicate-trap"    # a wrong-service or wrong-failure twin exists
VOCAB = "vocabulary-mismatch"   # shares almost no tokens with its target doc
GENERAL = "general-doc"         # answered by a policy doc with no service field
NO_MATCH = "no-match"           # nothing in the corpus answers this


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    question: str
    expected_doc_ids: list[str]  # empty means: the correct answer is no_match
    acceptable_extra_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def is_no_match(self) -> bool:
        return not self.expected_doc_ids


# --------------------------------------------------------------------------
# The five from the brief.
# --------------------------------------------------------------------------
GIVEN_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id=1,
        question="checkout-api is running hot on CPU - what should I check first?",
        expected_doc_ids=["RB-001"],
        tags=[GIVEN, TRAP],
        note="RB-003 (payments/cpu) and RB-004 (checkout/memory) are the traps.",
    ),
    EvalQuestion(
        id=2,
        question=(
            "We are seeing 'too many connections' errors on checkout-api - "
            "what is the likely cause?"
        ),
        expected_doc_ids=["RB-002"],
        acceptable_extra_ids=["RB-012"],
        tags=[GIVEN, TRAP],
        note=(
            "The brief says citing RB-012 as well is good but not required, so "
            "it must not be scored as noise. RB-007 (inventory) is the trap."
        ),
    ),
    EvalQuestion(
        id=3,
        question="How do I safely roll back checkout-api to a previous version?",
        expected_doc_ids=["RB-005"],
        acceptable_extra_ids=["RB-010"],
        tags=[GIVEN, TRAP],
        note="RB-006 (payments rollback) is a 77% token match. The hardest trap.",
    ),
    EvalQuestion(
        id=4,
        question=(
            "We had a checkout-api incident on 2026-08-10 - what was the root "
            "cause and how was it fixed?"
        ),
        expected_doc_ids=["RB-012"],
        tags=[GIVEN, TRAP],
        note=(
            "RB-002 is the same service and the same failure mode. Only "
            "doc_type and the date separate them."
        ),
    ),
    EvalQuestion(
        id=5,
        question="What is our policy for communicating an incident to customers?",
        expected_doc_ids=["RB-011"],
        tags=[GIVEN, GENERAL],
        note=(
            "RB-011 has service=None. It must survive the filter even though "
            "the question names no service at all."
        ),
    ),
]

# --------------------------------------------------------------------------
# Held-out: answerable. Written blind, before seeing any score.
# --------------------------------------------------------------------------
HELD_OUT_ANSWERABLE: list[EvalQuestion] = [
    EvalQuestion(
        id=6,
        question="payments-api CPU is pegged above 85%. Where do I start?",
        expected_doc_ids=["RB-003"],
        tags=[TRAP],
        note="The mirror of Q1. Checks the filter discriminates in both directions.",
    ),
    EvalQuestion(
        id=7,
        question=(
            "checkout-api containers keep restarting with exit code 137. "
            "What is going on?"
        ),
        expected_doc_ids=["RB-004"],
        tags=[TRAP],
        note="Same service as RB-001, different failure mode.",
    ),
    EvalQuestion(
        id=8,
        question="What is the rollback procedure for payments-api?",
        expected_doc_ids=["RB-006"],
        acceptable_extra_ids=["RB-010"],
        tags=[TRAP],
    ),
    EvalQuestion(
        id=9,
        question=(
            "inventory-api is refusing connections with 'too many clients "
            "already'. What should I do?"
        ),
        expected_doc_ids=["RB-007"],
        tags=[TRAP],
        note="RB-002 is a 73% token match on the wrong service.",
    ),
    EvalQuestion(
        id=10,
        question="The inventory sync lag alert is firing. How do I stop overselling?",
        expected_doc_ids=["RB-008"],
    ),
    EvalQuestion(
        id=11,
        question=(
            "The primary on-call has not acknowledged a page. How long before "
            "it escalates, and to whom?"
        ),
        expected_doc_ids=["RB-009"],
        tags=[GENERAL],
    ),
    EvalQuestion(
        id=12,
        question="When are we allowed to deploy to production?",
        expected_doc_ids=["RB-010"],
        tags=[GENERAL],
    ),
    EvalQuestion(
        id=13,
        question="What is expand-and-contract and why do migrations need two releases?",
        expected_doc_ids=["RB-010"],
        acceptable_extra_ids=["RB-005", "RB-006"],
        tags=[GENERAL],
        note="The rollback runbooks discuss it too, so citing them as well is fair.",
    ),
    EvalQuestion(
        id=14,
        question="The checkout service is dragging its feet and the boxes are working too hard.",
        expected_doc_ids=["RB-001"],
        tags=[VOCAB],
        note=(
            "Deliberate vocabulary mismatch: no 'CPU', no 'latency', no "
            "'high'. Expected to FAIL under lexical-only retrieval and to be "
            "rescued by the dense arm in Phase 5. This question is the entire "
            "justification for adding embeddings."
        ),
    ),
    EvalQuestion(
        id=15,
        question=(
            "We are taking orders for things that are not actually sitting in "
            "the warehouse. What is the fix?"
        ),
        expected_doc_ids=["RB-008"],
        tags=[VOCAB],
        note=(
            "Second vocabulary mismatch: describes oversell without using "
            "'sync', 'lag', or 'Kafka'."
        ),
    ),
]

# --------------------------------------------------------------------------
# Held-out: the correct answer is no_match.
#
# The five given questions all have answers, so on its own that set cannot
# detect an agent that never says "I do not know" - which is the single failure
# the brief warns about most directly.
# --------------------------------------------------------------------------
HELD_OUT_NO_MATCH: list[EvalQuestion] = [
    EvalQuestion(
        id=16,
        question="What is our refund policy for orders over $500?",
        expected_doc_ids=[],
        tags=[NO_MATCH],
        note=(
            "The word 'refund' appears nowhere in the corpus. RB-011 is the "
            "trap: it matches on 'policy' and 'customer' and is exactly the "
            "closest-sounding document the brief tells us not to cite."
        ),
    ),
    EvalQuestion(
        id=17,
        question="How do I restart the recommendation-engine service?",
        expected_doc_ids=[],
        tags=[NO_MATCH],
        note="A service that does not exist in the corpus.",
    ),
    EvalQuestion(
        id=18,
        question="What is the runbook for search-api high latency?",
        expected_doc_ids=[],
        tags=[NO_MATCH],
        note=(
            "The nastiest one. It is phrased exactly like a question that "
            "should work, and every token except 'search-api' appears "
            "throughout the corpus."
        ),
    ),
    EvalQuestion(
        id=19,
        question="How many vacation days do engineers get?",
        expected_doc_ids=[],
        tags=[NO_MATCH],
        note="Entirely off-domain. Should be the easiest no_match to get right.",
    ),
    EvalQuestion(
        id=20,
        question="What is our process for renewing SSL certificates before they expire?",
        expected_doc_ids=[],
        tags=[NO_MATCH],
        note=(
            "Plausibly operational, and both 'process' and 'expire' appear in "
            "the corpus, but nothing documents certificate renewal."
        ),
    ),
]

ALL_QUESTIONS: list[EvalQuestion] = (
    GIVEN_QUESTIONS + HELD_OUT_ANSWERABLE + HELD_OUT_NO_MATCH
)


def by_tag(tag: str) -> list[EvalQuestion]:
    return [q for q in ALL_QUESTIONS if tag in q.tags]
