"""What flows between the graph's nodes.

LangGraph merges each node's returned dict into this state. Two fields need to
*accumulate* across nodes rather than be overwritten, so they carry reducers:

- `trace` appends, giving a readable record of what each stage decided.
- `llm_calls` sums, so we can assert that a `no_match` question cost zero calls.
  That assertion matters: on a rate-limited free tier, and because a model that
  is never shown a document cannot invent a citation for one.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agent.core.models import Candidate, Confidence, QuerySpec


class AgentState(TypedDict, total=False):
    # Input
    question: str
    model_role: str  # "generator" (default) or "reasoner"

    # Carried through state rather than read from the module, so a test or the
    # harness can run the graph with different thresholds without mutating
    # global configuration.
    settings: object  # agent.config.Settings

    # The text retrieval actually searches for. Equal to `question` until the
    # corrective loop rewrites it. The original is kept separately because the
    # grounding model must be shown what the user asked, not our paraphrase.
    search_query: str

    # Set by the analyze node
    spec: QuerySpec

    # Set by the retrieve node; narrowed by the metadata filter
    candidates: list[Candidate]
    gate_passed: bool

    # Set by the ground node
    raw_answer: str
    raw_cited_ids: list[str]

    # Set by the grade node: the candidates the relevance grader kept. Distinct
    # from `candidates` so the trace can show what each stage discarded.
    graded: list[Candidate]

    # How many rewrites the corrective loop has spent. Read by the edge that
    # bounds the cycle - which is the only thing stopping it running forever.
    rewrites: int

    # Output, set by the finalize node
    answer: str
    cited_doc_ids: list[str]
    confidence: Confidence

    # Diagnostics. Not part of the brief's contract, but the difference between
    # debugging by reading a line of text and debugging by archaeology.
    trace: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]


def new_state(question: str, model_role: str = "generator") -> AgentState:
    return AgentState(
        question=question,
        search_query=question,
        model_role=model_role,
        candidates=[],
        gate_passed=False,
        rewrites=0,
        trace=[],
        llm_calls=0,
    )
