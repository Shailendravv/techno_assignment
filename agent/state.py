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

    # Set by the analyze node
    spec: QuerySpec

    # Set by the retrieve node; narrowed by the metadata filter
    candidates: list[Candidate]
    gate_passed: bool

    # Set by the ground node
    raw_answer: str
    raw_cited_ids: list[str]

    # Set by the grade node (Phase 5); how many rewrites we have spent
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
        model_role=model_role,
        candidates=[],
        gate_passed=False,
        rewrites=0,
        trace=[],
        llm_calls=0,
    )
