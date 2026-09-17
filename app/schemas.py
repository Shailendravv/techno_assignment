"""Request and response models for the HTTP surface.

These mirror the contract in `agent.core.models.Answer` rather than replacing
it. The agent package is the source of truth; this layer only validates what
arrives over the wire and shapes what goes back.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    # "generator" is the production model; "reasoner" is the slower, preview
    # quality arm. Exposed so the demo can show the difference side by side.
    model_role: str = Field(default="generator", pattern="^(generator|reasoner)$")


class AskResponse(BaseModel):
    """Exactly the three fields the brief specifies, plus timing.

    `confidence` is one of high | medium | low | no_match. `no_match` is a
    legitimate, expected result - not an error - and clients should render it
    as an answer rather than a failure.
    """

    answer: str
    cited_doc_ids: list[str]
    confidence: str
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    corpus_docs: int
    store: str
    retrieval_mode: str
    groq_configured: bool
