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

    # Return the per-stage trace. Off by default because it is not part of the
    # brief's contract, and on in the UI because "why did it decline" is the
    # most interesting thing this system has to say.
    explain: bool = False


class AskResponse(BaseModel):
    """Exactly the three fields the brief specifies, plus diagnostics.

    `confidence` is one of high | medium | low | no_match. `no_match` is a
    legitimate, expected result - not an error - and clients should render it
    as an answer rather than a failure.
    """

    answer: str
    cited_doc_ids: list[str]
    confidence: str

    elapsed_ms: int
    trace: list[str] = Field(default_factory=list)
    llm_calls: int = 0


class HealthResponse(BaseModel):
    """What this instance is, and whether it can actually serve.

    Every credential field is a boolean. This endpoint is public on a deployed
    app, so it reports presence and never a value.
    """

    status: str
    profile: str

    corpus_docs: int
    store: str
    store_reachable: bool
    store_error: str = ""

    retrieval_mode: str
    grader_enabled: bool
    embedder: str
    embedding_dims: int

    groq_configured: bool
    gemini_configured: bool
    supabase_configured: bool
    cloudinary_configured: bool


class SignRequest(BaseModel):
    # Optional: let Cloudinary assign one if the client does not care. When
    # supplied it is signed, so the browser cannot upload somewhere else.
    public_id: str | None = Field(default=None, max_length=200)


class SignResponse(BaseModel):
    """Everything the browser needs to upload directly - and no secret.

    The signature authorises one upload into one folder. It is not a credential
    that can be reused for anything else, which is the difference between this
    and shipping an unsigned upload preset.
    """

    cloud_name: str
    api_key: str
    resource_type: str
    upload_url: str
    signature: str
    timestamp: int
    folder: str
    public_id: str | None = None
