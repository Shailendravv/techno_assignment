"""Request and response models for the HTTP surface.

These mirror the contract in `agent.core.models.Answer` rather than replacing
it. The agent package is the source of truth; this layer only validates what
arrives over the wire and shapes what goes back.
"""

from __future__ import annotations

import re

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

    # Observability only - neither changes an answer. `session_id` groups the
    # questions of one sitting into a Langfuse session, which is what makes a
    # follow-up question readable as a follow-up rather than as an unrelated
    # trace that happens to be nearby. `user_id` attributes cost and quality.
    #
    # Both are client-supplied and therefore untrusted: they are opaque labels
    # on a trace, never an authorisation claim, and nothing here reads them
    # back. Length-capped so a caller cannot use them as a data channel into
    # the observability backend.
    session_id: str | None = Field(default=None, max_length=200)
    user_id: str | None = Field(default=None, max_length=200)


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

    # What is actually serving, and what was asked for. They differ when
    # Supabase is selected but not configured and the files backend takes over;
    # a health endpoint that reported only the request would let a
    # misconfigured deployment look correct.
    store: str
    store_requested: str
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
    langfuse_configured: bool

    # Whether signed uploads are actually available: a shared secret is set
    # *and* Cloudinary is configured. Reported because "the route 404s" and
    # "the route is not configured" look identical from outside, and an
    # operator needs to tell them apart.
    ingest_enabled: bool


# Uploads live in their own namespace, kept away from the curated corpus.
#
# `on_conflict=doc_id` in the ingestion pipeline means an uploaded document
# whose front-matter claims `doc_id: RB-001` *overwrites* RB-001. Pinning the
# `public_id` under a prefix the runbooks never use is the first of the two
# defences against that; `ingest.pipeline` namespacing the `doc_id` it derives
# is the second.
UPLOAD_PREFIX = "upload-"
_PUBLIC_ID = re.compile(rf"^{re.escape(UPLOAD_PREFIX)}[a-z0-9]{{8,64}}$")


class SignRequest(BaseModel):
    """One requested upload.

    `public_id` is optional - Cloudinary will assign one - but when supplied it
    is signed, and therefore it is what the browser is authorised to write. It
    used to be any string of up to 200 characters, which meant a caller could
    request a signature for `RB-001` and overwrite a real runbook's asset. It is
    now constrained to the upload namespace.
    """

    public_id: str | None = Field(default=None, max_length=80)


def valid_public_id(value: str | None) -> bool:
    """Whether this `public_id` is inside the upload namespace.

    Checked in the route rather than as a pydantic validator, deliberately.
    FastAPI validates the request body before the handler runs and returns 422
    for a bad one - which would answer an unauthenticated caller with a 422
    where a valid body gets a 404, and that difference tells them the route
    exists. Authentication decides first; this runs after.
    """
    return value is None or bool(_PUBLIC_ID.match(value))


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
