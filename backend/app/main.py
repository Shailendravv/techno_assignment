"""The HTTP surface. Deliberately thin.

Every route here delegates to `agent.answer_question()` - the same function the
CLI and the evaluation harness call. That is the point: all three entry points
exercise identical code, so a harness score is evidence about what the API will
actually do.

The one route that is not a delegation is `POST /ingest/sign`, and it is
deliberately the *only* thing this API does with uploads. Vercel caps request
and response bodies at 4.5 MB, so files go browser -> Cloudinary directly and
we only ever handle the signature and the resulting `public_id`.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.api import answer_question
from agent.config import ROOT, settings
from agent.llm import LLMUnavailable
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SignRequest,
    SignResponse,
)
from logger.zap import create_logger

log = create_logger()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    log.info("app_startup", profile=settings.profile, store=settings.store)
    yield


app = FastAPI(
    title="Runbook Agent",
    description="Grounded question answering over operational runbooks.",
    version="1.0.0",
    lifespan=_lifespan,
)

# The frontend is a separate origin (the Vite dev server, or a deployed static
# site) calling this API from the browser, so it needs an explicit allow-list
# rather than the same-origin default. Origins come from settings, never
# hardcoded, so a deployed frontend only needs an env var, not a code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness, and a readable summary of how this instance is configured.

    Reports whether each credential is *present*, never what it is - this
    endpoint is public on a deployed app.

    It also doubles as the endpoint the weekly keep-alive cron hits. A Supabase
    free project pauses after seven idle days and unpausing is manual through
    the dashboard, which is a bad thing to discover the night before a demo.
    """
    from agent.store import get_store

    try:
        store = get_store(settings)
        store_health = store.health()
        mounted = store.name
    except Exception as exc:  # noqa: BLE001 - health must never 500
        store_health = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
        mounted = "none"

    described = settings.describe()

    # Report what is actually mounted, not what was asked for. Supabase falls
    # back to files when it is not configured, and a health endpoint that hid
    # that would let a misconfigured deployment look correct - which is the one
    # thing a health endpoint exists to prevent.
    return HealthResponse(
        status="ok",
        corpus_docs=store_health.get("documents", 0),
        store_reachable=bool(store_health.get("reachable")),
        store_error=str(store_health.get("error") or ""),
        store_requested=described.pop("store"),
        store=mounted,
        **described,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the runbooks, or decline to.

    A `no_match` response is a 200, not a 404. The agent declining to answer is
    a successful outcome - arguably the most valuable one it produces - and
    signalling it as an error would invite clients to treat it as a fault and
    retry, or to hide it.
    """
    started = time.perf_counter()
    try:
        result = answer_question(
            request.question,
            model_role=request.model_role,
            with_trace=request.explain,
            # Always, regardless of `explain`: this is what the response and the
            # access log both report, and a cost figure that silently defaults
            # to zero is worse than no figure at all.
            with_metrics=True,
        )
    except LLMUnavailable as exc:
        # Configuration, not a bug: no key, so the grounding step cannot run.
        log.warning("ask_unavailable", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # The question text itself is not logged - it is user input, not
    # something this module should be deciding is safe to persist to disk.
    log.info(
        "ask",
        confidence=result["confidence"],
        cited_doc_count=len(result["cited_doc_ids"]),
        elapsed_ms=elapsed_ms,
        llm_calls=result.get("llm_calls", 0),
    )

    return AskResponse(
        answer=result["answer"],
        cited_doc_ids=result["cited_doc_ids"],
        confidence=result["confidence"],
        elapsed_ms=elapsed_ms,
        trace=result.get("trace") or [],
        llm_calls=result.get("llm_calls", 0),
    )


@app.post("/ingest/sign", response_model=SignResponse)
def ingest_sign(request: SignRequest) -> SignResponse:
    """Authorise one direct browser upload to Cloudinary.

    The file never touches this API. Vercel caps request bodies at 4.5 MB, and
    an upload path that works until somebody attaches something large is worse
    than one that never worked - so the browser uploads directly and we hand it
    a signature scoped to one folder and one `public_id`.

    The API secret is used to compute the signature and is never returned.
    """
    from ingest.cloudinary_client import CloudinaryUnavailable, build_upload_signature

    try:
        signed = build_upload_signature(settings, public_id=request.public_id)
    except CloudinaryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SignResponse(**signed)


@app.get("/eval/latest")
def eval_latest() -> JSONResponse:
    """The most recent committed evaluation report.

    Served from the repository rather than recomputed. A full harness run makes
    twenty grounding calls against a tier allowing about two a minute - not
    something to do inside a request - and the committed report is the actual
    evidence anyway.
    """
    import json

    for name in ("harness_output.json", "retrieval_comparison.json"):
        path = ROOT / name
        if path.is_file():
            return JSONResponse(
                {"report": name, **json.loads(path.read_text(encoding="utf-8"))}
            )

    return JSONResponse(
        {"detail": "No evaluation report has been committed yet."}, status_code=404
    )


@app.get("/")
def index():
    """No bundled UI here - the frontend/ React app is the client.

    This API is CORS-enabled for it (see `settings.cors_origin_list`) and
    otherwise self-describes through OpenAPI.
    """
    return JSONResponse({"detail": "See /docs for the API.", "health": "/health"})
