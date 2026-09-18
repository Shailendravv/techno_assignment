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

import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.api import answer_question
from agent.config import ROOT, current_settings
from agent.llm import LLMUnavailable, is_upstream_rate_limit as _is_upstream_rate_limit
from agent.store import StoreUnavailable
from app.ratelimit import RateLimiter
from app.schemas import (
    UPLOAD_PREFIX,
    valid_public_id,
    AskRequest,
    AskResponse,
    HealthResponse,
    SignRequest,
    SignResponse,
)
from logger.zap import create_logger

log = create_logger()


def _authorised(supplied: str | None, expected: str) -> bool:
    """Constant-time comparison of the ingest shared secret.

    `secrets.compare_digest` rather than `==` because a plain comparison exits
    at the first differing byte, and the timing of that is enough to recover a
    secret one byte at a time given enough requests. Cheap to do correctly.
    """
    if not supplied or not expected:
        return False
    return secrets.compare_digest(supplied, expected)


def _flush_traces() -> None:
    """Push any queued Langfuse events out before this request returns.

    A no-op when tracing is not configured, which is the default. Never raises:
    the rule `agent/observability.py` states applies here too - losing a trace
    is acceptable, failing a request to deliver one is not.
    """
    try:
        from agent import observability

        observability.flush()
    except Exception:  # noqa: BLE001 - observability may never fail a request
        pass


@asynccontextmanager
async def _lifespan(_: FastAPI):
    from agent import observability

    log.info(
        "app_startup",
        profile=current_settings().profile,
        store=current_settings().store,
        tracing="langfuse" if observability.enabled() else "off",
    )
    yield
    # The other half of the serverless problem: a long-lived process (uvicorn
    # locally, a warm container) should not lose whatever is still queued when
    # it is finally told to stop.
    _flush_traces()


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
    allow_origins=current_settings().cors_origin_list,
    # No cookie, header or credential is used by any route, so allowing them
    # buys nothing and constrains what `CORS_ORIGINS` can safely become: with
    # credentials on, a wildcard origin is a cross-site read of authenticated
    # responses. Off is both correct today and safe to widen later.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Failure handling.
#
# There were no exception handlers at all. Everything except `LLMUnavailable`
# reached Starlette's default and became a bare `500 Internal Server Error`
# with no body, no log line and no trace flush - verified by pointing the app
# at an unreachable store. The two most likely production failures on this
# stack both landed there: Supabase refusing a connection, and Groq's free tier
# still returning 429 after the retry budget was spent.
#
# A correlation id is returned to the caller and logged, so "it broke at 3am"
# and the log line can be joined without guessing.
# ---------------------------------------------------------------------------

def _correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def _unavailable(request: Request, exc: Exception, reason: str) -> JSONResponse:
    incident = _correlation_id()
    log.error(
        "request_failed",
        incident=incident,
        path=request.url.path,
        reason=reason,
        error=f"{type(exc).__name__}: {exc}",
    )
    _flush_traces()
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                f"{reason} This is usually transient; retry shortly. "
                f"Incident {incident}."
            ),
            "incident": incident,
        },
        headers={"Retry-After": "30"},
    )


@app.exception_handler(StoreUnavailable)
def _store_unavailable(request: Request, exc: StoreUnavailable) -> JSONResponse:
    """The corpus could not be read.

    503 rather than 500: nothing is wrong with the request, the dependency is
    down. A Supabase free project also pauses after seven idle days, which
    presents exactly like this.
    """
    return _unavailable(request, exc, "The document store is unreachable.")


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Everything else.

    Returns an opaque message plus the correlation id - never the exception
    text, which on this stack can carry a Supabase URL or a fragment of a
    provider response.
    """
    if _is_upstream_rate_limit(exc):
        return _unavailable(
            request, exc, "The language model is rate limited right now."
        )

    incident = _correlation_id()
    log.error(
        "request_failed",
        incident=incident,
        path=request.url.path,
        reason="unhandled",
        error=f"{type(exc).__name__}: {exc}",
    )
    _flush_traces()
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Something went wrong. Incident {incident}.",
            "incident": incident,
        },
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
        store = get_store(current_settings())
        store_health = store.health()
        mounted = store.name
    except Exception as exc:  # noqa: BLE001 - health must never 500
        store_health = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
        mounted = "none"

    described = current_settings().describe()

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


_ask_limiter = RateLimiter(current_settings().ask_rate_limit_per_minute)


def _client_key(request: Request) -> str:
    """Who to count this request against.

    `x-forwarded-for` because the app runs behind Vercel's proxy, where
    `request.client` is the proxy for every caller and would rate-limit the
    whole world as one client. The leftmost entry is the original client and is
    spoofable - which is fine for a quota guard and would not be for an
    authorisation decision, so this is used for nothing else.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest, http_request: Request, background: BackgroundTasks
) -> AskResponse:
    """Answer a question from the runbooks, or decline to.

    A `no_match` response is a 200, not a 404. The agent declining to answer is
    a successful outcome - arguably the most valuable one it produces - and
    signalling it as an error would invite clients to treat it as a fault and
    retry, or to hide it.
    """
    allowed, retry_after = _ask_limiter.check(_client_key(http_request))
    if not allowed:
        log.warning("ask_rate_limited", retry_after_s=retry_after)
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many requests. This API answers from a shared free-tier "
                "model quota, so it is rate limited per client."
            ),
            headers={"Retry-After": str(retry_after)},
        )

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
            session_id=request.session_id,
            user_id=request.user_id,
        )
    except LLMUnavailable as exc:
        # Configuration, not a bug: no key, so the grounding step cannot run.
        log.warning("ask_unavailable", error=str(exc))
        # Inline, because an HTTPException leaves the normal response path and
        # takes any background task registered on it with it. A request that
        # failed for want of a key is exactly the one somebody goes looking for
        # in the traces, so this one pays the flush rather than losing it.
        _flush_traces()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Groq's free tier still 429s after the retry budget in `agent.llm` is
        # spent, and that is the single most likely production failure on this
        # stack. It is a dependency being busy, not a fault in the request, so
        # it is a 503 with a `Retry-After` rather than an opaque 500. Caught
        # here rather than left to the application-wide handler so the answer
        # path states its own failure modes.
        if not _is_upstream_rate_limit(exc):
            raise
        incident = _correlation_id()
        log.warning(
            "ask_rate_limited_upstream",
            incident=incident,
            error=f"{type(exc).__name__}: {exc}",
        )
        _flush_traces()
        raise HTTPException(
            status_code=503,
            detail=(
                "The language model is rate limited right now. This is a free "
                f"tier; retry shortly. Incident {incident}."
            ),
            headers={"Retry-After": "30"},
        ) from exc

    # Not inside the request, and not left to the exporter's own timer either.
    # Langfuse batches on a background thread and a serverless function can be
    # frozen the moment it returns, so "flush eventually" means "do not flush";
    # but flushing before the response bills the user for a round trip to
    # Langfuse. A Starlette background task runs after the body is sent and
    # before the ASGI cycle completes, which is both.
    background.add_task(_flush_traces)

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
def ingest_sign(
    request: SignRequest,
    x_ingest_key: str | None = Header(default=None, alias="X-Ingest-Key"),
) -> SignResponse:
    """Authorise one direct browser upload to Cloudinary.

    The file never touches this API. Vercel caps request bodies at 4.5 MB, and
    an upload path that works until somebody attaches something large is worse
    than one that never worked - so the browser uploads directly and we hand it
    a signature scoped to one folder and one `public_id`.

    The API secret is used to compute the signature and is never returned.

    **Authenticated, because of what is on the other end of it.** This route
    used to be open. A signature is a write capability against our object
    store, and `ingest.pipeline --from-cloudinary` turns whatever lands in that
    folder into corpus documents - taking `doc_id`, `service` and
    `failure_mode` from the uploaded file's own front-matter, and handing its
    body to a model as ground truth. So an unauthenticated caller could choose
    what the agent believes. The `public_id` is validated separately, in
    `SignRequest`, so a caller also cannot aim an upload at an existing runbook.

    A missing or wrong key is a **404, not a 403**: an unconfigured deployment
    should not advertise that this route exists at all.
    """
    from ingest.cloudinary_client import CloudinaryUnavailable, build_upload_signature

    cfg = current_settings()

    if not cfg.ingest_enabled or not _authorised(x_ingest_key, cfg.ingest_api_key):
        log.warning("ingest_sign_denied", configured=cfg.ingest_enabled)
        raise HTTPException(status_code=404, detail="Not Found")

    if not valid_public_id(request.public_id):
        raise HTTPException(
            status_code=422,
            detail=(
                f"public_id must look like {UPLOAD_PREFIX}<8-64 lowercase "
                "alphanumerics>; it cannot name an existing document"
            ),
        )

    try:
        signed = build_upload_signature(cfg, public_id=request.public_id)
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

    This API is CORS-enabled for it (see `CORS_ORIGINS`) and
    otherwise self-describes through OpenAPI.
    """
    return JSONResponse({"detail": "See /docs for the API.", "health": "/health"})
