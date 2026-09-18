"""Tracing, exported to Langfuse when it is configured and nowhere otherwise.

The agent already produces a trace: every node appends a line saying what it
decided, `AgentState.trace` accumulates them, and `agent/stages.py` records all
twenty declared stages - including the ones that never ran. That is the
substance. This module gives it a second home, one you can filter, aggregate
and attach scores to, instead of grepping `logs/app.log`.

Three decisions worth stating.

**The SDK, not a hand-rolled POST.** An earlier version of this file sent one
`ingestion` batch over stdlib `urllib` to keep the deployed bundle small. It
cost about forty lines and it bought a trace that was structurally wrong: every
span carried the same timestamp, so nothing had a duration; the Groq calls were
strings in a list rather than `generation` observations, so no model name, no
token usage, and therefore no cost anywhere in Langfuse. Those are not
cosmetic - they are most of what the product is for. `langfuse` is worth its
place in `requirements.txt` for them.

**Nesting comes from the stage ledger, not from a second set of call sites.**
`StageRecorder.stage()` already wraps every stage of both pipelines in a
context manager. Hooking Langfuse in there means one integration point instead
of twenty, and it means the trace tree and the log ledger cannot drift: they
are the same events. `STAGE_OBSERVATIONS` below maps each stage to a Langfuse
observation type, because "this step only looked something up" (`retriever`)
and "this step called a model" (`generation`) are the distinctions every
filter, evaluator and dashboard in Langfuse is built on.

**Fire-and-forget, and silent on failure.** Same rule as before, and the reason
has not changed: an observability backend that can fail a request is worse than
no observability. Every public function here swallows its own exceptions and
degrades to a no-op handle, so a call site never needs a None check and a
broken Langfuse costs a trace and nothing else.

Off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, which means
the default configuration - and the whole test suite - makes no network calls.
"""

from __future__ import annotations

import os
import re
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from agent.config import Settings, settings as default_settings

DEFAULT_HOST = "https://cloud.langfuse.com"

# How long a flush may block a request before we give up on the batch. The
# trade is stated rather than defaulted: on Vercel the function can be frozen
# the moment a response is returned, so a background flush may never run, and
# an un-flushed trace is a lost trace. Two seconds of tail latency for a
# request that already spent seconds inside Groq is a price worth paying; ten
# would not be.
FLUSH_TIMEOUT_S = 2


# ---------------------------------------------------------------------------
# Stage -> observation mapping
# ---------------------------------------------------------------------------
# Names are Langfuse's, not ours. They are verb-first and low-cardinality
# because Langfuse treats an observation name as an API: evaluators target
# observations by name, dashboards group by it, and saved filters match on it.
# Renaming one silently breaks all three, so these are chosen to outlive the
# internal stage names they are derived from.
#
# The types matter as much as the names. `retriever` marks a step that only
# looks something up, `embedding` and `generation` carry model and token usage,
# and `guardrail` marks the two steps whose entire job is to refuse. A tree of
# undifferentiated spans would render, and would tell you nothing.

STAGE_OBSERVATIONS: dict[str, tuple[str, str]] = {
    # -- offline: ingestion -------------------------------------------------
    "extract_text": ("extract-text", "span"),
    "clean": ("clean-text", "span"),
    "extract_metadata": ("extract-metadata", "span"),
    "chunk": ("chunk-document", "span"),
    "embed_chunks": ("embed-chunks", "embedding"),
    "store_embeddings": ("store-embeddings", "span"),

    # -- online: the query path --------------------------------------------
    "query_enhance": ("analyze-query", "span"),
    "embed_query": ("embed-query", "embedding"),
    "sparse_retrieve": ("search-lexical", "retriever"),
    "dense_retrieve": ("search-dense", "retriever"),
    "rrf_fuse": ("fuse-rankings", "chain"),
    # The gate lives inside this stage, and refusing to answer is the outcome
    # this system exists to produce - so it is typed as what it is.
    "relevance_filter": ("filter-and-gate", "guardrail"),
    "relevance_grade": ("grade-relevance", "evaluator"),
    "rewrite_query": ("rewrite-query", "span"),
    "build_prompt": ("build-prompt", "span"),
    "generate": ("generate-answer", "span"),
    # `llm_gateway` is deliberately absent. It is recorded by `agent/llm.py`
    # after the fact, and the `generation` that call site opens is the same
    # event with model, tokens and cost attached. Two observations for one
    # round trip would double-count in every latency and cost view.
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: Any = None
_client_failed = False

# Whether a root span is open on this execution context. Stage observations are
# gated on it: an observation created with no parent becomes its own one-span
# trace in Langfuse, so a stray `recorder.stage()` outside a run would litter
# the project with orphans rather than produce anything readable.
_TRACING: ContextVar[bool] = ContextVar("langfuse_tracing", default=False)


def _credentials(cfg: Settings | None = None) -> tuple[str, str, str] | None:
    """Keys and host, from settings - which is `.env`, the profile, or neither.

    `LANGFUSE_BASE_URL` is accepted as well as `LANGFUSE_HOST` because the
    Langfuse docs and CLI have used both names, and a key pair that silently
    does nothing because the host was spelled the other way is a bad half hour.
    """
    cfg = cfg or default_settings

    # The environment first, and settings only as the fallback. `Settings` is a
    # frozen snapshot taken at import; the environment is what is true now, and
    # this module is read from a long-lived server process where the difference
    # is real. It is also what lets a test turn tracing off without editing the
    # `.env` the developer runs the app with.
    if os.getenv("LANGFUSE_TRACING_ENABLED", "").strip().lower() in ("0", "false", "no", "off"):
        return None
    if not getattr(cfg, "langfuse_tracing", True) and "LANGFUSE_TRACING_ENABLED" not in os.environ:
        return None

    public = os.getenv("LANGFUSE_PUBLIC_KEY", "") or getattr(cfg, "langfuse_public_key", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "") or getattr(cfg, "langfuse_secret_key", "")
    if not (public and secret):
        return None

    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or getattr(cfg, "langfuse_host", "")
        or DEFAULT_HOST
    )
    return public, secret, host.rstrip("/")


def enabled(cfg: Settings | None = None) -> bool:
    """Whether tracing is configured. Credentials only - no network call."""
    return _credentials(cfg) is not None


_SECRET = re.compile(
    # Long opaque tokens, and the prefixes the services in this project use.
    r"\b(?:sk-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9_\-]{8,}|pk-lf-[A-Za-z0-9\-]{8,}"
    r"|AIza[A-Za-z0-9_\-]{8,}|eyJ[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-.]{16,})"
)
_EMAIL = re.compile(r"\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b")


def mask(*, data: Any, **_: Any) -> Any:
    """Redact credentials and addresses from anything on its way to Langfuse.

    The question text itself is not masked and is not meant to be: it is the
    trace input, and a trace whose input you cannot read answers nothing. What
    this catches is the accident - a key pasted into a question, an address in
    a runbook - reaching a third party because somebody enabled tracing.

    Runs on every string in every observation, so it is a regex pass and not a
    parser, and it is not a compliance control. A corpus with real PII in it
    needs a decision about tracing, not a wider pattern here.
    """
    try:
        if isinstance(data, str):
            return _EMAIL.sub("[email]", _SECRET.sub("[redacted]", data))
        if isinstance(data, dict):
            return {key: mask(data=value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [mask(data=item) for item in data]
        return data
    except Exception:  # noqa: BLE001 - masking must never fail an export
        return "[unmaskable]"


def client(cfg: Settings | None = None):
    """The Langfuse client, or None when tracing is off or broken.

    Constructed lazily and exactly once. Lazily because importing `langfuse`
    pulls in the OpenTelemetry SDK and the CLI, the tests and the ingest
    pipeline should not pay for that when no keys are set; once because the
    SDK keys its own singleton on the public key and a second construction is
    silently ignored anyway.
    """
    global _client, _client_failed

    cfg = cfg or default_settings

    if _client is not None or _client_failed:
        return _client

    credentials = _credentials(cfg)
    if credentials is None:
        return None

    public, secret, host = credentials

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public,
            secret_key=secret,
            base_url=host,
            # The profile is the environment, so a `local` run and a deployed
            # `dev` one land in separate buckets. Without this, every
            # experiment run locally would pollute the dashboards and the
            # evaluations of the deployed app - which is the failure mode that
            # makes teams stop trusting their own numbers.
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", cfg.profile),
            release=os.getenv("VERCEL_GIT_COMMIT_SHA") or None,
            mask=mask,
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        # Remembered, so a broken install or an unreachable host costs one
        # failed construction rather than one per request.
        _client_failed = True
        _client = None

    return _client


def reset_client() -> None:
    """Forget the cached client. For tests, and for a changed key in `.env`."""
    global _client, _client_failed
    _client = None
    _client_failed = False


def flush() -> None:
    """Send whatever is queued. Never raises.

    Called at the end of a request and at the end of a CLI or harness run.
    Langfuse batches in a background thread, and both a serverless function and
    a short script can exit before that thread gets its turn.
    """
    active = _client
    if active is None:
        return
    try:
        active.flush()
    except Exception:  # noqa: BLE001 - see the module docstring
        pass


# ---------------------------------------------------------------------------
# The no-op handle
# ---------------------------------------------------------------------------

class _Noop:
    """What every helper here yields when tracing is off.

    A working object rather than None, so no call site needs a conditional and
    `agent/llm.py` does not grow two code paths for "traced" and "not traced".
    """

    def update(self, **_: Any) -> "_Noop":
        return self

    def score(self, **_: Any) -> None:
        pass

    def score_trace(self, **_: Any) -> None:
        pass

    def end(self, **_: Any) -> None:
        pass


NOOP = _Noop()


def _quiet_exit(manager: Any):
    """A callback that closes `manager` and swallows only its own failure.

    Every context manager below opens a Langfuse observation that may fail to
    open, may fail to close, and must do neither to the caller. The obvious
    shape - `try: yield ... except: yield NOOP` - is wrong, and wrong in a way
    that only shows up on the path that matters: when the body raises, the
    exception is thrown in at the `yield`, the handler yields a second time,
    and Python replaces the real exception with `RuntimeError: generator didn't
    stop after throw()`. An instrumentation bug would then be reported as an
    agent bug, with the actual cause gone.

    So setup is guarded separately from the body, and this runs on the way out.
    It returns False without exception: an observation never suppresses what
    the code it was watching raised.
    """

    def _exit(exc_type, exc, traceback) -> bool:
        try:
            manager.__exit__(exc_type, exc, traceback)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass
        return False

    return _exit


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@contextmanager
def trace_run(
    question: str,
    cfg: Settings | None = None,
    *,
    name: str = "answer-question",
    session_id: str | None = None,
    user_id: str | None = None,
    model_role: str = "generator",
) -> Iterator[Any]:
    """The root span for one question. Everything else nests inside it.

    One question is one trace, which is the unit Langfuse's data model is built
    around: small enough to read, complete enough to judge. Multi-turn grouping
    is `session_id`'s job, not a bigger trace's.

    Tags are set here and only here, because Langfuse fixes them when the
    observation is created. That rules out the tag this system most wants -
    the outcome - so `confidence` and `answered`/`declined` are recorded as
    scores instead, in `finish()` below. What is left is what is known before
    the pipeline runs: how this instance is configured.
    """
    cfg = cfg or default_settings
    active = client(cfg)
    if active is None:
        yield NOOP
        return

    with ExitStack() as stack:
        token = _TRACING.set(True)
        # Pushed first so it runs last: the flag stays true until every
        # observation opened under it has closed.
        stack.callback(_TRACING.reset, token)

        root: Any = NOOP
        try:
            from langfuse import propagate_attributes

            span = active.start_as_current_observation(
                as_type="span",
                name=name,
                # The user's question and nothing else. The default would be
                # every argument this function received, which here means the
                # whole Settings object - a config dump where a reviewer needs
                # one line.
                input={"question": question},
                metadata={
                    "profile": cfg.profile,
                    "store": cfg.store,
                    "retrieval_mode": cfg.retrieval.mode,
                    "embedder": cfg.embedding.backend,
                    "grader_enabled": cfg.retrieval.grader_enabled,
                    "model_role": model_role,
                    "model": getattr(cfg.models, model_role, ""),
                    "max_rewrites": cfg.max_rewrites,
                },
            )
            root = span.__enter__()
            stack.push(_quiet_exit(span))

            attributes = propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                trace_name=name,
                tags=[
                    f"mode:{cfg.retrieval.mode}",
                    f"store:{cfg.store}",
                    f"role:{model_role}",
                ],
            )
            attributes.__enter__()
            stack.push(_quiet_exit(attributes))
        except Exception:  # noqa: BLE001 - see the module docstring
            root = NOOP

        yield root


def finish(root: Any, result: dict, *, llm_calls: int, elapsed_ms: int, cached: bool = False) -> None:
    """Close out the root span: the answer, the cost, and two scores.

    The scores are the part that earns its keep. Confidence and whether we
    answered at all are only known once the pipeline has run, and Langfuse tags
    are immutable from creation - so the one question this project most wants
    to ask its traces ("show me every question we declined") has to be a score
    or it cannot be asked at all.
    """
    try:
        cited = result.get("cited_doc_ids") or []
        confidence = str(result.get("confidence", "unknown"))

        root.update(
            output={
                "answer": result.get("answer", ""),
                "cited_doc_ids": cited,
                "confidence": confidence,
            },
            metadata={
                "llm_calls": llm_calls,
                "elapsed_ms": elapsed_ms,
                "cited_doc_count": len(cited),
                "answer_cache": "hit" if cached else "miss",
            },
        )

        root.score_trace(
            name="confidence",
            value=confidence,
            data_type="CATEGORICAL",
        )
        root.score_trace(
            name="answered",
            value=1.0 if cited else 0.0,
            data_type="BOOLEAN",
            comment=(
                "cited " + ", ".join(cited) if cited
                else "declined: nothing in the corpus applied"
            ),
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        pass


@contextmanager
def stage_observation(stage_name: str) -> Iterator[Any]:
    """The Langfuse half of a `recorder.stage(...)` block.

    Yields a live observation when a run is being traced and a no-op otherwise.
    Nesting is not managed here: the SDK is OpenTelemetry-native, so whatever
    context manager is open on this thread is the parent. That is why the store
    ends up inside `retrieve-documents` without either of them knowing.
    """
    if not _TRACING.get():
        yield NOOP
        return

    active = client()
    if active is None:
        yield NOOP
        return

    name, as_type = STAGE_OBSERVATIONS.get(stage_name, (stage_name.replace("_", "-"), "span"))
    with ExitStack() as stack:
        observation: Any = NOOP
        try:
            span = active.start_as_current_observation(as_type=as_type, name=name)
            observation = span.__enter__()
            stack.push(_quiet_exit(span))
        except Exception:  # noqa: BLE001 - see the module docstring
            observation = NOOP

        yield observation


def tracing() -> bool:
    """Whether a traced run is in progress on this execution context."""
    return _TRACING.get()


def event(name: str, *, level: str = "DEFAULT", status_message: str = "", **attributes: Any) -> None:
    """A point in the trace with no duration - something that happened.

    Used for the degradations: a stage that ran in a reduced form did not
    occupy an interval of its own, so a span with an invented duration would be
    a worse record than an event with none.
    """
    if not _TRACING.get():
        return
    active = client()
    if active is None:
        return
    try:
        active.create_event(
            name=name, level=level, status_message=status_message, **attributes
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        pass


@contextmanager
def observation(name: str, as_type: str = "span", **attributes: Any) -> Iterator[Any]:
    """An observation that does not correspond to a declared stage.

    There is exactly one of these: the `retriever` span that groups the four
    ranking stages and the gate under one node. The stages are declared
    separately in `agent/stages.py` because they are separately skippable, but
    in a trace tree they are one retrieval step, and reading them as five
    siblings of the generation is reading the pipeline wrong.
    """
    if not _TRACING.get():
        yield NOOP
        return

    active = client()
    if active is None:
        yield NOOP
        return

    with ExitStack() as stack:
        obs: Any = NOOP
        try:
            span = active.start_as_current_observation(
                as_type=as_type, name=name, **attributes
            )
            obs = span.__enter__()
            stack.push(_quiet_exit(span))
        except Exception:  # noqa: BLE001 - see the module docstring
            obs = NOOP

        yield obs


@contextmanager
def generation(
    *,
    role: str,
    model: str,
    messages: list[dict],
    attempt: int,
    temperature: float,
    max_tokens: int,
) -> Iterator[Any]:
    """One round trip to Groq, as a `generation`.

    Per attempt, not per logical call: a 429 that was retried is two requests
    that both counted against the quota, and a trace that showed one would
    disagree with the bill. The name is the role rather than the model, because
    models get swapped in config and every filter naming one would break.
    """
    if not _TRACING.get():
        yield NOOP
        return

    active = client()
    if active is None:
        yield NOOP
        return

    with ExitStack() as stack:
        gen: Any = NOOP
        try:
            span = active.start_as_current_observation(
                as_type="generation",
                name=f"call-{role}",
                model=model,
                # The messages as sent. Langfuse renders a role-labelled
                # conversation from this shape, so the prompt is readable as a
                # prompt - which is the thing you came to look at when an
                # answer was wrong.
                input=messages,
                model_parameters={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                metadata={"attempt": attempt},
            )
            gen = span.__enter__()
            stack.push(_quiet_exit(span))
        except Exception:  # noqa: BLE001 - see the module docstring
            gen = NOOP

        yield gen


def trace_url() -> str:
    """A link to the trace in progress, for a log line. Empty when untraced."""
    try:
        active = client()
        if active is None or not _TRACING.get():
            return ""
        return active.get_trace_url() or ""
    except Exception:  # noqa: BLE001 - see the module docstring
        return ""
