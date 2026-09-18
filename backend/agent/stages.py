"""The stage ledger: what the pipeline did, and what it didn't.

Every stage of both pipelines is declared here, and every declared stage emits
exactly one log line per run - including the ones that did not execute. That is
the whole design, and it is worth saying why, because a log that records only
what happened is the obvious thing to build and it is not enough.

Three failures this codebase can have are all invisible in a conventional log:

1. **A stage disabled by profile.** `GRADER_ENABLED=false` locally means the
   CRAG relevance grader never runs. Nothing says so. The answer looks the same.
2. **A stage that silently degraded.** `RETRIEVAL_MODE=hybrid` with no usable
   embedder falls back to lexical-only - deliberately, because a working
   retriever beats a 500. But a hybrid run that served lexical results is not
   the thing the configuration claims, and any comparison between arms that
   doesn't know the difference is measuring noise.
3. **A stage that was never built.** OCR, document classification, and parallel
   retrieval are not in this system. A reader of the logs cannot distinguish
   "absent" from "ran and logged nothing".

So the registry declares all three cases up front, and an unrecorded stage is
reported with its declared status and reason rather than omitted. `grep
not_implemented logs/app.log` answers "what is this pipeline missing", and
`grep skipped` answers "what did this run not do", without reading any source.

**Lifecycles are reported separately.** Steps 1-8 are the offline ingestion
pipeline, run by `python -m ingest.pipeline` and never inside a request. Steps
9-20 are the online query path. A query run reports twelve lines and an ingest
run reports eight; neither claims to have skipped the other's work, because
"skipped" would be a lie about a stage that was never on that path.

**This may never fail a request.** Same rule `agent/observability.py` states for
tracing and `agent/cache.py` for caching: the pipeline is the product, and the
instrumentation around it is not. Every public method here swallows its own
exceptions, and `current_recorder()` returns a working no-op when no run is in
progress - so a call site never needs a None check, and `agent.core` can stay
pure functions that know nothing about logging.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional


class Status(str, Enum):
    """What became of a stage on one run."""

    RAN = "ran"
    SKIPPED = "skipped"  # exists, but configuration or routing turned it off
    DEGRADED = "degraded"  # ran in a reduced form - the dangerous one
    NOT_IMPLEMENTED = "not_implemented"  # never built; see `reason`
    FAILED = "failed"  # raised; the exception was re-raised to the caller


INGEST = "ingest"
QUERY = "query"


@dataclass(frozen=True)
class Stage:
    id: int
    name: str
    lifecycle: str
    declared: Status = Status.SKIPPED
    reason: str = ""


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# `declared` is the status reported when a run never touches the stage. For a
# stage that exists, that means "skipped" and the call site is expected to say
# why. For a stage that was never built, the reason is fixed and lives here -
# nobody is going to call `skip()` for code that does not exist.
#
# The reasons for the three gaps are the project's own, taken from
# `documents/UNDERSTANDING.md` and the docstrings at the relevant call sites.
# They are recorded here so the log explains itself.

STAGES: tuple[Stage, ...] = (
    # -- Offline: ingestion. `python -m ingest.pipeline`, never in a request. --
    Stage(1, "extract_text", INGEST),
    Stage(
        2,
        "ocr",
        INGEST,
        Status.NOT_IMPLEMENTED,
        "no scanned-document path: the corpus is authored markdown, and the PDF "
        "route uses pymupdf4llm text extraction, which assumes a text layer",
    ),
    Stage(3, "clean", INGEST),
    Stage(4, "extract_metadata", INGEST),
    Stage(
        5,
        "classify_document",
        INGEST,
        Status.NOT_IMPLEMENTED,
        "metadata is authored, not inferred: a guessed service is worse than an "
        "absent one, because a wrong one makes the filter drop the document for "
        "every question it actually answers",
    ),
    Stage(6, "chunk", INGEST),
    Stage(7, "embed_chunks", INGEST),
    Stage(8, "store_embeddings", INGEST),

    # -- Online: the query path. `answer_question()`. --
    Stage(9, "query_enhance", QUERY),
    Stage(10, "embed_query", QUERY),
    Stage(
        11,
        "parallel_retrieve",
        QUERY,
        Status.NOT_IMPLEMENTED,
        "the two arms run sequentially: at twelve documents the lexical pass is "
        "single-digit milliseconds, so concurrency would buy noise and cost a "
        "thread pool in a serverless function",
    ),
    Stage(12, "sparse_retrieve", QUERY),
    Stage(13, "dense_retrieve", QUERY),
    Stage(14, "rrf_fuse", QUERY),
    Stage(15, "relevance_filter", QUERY),
    Stage(16, "relevance_grade", QUERY),
    Stage(17, "rewrite_query", QUERY),
    Stage(18, "build_prompt", QUERY),
    Stage(19, "llm_gateway", QUERY),
    Stage(20, "generate", QUERY),
)

_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}
_BY_ID: dict[int, Stage] = {s.id: s for s in STAGES}

INGEST_STAGES: tuple[Stage, ...] = tuple(s for s in STAGES if s.lifecycle == INGEST)
QUERY_STAGES: tuple[Stage, ...] = tuple(s for s in STAGES if s.lifecycle == QUERY)

# Widest name, so the message column lines up in `logs/app.log` and a run can be
# read down the page rather than parsed.
_NAME_WIDTH = max(len(s.name) for s in STAGES)
_STATUS_WIDTH = max(len(s.value) for s in Status)


def stage_id(name: str) -> Optional[int]:
    stage = _BY_NAME.get(name)
    return stage.id if stage else None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class _NullObservation:
    """An observation that records nothing, for when Langfuse is not there."""

    def update(self, **_):
        return self

    def end(self, **_):
        return None


_NULL_OBSERVATION = _NullObservation()


class _NoTracing:
    """The stand-in for `agent.observability` when it cannot be imported.

    Instrumentation that breaks on an import error is worse than absent
    instrumentation, because it takes the pipeline down with it. This keeps the
    part of that module's contract this one uses - `NOOP` and
    `stage_observation` - with no Langfuse and no import.
    """

    NOOP = _NULL_OBSERVATION

    @staticmethod
    @contextmanager
    def stage_observation(_name: str):
        yield _NULL_OBSERVATION


def _observability():
    """`agent.observability`, or a silent stand-in for it."""
    try:
        from agent import observability

        return observability
    except Exception:  # noqa: BLE001 - see the module docstring
        return _NoTracing


def _default_logger():
    try:
        from logger.zap import create_logger

        return create_logger()
    except Exception:  # noqa: BLE001 - a run must not fail for want of a logger
        return None


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------

class _Open:
    """The handle a `with recorder.stage(...)` block writes its detail to.

    It carries the Langfuse observation for the same stage, when there is one.
    `detail()` is the log line; `io()` is what Langfuse shows as the step's
    input and output. Both are optional and both go to the same place - the
    stage that is currently open - so a call site never has to know whether
    tracing is on.
    """

    def __init__(self, observation=None):
        self._detail = ""
        self._observation = observation or _observability().NOOP

    def detail(self, text: str) -> None:
        self._detail = str(text)
        # The ledger line doubles as the observation's status message, so a
        # trace opened from a log line says the same thing the log line did.
        self._observation.update(metadata={"detail": self._detail})

    def io(self, input=None, output=None, **metadata) -> None:
        """Set what this step was given and what it produced, for Langfuse.

        A no-op when tracing is off, which is most of the time. Only the stages
        whose input and output a human would actually want to read call this -
        an observation with neither is noise, and one carrying a dump of
        everything in scope is worse.
        """
        fields = {}
        if input is not None:
            fields["input"] = input
        if output is not None:
            fields["output"] = output
        if metadata:
            fields["metadata"] = metadata
        if fields:
            self._observation.update(**fields)


class StageRecorder:
    """Collects one record per stage for a single run, and emits them.

    Lines stream as each stage completes rather than being buffered to the end,
    so a run that hangs still shows the stage it hung in. `flush()` then emits
    whatever was never reached, which is what makes the ledger complete.
    """

    def __init__(self, lifecycle: str, log=None, enabled: bool = True, **fields):
        self.lifecycle = lifecycle
        self.enabled = enabled
        self._fields = fields
        self._seen: set[int] = set()
        self._flushed = False
        self._log = log if log is not None else _default_logger()

    # -- recording ---------------------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Iterator[_Open]:
        """Time a block, record it as `ran`, and re-raise anything it throws.

        A stage that raises is recorded as `failed` and the exception continues
        on its way. Instrumentation reports the failure; it does not handle it.

        This is also the one place Langfuse observations are created, for the
        reason `agent/observability.py` states: the trace tree and this ledger
        are the same events, and building them from two sets of call sites is
        how they drift. The observation is opened around the same block, so the
        duration Langfuse reports is the duration logged here, and nesting
        follows the call stack without anybody passing a parent around.
        """
        with _observability().stage_observation(name) as observation:
            handle = _Open(observation)
            started = time.perf_counter()
            try:
                yield handle
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                self._emit(
                    name,
                    Status.FAILED,
                    ms=_elapsed_ms(started),
                    reason=f"{type(exc).__name__}: {exc}",
                )
                # Langfuse renders an errored observation differently, which is
                # the difference between finding the failure and reading twelve
                # green spans looking for it.
                try:
                    observation.update(
                        level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                    )
                except Exception:  # noqa: BLE001 - see the module docstring
                    pass
                raise
            self._emit(name, Status.RAN, ms=_elapsed_ms(started), detail=handle._detail)

    def ran(self, name: str, detail: str = "", ms: int = 0) -> None:
        """Record a stage that ran, where wrapping it in a block does not fit."""
        self._emit(name, Status.RAN, ms=ms, detail=detail)

    def skip(self, name: str, reason: str) -> None:
        """Record a stage that exists but did not run on this path."""
        self._emit(name, Status.SKIPPED, reason=reason)

    def degrade(self, name: str, reason: str) -> None:
        """Record a stage that ran in a reduced form.

        Distinct from `skip` on purpose: a degraded stage produced output, and
        anything comparing this run against another needs to know the output
        was not the thing the configuration asked for.

        This is the one status that also reaches Langfuse outside a block, as a
        warning-level event. A skipped stage is usually just the profile doing
        what it was told; a degraded one means the run served something the
        configuration did not ask for, and that is worth finding in a trace
        rather than only in a log nobody greps until afterwards.
        """
        self._emit(name, Status.DEGRADED, reason=reason)
        try:
            observability = _observability()
            if getattr(observability, "tracing", lambda: False)():
                observability.event(
                    name=f"degraded-{name.replace('_', '-')}",
                    level="WARNING",
                    status_message=reason,
                )
        except Exception:  # noqa: BLE001 - see the module docstring
            pass

    def skip_remaining(self, reason: str) -> None:
        """Attribute every stage not yet reported to one cause.

        For the short-circuits: the gate refusing means steps 16-20 never run,
        and they all did not run for that one reason. Recording it beats twelve
        blank `skipped` lines that make the reader go and find out why.
        Stages declared `not_implemented` keep their own reason - they were not
        skipped by this run, they do not exist.
        """
        try:
            for stage in STAGES:
                if stage.lifecycle != self.lifecycle or stage.id in self._seen:
                    continue
                if stage.declared is Status.NOT_IMPLEMENTED:
                    continue
                self._emit(stage.name, Status.SKIPPED, reason=reason)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass

    # -- emission ----------------------------------------------------------

    def flush(self) -> None:
        """Emit every stage on this lifecycle that nothing has reported yet."""
        if self._flushed:
            return
        self._flushed = True
        try:
            for stage in STAGES:
                if stage.lifecycle != self.lifecycle or stage.id in self._seen:
                    continue
                self._write(stage, stage.declared, reason=stage.reason)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass

    def _emit(self, name: str, status: Status, ms: int = 0, detail: str = "", reason: str = "") -> None:
        try:
            stage = _BY_NAME.get(name)
            # An unknown name, or one belonging to the other pipeline, is a
            # call-site bug. Dropping it keeps the ledger honest about this
            # lifecycle and keeps a typo from becoming an exception in a
            # request path that has nothing to do with logging.
            if stage is None or stage.lifecycle != self.lifecycle:
                return
            if stage.id in self._seen:
                return
            self._seen.add(stage.id)
            self._write(stage, status, ms=ms, detail=detail, reason=reason)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass

    def _write(self, stage: Stage, status: Status, ms: int = 0, detail: str = "", reason: str = "") -> None:
        if not self.enabled or self._log is None:
            return

        # The message is column-aligned for reading; the same values repeat as
        # structured fields for the JSON handler the deployed profile uses.
        message = (
            f"stage {stage.id:02d} {stage.name:<{_NAME_WIDTH}} "
            f"{status.value:<{_STATUS_WIDTH}} {ms:>5}ms {detail or reason}".rstrip()
        )
        # `detail` and `reason` are deliberately not repeated as structured
        # fields. Both formatters in `logger/zap.py` already carry the message -
        # the JSON one as `message`, the text one as the line itself - and
        # `TextFormatter` appends every extra as JSON, so passing them twice
        # doubles the length of the one log line a human is going to read.
        self._log.info(
            message,
            step=stage.id,
            stage=stage.name,
            lifecycle=stage.lifecycle,
            status=status.value,
            ms=ms,
            **self._fields,
        )


class _NullRecorder(StageRecorder):
    """What `current_recorder()` hands back when no run is in progress.

    A real recorder with logging off, rather than a separate no-op type, so
    there is one code path and a call site cannot accidentally depend on the
    difference.
    """

    def __init__(self):
        super().__init__(QUERY, log=None, enabled=False)


_NULL = _NullRecorder()
_current: ContextVar[Optional[StageRecorder]] = ContextVar("stage_recorder", default=None)


def current_recorder() -> StageRecorder:
    """The recorder for the run in progress, or a silent one.

    Never returns None. That is what lets a store or a node record a stage
    without threading a recorder through every signature and without a None
    check at every call site.
    """
    return _current.get() or _NULL


@contextmanager
def using_recorder(recorder: StageRecorder) -> Iterator[StageRecorder]:
    """Scope a recorder to a block, and flush it on the way out.

    Flushes even when the body raises: a run that blew up is precisely the run
    whose ledger you want, and it is the one a `finally`-less implementation
    would lose.
    """
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        recorder.flush()
        _current.reset(token)


def new_recorder(lifecycle: str, cfg=None, **fields) -> StageRecorder:
    """A recorder configured from settings - `STAGE_LOG` decides if it writes."""
    try:
        from agent.config import settings as default_settings

        cfg = cfg or default_settings
        enabled = bool(getattr(cfg, "stage_log", True))
    except Exception:  # noqa: BLE001 - never block a run on configuration
        enabled = True

    return StageRecorder(lifecycle, enabled=enabled, **fields)
