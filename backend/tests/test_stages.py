"""The stage ledger: one log line per pipeline step, every run, no exceptions.

The property under test throughout is **completeness, not correctness of
detail**. A log that prints what ran is easy; the reason this module exists is
the steps that *don't* run. A grader disabled by profile, a dense arm that fell
back to lexical because no embedder was installed, an OCR step that was never
built - all three are invisible in a log that only records what happened, and
all three change what an answer is worth.

So the ledger declares its stages up front and reports every one of them with a
status. A stage that never executed still emits a line saying why. That turns
"which part of the pipeline is actually running?" from an archaeology exercise
into one `grep`.

The second property is the one `agent/observability.py` already states for
tracing and `agent/cache.py` for caching: **instrumentation may never fail a
request.** A recorder whose logger throws must lose its line, not the answer.
"""

from __future__ import annotations

import pytest

from agent.stages import (
    INGEST_STAGES,
    QUERY_STAGES,
    STAGES,
    Status,
    StageRecorder,
    current_recorder,
    stage_id,
    using_recorder,
)


class _CollectingLogger:
    """Stands in for `logger.zap.Logger`, keeping records instead of writing."""

    def __init__(self):
        self.lines: list[tuple[str, dict]] = []

    def info(self, msg: str, **kwargs) -> None:
        self.lines.append((msg, kwargs))

    debug = warning = error = info


class _BrokenLogger:
    def info(self, msg: str, **kwargs) -> None:
        raise RuntimeError("the logging backend is down")

    debug = warning = error = info


@pytest.fixture
def log():
    return _CollectingLogger()


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def test_the_registry_covers_twenty_steps_numbered_one_to_twenty():
    assert len(STAGES) == 20
    assert [s.id for s in STAGES] == list(range(1, 21))


def test_every_stage_belongs_to_exactly_one_lifecycle():
    assert len(INGEST_STAGES) + len(QUERY_STAGES) == len(STAGES)
    assert {s.id for s in INGEST_STAGES}.isdisjoint({s.id for s in QUERY_STAGES})


def test_stage_names_are_unique():
    names = [s.name for s in STAGES]
    assert len(set(names)) == len(names)


def test_a_stage_declared_unimplemented_says_why():
    """An unexplained gap in the log is a question, not an answer."""
    unimplemented = [s for s in STAGES if s.declared is Status.NOT_IMPLEMENTED]

    assert unimplemented, "the registry should declare the known gaps"
    for stage in unimplemented:
        assert stage.reason, f"stage {stage.id} ({stage.name}) declares no reason"


# ---------------------------------------------------------------------------
# Completeness - the point of the whole module
# ---------------------------------------------------------------------------

def test_a_query_run_that_records_nothing_still_reports_every_query_stage(log):
    """The empty run is the honest test: no line may go missing by default."""
    recorder = StageRecorder("query", log=log)
    recorder.flush()

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert reported == {s.id for s in QUERY_STAGES}


def test_an_ingest_run_reports_every_ingest_stage(log):
    recorder = StageRecorder("ingest", log=log)
    recorder.flush()

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert reported == {s.id for s in INGEST_STAGES}


def test_a_lifecycle_reports_only_its_own_stages(log):
    """A query run must not claim it skipped OCR - OCR is not on its path."""
    StageRecorder("query", log=log).flush()

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert reported.isdisjoint({s.id for s in INGEST_STAGES})


def test_no_stage_is_reported_twice(log):
    recorder = StageRecorder("query", log=log)
    with recorder.stage("sparse_retrieve"):
        pass
    recorder.flush()

    steps = [kwargs["step"] for _, kwargs in log.lines]
    assert len(steps) == len(set(steps))


def test_flushing_twice_emits_nothing_the_second_time(log):
    recorder = StageRecorder("query", log=log)
    recorder.flush()
    before = len(log.lines)
    recorder.flush()

    assert len(log.lines) == before


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------

def test_a_stage_that_ran_is_recorded_with_its_detail_and_timing(log):
    recorder = StageRecorder("query", log=log)
    with recorder.stage("sparse_retrieve") as stage:
        stage.detail("top=RB-001(8.2)")
    recorder.flush()

    entry = _entry(log, stage_id("sparse_retrieve"))
    assert entry["status"] == "ran"
    assert "top=RB-001(8.2)" in entry["message"]
    assert entry["ms"] >= 0


def test_a_disabled_grader_is_reported_as_skipped_with_its_reason(log):
    """The case that prompted this module. Silence here reads as 'it ran'."""
    recorder = StageRecorder("query", log=log)
    recorder.skip("relevance_grade", "GRADER_ENABLED=false")
    recorder.flush()

    entry = _entry(log, stage_id("relevance_grade"))
    assert entry["status"] == "skipped"
    assert "GRADER_ENABLED=false" in entry["message"]


def test_a_dense_arm_that_fell_back_is_reported_as_degraded(log):
    """Hybrid-requested-but-lexical-served is the failure that hides itself."""
    recorder = StageRecorder("query", log=log)
    recorder.degrade("dense_retrieve", "no embedder available")
    recorder.flush()

    entry = _entry(log, stage_id("dense_retrieve"))
    assert entry["status"] == "degraded"
    assert "no embedder available" in entry["message"]


def test_an_unimplemented_stage_reports_its_declared_reason_unprompted(log):
    """Nobody calls `skip()` for a step that was never written - the registry does."""
    StageRecorder("query", log=log).flush()

    entry = _entry(log, stage_id("parallel_retrieve"))
    assert entry["status"] == "not_implemented"
    assert entry["message"].strip() != f"stage {entry['step']:02d}"


def test_skip_remaining_attributes_untouched_stages_to_one_cause(log):
    recorder = StageRecorder("query", log=log)
    with recorder.stage("sparse_retrieve"):
        pass
    recorder.skip_remaining("gate rejected; the model was never called")
    recorder.flush()

    assert _entry(log, stage_id("sparse_retrieve"))["status"] == "ran"
    assert "gate rejected; the model was never called" in _entry(
        log, stage_id("generate")
    )["message"]


def test_skip_remaining_leaves_unimplemented_stages_with_their_own_reason(log):
    """A step that does not exist was not skipped by this run."""
    recorder = StageRecorder("query", log=log)
    recorder.skip_remaining("cache hit")
    recorder.flush()

    entry = _entry(log, stage_id("parallel_retrieve"))
    assert entry["status"] == "not_implemented"
    assert "sequentially" in entry["message"]


def test_a_stage_that_raises_is_recorded_as_failed_and_the_error_propagates(log):
    recorder = StageRecorder("query", log=log)

    with pytest.raises(ValueError, match="boom"):
        with recorder.stage("dense_retrieve"):
            raise ValueError("boom")
    recorder.flush()

    entry = _entry(log, stage_id("dense_retrieve"))
    assert entry["status"] == "failed"
    assert "boom" in entry["message"]


# ---------------------------------------------------------------------------
# It may never fail a request
# ---------------------------------------------------------------------------

def test_a_broken_logger_never_reaches_the_caller():
    recorder = StageRecorder("query", log=_BrokenLogger())

    with recorder.stage("sparse_retrieve") as stage:
        stage.detail("anything")
    recorder.flush()  # must not raise


def test_recording_an_unknown_stage_is_ignored_rather_than_raising(log):
    recorder = StageRecorder("query", log=log)
    recorder.skip("no_such_stage", "typo in a call site")  # must not raise

    assert stage_id("no_such_stage") is None


def test_a_stage_on_another_lifecycle_is_ignored_rather_than_raising(log):
    """`chunk` is real, but it is not on the query path."""
    recorder = StageRecorder("query", log=log)
    recorder.skip("chunk", "wrong lifecycle")
    recorder.flush()

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert stage_id("chunk") not in reported


def test_disabling_stage_logging_emits_nothing(log):
    recorder = StageRecorder("query", log=log, enabled=False)
    with recorder.stage("sparse_retrieve"):
        pass
    recorder.flush()

    assert log.lines == []


# ---------------------------------------------------------------------------
# The context-scoped recorder
# ---------------------------------------------------------------------------

def test_without_a_run_in_progress_the_recorder_is_a_silent_no_op():
    """Call sites must never need a None check - `agent.core` stays pure."""
    recorder = current_recorder()

    with recorder.stage("sparse_retrieve") as stage:
        stage.detail("no run is in progress")
    recorder.skip("relevance_grade", "still fine")
    recorder.flush()  # must not raise


def test_using_recorder_scopes_the_current_recorder_and_restores_it(log):
    recorder = StageRecorder("query", log=log)

    with using_recorder(recorder):
        assert current_recorder() is recorder

    assert current_recorder() is not recorder


def test_using_recorder_flushes_on_the_way_out_even_when_the_body_raises(log):
    recorder = StageRecorder("query", log=log)

    with pytest.raises(ValueError):
        with using_recorder(recorder):
            raise ValueError("pipeline blew up")

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert reported == {s.id for s in QUERY_STAGES}


# ---------------------------------------------------------------------------
# Wiring - that the pipeline actually reports itself
# ---------------------------------------------------------------------------
# Unit tests above prove the recorder works. These prove it is plugged in, which
# is the failure mode that matters: a ledger nobody calls is worse than none,
# because it reads as "this stage did not run".

def _run_and_capture(question: str, monkeypatch) -> _CollectingLogger:
    """Answer a question with the ledger pointed at a collecting logger."""
    from agent import stages

    log = _CollectingLogger()
    monkeypatch.setattr(stages, "_default_logger", lambda: log)

    from agent.api import answer_question
    from agent.config import load_settings

    # The local profile has ANSWER_CACHE off, so every call runs the pipeline -
    # a cache hit would legitimately report twelve skipped stages and make these
    # assertions pass for the wrong reason.
    answer_question(question, cfg=load_settings("local"))
    return log


def test_a_refused_question_reports_the_whole_query_ledger(monkeypatch):
    """No network: the gate rejects this before any model is reached."""
    log = _run_and_capture("How do I get a refund on my subscription?", monkeypatch)

    reported = {kwargs["step"] for _, kwargs in log.lines}
    assert reported == {s.id for s in QUERY_STAGES}


def test_a_refused_question_shows_the_model_was_never_called(monkeypatch):
    """The safety property the graph's edges guarantee, now visible in the log."""
    log = _run_and_capture("How do I get a refund on my subscription?", monkeypatch)

    entry = _entry(log, stage_id("llm_gateway"))
    assert entry["status"] == "skipped"
    assert "gate rejected" in entry["message"]


def test_the_dense_arm_reports_itself_on_a_hybrid_run(monkeypatch):
    """Guards the question that started this: is dense retrieval running?"""
    log = _run_and_capture("payments-api latency runbook", monkeypatch)

    assert _entry(log, stage_id("dense_retrieve"))["status"] in ("ran", "degraded")
    assert _entry(log, stage_id("sparse_retrieve"))["status"] == "ran"


# ---------------------------------------------------------------------------

def _entry(log: _CollectingLogger, step: int) -> dict:
    """The structured fields for one step, plus the message line it emitted.

    `detail` and `reason` live in the message rather than in the extras - see
    `StageRecorder._write` - so assertions about them read the line a human
    would read.
    """
    for message, kwargs in log.lines:
        if kwargs["step"] == step:
            return {**kwargs, "message": message}
    raise AssertionError(f"step {step} was never reported")
