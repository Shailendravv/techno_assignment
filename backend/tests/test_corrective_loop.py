"""Phase 5: the CRAG grader, the rewrite cycle, and the bound on it.

Every model call is stubbed, so these run with no key and no network. What is
being tested is the *wiring* - which nodes run, in what order, how many times,
and whether the loop can run away.

The assertion the free tier depends on is
`test_the_rewrite_loop_is_bounded_by_max_rewrites`. An unbounded corrective
cycle against a tier allowing roughly two questions a minute is not a
theoretical hazard, and the bound lives in a routing edge specifically so it
can be read rather than trusted.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.api import answer_question
from agent.config import load_settings
from agent.core.corpus import load_corpus
from agent.core.models import Candidate
from agent.nodes.grade import grade_candidates, rewrite_query


@pytest.fixture
def cfg():
    """Grading on, lexical retrieval, so these tests exercise one thing."""
    base = load_settings("local")
    return replace(
        base,
        retrieval=replace(base.retrieval, grader_enabled=True, mode="lexical"),
        groq_api_key="test-key-not-used-because-everything-is-stubbed",
    )


@pytest.fixture
def docs():
    return load_corpus("runbooks")


class Script:
    """A stubbed `chat_json` that replies from a queue and records its calls."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.roles: list[str] = []
        self.prompts: list[str] = []

    def __call__(self, messages, role="generator", cfg=None, max_tokens=1024):
        self.roles.append(role)
        self.prompts.append("\n".join(m["content"] for m in messages))
        reply = self.replies.pop(0) if self.replies else {"answer": "", "cited_doc_ids": []}
        return dict(reply), 1

    @property
    def calls(self) -> int:
        return len(self.roles)


def _patch_all(monkeypatch, script: Script) -> None:
    """One script drives grading, rewriting and grounding alike."""
    monkeypatch.setattr("agent.nodes.grade.chat_json", script)
    monkeypatch.setattr("agent.nodes.ground.chat_json", script)


# --------------------------------------------------------------------------
# The grader, on its own.
# --------------------------------------------------------------------------

def test_the_grader_keeps_only_what_it_judges_relevant(monkeypatch, docs, cfg):
    script = Script({"relevant": ["RB-001"], "reason": "matches service and failure"})
    monkeypatch.setattr("agent.nodes.grade.chat_json", script)

    candidates = [Candidate(doc=d) for d in docs[:3]]
    kept, reason, calls = grade_candidates("checkout-api CPU", candidates, cfg=cfg)

    assert [c.doc_id for c in kept] == ["RB-001"]
    assert calls == 1
    assert "matches service" in reason


def test_the_grader_records_why_it_dropped_each_candidate(monkeypatch, docs, cfg):
    script = Script({"relevant": [], "reason": "all about other services"})
    monkeypatch.setattr("agent.nodes.grade.chat_json", script)

    candidates = [Candidate(doc=d) for d in docs[:2]]
    grade_candidates("q", candidates, cfg=cfg)

    assert all(c.verdict == "dropped" for c in candidates)
    assert "other services" in candidates[0].reason


def test_the_grader_cannot_resurrect_a_document_it_was_not_shown(
    monkeypatch, docs, cfg
):
    """An invented ID here would undo the metadata filter's hard drop."""
    script = Script({"relevant": ["RB-001", "RB-999"], "reason": "x"})
    monkeypatch.setattr("agent.nodes.grade.chat_json", script)

    candidates = [Candidate(doc=d) for d in docs[:3]]
    kept, _, _ = grade_candidates("q", candidates, cfg=cfg)

    assert [c.doc_id for c in kept] == ["RB-001"]


def test_a_broken_grader_fails_open_rather_than_refusing_everything(
    monkeypatch, docs, cfg
):
    """The grader is one of three gates. Failing closed would silently turn
    every answerable question into no_match the moment it had a bad day."""
    from agent.llm import LLMBadJSON

    def explode(*args, **kwargs):
        raise LLMBadJSON("not json")

    monkeypatch.setattr("agent.nodes.grade.chat_json", explode)

    candidates = [Candidate(doc=d) for d in docs[:3]]
    kept, reason, calls = grade_candidates("q", candidates, cfg=cfg)

    assert [c.doc_id for c in kept] == [c.doc_id for c in candidates]
    assert calls == 0
    assert "unavailable" in reason


def test_grading_nothing_costs_no_call(cfg):
    kept, _, calls = grade_candidates("q", [], cfg=cfg)

    assert kept == [] and calls == 0


# --------------------------------------------------------------------------
# The rewriter.
# --------------------------------------------------------------------------

def test_a_rewrite_replaces_the_query(monkeypatch, cfg):
    monkeypatch.setattr(
        "agent.nodes.grade.chat_json",
        Script({"query": "checkout-api high CPU utilisation"}),
    )

    rewritten, calls = rewrite_query("the boxes are working too hard", cfg=cfg)

    assert rewritten == "checkout-api high CPU utilisation"
    assert calls == 1


def test_an_empty_rewrite_falls_back_to_the_original(monkeypatch, cfg):
    monkeypatch.setattr("agent.nodes.grade.chat_json", Script({"query": "   "}))

    rewritten, _ = rewrite_query("original question", cfg=cfg)

    assert rewritten == "original question"


def test_a_runaway_rewrite_is_rejected(monkeypatch, cfg):
    """A rewrite far longer than the question is a model that misunderstood the
    task, not a better query."""
    monkeypatch.setattr(
        "agent.nodes.grade.chat_json", Script({"query": "word " * 500})
    )

    rewritten, _ = rewrite_query("short question", cfg=cfg)

    assert rewritten == "short question"


def test_a_broken_rewriter_returns_the_original_question(monkeypatch, cfg):
    from agent.llm import LLMUnavailable

    def explode(*args, **kwargs):
        raise LLMUnavailable("no key")

    monkeypatch.setattr("agent.nodes.grade.chat_json", explode)

    rewritten, calls = rewrite_query("original", cfg=cfg)

    assert rewritten == "original" and calls == 0


# --------------------------------------------------------------------------
# The loop, through the graph.
# --------------------------------------------------------------------------

def test_relevant_candidates_route_straight_to_grounding(monkeypatch, cfg):
    script = Script(
        {"relevant": ["RB-001"], "reason": "matches"},
        {"answer": "Check the deploy log first.", "cited_doc_ids": ["RB-001"]},
    )
    _patch_all(monkeypatch, script)

    result = answer_question(
        "checkout-api is running hot on CPU - what should I check first?",
        cfg=cfg,
        with_trace=True,
    )

    assert result["cited_doc_ids"] == ["RB-001"]
    assert script.roles == ["grader", "generator"]
    assert not any("rewrite" in line for line in result["trace"])


def test_nothing_relevant_triggers_exactly_one_rewrite(monkeypatch, cfg):
    script = Script(
        {"relevant": [], "reason": "nothing applies"},
        {"query": "checkout-api high CPU"},
        {"relevant": ["RB-001"], "reason": "now it matches"},
        {"answer": "Check the deploy log.", "cited_doc_ids": ["RB-001"]},
    )
    _patch_all(monkeypatch, script)

    result = answer_question(
        "checkout-api is running hot on CPU - what should I check first?",
        cfg=cfg,
        with_trace=True,
    )

    assert script.roles == ["grader", "grader", "grader", "generator"]
    assert result["cited_doc_ids"] == ["RB-001"]
    assert sum("rewrite:" in line for line in result["trace"]) == 1


def test_the_rewrite_loop_is_bounded_by_max_rewrites(monkeypatch, cfg):
    """The assertion the free tier depends on.

    The grader refuses everything, every time, and the rewrite it produces is a
    good one that retrieves successfully - so nothing except the bound stops
    this cycling forever.
    """
    script = Script(
        *([
            {"relevant": [], "reason": "no"},
            {"query": "checkout-api high CPU utilisation runbook"},
        ] * 20)
    )
    _patch_all(monkeypatch, script)

    result = answer_question(
        "checkout-api is running hot on CPU - what should I check first?",
        cfg=cfg,
        with_trace=True,
    )

    assert result["confidence"] == "no_match"
    assert result["cited_doc_ids"] == []
    # grade, rewrite, grade - then the budget is spent and the edge routes out.
    assert script.calls == 3
    assert sum("rewrite:" in line for line in result["trace"]) == cfg.max_rewrites


def test_a_grader_that_rejects_everything_never_reaches_the_generator(
    monkeypatch, cfg
):
    """The same guarantee the retrieval gate gives, one stage later: a model
    that is never shown a document cannot invent a citation for one."""
    script = Script(*([{"relevant": [], "reason": "no"}, {"query": "retry"}] * 4))
    _patch_all(monkeypatch, script)

    result = answer_question(
        "checkout-api is running hot on CPU - what should I check first?",
        cfg=cfg,
        with_trace=True,
    )

    assert "generator" not in script.roles
    assert result["confidence"] == "no_match"


def test_the_gate_still_short_circuits_before_the_grader(monkeypatch, cfg):
    """An off-topic question must cost zero calls - grading included."""
    script = Script()
    _patch_all(monkeypatch, script)

    result = answer_question(
        "How many vacation days do engineers get?", cfg=cfg, with_trace=True
    )

    assert script.calls == 0
    assert result["llm_calls"] == 0
    assert result["confidence"] == "no_match"


def test_the_grounding_model_sees_the_original_question_not_the_rewrite(
    monkeypatch, cfg
):
    """The rewrite is a retrieval device. The user asked their question, and
    answering a paraphrase of it would be answering the wrong question."""
    script = Script(
        {"relevant": [], "reason": "no"},
        {"query": "checkout-api high CPU utilisation"},
        {"relevant": ["RB-001"], "reason": "yes"},
        {"answer": "ok", "cited_doc_ids": ["RB-001"]},
    )
    _patch_all(monkeypatch, script)

    original = "checkout-api is running hot on CPU - what should I check first?"
    answer_question(original, cfg=cfg, with_trace=True)

    assert original in script.prompts[-1]
    assert "checkout-api high CPU utilisation" not in script.prompts[-1]


def test_the_grader_can_be_switched_off_entirely(monkeypatch):
    """`GRADER_ENABLED=false` must cost nothing, not merely keep everything."""
    base = load_settings("local")
    off = replace(
        base,
        retrieval=replace(base.retrieval, grader_enabled=False, mode="lexical"),
        groq_api_key="stub",
    )
    script = Script({"answer": "ok", "cited_doc_ids": ["RB-001"]})
    _patch_all(monkeypatch, script)

    result = answer_question(
        "checkout-api is running hot on CPU - what should I check first?",
        cfg=off,
        with_trace=True,
    )

    assert script.roles == ["generator"]
    assert result["cited_doc_ids"] == ["RB-001"]
