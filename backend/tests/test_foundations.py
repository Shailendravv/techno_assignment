"""Phase 0: the skeleton holds together.

These are cheap structural checks, not behaviour tests. Their job is to fail
loudly if the contracts drift - particularly the dict shape that
`answer_question()` must return, which is specified by the brief and is not
ours to change.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent.config import NO_MATCH_MESSAGE, Settings, load_settings
from agent.core.models import Answer, Candidate, Chunk, Doc, QuerySpec
from agent.llm import LLMBadJSON, extract_json
from agent.state import new_state
from app.main import app


def test_answer_serialises_to_the_three_required_fields():
    result = Answer("some text", ["RB-001"], "high").as_dict()
    assert set(result) == {"answer", "cited_doc_ids", "confidence"}
    assert result["cited_doc_ids"] == ["RB-001"]


def test_answer_defaults_to_no_match_with_no_citations():
    result = Answer(NO_MATCH_MESSAGE).as_dict()
    assert result["confidence"] == "no_match"
    assert result["cited_doc_ids"] == []


def test_docs_without_a_service_are_general():
    """General policy docs apply to every service, so they must never be
    dropped for 'mismatching' one. This flag is what the filter keys off."""
    policy = Doc("RB-011", "Comms policy", None, None, "policy", None, "...")
    runbook = Doc("RB-001", "CPU", "checkout-api", "cpu", "runbook", None, "...")
    assert policy.is_general
    assert not runbook.is_general


def test_dropping_a_candidate_records_why():
    doc = Doc("RB-003", "CPU", "payments-api", "cpu", "runbook", None, "...")
    candidate = Candidate(doc=doc, lexical_score=7.1)
    candidate.drop("service mismatch: payments-api != checkout-api")
    assert candidate.verdict == "dropped"
    assert "payments-api" in candidate.reason


def test_chunk_carries_its_parent_document():
    """Retrieval is chunk-level; citation is document-level."""
    chunk = Chunk("RB-001#first-checks", "RB-001", "First checks", "...")
    assert chunk.doc_id == "RB-001"


def test_state_starts_with_zero_llm_calls():
    state = new_state("anything")
    assert state["llm_calls"] == 0
    assert state["trace"] == []
    assert state["rewrites"] == 0


def test_models_are_addressed_by_role_not_name():
    cfg = load_settings()
    assert isinstance(cfg, Settings)
    for role in ("generator", "reasoner", "grader"):
        assert getattr(cfg.models, role)


def test_query_spec_accepts_a_fully_unknown_question():
    """A question naming no service and no failure mode is normal, not an
    error - it is how off-topic questions arrive."""
    spec = QuerySpec(raw="what's our refund policy?", service=None,
                     failure_mode=None, intent="policy", date=None)
    assert spec.service is None


class TestJSONExtraction:
    """Models fence JSON, prefix it with prose, and - if they reason out loud -
    bury it in a <think> block. All three arrive in practice."""

    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_with_surrounding_prose(self):
        text = 'Sure!\n{"cited_doc_ids": ["RB-001"]}\nHope that helps.'
        assert extract_json(text) == {"cited_doc_ids": ["RB-001"]}

    def test_after_a_think_block(self):
        assert extract_json("<think>weighing it up</think>{\"a\": 3}") == {"a": 3}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('x {"s": "a}b", "n": {"c": 1}} y') == {
            "s": "a}b", "n": {"c": 1}
        }

    def test_unparseable_raises(self):
        try:
            extract_json("no json here")
        except LLMBadJSON:
            return
        raise AssertionError("expected LLMBadJSON")


def test_health_reports_configuration():
    """Health must never 500, even before the corpus exists - it is also the
    endpoint the keep-alive cron will hit."""
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["retrieval_mode"] in {"lexical", "hybrid"}
