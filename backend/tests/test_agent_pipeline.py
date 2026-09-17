"""Phase 3: the graph, end to end, with the model stubbed.

The model is replaced by a fake so these tests run with no API key, no network,
and no rate limit - and so that they test *our* behaviour rather than Groq's.
What matters here is the wiring: that the gate routes around the grounding node
entirely, that invented citations are stripped, and that the dict coming out
has exactly the shape the brief specifies.

The one assertion worth singling out is `llm_calls == 0` on the no_match path.
That is not a performance check. It is the guarantee that a model which was
never shown a document cannot have invented a citation for one.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agent.api import answer_question
from agent.core.corpus import by_id, load_corpus
from agent.core.models import Candidate
from agent.nodes.ground import build_messages, verify_citations
from app.main import app


@pytest.fixture(scope="module")
def lookup():
    return by_id(load_corpus("runbooks"))


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the Groq call with a scripted reply.

    Returns a recorder so tests can assert on what the model was actually
    shown - which documents reached it, and whether it was called at all.
    """

    class Recorder:
        def __init__(self):
            self.calls: list[list[dict]] = []
            self.reply: dict = {"answer": "stub", "cited_doc_ids": []}

        def __call__(self, messages, role="generator", cfg=None, max_tokens=1024):
            self.calls.append(messages)
            return dict(self.reply), 1

        @property
        def prompt(self) -> str:
            return "\n".join(m["content"] for m in self.calls[-1])

    recorder = Recorder()
    monkeypatch.setattr("agent.nodes.ground.chat_json", recorder)
    return recorder


# --------------------------------------------------------------------------
# The contract.
# --------------------------------------------------------------------------

def test_the_result_has_exactly_the_three_specified_fields(fake_llm):
    fake_llm.reply = {"answer": "Check the deploy log.", "cited_doc_ids": ["RB-001"]}
    result = answer_question("checkout-api is running hot on CPU")

    assert set(result) == {"answer", "cited_doc_ids", "confidence"}
    assert result["confidence"] in {"high", "medium", "low", "no_match"}


def test_an_empty_question_is_handled_without_calling_the_model(fake_llm):
    result = answer_question("   ")

    assert result["confidence"] == "no_match"
    assert fake_llm.calls == []


# --------------------------------------------------------------------------
# The gate routes around the grounding node. This is the safety property.
# --------------------------------------------------------------------------

def test_a_gated_question_never_reaches_the_model(fake_llm):
    result = answer_question(
        "What is the runbook for search-api high latency?", with_trace=True
    )

    assert result["confidence"] == "no_match"
    assert result["cited_doc_ids"] == []
    assert result["llm_calls"] == 0
    assert fake_llm.calls == [], "the model was called on a question that was gated"


def test_an_off_topic_question_is_gated(fake_llm):
    result = answer_question("How many vacation days do engineers get?",
                             with_trace=True)

    assert result["confidence"] == "no_match"
    assert result["llm_calls"] == 0


def test_the_trace_records_why_the_gate_rejected(fake_llm):
    result = answer_question(
        "How do I restart the recommendation-engine service?", with_trace=True
    )

    trace = " ".join(result["trace"])
    assert "REJECT" in trace
    assert "recommendation-engine" in trace


# --------------------------------------------------------------------------
# What the model is shown.
# --------------------------------------------------------------------------

def test_the_model_only_sees_documents_that_survived_the_filter(fake_llm):
    """RB-003 is a 69% token match for RB-001 on the wrong service. It must not
    reach the model at all - the filter is a hard drop, not a re-rank."""
    fake_llm.reply = {"answer": "Check the deploy log.", "cited_doc_ids": ["RB-001"]}
    answer_question("checkout-api is running hot on CPU")

    prompt = fake_llm.prompt
    assert "BEGIN RB-001" in prompt
    assert "BEGIN RB-003" not in prompt, "a wrong-service document reached the model"
    assert "BEGIN RB-004" not in prompt, "a wrong-failure document reached the model"


def test_the_prompt_tells_the_model_that_declining_is_correct(fake_llm):
    """The second gate only works if the model has explicit permission to use
    it. Without this the prompt is an instruction to find an answer."""
    fake_llm.reply = {"answer": "x", "cited_doc_ids": ["RB-001"]}
    answer_question("checkout-api is running hot on CPU")

    prompt = fake_llm.prompt.lower()
    assert "correct and expected" in prompt
    assert "empty citation list" in prompt


def test_the_prompt_carries_structured_metadata_for_each_document(fake_llm):
    fake_llm.reply = {"answer": "x", "cited_doc_ids": ["RB-001"]}
    answer_question("checkout-api is running hot on CPU")

    prompt = fake_llm.prompt
    assert "service: checkout-api" in prompt
    assert "failure mode: cpu" in prompt


# --------------------------------------------------------------------------
# Citation verification.
# --------------------------------------------------------------------------

def test_invented_citations_are_stripped(fake_llm):
    """RB-007 is an easy string to invent after reading RB-001 to RB-005. A
    citation to a document that was never supplied is worse than no citation,
    because it looks authoritative."""
    fake_llm.reply = {"answer": "See RB-007.", "cited_doc_ids": ["RB-007"]}
    result = answer_question("checkout-api is running hot on CPU", with_trace=True)

    assert result["cited_doc_ids"] == []
    assert result["confidence"] == "no_match"
    assert "invented" in " ".join(result["trace"])


def test_a_real_citation_alongside_an_invented_one_survives(fake_llm):
    fake_llm.reply = {
        "answer": "Check the deploy log.",
        "cited_doc_ids": ["RB-001", "RB-099"],
    }
    result = answer_question("checkout-api is running hot on CPU")

    assert result["cited_doc_ids"] == ["RB-001"]


def test_citation_ids_are_normalised(lookup):
    candidates = [Candidate(doc=lookup["RB-001"])]
    verified, invented = verify_citations([" rb-001 ", "RB-001"], candidates)

    assert verified == ["RB-001"], "case and whitespace should not matter"
    assert invented == []


def test_a_model_that_declines_produces_no_match(fake_llm):
    """The second gate doing its job: candidates passed the lexical gate, the
    model read them and said none applied."""
    fake_llm.reply = {
        "answer": "None of these documents cover refunds.",
        "cited_doc_ids": [],
    }
    result = answer_question("What is our refund policy for orders over $500?")

    assert result["confidence"] == "no_match"
    assert result["cited_doc_ids"] == []
    assert "refund" in result["answer"].lower()


def test_a_string_instead_of_a_list_is_tolerated(fake_llm):
    """Models return `"RB-001"` instead of `["RB-001"]` often enough to handle."""
    fake_llm.reply = {"answer": "Check the deploy log.", "cited_doc_ids": "RB-001"}
    result = answer_question("checkout-api is running hot on CPU")

    assert result["cited_doc_ids"] == ["RB-001"]


# --------------------------------------------------------------------------
# Confidence.
# --------------------------------------------------------------------------

def test_matching_service_and_failure_mode_gives_high_confidence(fake_llm):
    fake_llm.reply = {"answer": "Check the deploy log.", "cited_doc_ids": ["RB-001"]}
    result = answer_question("checkout-api is running hot on CPU")

    assert result["confidence"] == "high"


def test_a_policy_question_answered_by_a_policy_doc_is_not_penalised(fake_llm):
    """Question 5 names no service and no failure mode, so there is nothing
    structural to match on. That is the expected shape of a good answer here,
    not a weak one."""
    fake_llm.reply = {"answer": "Tell customers early.", "cited_doc_ids": ["RB-011"]}
    result = answer_question("What is our policy for communicating an incident to customers?")

    assert result["confidence"] in {"high", "medium"}


# --------------------------------------------------------------------------
# The HTTP surface calls the same function.
# --------------------------------------------------------------------------

def test_post_ask_returns_the_contract_fields(fake_llm):
    fake_llm.reply = {"answer": "Check the deploy log.", "cited_doc_ids": ["RB-001"]}
    response = TestClient(app).post(
        "/ask", json={"question": "checkout-api is running hot on CPU"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cited_doc_ids"] == ["RB-001"]
    assert body["confidence"] == "high"
    assert body["elapsed_ms"] >= 0


def test_post_ask_returns_200_for_no_match_not_an_error(fake_llm):
    """Declining is a successful outcome. Signalling it as an error would
    invite clients to retry it or hide it."""
    response = TestClient(app).post(
        "/ask", json={"question": "How many vacation days do engineers get?"}
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == "no_match"


def test_post_ask_rejects_an_empty_question():
    response = TestClient(app).post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_post_ask_rejects_an_unknown_model_role():
    response = TestClient(app).post(
        "/ask", json={"question": "anything", "model_role": "gpt-9"}
    )
    assert response.status_code == 422
