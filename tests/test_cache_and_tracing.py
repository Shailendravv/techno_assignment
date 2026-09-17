"""Phase 8: the answer cache and the trace exporter.

Both are optimisations wrapped around the pipeline, and both are tested for the
same property above all others: **neither may ever fail a request.** A cache
that raises is worse than no cache; an observability backend that raises is
worse than no observability. The pipeline is the product, and these are not.

The normalisation tests are the sharp ones. This corpus is built out of
near-duplicates, so any normalisation aggressive enough to collapse
"checkout-api" and "payments-api" would let the cache serve the wrong document
with every downstream defence bypassed - see the module docstring in
`agent/cache.py` for why there is deliberately no semantic cache.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.cache import NullCache, cache_key, get_cache, normalise_question
from agent.config import Supabase, load_settings
from agent.observability import build_payload, enabled, export_trace


@pytest.fixture
def cfg():
    return load_settings("local")


@pytest.fixture
def cached_cfg():
    base = load_settings("dev")
    return replace(
        base,
        answer_cache=True,
        supabase=Supabase(url="https://example.supabase.co", service_key="k"),
    )


# --------------------------------------------------------------------------
# Normalisation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b",
    [
        ("How do I roll back checkout-api?", "how do i roll back checkout-api"),
        ("  extra   spaces  here ", "extra spaces here"),
        ("Is it down?", "Is it down"),
        ("What’s the policy", "What's the policy"),
    ],
)
def test_spellings_of_the_same_question_collapse(a, b):
    assert normalise_question(a) == normalise_question(b)


@pytest.mark.parametrize(
    "a,b",
    [
        # The one that matters. One token apart, different documents.
        ("roll back checkout-api", "roll back payments-api"),
        ("checkout-api high CPU", "checkout-api high memory"),
        ("incident on 2026-08-10", "incident on 2026-08-11"),
    ],
)
def test_questions_about_different_documents_never_collapse(a, b):
    """No stemming, no stopword removal. The more aggressive the normalisation,
    the closer these get - and a collision here serves the wrong document with
    the filter, the gate and the grader all bypassed."""
    assert normalise_question(a) != normalise_question(b)


def test_the_key_covers_the_configuration_not_just_the_question(cfg):
    """A lexical-arm answer served to a hybrid request would make an A/B
    comparison read its own cache."""
    lexical = replace(cfg, retrieval=replace(cfg.retrieval, mode="lexical"))
    hybrid = replace(cfg, retrieval=replace(cfg.retrieval, mode="hybrid"))

    assert cache_key("same question", lexical) != cache_key("same question", hybrid)


def test_the_key_changes_when_the_generator_changes(cfg):
    from agent.config import Models

    other = replace(cfg, models=Models(generator="some/other-model"))

    assert cache_key("q", cfg) != cache_key("q", other)


def test_the_key_is_stable_for_the_same_question_and_configuration(cfg):
    assert cache_key("checkout-api CPU", cfg) == cache_key("checkout-api CPU", cfg)


# --------------------------------------------------------------------------
# Selection and failure behaviour.
# --------------------------------------------------------------------------

def test_caching_is_off_by_default_locally(cfg):
    """The harness must measure the pipeline, not the cache."""
    assert get_cache(cfg).enabled is False


def test_caching_is_off_when_there_is_nowhere_to_put_it():
    """A cache that silently does nothing is worse than one plainly absent."""
    cfg = replace(load_settings("dev"), answer_cache=True)

    assert get_cache(cfg).enabled is False


def test_the_cache_is_selected_when_configured(cached_cfg):
    assert get_cache(cached_cfg).enabled is True


def test_a_broken_cache_read_is_a_miss_not_an_error(cached_cfg, monkeypatch):
    cache = get_cache(cached_cfg)
    monkeypatch.setattr(
        cache.store, "_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )

    assert cache.get("anything") is None


def test_a_broken_cache_write_does_not_raise(cached_cfg, monkeypatch):
    cache = get_cache(cached_cfg)
    monkeypatch.setattr(
        cache.store, "_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )

    cache.put("q", {"answer": "a", "cited_doc_ids": [], "confidence": "no_match"})


def test_a_cache_hit_returns_the_three_contract_fields(cached_cfg, monkeypatch):
    cache = get_cache(cached_cfg)
    monkeypatch.setattr(
        cache.store,
        "_request",
        lambda *a, **k: [
            {
                "answer": "Check the deploy log.",
                "cited_doc_ids": ["RB-001"],
                "confidence": "high",
            }
        ],
    )

    hit = cache.get("checkout-api CPU")

    assert hit["cited_doc_ids"] == ["RB-001"]
    assert hit["confidence"] == "high"


def test_a_cache_hit_short_circuits_the_pipeline(cached_cfg, monkeypatch):
    """The whole point: a repeated question costs no model call."""
    from agent.api import answer_question

    monkeypatch.setattr(
        "agent.cache.get_cache",
        lambda cfg=None: type(
            "Hit",
            (),
            {
                "enabled": True,
                "get": lambda self, q: {
                    "answer": "cached", "cited_doc_ids": ["RB-001"], "confidence": "high"
                },
                "put": lambda self, q, r: None,
            },
        )(),
    )

    def explode(*args, **kwargs):
        raise AssertionError("the graph must not run on a cache hit")

    monkeypatch.setattr("agent.graph.COMPILED.invoke", explode)

    result = answer_question("checkout-api CPU", cfg=cached_cfg, with_trace=True)

    assert result["answer"] == "cached"
    assert result["llm_calls"] == 0


def test_the_null_cache_never_hits():
    assert NullCache().get("anything") is None


# --------------------------------------------------------------------------
# Tracing.
# --------------------------------------------------------------------------

def test_tracing_is_off_without_credentials(monkeypatch):
    """The default configuration, and the whole test suite, makes no network
    calls to an observability backend."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert enabled() is False


def test_exporting_without_credentials_is_a_no_op(cfg, monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert export_trace("q", {"confidence": "high"}, [], 0, 12, cfg) is False


def test_an_unreachable_backend_never_raises(cfg, monkeypatch):
    """Losing a trace is an acceptable cost; losing an answer is not."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")

    assert export_trace("q", {"confidence": "high"}, ["analyze: x"], 1, 5, cfg) is False


def test_the_payload_carries_one_span_per_pipeline_stage(cfg):
    trace = ["analyze: x", "retrieve: y", "gate: REJECT - z", "finalize: no_match"]
    payload = build_payload("q", {"confidence": "no_match"}, trace, 0, 40, cfg)

    spans = [e for e in payload["batch"] if e["type"] == "span-create"]
    assert len(spans) == len(trace)
    assert "analyze" in spans[0]["body"]["name"]


def test_every_span_belongs_to_the_one_trace(cfg):
    payload = build_payload("q", {"confidence": "high"}, ["a: 1", "b: 2"], 1, 10, cfg)

    trace_event = next(e for e in payload["batch"] if e["type"] == "trace-create")
    trace_id = trace_event["body"]["id"]

    assert all(
        e["body"]["traceId"] == trace_id
        for e in payload["batch"]
        if e["type"] == "span-create"
    )


def test_declined_answers_are_tagged_so_they_can_be_found(cfg):
    """"Show me every question we declined" is the query worth having - it is
    the outcome this design exists to produce."""
    payload = build_payload(
        "q", {"confidence": "no_match", "cited_doc_ids": []}, [], 0, 10, cfg
    )
    tags = payload["batch"][0]["body"]["tags"]

    assert "declined" in tags
    assert "confidence:no_match" in tags


def test_answered_questions_are_tagged_separately(cfg):
    payload = build_payload(
        "q", {"confidence": "high", "cited_doc_ids": ["RB-001"]}, [], 1, 10, cfg
    )

    assert "answered" in payload["batch"][0]["body"]["tags"]


def test_the_payload_records_which_configuration_produced_it(cfg):
    """A trace you cannot attribute to an arm is a trace you cannot compare."""
    payload = build_payload("q", {"confidence": "high"}, [], 1, 10, cfg)
    metadata = payload["batch"][0]["body"]["metadata"]

    assert metadata["retrieval_mode"] == cfg.retrieval.mode
    assert metadata["profile"] == "local"


def test_the_payload_is_json_serialisable(cfg):
    import json

    json.dumps(build_payload("q", {"confidence": "high"}, ["a: 1"], 1, 10, cfg))
