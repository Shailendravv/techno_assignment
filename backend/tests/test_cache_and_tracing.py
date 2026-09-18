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

from agent import observability
from agent.cache import NullCache, cache_key, get_cache, normalise_question
from agent.config import Supabase, load_settings


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

    result = answer_question(
        "checkout-api CPU", cfg=cached_cfg, with_trace=True, with_metrics=True
    )

    assert result["answer"] == "cached"
    assert result["llm_calls"] == 0


def test_the_null_cache_never_hits():
    assert NullCache().get("anything") is None


# --------------------------------------------------------------------------
# Tracing.
# --------------------------------------------------------------------------
# What these check is not "does Langfuse receive a trace" - that needs a
# project and a network, and it is verified by running the thing. They check
# the two properties the pipeline depends on: that tracing is off unless it is
# configured, and that no failure of it can reach the caller.


def test_tracing_is_off_without_credentials(cfg, monkeypatch):
    """The default configuration, and the whole test suite, makes no network
    calls to an observability backend."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)

    blank = replace(cfg, langfuse_public_key="", langfuse_secret_key="")
    assert observability.enabled(blank) is False
    assert observability.client(blank) is None


def test_tracing_can_be_turned_off_without_deleting_the_keys(monkeypatch):
    """Keys are shared between the app, the CLI and this suite. Turning
    tracing off must not mean editing credentials."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    assert observability.enabled() is False


def test_an_unconfigured_run_yields_working_no_op_handles(cfg):
    """Call sites never branch on whether tracing is on, so the handles have
    to behave when it is off."""
    with observability.trace_run("q", cfg) as root:
        root.update(output={"answer": "x"})
        observability.finish(
            root, {"confidence": "high", "cited_doc_ids": ["RB-001"]},
            llm_calls=1, elapsed_ms=10,
        )

    with observability.stage_observation("generate") as observation:
        observation.update(output="anything")

    with observability.generation(
        role="generator", model="m", messages=[], attempt=1,
        temperature=0.0, max_tokens=10,
    ) as generation:
        generation.update(output="anything", usage_details={"input": 1})

    assert observability.tracing() is False


def test_a_broken_client_never_reaches_the_caller(cfg, monkeypatch):
    """Losing a trace is an acceptable cost; losing an answer is not."""
    class Exploding:
        def start_as_current_observation(self, **_):
            raise RuntimeError("langfuse is having a bad day")

        def create_event(self, **_):
            raise RuntimeError("still bad")

        def flush(self):
            raise RuntimeError("worse")

    monkeypatch.setattr(observability, "client", lambda *_, **__: Exploding())

    with observability.trace_run("q", cfg) as root:
        observability.finish(root, {"confidence": "no_match"}, llm_calls=0, elapsed_ms=1)

    observability.flush()  # must not raise even though the client does


def test_stages_outside_a_run_create_no_orphan_traces(cfg, monkeypatch):
    """An observation with no parent becomes its own one-span trace. A stage
    recorded outside a run - in ingest, or in a unit test - must not litter the
    project with them."""
    created = []

    class Recording:
        def start_as_current_observation(self, **kwargs):
            created.append(kwargs)
            raise AssertionError("should not have been called")

    monkeypatch.setattr(observability, "client", lambda *_, **__: Recording())

    with observability.stage_observation("generate") as observation:
        observation.update(output="x")

    assert created == []


def test_every_query_stage_is_represented_in_a_trace():
    """A stage added to the ledger without a Langfuse mapping would run, log,
    and be invisible in the trace tree. `llm_gateway` is the one deliberate
    exclusion: `agent/llm.py` emits a `generation` for the same round trip, and
    two observations for one call double-count cost."""
    from agent.stages import QUERY_STAGES, Status

    expected = {
        stage.name
        for stage in QUERY_STAGES
        if stage.declared is not Status.NOT_IMPLEMENTED
    } - {"llm_gateway"}

    assert expected <= set(observability.STAGE_OBSERVATIONS)


def test_observation_types_are_ones_langfuse_accepts():
    """The type is what makes a retrieval step filterable as retrieval. A typo
    would be accepted locally and be wrong in the UI."""
    valid = {
        "span", "generation", "agent", "tool", "chain",
        "retriever", "evaluator", "embedding", "guardrail", "event",
    }

    for name, as_type in observability.STAGE_OBSERVATIONS.values():
        assert as_type in valid, f"{name} has an unknown observation type {as_type}"


def test_observation_names_are_stable_and_low_cardinality():
    """Langfuse treats a name as an API: evaluators target it, dashboards group
    by it. A name carrying a run-specific value silently breaks all of them."""
    for name, _ in observability.STAGE_OBSERVATIONS.values():
        assert name == name.lower()
        assert " " not in name
        assert not any(char.isdigit() for char in name)


@pytest.mark.parametrize(
    "secret",
    [
        "gsk_abcdefghijklmnopqrstuvwx",
        "sk-abcdefghijklmnopqrstuvwx",
        "AIzaSyABCDEFGHIJKLMNOPQRSTUV",
    ],
)
def test_credentials_are_redacted_before_they_leave_the_process(secret):
    """Enabling tracing must not turn a key pasted into a question into a key
    sitting in a third party's database."""
    masked = observability.mask(data=f"my key is {secret} please help")

    assert secret not in masked
    assert "[redacted]" in masked


def test_masking_reaches_into_nested_structures():
    payload = {
        "messages": [{"role": "user", "content": "mail me at ops@example.com"}],
        "count": 3,
    }
    masked = observability.mask(data=payload)

    assert "ops@example.com" not in str(masked)
    assert masked["count"] == 3


def test_masking_leaves_an_ordinary_question_alone():
    """The question is the trace input. A masker aggressive enough to redact it
    would leave a trace nobody can read."""
    question = "What is the rollback procedure for payments-api?"

    assert observability.mask(data=question) == question
