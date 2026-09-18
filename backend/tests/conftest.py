"""Suite-wide defaults.

One fixture, and it exists to make a claim in `agent/observability.py` true
rather than merely intended: **the test suite sends nothing to Langfuse.**

That claim used to rest on "the default configuration has no keys". It does -
but a developer's `.env` does, `agent/config.py` loads `.env` into the process
environment on import, and so running the suite on the machine the app is
developed on would have quietly exported a trace for every test that answers a
question. Traces from a test run are worse than no traces: they land in the
same project as the real ones, with the same names, and they are indistinguishable
after the fact.

Setting the off switch here means the suite is offline by construction, on
every machine, and a test that wants to exercise tracing has to say so.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    # The same rule, for the same reason, applied to the answer cache.
    #
    # `ANSWER_CACHE` is on in the `dev` profile and the cache lives in Supabase,
    # so on a machine whose `.env` selects `dev` the suite wrote its fake-LLM
    # answers into the shared cache and read them back. Two things went wrong:
    # a test failed because an earlier test's stub was served to it, and - the
    # one that matters - a real request for one of those questions was answered
    # from a test. The cache has no TTL, so that persists until deleted.
    #
    # The environment variable alone does not fix it. `agent/config.py` builds
    # its `settings` singleton at import, which happens before any fixture
    # runs, so by the time this executes the profile's `answer_cache=True` is
    # already baked into the object `answer_question()` falls back to. It is
    # set anyway, for the code paths that build a fresh `Settings` mid-test.
    monkeypatch.setenv("ANSWER_CACHE", "false")

    # This is the seam that actually holds: `agent/api.py` imports `get_cache`
    # inside the function, so it resolves through the module every call. A test
    # that wants a cache patches the same attribute afterwards and wins - which
    # is how `test_a_cache_hit_short_circuits_the_pipeline` still works - and
    # the cache's own tests bound `get_cache` at import and never see this.
    from agent.cache import NullCache

    monkeypatch.setattr("agent.cache.get_cache", lambda cfg=None: NullCache())

    # The client is a module-level singleton, so a test that built one must not
    # leak it into the next test's assertions about whether tracing is on.
    from agent import observability

    observability.reset_client()
    yield
    observability.reset_client()
