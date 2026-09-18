"""Suite-wide defaults.

Two fixtures, and between them they make one claim true rather than merely
intended: **the test suite is offline, on every machine, whatever is in
`.env`.**

That claim used to rest on "the default configuration has no keys". It does -
but a developer's `.env` does not, `agent/config.py` loads `.env` into the
process environment on import, and on a machine where this system had actually
been configured the suite reached live Supabase and live Groq. Ten tests failed
for it, and the ones that passed were passing for the wrong reason.

The failure is worth naming precisely, because the obvious fix does not work.
Clearing a variable with `monkeypatch.delenv` is undone the moment anything
calls `load_settings()`, because that re-invokes `_load_dotenv()` and copies
`.env` back over the environment. So the environment cannot be cleaned while
the loader is still armed; the loader has to be disarmed first.
"""

from __future__ import annotations

import os

import pytest

# Everything a developer might have configured locally. Cleared for every test,
# because a test that asserts on an *absent* credential is otherwise asserting
# on a property of the machine it runs on.
CREDENTIAL_VARS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
    "CLOUDINARY_FOLDER",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "INGEST_API_KEY",
)

# Profile-shaped overrides. A developer running `APP_ENV=dev` must not thereby
# run a different test suite from CI's.
PROFILE_VARS = (
    "STORE_BACKEND",
    "EMBEDDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "RETRIEVAL_MODE",
    "GRADER_ENABLED",
    "ANSWER_CACHE",
    "COVERAGE_FLOOR",
    "LEXICAL_FLOOR",
    "COSINE_FLOOR",
    "MAX_REWRITES",
)


def dotenv_values() -> dict[str, str]:
    """Parse `backend/.env` directly.

    Deliberately not `agent.config._load_dotenv`: `offline_environment` disarms
    that, and a network-marked test needs the credentials back without
    re-arming the loader for everything else in the session.
    """
    from agent.config import ROOT

    path = ROOT / ".env"
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip("\"'")
    return values


@pytest.fixture
def live_dev_config(monkeypatch, offline_environment):
    """Real credentials and the `dev` profile, for `@pytest.mark.network` only.

    Skips rather than fails when nothing is configured, so the marker stays
    runnable on a machine that has never been set up. Everything else in the
    session remains offline - this fixture is opt-in and per-test.
    """
    values = dotenv_values()
    url = os.environ.get("SUPABASE_URL") or values.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or values.get("SUPABASE_SERVICE_KEY", "")
    if not (url and key):
        pytest.skip("needs SUPABASE_URL and SUPABASE_SERVICE_KEY")

    import agent.config as config

    monkeypatch.setenv("SUPABASE_URL", url)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", key)
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        value = os.environ.get(name) or values.get(name, "")
        if value:
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(config, "_profile", config.load_profile("dev"))
    monkeypatch.setattr(config, "settings", config.Settings(profile="dev"))
    return config.current_settings()


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Pin every test to the `local` profile with no credentials.

    Ordering is the whole fixture and it is not interchangeable:

    1. **Disarm `_load_dotenv` first.** It is re-invoked by every
       `load_settings()` call, so while it is armed any variable we clear is
       restored by the next thing that reads configuration.
    2. **Then clear the environment**, now that nothing will put it back.
    3. **Then rebuild the settings singleton**, because `agent/config.py` built
       it at import - long before any fixture runs - so the object the rest of
       the code falls back to still holds whatever `.env` said.

    Step 3 works because every module now resolves its default through
    `agent.config.current_settings()` rather than binding `settings` into its
    own namespace at import. One indirection, one place to patch.
    """
    monkeypatch.setattr("agent.config._load_dotenv", lambda: None)

    for name in CREDENTIAL_VARS + PROFILE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    import agent.config as config

    monkeypatch.setattr(config, "_profile", config.load_profile("local"))
    monkeypatch.setattr(config, "settings", config.Settings())

    # Stores are memoised per configuration, so one built under a previous
    # test's settings must not be handed to the next.
    from agent.store import reset_stores
    from agent.store.file_store import reset_dense_cache

    reset_stores()
    reset_dense_cache()
    yield
    reset_stores()


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch, offline_environment):
    """Belt and braces around the two things that reach the network anyway.

    The answer cache lives in Supabase, and a suite that wrote its fake-LLM
    answers into the shared cache once read them back: a test failed because an
    earlier test's stub was served to it, and - the one that mattered - a real
    request was later answered from a test. The cache has no TTL, so that
    persists until deleted.

    `offline_environment` already removes the credentials that would let either
    happen. These two lines stay because they are the seams the cache's own
    tests use, and because defence in depth is cheap here.
    """
    from agent.cache import NullCache

    monkeypatch.setattr("agent.cache.get_cache", lambda cfg=None: NullCache())

    # The client is a module-level singleton, so a test that built one must not
    # leak it into the next test's assertions about whether tracing is on.
    from agent import observability

    observability.reset_client()
    yield
    observability.reset_client()
