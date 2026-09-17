"""The `config/*.json` profile layer.

Profiles are how local and deployed get genuinely different components -
`fastembed` against Gemini, files against Postgres - without a branch in the
code. That makes them configuration that can break a deployment, so they are
tested like code.

Two of these tests are the ones that would actually catch a bad day:

- `test_no_profile_contains_a_secret` - profiles are committed, so a key
  written into one is a key published to the repository.
- `test_a_missing_profile_degrades_to_defaults` - the deployed function may be
  built without the config directory, and "no profile" must mean "the
  defaults", not a crash inside a request handler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.config import ROOT, Settings, load_profile, load_settings

PROFILE_DIR = ROOT / "config"
PROFILE_NAMES = ("local", "dev")

# Substrings that betray a credential. Profiles carry tuning, never secrets.
SECRET_MARKERS = ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "SUPABASE_URL")


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_every_profile_is_valid_json(name):
    data = json.loads((PROFILE_DIR / f"{name}.json").read_text(encoding="utf-8"))

    assert isinstance(data, dict)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_no_profile_contains_a_secret(name):
    """These files are committed. A key here is a key on GitHub."""
    keys = load_profile(name)

    offenders = [k for k in keys if any(marker in k.upper() for marker in SECRET_MARKERS)]
    assert offenders == [], f"{name}.json carries what look like credentials: {offenders}"


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_every_profile_key_is_one_the_code_actually_reads(name):
    """A typo'd key in a profile is silent: the setting keeps its default and
    nothing reports that the profile asked for something else."""
    source = (ROOT / "agent" / "config.py").read_text(encoding="utf-8")

    unknown = [key for key in load_profile(name) if f'"{key}"' not in source]
    assert unknown == [], f"{name}.json sets keys nothing reads: {unknown}"


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_every_profile_builds_a_usable_settings(name):
    settings = load_settings(name)

    assert settings.profile == name
    assert settings.retrieval.mode in ("lexical", "hybrid")
    assert settings.embedding.backend in ("local", "gemini", "none")
    assert settings.store in ("files", "supabase")


def test_comment_keys_are_stripped():
    """JSON has no comment syntax, so `_comment` carries the explanation - and
    must never reach the settings as if it were one."""
    assert not any(k.startswith("_") for k in load_profile("local"))


def test_the_local_profile_runs_offline_and_deterministically():
    """The harness's requirement: no network, no rate limit, same answer twice."""
    local = load_settings("local")

    assert local.embedding.backend == "local"
    assert local.store == "files"


def test_the_deployed_profile_keeps_the_onnx_model_out_of_the_bundle():
    """`fastembed` is ~200MB of onnxruntime plus a model download. Vercel's
    Python bundler does no tree-shaking, so the deployed profile must not ask
    for it."""
    dev = load_settings("dev")

    assert dev.embedding.backend == "gemini"
    assert dev.store == "supabase"


def test_the_two_profiles_disagree_about_the_things_they_exist_to_disagree_about():
    local, dev = load_settings("local"), load_settings("dev")

    assert local.embedding.backend != dev.embedding.backend
    assert local.store != dev.store


def test_an_environment_variable_beats_the_profile(monkeypatch):
    """Vercel sets environment variables; they must win over a committed file."""
    monkeypatch.setenv("RETRIEVAL_MODE", "lexical")

    assert load_settings("dev").retrieval.mode == "lexical"


def test_a_missing_profile_degrades_to_defaults():
    """The deployed function may be built without `config/`. That must mean
    "use the defaults", not a 500 from a request handler."""
    settings = load_settings("a-profile-that-does-not-exist")

    assert isinstance(settings, Settings)
    assert settings.retrieval.mode in ("lexical", "hybrid")


def test_a_malformed_profile_says_which_file_is_wrong(tmp_path, monkeypatch):
    bad = PROFILE_DIR / "broken-test-profile.json"
    bad.write_text("{not json", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not valid JSON"):
            load_profile("broken-test-profile")
    finally:
        bad.unlink()


def test_loading_a_profile_does_not_leak_into_the_next_load():
    """`load_settings(name)` swaps the active profile. If it failed to put the
    previous one back, every later call would silently read the wrong file."""
    before = load_settings().store
    load_settings("dev")

    assert load_settings().store == before


def test_describe_reports_configuration_without_leaking_credentials():
    """`/health` is public on a deployed app."""
    described = load_settings("dev").describe()

    assert described["groq_configured"] in (True, False)
    assert not any(
        isinstance(v, str) and len(v) > 40 for v in described.values()
    ), "describe() should report presence, never a credential"


def test_the_embedding_signature_distinguishes_the_two_backends():
    """The signature keys the vector cache. If the two backends shared one,
    384-dim vectors would be read back as 768-dim ones."""
    assert load_settings("local").embedding.signature != (
        load_settings("dev").embedding.signature
    )
