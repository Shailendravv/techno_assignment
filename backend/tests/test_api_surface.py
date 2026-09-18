"""The HTTP surface added in Phases 7-8: health, signing, and the report.

`/ask` itself is covered in test_agent_pipeline.py, where the graph is stubbed.
What is tested here is the thin layer around it - and in particular the two
things that would be embarrassing to get wrong in public:

- `/health` reports credential *presence*, never a credential. It is public.
- `/ingest/sign` returns a signature, never the API secret.
"""

from __future__ import annotations

import json

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------

def test_health_reports_how_this_instance_is_configured(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["profile"] in ("local", "dev")
    assert body["corpus_docs"] == 12
    assert body["retrieval_mode"] in ("lexical", "hybrid")


def test_health_reports_credential_presence_not_credentials(client):
    """This endpoint is public on a deployed app."""
    body = client.get("/health").json()

    for key in (
        "groq_configured",
        "gemini_configured",
        "supabase_configured",
        "cloudinary_configured",
    ):
        assert isinstance(body[key], bool)

    # Nothing long enough to be a key should appear anywhere in the response.
    assert not any(
        isinstance(value, str) and len(value) > 60 for value in body.values()
    )


def test_health_reports_whether_the_store_is_actually_reachable(client):
    """"Configured" and "working" are different questions, and the keep-alive
    cron needs the second one."""
    body = client.get("/health").json()

    assert body["store_reachable"] is True


def test_health_distinguishes_the_store_mounted_from_the_store_requested(client):
    """Supabase falls back to files when it is not configured. A health endpoint
    that reported only the request would let a misconfigured deployment look
    correct, which is the one thing it exists to prevent."""
    body = client.get("/health").json()

    assert body["store"] in ("files", "supabase")
    assert body["store_requested"] in ("files", "supabase")


def test_health_never_500s_even_when_the_store_is_broken(client, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr("agent.store.get_store", explode)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["store_reachable"] is False


# --------------------------------------------------------------------------
# /ingest/sign
# --------------------------------------------------------------------------

INGEST_KEY = "test-ingest-secret"


@pytest.fixture
def ingest_enabled(monkeypatch):
    """Configure signed uploads, and hand back the key a caller needs.

    The route is off unless both a shared secret and Cloudinary are configured,
    so a test that wants to exercise it has to say so - which is the point.
    """
    from agent.config import Cloudinary, current_settings

    cfg = current_settings()
    monkeypatch.setattr(
        "agent.config.settings",
        replace(
            cfg,
            ingest_api_key=INGEST_KEY,
            cloudinary=Cloudinary(
                cloud_name="demo", api_key="123", api_secret="shhh"
            ),
        ),
    )
    return {"X-Ingest-Key": INGEST_KEY}


def test_signing_is_not_available_without_a_key(client):
    """An unauthenticated caller must not be able to mint an upload signature.

    404 rather than 401 or 403: an unconfigured deployment should not advertise
    that this route exists. This is the finding that mattered most about this
    endpoint - it used to hand anyone a valid Cloudinary signature, and the
    ingestion pipeline turns whatever is uploaded into corpus documents that a
    model is then asked to answer from.
    """
    response = client.post("/ingest/sign", json={})

    assert response.status_code == 404


def test_signing_with_the_wrong_key_is_also_a_404(client, ingest_enabled):
    response = client.post(
        "/ingest/sign", json={}, headers={"X-Ingest-Key": "not-the-key"}
    )

    assert response.status_code == 404


def test_a_public_id_may_not_name_an_existing_document(client, ingest_enabled):
    """`on_conflict=doc_id` means naming a runbook is overwriting it."""
    response = client.post(
        "/ingest/sign", json={"public_id": "RB-001"}, headers=ingest_enabled
    )

    assert response.status_code == 422


def test_a_signed_response_never_carries_the_api_secret(
    client, monkeypatch, ingest_enabled
):
    monkeypatch.setattr(
        "ingest.cloudinary_client.build_upload_signature",
        lambda *a, **k: {
            "cloud_name": "demo",
            "api_key": "123",
            "resource_type": "raw",
            "upload_url": "https://api.cloudinary.com/v1_1/demo/raw/upload",
            "signature": "a" * 40,
            "timestamp": 1700000000,
            "folder": "runbooks",
            "public_id": None,
        },
    )

    body = client.post(
        "/ingest/sign",
        json={"public_id": "upload-abc12345"},
        headers=ingest_enabled,
    ).json()

    assert "api_secret" not in body
    assert body["resource_type"] == "raw"


def test_an_over_long_public_id_is_rejected_by_validation(client):
    response = client.post("/ingest/sign", json={"public_id": "x" * 500})

    assert response.status_code == 422


# --------------------------------------------------------------------------
# /eval/latest
# --------------------------------------------------------------------------

def test_the_latest_evaluation_report_is_served_from_the_repository(client):
    """Recomputing it would mean twenty grounding calls inside a request, and
    the committed report is the actual evidence anyway."""
    response = client.get("/eval/latest")

    assert response.status_code in (200, 404)
    if response.status_code == 200:
        body = response.json()
        assert body["report"].endswith(".json")


# --------------------------------------------------------------------------
# /
# --------------------------------------------------------------------------

def test_the_root_serves_the_ui_or_points_at_the_docs(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(("text/html", "application/json"))


# ---------------------------------------------------------------------------
# Rate limiting.
#
# `/ask` takes no credential and every request that clears the gate spends Groq
# tokens from a shared free-tier quota, so an unlimited endpoint is a
# quota-exhaustion button. In-process and per-instance, which is stated in
# `app/ratelimit.py` rather than implied.
# ---------------------------------------------------------------------------

def test_the_limiter_counts_a_window_and_then_refuses():
    from app.ratelimit import RateLimiter

    limiter = RateLimiter(per_minute=3)

    verdicts = [limiter.check("1.2.3.4")[0] for _ in range(4)]

    assert verdicts == [True, True, True, False]


def test_clients_are_counted_separately():
    from app.ratelimit import RateLimiter

    limiter = RateLimiter(per_minute=1)
    limiter.check("1.2.3.4")

    allowed, _ = limiter.check("5.6.7.8")

    assert allowed is True


def test_the_window_reopens(monkeypatch):
    from app.ratelimit import RateLimiter

    limiter = RateLimiter(per_minute=1, window_s=60.0)
    assert limiter.check("1.2.3.4", now=0.0)[0] is True
    assert limiter.check("1.2.3.4", now=30.0)[0] is False

    assert limiter.check("1.2.3.4", now=61.0)[0] is True


def test_a_zero_limit_disables_the_check():
    """So a deployment can turn it off without deleting the code path."""
    from app.ratelimit import RateLimiter

    limiter = RateLimiter(per_minute=0)

    assert all(limiter.check("1.2.3.4")[0] for _ in range(50))


def test_an_over_limit_request_gets_a_429_with_retry_after(client, monkeypatch):
    from app.ratelimit import RateLimiter

    monkeypatch.setattr("app.main._ask_limiter", RateLimiter(per_minute=1))

    first = client.post("/ask", json={"question": "checkout-api high CPU"})
    second = client.post("/ask", json={"question": "checkout-api high CPU"})

    assert first.status_code in (200, 503)  # 503 when no GROQ key; either counts
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0


# ---------------------------------------------------------------------------
# Failure responses.
#
# There were no exception handlers at all, so everything except LLMUnavailable
# became a bare `500 Internal Server Error` with no body and no log line. Both
# of this stack's most likely production failures landed there: Supabase
# refusing a connection, and Groq still returning 429 after the retry budget.
# ---------------------------------------------------------------------------

def test_an_unreachable_store_is_a_503_with_an_incident_id(client, monkeypatch):
    import agent.graph
    from agent.store import StoreUnavailable

    class Unreachable:
        name = "broken"

        def documents(self):
            raise StoreUnavailable("Supabase returned 500 for /rest/v1/documents")

        def lexical_index(self):
            raise StoreUnavailable("down")

        def retrieve(self, *a, **k):
            raise StoreUnavailable("down")

        def health(self):
            return {"reachable": False, "error": "down"}

    monkeypatch.setattr(agent.graph, "get_store", lambda cfg=None: Unreachable())

    response = client.post("/ask", json={"question": "checkout-api high CPU"})

    assert response.status_code == 503
    assert response.headers["Retry-After"]
    assert response.json()["incident"]


def test_a_failure_does_not_leak_the_upstream_error(client, monkeypatch):
    """The message can carry a Supabase URL or a fragment of a provider
    response, so the caller gets a correlation id and the detail goes to the
    log."""
    import agent.graph
    from agent.store import StoreUnavailable

    secret_ish = "https://verysecret.supabase.co/rest/v1/documents"

    class Unreachable:
        name = "broken"

        def documents(self):
            raise StoreUnavailable(secret_ish)

        def lexical_index(self):
            raise StoreUnavailable(secret_ish)

        def retrieve(self, *a, **k):
            raise StoreUnavailable(secret_ish)

        def health(self):
            return {"reachable": False, "error": "down"}

    monkeypatch.setattr(agent.graph, "get_store", lambda cfg=None: Unreachable())

    response = client.post("/ask", json={"question": "checkout-api high CPU"})

    assert secret_ish not in response.text


def test_an_exhausted_rate_limit_is_a_503_not_a_500(client, monkeypatch):
    """Groq's free tier still 429s after the retry budget is spent. That is a
    dependency being busy, not a bug in the request."""
    import agent.graph

    class RateLimitError(Exception):
        status_code = 429

    def _rate_limited(*a, **k):
        raise RateLimitError("rate limit exceeded")

    monkeypatch.setattr("agent.nodes.ground.chat_json", _rate_limited)

    response = client.post(
        "/ask", json={"question": "checkout-api is running hot on CPU"}
    )

    assert response.status_code == 503
    assert "rate limited" in response.json()["detail"].lower()
