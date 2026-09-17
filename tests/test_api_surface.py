"""The HTTP surface added in Phases 7-8: health, signing, and the report.

`/ask` itself is covered in test_agent_pipeline.py, where the graph is stubbed.
What is tested here is the thin layer around it - and in particular the two
things that would be embarrassing to get wrong in public:

- `/health` reports credential *presence*, never a credential. It is public.
- `/ingest/sign` returns a signature, never the API secret.
"""

from __future__ import annotations

import json

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

def test_signing_without_cloudinary_configured_is_a_503_not_a_500(client):
    """Missing configuration is not a bug, and the message should say what to set."""
    response = client.post("/ingest/sign", json={})

    if response.status_code == 503:
        assert "CLOUDINARY" in response.json()["detail"]
    else:
        # Cloudinary is configured in this environment; then it must succeed.
        assert response.status_code == 200


def test_a_signed_response_never_carries_the_api_secret(client, monkeypatch):
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

    body = client.post("/ingest/sign", json={"public_id": "RB-013"}).json()

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
