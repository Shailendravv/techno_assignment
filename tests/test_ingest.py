"""Phase 7: parsing, chunking, idempotency, and the upload signature.

The signature test is the one worth having. Cloudinary's algorithm is fixed and
unforgiving - wrong parameter set, wrong sort order, or a stray empty value and
it returns a bare 401 with no hint as to which. Pinning it against a
hand-computed vector turns "debug the upload live" into "read a failing test".

Everything here runs offline. Nothing contacts Cloudinary or Supabase.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent.config import Cloudinary, load_settings
from agent.core.models import Doc
from ingest.cloudinary_client import (
    CloudinaryUnavailable,
    build_upload_signature,
    delivery_url,
    sign_params,
)
from ingest.pipeline import (
    IngestReport,
    _doc_from_markdown,
    collect_from_repository,
    content_hash,
    ingest_documents,
    parse_bytes,
)


@pytest.fixture
def cfg():
    base = load_settings("local")
    return replace(
        base,
        cloudinary=Cloudinary(
            cloud_name="demo-cloud",
            api_key="123456789",
            api_secret="abcdefghijklmnop",
            folder="runbooks",
        ),
    )


# --------------------------------------------------------------------------
# The upload signature.
# --------------------------------------------------------------------------

def test_the_signature_matches_cloudinarys_documented_algorithm():
    """Sorted `k=v` pairs joined by `&`, secret appended with no separator."""
    params = {"timestamp": 1700000000, "folder": "runbooks"}

    expected = hashlib.sha1(
        b"folder=runbooks&timestamp=1700000000abcdefghijklmnop"
    ).hexdigest()

    assert sign_params(params, "abcdefghijklmnop") == expected


def test_parameters_are_sorted_not_taken_in_insertion_order():
    unsorted = {"timestamp": 1, "folder": "a", "public_id": "b"}
    sorted_differently = {"public_id": "b", "folder": "a", "timestamp": 1}

    assert sign_params(unsorted, "s") == sign_params(sorted_differently, "s")


def test_transport_parameters_are_excluded_from_the_signature():
    """`file`, `api_key`, `resource_type` and `cloud_name` are sent with the
    upload but are not signed. Including one produces a 401 with no clue why."""
    base = {"timestamp": 1, "folder": "runbooks"}
    noisy = {**base, "file": "x", "api_key": "k", "resource_type": "raw",
             "cloud_name": "c"}

    assert sign_params(base, "s") == sign_params(noisy, "s")


def test_empty_values_are_dropped_rather_than_signed_as_empty():
    """Signing `public_id=` and then omitting it from the upload is a mismatch."""
    assert sign_params({"timestamp": 1, "public_id": ""}, "s") == sign_params(
        {"timestamp": 1}, "s"
    )


def test_the_signed_payload_never_contains_the_api_secret(cfg):
    signed = build_upload_signature(cfg, public_id="RB-013", timestamp=1700000000)

    assert "abcdefghijklmnop" not in str(signed.values())
    assert "api_secret" not in signed


def test_the_signed_payload_carries_what_the_browser_needs(cfg):
    signed = build_upload_signature(cfg, timestamp=1700000000)

    assert signed["cloud_name"] == "demo-cloud"
    assert signed["resource_type"] == "raw"
    assert signed["upload_url"].endswith("/demo-cloud/raw/upload")
    assert len(signed["signature"]) == 40


def test_the_signature_covers_the_public_id_so_it_cannot_be_swapped(cfg):
    """A signature that ignored `public_id` would let a client upload anywhere."""
    a = build_upload_signature(cfg, public_id="RB-013", timestamp=1700000000)
    b = build_upload_signature(cfg, public_id="RB-999", timestamp=1700000000)

    assert a["signature"] != b["signature"]


def test_signing_without_configuration_says_which_variables_are_missing():
    with pytest.raises(CloudinaryUnavailable, match="CLOUDINARY_CLOUD_NAME"):
        build_upload_signature(load_settings("local"))


def test_delivery_urls_use_raw_not_pdf(cfg):
    """Free Cloudinary accounts block PDF delivery by default. Raw is not
    subject to that rule, which is why everything is stored as raw."""
    assert "/raw/upload/" in delivery_url("runbooks/RB-001", cfg)


# --------------------------------------------------------------------------
# Parsing uploads.
# --------------------------------------------------------------------------

def test_markdown_with_front_matter_parses_into_a_document():
    raw = b"""---
doc_id: RB-013
title: "search-api - High latency"
service: search-api
failure_mode: latency
doc_type: runbook
---

## Symptoms

Latency is up.
"""
    doc = parse_bytes(raw, "RB-013.md")

    assert (doc.doc_id, doc.service, doc.failure_mode) == (
        "RB-013", "search-api", "latency"
    )


def test_a_document_without_front_matter_still_parses():
    doc = parse_bytes(b"# Some notes\n\nSomething happened.", "notes.md")

    assert doc.doc_id == "notes"
    assert doc.doc_type == "runbook"


def test_a_missing_service_is_left_as_none_rather_than_guessed():
    """A guessed service is worse than an absent one. Absent means "applies to
    everything" and is merely imprecise; wrong makes the filter drop the
    document for every question it actually answers."""
    doc = parse_bytes(b"checkout-api is mentioned throughout this text.", "x.md")

    assert doc.service is None


def test_yamls_several_spellings_of_absent_all_become_none():
    doc = _doc_from_markdown(
        '---\ndoc_id: RB-013\nservice: "null"\nfailure_mode: ~\n---\n\nbody',
        "RB-013.md",
    )

    assert doc.service is None and doc.failure_mode is None


def test_a_pdf_without_pymupdf_installed_says_why(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pymupdf(name, *args, **kwargs):
        if name == "pymupdf4llm":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pymupdf)

    with pytest.raises(RuntimeError, match="deployed bundle"):
        parse_bytes(b"%PDF-1.4", "upload.pdf")


# --------------------------------------------------------------------------
# Idempotency.
# --------------------------------------------------------------------------

def _doc(**overrides) -> Doc:
    base = dict(
        doc_id="RB-001",
        title="t",
        service="checkout-api",
        failure_mode="cpu",
        doc_type="runbook",
        date=None,
        text="body",
    )
    return Doc(**{**base, **overrides})


def test_the_same_document_hashes_the_same_twice():
    assert content_hash(_doc()) == content_hash(_doc())


def test_changed_prose_changes_the_hash():
    assert content_hash(_doc()) != content_hash(_doc(text="different"))


def test_changed_metadata_changes_the_hash_even_when_the_prose_is_identical():
    """The filter acts on metadata, so a stale `service` would keep deciding
    which questions the document answers."""
    assert content_hash(_doc()) != content_hash(_doc(service="payments-api"))


def test_a_dry_run_writes_nothing_and_embeds_nothing(cfg):
    """Embedding is the expensive step and it is metered against a daily quota.
    A run that writes nothing has no use for vectors it computed."""
    report = ingest_documents(collect_from_repository(cfg), cfg, dry_run=True)

    assert report.dry_run is True
    assert report.embedded == 0
    assert report.documents_seen == 12
    assert report.chunks_written > 12  # every document splits into sections


def test_a_dry_run_reports_what_it_would_have_written(cfg):
    report = ingest_documents(collect_from_repository(cfg), cfg, dry_run=True)

    assert report.documents_written == 12
    assert report.errors == []


def test_unchanged_documents_are_skipped(cfg, monkeypatch):
    """Re-running must be free, or nobody re-runs it and the index drifts."""
    docs = collect_from_repository(cfg)
    hashes = {doc.doc_id: content_hash(doc) for doc, _ in docs}

    class Writer:
        def __init__(self, _cfg):
            self.wrote = []

        def existing_hashes(self):
            return hashes

        def upsert_document(self, doc, digest, source):
            self.wrote.append(doc.doc_id)

        def replace_chunks(self, *a):
            pass

        def bump_version(self, *a):
            return 2

        def record_job(self, *a, **k):
            pass

    monkeypatch.setattr("ingest.pipeline.SupabaseWriter", Writer)
    report = ingest_documents(docs, cfg)

    assert report.documents_unchanged == 12
    assert report.documents_written == 0
    assert report.embedded == 0


def test_force_re_ingests_even_unchanged_documents(cfg, monkeypatch):
    calls = {"upserts": 0}

    class Writer:
        def __init__(self, _cfg):
            pass

        def existing_hashes(self):
            raise AssertionError("--force must not consult existing hashes")

        def upsert_document(self, doc, digest, source):
            calls["upserts"] += 1

        def replace_chunks(self, *a):
            pass

        def bump_version(self, *a):
            return 2

        def record_job(self, *a, **k):
            pass

    monkeypatch.setattr("ingest.pipeline.SupabaseWriter", Writer)
    # EMBEDDER=none so this exercises the write path, not the embedder.
    from agent.config import Embedding

    report = ingest_documents(
        collect_from_repository(cfg),
        replace(cfg, embedding=Embedding(backend="none", model="-", dims=0)),
        force=True,
    )

    assert calls["upserts"] == 12
    assert report.documents_written == 12


def test_one_failing_document_does_not_end_the_run(cfg, monkeypatch):
    from agent.config import Embedding

    class Writer:
        def __init__(self, _cfg):
            self.seen = 0

        def existing_hashes(self):
            return {}

        def upsert_document(self, doc, digest, source):
            self.seen += 1
            if doc.doc_id == "RB-003":
                raise RuntimeError("constraint violation")

        def replace_chunks(self, *a):
            pass

        def bump_version(self, *a):
            return 2

        def record_job(self, *a, **k):
            pass

    monkeypatch.setattr("ingest.pipeline.SupabaseWriter", Writer)
    report = ingest_documents(
        collect_from_repository(cfg),
        replace(cfg, embedding=Embedding(backend="none", model="-", dims=0)),
    )

    assert report.documents_written == 11
    assert any("RB-003" in error for error in report.errors)


def test_an_unusable_embedder_is_reported_not_silently_ignored(cfg, monkeypatch):
    """Chunks written without vectors leave the dense arm inert. That must be
    visible, because retrieval would still appear to work."""
    from agent.config import Embedding

    class Writer:
        def __init__(self, _cfg):
            pass

        def existing_hashes(self):
            return {}

        def upsert_document(self, *a):
            pass

        def replace_chunks(self, *a):
            pass

        def bump_version(self, *a):
            return 2

        def record_job(self, *a, **k):
            pass

    monkeypatch.setattr("ingest.pipeline.SupabaseWriter", Writer)
    report = ingest_documents(
        collect_from_repository(cfg),
        # gemini with no key: enabled, but not available.
        replace(cfg, embedding=Embedding(backend="gemini", model="g", dims=768)),
    )

    assert any("not usable" in error for error in report.errors)


def test_the_report_is_json_serialisable():
    import json

    json.dumps(IngestReport().as_dict())
