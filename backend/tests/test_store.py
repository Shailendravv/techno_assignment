"""Phase 6: the store abstraction, and the two backends behind it.

**What these tests do and do not establish, stated up front.**

The Supabase tests here stub the HTTP transport. They verify the marshalling -
rows to `Doc`s, spec to RPC arguments, errors to a usable message, and the
divergence check - which is the code in this repository. They do **not**
execute `supabase/migrations/*.sql`, because that needs a Postgres with
pgvector and this suite must run offline with no credentials.

The SQL is therefore covered two other ways:

1. `TestMigrationsMatchThePythonRules` reads the migration as text and asserts
   the clauses that encode the metadata rules are present. That is a blunt
   instrument, and it is aimed at one specific regression: deleting the
   `service is null` disjunct. Every question answered by a general policy
   document breaks if that clause goes, and it is exactly the kind of line
   somebody "simplifies" away.
2. `test_supabase_scores_identically_to_files`, which runs only when
   SUPABASE_URL is set, is the real equivalence check and the Phase 6 exit
   criterion. It is skipped by default and that skip is not a pass.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.config import ROOT, load_settings
from agent.core.models import Candidate, QuerySpec
from agent.core.query import analyze_query
from agent.core.retrieve import metadata_filter
from agent.store import StoreUnavailable, get_store, store_kind
from agent.store.file_store import FileStore

MIGRATIONS = ROOT / "supabase" / "migrations"


@pytest.fixture
def files_cfg():
    base = load_settings("local")
    return replace(base, store="files")


@pytest.fixture
def supabase_cfg():
    from agent.config import Supabase

    base = load_settings("dev")
    return replace(
        base,
        store="supabase",
        supabase=Supabase(url="https://example.supabase.co", service_key="test-key"),
    )


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------

def test_the_files_store_is_selected_by_default(files_cfg):
    assert get_store(files_cfg).name == "files"


def test_supabase_without_credentials_falls_back_to_files():
    """A missing credential must leave a working system reading the repository,
    not a 500 from every request. The corpus is checked in, so files is always
    available."""
    cfg = replace(load_settings("dev"), store="supabase")

    assert get_store(cfg).name == "files"
    assert "not configured" in store_kind(cfg)


def test_constructing_supabase_without_credentials_says_what_is_missing():
    from agent.store.supabase_store import SupabaseStore

    cfg = replace(load_settings("dev"), store="supabase")
    with pytest.raises(StoreUnavailable, match="SUPABASE_URL"):
        SupabaseStore(cfg)


# --------------------------------------------------------------------------
# FileStore - the reference implementation.
# --------------------------------------------------------------------------

def test_the_file_store_loads_the_whole_corpus(files_cfg):
    assert len(FileStore(files_cfg).documents()) == 12


def test_the_file_store_ranks_without_filtering(files_cfg):
    """The store ranks; Python filters. A store that filtered would mean the
    metadata rule existed in two places."""
    store = FileStore(replace(files_cfg, retrieval=replace(files_cfg.retrieval, mode="lexical")))
    docs = store.documents()
    spec = analyze_query("checkout-api is running hot on CPU", docs)

    ranked, _ = store.retrieve(spec, "checkout-api is running hot on CPU", store.cfg)

    # RB-003 is payments-api/cpu - the near-duplicate trap. Ranking must still
    # surface it; dropping it is the filter's job, one stage later.
    assert all(c.verdict == "kept" for c in ranked)
    assert len(ranked) > 1


def test_the_file_store_reports_its_health(files_cfg):
    health = FileStore(files_cfg).health()

    assert health["reachable"] is True
    assert health["documents"] == 12


# --------------------------------------------------------------------------
# SupabaseStore - marshalling, with the transport stubbed.
# --------------------------------------------------------------------------

def _rows() -> list[dict]:
    """Two documents as PostgREST would return them, including the NULL
    service that makes a document general."""
    return [
        {
            "doc_id": "RB-001",
            "title": "checkout-api - High CPU",
            "service": "checkout-api",
            "failure_mode": "cpu",
            "doc_type": "runbook",
            "date": None,
            "content": "Check the deploy log first.",
            "lexical_score": 0.42,
            "dense_score": 0.81,
            "fused_score": 0.0328,
            "matched_section": "First checks",
        },
        {
            "doc_id": "RB-011",
            "title": "Incident escalation policy",
            "service": None,
            "failure_mode": None,
            "doc_type": "policy",
            "date": None,
            "content": "Page the secondary after fifteen minutes.",
            "lexical_score": 0.10,
            "dense_score": 0.60,
            "fused_score": 0.0161,
            "matched_section": "The paging chain",
        },
    ]


@pytest.fixture
def stub_store(supabase_cfg, monkeypatch):
    from agent.store.supabase_store import SupabaseStore

    store = SupabaseStore(supabase_cfg)
    calls: list[tuple] = []

    def fake_request(path, payload=None, method="POST"):
        calls.append((path, payload, method))
        return _rows()

    monkeypatch.setattr(store, "_request", fake_request)
    store.calls = calls
    return store


def test_a_null_service_becomes_none_not_the_string_null(stub_store):
    """This equivalence is what lets one metadata filter serve both backends:
    an absent front-matter field and a SQL NULL must produce the same value."""
    docs = stub_store.documents()

    assert docs[1].service is None
    assert docs[1].is_general is True


def test_rows_become_documents_with_every_field(stub_store):
    doc = stub_store.documents()[0]

    assert (doc.doc_id, doc.service, doc.failure_mode, doc.doc_type) == (
        "RB-001", "checkout-api", "cpu", "runbook"
    )


def test_the_spec_is_passed_to_sql_as_filter_arguments(stub_store, supabase_cfg):
    docs = stub_store.documents()
    spec = analyze_query("checkout-api is running hot on CPU", docs)
    stub_store.calls.clear()

    stub_store.retrieve(spec, spec.raw, replace(
        supabase_cfg, retrieval=replace(supabase_cfg.retrieval, mode="lexical")
    ))

    path, payload, _ = stub_store.calls[-1]
    assert path.endswith("/rpc/hybrid_search")
    assert payload["p_service"] == "checkout-api"
    assert payload["p_failure_mode"] == "cpu"


def test_scores_survive_the_round_trip(stub_store, supabase_cfg):
    """The gate reads these, so losing them would silently disable it."""
    spec = analyze_query("checkout-api CPU", stub_store.documents())
    candidates, _ = stub_store.retrieve(spec, spec.raw, supabase_cfg)

    assert candidates[0].lexical_score == pytest.approx(0.42)
    assert candidates[0].dense_score == pytest.approx(0.81)


def test_a_disagreement_with_the_python_filter_is_reported_not_hidden(
    stub_store, supabase_cfg, monkeypatch
):
    """If SQL admits a row Python would drop, the two implementations of the
    rule this exercise turns on have drifted. Python still wins - but silently
    winning would hide the drift until it mattered."""
    # Resolve the corpus first: the analyser derives its service vocabulary
    # from it, so it has to be the real corpus rather than the bad row.
    spec = analyze_query("checkout-api is running hot on CPU", stub_store.documents())
    assert spec.service == "checkout-api"

    wrong_service = dict(_rows()[0], doc_id="RB-003", service="payments-api")
    monkeypatch.setattr(stub_store, "_request", lambda *a, **k: [wrong_service])

    candidates, trace = stub_store.retrieve(spec, spec.raw, supabase_cfg)

    assert any("WARNING" in line and "drifted" in line for line in trace)
    # Python is authoritative, and drops it.
    assert metadata_filter(spec, candidates, 4) == []


def test_an_unreachable_supabase_gives_a_message_naming_the_url(supabase_cfg):
    from agent.store.supabase_store import SupabaseStore

    store = SupabaseStore(replace(
        supabase_cfg,
        supabase=replace(supabase_cfg.supabase, url="http://127.0.0.1:9"),
    ))

    with pytest.raises(StoreUnavailable, match="127.0.0.1:9"):
        store.documents()


def test_an_empty_table_says_to_run_ingestion(stub_store, monkeypatch):
    monkeypatch.setattr(stub_store, "_request", lambda *a, **k: [])

    with pytest.raises(StoreUnavailable, match="ingest"):
        stub_store.documents()


def test_health_flags_a_dimension_mismatch_with_the_schema(stub_store, supabase_cfg):
    """`chunks.embedding` is vector(768). A profile asking for 384 produces a
    cast error inside a live query; this turns it into a line on /health."""
    from agent.config import Embedding
    from agent.store.supabase_store import SupabaseStore

    store = SupabaseStore(replace(
        supabase_cfg,
        embedding=Embedding(backend="local", model="bge", dims=384),
    ))
    store._documents = stub_store.documents()

    assert "768" in store.health()["error"]


# --------------------------------------------------------------------------
# The migrations, as text.
# --------------------------------------------------------------------------

class TestMigrationsMatchThePythonRules:
    """Blunt, and aimed at one regression in particular.

    The `service is null` disjunct is the SQL form of the rule that a general
    policy document is never dropped for mismatching a service. It looks
    redundant, it reads like a nullability oversight, and deleting it breaks
    every question a policy document answers - silently, because the query
    still runs and still returns rows.
    """

    @pytest.fixture(scope="class")
    def search_sql(self):
        """The migration that *currently* defines `hybrid_search`.

        Resolved by number rather than named, because these assertions guard the
        definition the database actually runs. Pinning them to `0002` meant that
        the moment `0004` redefined the function, the tests carried on passing
        while guarding a superseded file - which is the failure mode they exist
        to prevent, reproduced in the tests themselves.
        """
        defining = sorted(
            path
            for path in MIGRATIONS.glob("*.sql")
            if "create or replace function hybrid_search"
            in path.read_text(encoding="utf-8").lower()
        )
        assert defining, "no migration defines hybrid_search"
        raw = defining[-1].read_text(encoding="utf-8")
        return " ".join(raw.lower().split())

    def test_the_lexical_query_is_disjunctive(self, search_sql):
        """The regression that made the deployed lexical arm return nothing.

        `websearch_to_tsquery` alone builds an AND of every content word, so a
        natural-language question matched no document and `ts_rank_cd` returned
        0.0 for all twelve - against a floor calibrated on BM25, which scores
        partial matches. Reverting to a bare conjunctive query would silently
        restore that.
        """
        assert "or_tsquery" in search_sql

    def test_documents_that_do_not_match_contribute_no_rank(self, search_sql):
        """Without this the CTE emitted every document ranked by doc_id after an
        all-zero sort, feeding alphabetical order into RRF as though it were a
        ranking signal."""
        assert "a.fts @@ q.tsq" in search_sql

    def test_a_general_document_is_never_dropped_for_its_service(self, search_sql):
        assert "d.service is null or d.service = p_service" in search_sql

    def test_a_document_without_a_failure_mode_is_never_dropped_for_it(self, search_sql):
        assert "d.failure_mode is null or d.failure_mode = p_failure_mode" in search_sql

    def test_only_postmortems_are_filtered_by_date(self, search_sql):
        assert "d.doc_type <> 'postmortem' or d.date = p_date" in search_sql

    def test_the_filter_is_applied_before_the_limit_not_after(self, search_sql):
        """Filtering after ranking means asking for the top 8 and keeping three."""
        allowed_at = search_sql.index("allowed as (")
        lexical_at = search_sql.index("lexical as (")
        dense_at = search_sql.index("dense_chunks as (")

        assert allowed_at < lexical_at < dense_at
        assert "join allowed" in search_sql

    def test_fusion_combines_ranks_not_raw_scores(self, search_sql):
        assert "1.0 / (rrf_k + l.rank)" in search_sql
        assert "1.0 / (rrf_k + d.rank)" in search_sql

    def test_a_document_is_scored_by_its_best_chunk(self, search_sql):
        assert "distinct on (c.doc_id)" in search_sql

    def test_cosine_distance_is_converted_to_similarity(self, search_sql):
        assert "1 - (c.embedding <=> query_embedding)" in search_sql

    def test_the_vector_index_uses_the_cosine_operator_class(self):
        schema = (MIGRATIONS / "0001_schema.sql").read_text(encoding="utf-8").lower()

        assert "vector_cosine_ops" in schema

    def test_the_title_is_weighted_above_the_body(self):
        """Mirrors the Python index, which counts the title twice."""
        schema = (MIGRATIONS / "0001_schema.sql").read_text(encoding="utf-8").lower()

        assert "setweight(to_tsvector('english', coalesce(title, '')), 'a')" in schema

    def test_every_migration_is_numbered_and_ordered(self):
        names = sorted(p.name for p in MIGRATIONS.glob("*.sql"))

        assert names == sorted(names)
        assert all(name[:4].isdigit() for name in names)


# --------------------------------------------------------------------------
# The real equivalence check. Skipped without credentials - and a skip is not
# a pass. This is the Phase 6 exit criterion.
# --------------------------------------------------------------------------

@pytest.mark.network
def test_supabase_does_not_rank_worse_than_files_on_the_lexical_arm(live_dev_config):
    """Behaviour preservation is the only thing that makes the migration safe.

    This is the Phase 6 exit criterion. Both stores go through the same filter
    and the same gate, so any difference is a difference in *ranking* - which is
    exactly what moving from `rank_bm25` to `ts_rank_cd` risks, since the two
    are not comparable scorers.

    **Run on the lexical arm deliberately.** The two profiles use different
    embedders - bge-small at 384 dimensions locally, Gemini at 768 deployed - so
    a hybrid comparison would vary the dense arm and the lexical arm at once and
    could not attribute a difference to either. Holding the embedder out makes
    this a controlled test of the one question that matters: does Postgres rank
    like BM25?

    **Asserted as "no worse", not as "identical", and that is deliberate.** An
    equality assertion sounds stricter and is actually the wrong shape: BM25 and
    `ts_rank_cd` are different algorithms and will never agree question for
    question, so equality would either fail forever or force the better backend
    down to the worse one. What must hold is that the swap costs nothing:

      1. every document the file backend retrieves is still retrieved, and
      2. the refusal behaviour is preserved exactly - which is the property this
         whole system exists to provide, and the one a more eager retriever
         would quietly destroy.

    Postgres currently does better than BM25 on Q14, the vocabulary-mismatch
    question written in Phase 1 to probe exactly this. That is allowed to
    improve; it is not allowed to regress.

    Two earlier versions of this test were wrong in ways worth recording. It
    built *both* arms from the `local` profile and then flipped `store`, so it
    sent a 384-dimension vector at a `vector(768)` column and Postgres answered
    `different vector dimensions 768 and 384`. And it was keyed on
    `SUPABASE_URL` being set rather than marked, so it ran by accident on any
    machine with a configured `.env` - and passed, because with the credentials
    cleared both arms silently fell back to `FileStore` and it compared the file
    backend against itself.
    """
    from eval.questions import ALL_QUESTIONS
    from eval.retrieval_eval import run_retrieval

    lexical = replace(live_dev_config.retrieval, mode="lexical")
    files = replace(live_dev_config, store="files", retrieval=lexical)
    supabase = replace(live_dev_config, store="supabase", retrieval=lexical)

    file_results, _ = run_retrieval(ALL_QUESTIONS, files)
    supabase_results, _ = run_retrieval(ALL_QUESTIONS, supabase)

    regressions = [
        (f.id, f.outcome, s.outcome)
        for f, s in zip(file_results, supabase_results)
        if f.outcome == "RETRIEVED" and s.outcome != "RETRIEVED"
    ]
    assert not regressions, f"Postgres lost documents the file backend found: {regressions}"

    # The refusal half, which must match exactly. A backend that admits an
    # unanswerable question the other stopped has traded away the behaviour the
    # brief scores most heavily.
    refusals = lambda results: [
        (r.id, r.outcome) for r in results if r.outcome in ("STOPPED", "ADMITTED")
    ]
    assert refusals(file_results) == refusals(supabase_results)
