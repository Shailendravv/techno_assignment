-- The answer cache (Phase 8) and the ingestion job table (Phase 7).
--
-- Both replace infrastructure the production case study specifies with a
-- Postgres table, on the grounds that a table we already have beats a service
-- we would have to run. The trade is written down rather than glossed over:
-- Redis would be faster and would expire keys for us; a table is free, already
-- deployed, and survives a restart.

-- --------------------------------------------------------------------------
-- Exact-answer cache.
--
-- Keyed on a normalised question, not the raw string, so that trailing
-- whitespace and capitalisation do not each get their own row.
--
-- Cached `no_match` results are the interesting case and they are cached
-- deliberately. A `no_match` is a real, considered result - it is the answer
-- the whole design exists to produce - and re-deriving it costs the same as
-- deriving it the first time. Refusing to cache it would make the cheapest
-- outcome the slowest.
-- --------------------------------------------------------------------------
create table if not exists answer_cache (
    question_key   text primary key,
    question       text not null,

    answer         text not null,
    cited_doc_ids  text[] not null default '{}',
    confidence     text not null,

    -- Which configuration produced this. A cached answer from the lexical arm
    -- must not be served to a hybrid request, or an A/B comparison would be
    -- reading its own cache. Part of the key in practice.
    arm            text not null default 'hybrid',
    model          text,

    -- Invalidation is by corpus version rather than by time. The answer to
    -- "how do I roll back checkout-api" does not go stale on a schedule; it
    -- goes stale when the runbook changes.
    corpus_version integer not null default 1,

    hits           integer not null default 0,
    created_at     timestamptz not null default now(),
    last_used_at   timestamptz not null default now()
);

create index if not exists answer_cache_corpus_idx
    on answer_cache (corpus_version, arm);

-- --------------------------------------------------------------------------
-- Ingestion jobs.
--
-- This is the case study's Redis Streams + Celery, replaced by one table and a
-- process that polls it. At this volume - a handful of documents, uploaded by
-- hand - a queue would be more moving parts than work.
--
-- The state is here rather than in the worker's memory for the usual reason:
-- ingestion runs offline, in a GitHub Action or on a laptop, and the thing that
-- started it is not around to be asked how it went.
-- --------------------------------------------------------------------------
create table if not exists ingestion_jobs (
    id           bigserial primary key,
    public_id    text not null,
    filename     text,
    status       text not null default 'queued'
                 check (status in ('queued', 'running', 'done', 'failed')),
    doc_id       text,
    chunks       integer not null default 0,
    error        text,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create index if not exists ingestion_jobs_status_idx
    on ingestion_jobs (status, created_at);

alter table answer_cache   enable row level security;
alter table ingestion_jobs enable row level security;

-- --------------------------------------------------------------------------
-- Corpus version.
--
-- One row, bumped whenever ingestion changes a document. The answer cache reads
-- it, so a re-ingest invalidates every cached answer without deleting a row -
-- which means a bad ingest can be rolled back by decrementing a number rather
-- than by rebuilding the cache.
-- --------------------------------------------------------------------------
create table if not exists corpus_meta (
    id              integer primary key default 1 check (id = 1),
    corpus_version  integer not null default 1,
    documents       integer not null default 0,
    chunks          integer not null default 0,
    embedder        text,
    updated_at      timestamptz not null default now()
);

insert into corpus_meta (id) values (1) on conflict (id) do nothing;
alter table corpus_meta enable row level security;

create or replace function bump_corpus_version(
    p_documents integer,
    p_chunks    integer,
    p_embedder  text
)
returns integer
language sql
as $$
    update corpus_meta
       set corpus_version = corpus_version + 1,
           documents      = p_documents,
           chunks         = p_chunks,
           embedder       = p_embedder,
           updated_at     = now()
     where id = 1
    returning corpus_version;
$$;
