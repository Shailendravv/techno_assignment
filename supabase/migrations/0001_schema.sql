-- Phase 6: the index moves out of memory and into Postgres.
--
-- Why Postgres with pgvector rather than a dedicated vector service: this
-- corpus needs three things applied to the same query - dense similarity, full
-- text ranking, and a hard metadata filter. A separate vector database can do
-- the first, and then you fetch candidates back and filter them in application
-- code, which is fetch-then-filter: you ask for the top 8, throw half away, and
-- have no way to ask for "the top 8 that are actually allowed". Doing all three
-- in one SQL statement is the entire reason for this choice, and 0002 is where
-- that statement lives.
--
-- Scale note, so nobody reads more into these indexes than is there: twelve
-- documents is about seventy chunks. HNSW on seventy rows is theatre - Postgres
-- will sequential-scan it and be right to. The index is here because the shape
-- of the system should be the shape it would need at a realistic corpus size,
-- and because getting the operator class wrong is a thing you want to discover
-- now rather than at scale.

create extension if not exists vector;

-- --------------------------------------------------------------------------
-- documents
--
-- The structured fields are the whole design. A service name buried in four
-- hundred words of near-identical prose is a weak signal that a similarity
-- score can out-vote; the same name in a `service` column is a fact the filter
-- acts on. These columns are why the near-duplicate pairs are separable.
-- --------------------------------------------------------------------------
create table if not exists documents (
    doc_id        text primary key,
    title         text not null,

    -- NULL means "applies to every service" - a general policy document.
    -- This is not missing data, it is a meaningful value, and the search
    -- function must never drop a NULL service for mismatching a named one.
    -- That single rule is what makes the policy questions answerable.
    service       text,
    failure_mode  text,
    doc_type      text not null,
    date          date,

    content       text not null,

    -- Ingestion is idempotent on this. Re-running the pipeline over unchanged
    -- documents must cost nothing, or nobody will re-run it.
    content_hash  text not null,

    -- Where the canonical file lives. Populated by Phase 7; NULL for documents
    -- seeded straight from the repository.
    source_public_id text,
    source_url       text,

    version       integer not null default 1,
    updated_at    timestamptz not null default now(),

    -- Title weighted above body, mirroring the Python index, which counts the
    -- title twice. The title is where the two facts that decide the answer live
    -- - the service and the failure mode - stated in four words rather than
    -- diluted across four hundred.
    fts tsvector generated always as (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) stored
);

create index if not exists documents_fts_idx on documents using gin (fts);
create index if not exists documents_service_idx on documents (service);
create index if not exists documents_failure_mode_idx on documents (failure_mode);

-- --------------------------------------------------------------------------
-- chunks
--
-- Parent-document retrieval: match at `##` section level, cite at document
-- level. The brief's contract is `cited_doc_ids`, so `doc_id` is carried on
-- every chunk and every chunk-level hit collapses back to it before anything
-- downstream sees it.
--
-- The embedding width is fixed at 768 because pgvector needs a concrete
-- dimension in the column type. That ties this schema to `config/dev.json`'s
-- EMBEDDING_DIMS, and the coupling is checked at startup rather than
-- discovered as a cast error mid-query - see SupabaseStore.check_health.
-- --------------------------------------------------------------------------
create table if not exists chunks (
    id         bigserial primary key,
    chunk_id   text not null unique,
    doc_id     text not null references documents (doc_id) on delete cascade,
    section    text not null,
    text       text not null,
    embedding  vector(768),

    fts tsvector generated always as (
        to_tsvector('english', coalesce(text, ''))
    ) stored
);

create index if not exists chunks_doc_id_idx on chunks (doc_id);
create index if not exists chunks_fts_idx on chunks using gin (fts);

-- vector_cosine_ops, because the embedders return unit-normalised vectors and
-- cosine is what the Python path scores with. An L2 index here would rank
-- differently from the local arm, and the whole point of the equivalence test
-- is that the two backends agree.
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Row-level security.
--
-- The service key bypasses RLS, and the API only ever talks to Supabase with
-- the service key from the server side. Enabling RLS with no permissive policy
-- means that if the anon key ever leaks into the browser bundle, it reads
-- nothing. This is single-tenant, so there is no per-tenant policy to write -
-- the case study's ACL isolation is explicitly declined in the write-up - but
-- "no policy" should be a decision, not an oversight.
-- --------------------------------------------------------------------------
alter table documents enable row level security;
alter table chunks    enable row level security;
