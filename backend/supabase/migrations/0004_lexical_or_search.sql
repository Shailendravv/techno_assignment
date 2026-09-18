-- Fix the lexical arm, which returned nothing.
--
-- `0002` ranked with `websearch_to_tsquery`, which builds a **conjunctive**
-- query: every content word must appear in a document for it to match at all.
-- On a natural-language question that is almost never true. Measured against
-- the live corpus:
--
--   websearch_to_tsquery('english',
--     'checkout-api is running hot on CPU what should I check first')
--   -> 'checkout-api' <-> 'checkout' <-> 'api' & 'run' & 'hot' & 'cpu'
--      & 'check' & 'first'
--   -> 0 documents match, ts_rank_cd = 0.000000 for all twelve
--
-- `rank_bm25`, which the file backend uses and against which the gate's
-- `lexical_floor` was calibrated, is **disjunctive**: it scores partial
-- matches. So the two backends were not two scorers on different scales, they
-- were different retrieval semantics, and the SQL one retrieved nothing.
--
-- Two consequences, both fixed here:
--
--   1. The lexical arm contributed no ranking signal at all.
--   2. The old `lexical` CTE had no `where score > 0`, so it still emitted all
--      twelve rows ranked by `doc_id` after the all-zero sort - feeding
--      **alphabetical order into RRF** as though it were evidence.
--
-- The companion half of this fix is in Python: the gate now scores candidates
-- against the local BM25 index rather than against whatever the store
-- returned, so one calibration is valid on both backends. See
-- `agent.core.retrieve.normalised_top_score`.

-- --------------------------------------------------------------------------
-- The query builder, as its own function so it can be tested and read.
--
-- `websearch_to_tsquery` is kept for its *parsing* - it handles quoted phrases
-- and hyphenated tokens like `checkout-api`, emitting them as `<->` phrase
-- groups - and only its top-level AND is rewritten to OR. Replacing ' & ' with
-- ' | ' leaves those phrase groups intact, so `checkout-api` still has to match
-- as a unit while the separate words no longer all have to be present.
-- --------------------------------------------------------------------------
create or replace function or_tsquery(q text)
returns tsquery
language sql
immutable
as $$
    select case
             when coalesce(trim(q), '') = '' then null::tsquery
             else nullif(
                    replace(websearch_to_tsquery('english', q)::text, ' & ', ' | '),
                    ''
                  )::tsquery
           end;
$$;

create or replace function hybrid_search(
    query_text       text,
    query_embedding  vector(768) default null,
    p_service        text default null,
    p_failure_mode   text default null,
    p_date           date default null,
    match_count      integer default 8,
    rrf_k            integer default 60
)
returns table (
    doc_id        text,
    title         text,
    service       text,
    failure_mode  text,
    doc_type      text,
    date          date,
    content       text,
    lexical_score real,
    dense_score   real,
    fused_score   real,
    matched_section text
)
language sql
stable
as $$
with
-- The metadata rules, in one place, exactly mirroring the Python filter:
--
--   question names X   document says Y            -> drop
--   question names X   document says nothing      -> KEEP  (general policy doc)
--   question names X   document says X            -> keep
--   question has date  postmortem with other date -> drop
--
-- The `service is null` disjunct is the SQL form of the general-policy rule.
-- Get it wrong and every question answered by a policy document breaks.
allowed as (
    select *
    from documents d
    where (p_service      is null or d.service      is null or d.service      = p_service)
      and (p_failure_mode is null or d.failure_mode is null or d.failure_mode = p_failure_mode)
      and (p_date         is null or d.doc_type <> 'postmortem' or d.date     = p_date)
),

-- Built once rather than re-parsed per row.
q as (select or_tsquery(query_text) as tsq),

-- Lexical arm. ts_rank_cd over the weighted tsvector, cover density enabled so
-- that terms appearing near each other score above terms scattered apart.
--
-- `a.fts @@ q.tsq` is the important addition: only documents that actually
-- match contribute a rank. Without it every document was ranked, so RRF
-- received twelve alphabetical positions instead of a ranking.
lexical as (
    select
        a.doc_id,
        ts_rank_cd(a.fts, q.tsq, 32)::real as score,
        row_number() over (
            order by ts_rank_cd(a.fts, q.tsq, 32) desc, a.doc_id
        ) as rank
    from allowed a, q
    where q.tsq is not null
      and a.fts @@ q.tsq
    order by score desc
    limit match_count
),

-- Dense arm, at chunk level. `<=>` is cosine *distance*, so similarity is
-- 1 - distance; both embedders return unit-normalised vectors, which is what
-- makes that identity hold.
--
-- distinct on (doc_id) collapses chunks to their parent, keeping each
-- document's single best section - not its average.
dense_chunks as (
    select distinct on (c.doc_id)
        c.doc_id,
        (1 - (c.embedding <=> query_embedding))::real as score,
        c.section
    from chunks c
    join allowed a on a.doc_id = c.doc_id
    where query_embedding is not null and c.embedding is not null
    order by c.doc_id, c.embedding <=> query_embedding
),
dense as (
    select
        dc.doc_id,
        dc.score,
        dc.section,
        row_number() over (order by dc.score desc, dc.doc_id) as rank
    from dense_chunks dc
    order by dc.score desc
    limit match_count
),

-- Reciprocal rank fusion. Ranks, not scores.
fused as (
    select
        coalesce(l.doc_id, d.doc_id) as doc_id,
        coalesce(l.score, 0)::real   as lexical_score,
        coalesce(d.score, 0)::real   as dense_score,
        (coalesce(1.0 / (rrf_k + l.rank), 0)
         + coalesce(1.0 / (rrf_k + d.rank), 0))::real as fused_score,
        d.section as matched_section
    from lexical l
    full outer join dense d on l.doc_id = d.doc_id
)
select
    a.doc_id,
    a.title,
    a.service,
    a.failure_mode,
    a.doc_type,
    a.date,
    a.content,
    f.lexical_score,
    f.dense_score,
    f.fused_score,
    f.matched_section
from fused f
join allowed a on a.doc_id = f.doc_id
order by f.fused_score desc, a.doc_id
limit match_count;
$$;
