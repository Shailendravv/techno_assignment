-- The single-query hybrid search.
--
-- Dense similarity, full-text ranking, RRF fusion and the metadata hard filter,
-- in one statement. This is the whole reason for choosing Postgres+pgvector
-- over a separate vector service, and it is worth being precise about why.
--
-- The filter is a **pre-filter**, not a post-filter. It is applied inside the
-- CTEs that rank, before `limit`, so each arm returns its best `match_count`
-- rows *from the allowed set*. Rank first and filter afterwards and you ask for
-- the top 8, discard five of them as wrong-service, and are left with three -
-- with no way to have asked for eight allowed ones. At twelve documents that is
-- invisible. At any real size it is the difference between a working retriever
-- and one that quietly returns less than you asked for.
--
-- --------------------------------------------------------------------------
-- On duplicating the metadata filter, which is a real cost and worth stating.
--
-- The same rules exist in Python (`agent.core.retrieve.metadata_filter`) and
-- here in SQL. Two implementations of the rule this exercise turns on is
-- exactly the kind of duplication that rots. The resolution is that **Python
-- stays authoritative**: SupabaseStore runs this function to pre-filter, then
-- runs the Python filter over what comes back. Because SQL has already dropped
-- everything Python would, that second pass is expected to drop nothing - and
-- if it ever does, the two have diverged and the store says so loudly instead
-- of silently preferring one. The SQL is an optimisation whose disagreement
-- with the source of truth is detectable.
-- --------------------------------------------------------------------------

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

-- Lexical arm. ts_rank_cd over the weighted tsvector, cover density enabled so
-- that terms appearing near each other score above terms scattered apart.
lexical as (
    select
        a.doc_id,
        ts_rank_cd(a.fts, websearch_to_tsquery('english', query_text), 32)::real as score,
        row_number() over (
            order by ts_rank_cd(a.fts, websearch_to_tsquery('english', query_text), 32) desc,
                     a.doc_id
        ) as rank
    from allowed a
    where query_text is not null and query_text <> ''
    order by score desc
    limit match_count
),

-- Dense arm, at chunk level. `<=>` is cosine *distance*, so similarity is
-- 1 - distance; both embedders return unit-normalised vectors, which is what
-- makes that identity hold.
--
-- distinct on (doc_id) collapses chunks to their parent, keeping each
-- document's single best section - not its average. A question about
-- escalation should be answered by the document whose escalation section
-- matches, not penalised for its four unrelated sections.
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

-- Reciprocal rank fusion. Ranks, not scores: a ts_rank_cd value and a cosine
-- similarity are not on the same scale and no conversion between them is
-- principled, so any weighting would be a constant fitted to one question set.
-- A document both arms rank gets both contributions, so agreement between two
-- independent signals is rewarded without either dominating.
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

-- Lexical-only search, for the arm that must work without an embedder at all.
-- A thin wrapper rather than a second implementation: passing a NULL embedding
-- makes the dense CTE empty and RRF degenerates to the lexical ranking, so
-- there is one query to keep correct rather than two.
create or replace function lexical_search(
    query_text     text,
    p_service      text default null,
    p_failure_mode text default null,
    p_date         date default null,
    match_count    integer default 8
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
    select * from hybrid_search(
        query_text, null, p_service, p_failure_mode, p_date, match_count, 60
    );
$$;
