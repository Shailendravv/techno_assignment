-- Replace a document's chunks in one transaction.
--
-- `ingest.pipeline.SupabaseWriter.replace_chunks` did this as two unrelated
-- HTTP calls: DELETE every chunk for a doc_id, then INSERT the new ones. The
-- delete-then-insert shape is right - a re-chunked document can produce fewer
-- sections than before, and an upsert keyed on chunk_id would leave the
-- orphans behind - but split across two requests there is a window where the
-- document has no chunks at all, and any failure in it is permanent.
--
-- The failure is silent, which is what makes it worth fixing. The `documents`
-- row still exists and still has its `content`, so the corpus looks complete
-- and the lexical arm keeps working; only the dense arm goes quiet for that one
-- document. Nothing in the trace says so, because retrieval genuinely ran.
-- Meanwhile the error was appended to `IngestReport.errors` for a run that had
-- already printed and exited.
--
-- One function, one transaction: either the document has its new chunks or it
-- keeps the old ones.

create or replace function replace_chunks(
    p_doc_id text,
    p_chunks jsonb
)
returns integer
language plpgsql
as $$
declare
    inserted integer;
begin
    -- Both statements run inside the function's transaction. A failure in the
    -- insert rolls the delete back with it.
    delete from chunks where doc_id = p_doc_id;

    if p_chunks is null or jsonb_array_length(p_chunks) = 0 then
        return 0;
    end if;

    insert into chunks (chunk_id, doc_id, section, text, embedding)
    select
        row ->> 'chunk_id',
        p_doc_id,                       -- from the argument, never from the payload
        row ->> 'section',
        row ->> 'text',
        case
            when row -> 'embedding' is null or jsonb_typeof(row -> 'embedding') = 'null'
                then null
            else (row ->> 'embedding')::vector
        end
    from jsonb_array_elements(p_chunks) as row;

    get diagnostics inserted = row_count;
    return inserted;
end;
$$;
