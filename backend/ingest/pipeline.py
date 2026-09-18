"""The offline ingestion pipeline: a document becomes queryable.

    python -m ingest.pipeline --seed            # runbooks/ -> Supabase
    python -m ingest.pipeline --from-cloudinary # uploaded files -> Supabase
    python -m ingest.pipeline --seed --dry-run  # parse and chunk, write nothing

    fetch -> parse -> chunk -> embed -> upsert

**This never runs inside a request handler, and that is a design decision
rather than an accident of where the code lives.** Parsing a PDF and embedding
seventy chunks is slow and memory-hungry; a serverless invocation is the wrong
place for both. Keeping it offline also keeps `pymupdf4llm` and `fastembed` out
of the deployed function bundle entirely, which is most of how the bundle stays
inside Vercel's limit. It is the same offline/online split a production
ingestion system makes, for the same reasons.

**Idempotent on content hash.** Re-running over unchanged documents costs
nothing - no re-embedding, no writes, no consumed Gemini quota. A pipeline that
is expensive to re-run is a pipeline nobody re-runs, and then the index drifts
from the corpus.

`--seed` is the path that matters for this exercise: it ingests the twelve
markdown runbooks straight from the repository, so the database can be
populated without a Cloudinary account existing at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field

from agent.config import Settings, load_settings
from agent.core.chunk import policy as chunk_policy, split_document
from agent.core.corpus import load_corpus, parse_doc
from agent.core.models import Chunk, Doc


@dataclass
class IngestReport:
    documents_seen: int = 0
    documents_written: int = 0
    documents_unchanged: int = 0
    chunks_written: int = 0
    embedded: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def as_dict(self) -> dict:
        return {
            "documents_seen": self.documents_seen,
            "documents_written": self.documents_written,
            "documents_unchanged": self.documents_unchanged,
            "chunks_written": self.chunks_written,
            "embedded": self.embedded,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


def content_hash(doc: Doc) -> str:
    """Identity for idempotency.

    Over the metadata *and* the body, because a document whose `service` field
    changed is a different document even when its prose is identical - and
    metadata is what the filter acts on, so getting this wrong would leave a
    stale `service` deciding which questions the document answers.
    """
    payload = "|".join(
        [
            doc.doc_id,
            doc.title,
            doc.service or "",
            doc.failure_mode or "",
            doc.doc_type,
            doc.date or "",
            doc.text,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_bytes(raw: bytes, filename: str) -> Doc:
    """Turn an uploaded file into a Doc.

    Markdown with front-matter is the native format. PDFs go through
    `pymupdf4llm`, which produces markdown rather than a wall of text, so the
    `##` headings the chunker splits on survive - a plain text extractor would
    destroy exactly the structure retrieval depends on.
    """
    from agent.stages import current_recorder

    recorder = current_recorder()

    with recorder.stage("extract_text") as ledger:
        if filename.lower().endswith(".pdf"):
            text = _pdf_to_markdown(raw)
            ledger.detail(f"{filename}: pymupdf4llm, layout-aware, {len(text)} chars")
        else:
            text = raw.decode("utf-8", "replace")
            ledger.detail(f"{filename}: markdown read directly, {len(text)} chars")

    return _doc_from_markdown(text, filename)


def _pdf_to_markdown(raw: bytes) -> str:
    import tempfile

    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError(
            "`pymupdf4llm` is not installed. It is a dev/ingestion dependency "
            "(requirements-dev.txt) and is deliberately excluded from the "
            "deployed bundle."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(raw)
        path = handle.name
    return pymupdf4llm.to_markdown(path)


def _doc_from_markdown(text: str, filename: str) -> Doc:
    """Parse front-matter, falling back to something usable when it is absent.

    An uploaded document without front-matter is the interesting case, and the
    fallback is deliberately conservative: `service` and `failure_mode` stay
    None rather than being guessed from the text. A guessed `service` is worse
    than an absent one - absent means "applies to everything" and is merely
    imprecise, while a wrong one makes the filter drop the document for every
    question it actually answers.
    """
    import frontmatter

    from agent.stages import current_recorder

    recorder = current_recorder()

    with recorder.stage("clean") as ledger:
        post = frontmatter.loads(text)
        meta = post.metadata
        stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        body = (post.content or text).strip()
        ledger.detail(
            f"front-matter split, whitespace trimmed: {len(text)} -> {len(body)} chars"
        )

    def optional(key: str) -> str | None:
        value = meta.get(key)
        if value is None:
            return None
        cleaned = str(value).strip()
        return None if cleaned in ("", "null", "None", "~") else cleaned

    with recorder.stage("extract_metadata") as ledger:
        doc = Doc(
            doc_id=str(meta.get("doc_id") or stem).strip(),
            title=str(meta.get("title") or stem).strip(),
            service=optional("service"),
            failure_mode=optional("failure_mode"),
            doc_type=str(meta.get("doc_type") or "runbook").strip(),
            date=optional("date"),
            text=body,
        )
        missing = [
            key for key in ("service", "failure_mode", "date")
            if getattr(doc, key) is None
        ]
        ledger.detail(
            f"declared in front-matter: doc_id={doc.doc_id} service={doc.service} "
            f"failure_mode={doc.failure_mode} type={doc.doc_type}"
            + (f"; absent={missing} (left None, never guessed)" if missing else "")
        )

    return doc


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

class SupabaseWriter:
    """Upserts into `documents` and `chunks`, over PostgREST."""

    def __init__(self, cfg: Settings):
        from agent.store.supabase_store import SupabaseStore

        self.cfg = cfg
        self.store = SupabaseStore(cfg)

    def existing_hashes(self) -> dict[str, str]:
        rows = self.store._request(
            "/rest/v1/documents?select=doc_id,content_hash", method="GET"
        )
        return {row["doc_id"]: row.get("content_hash", "") for row in rows}

    def upsert_document(self, doc: Doc, digest: str, source: dict | None) -> None:
        payload = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "service": doc.service,
            "failure_mode": doc.failure_mode,
            "doc_type": doc.doc_type,
            "date": doc.date,
            "content": doc.text,
            "content_hash": digest,
            "source_public_id": (source or {}).get("public_id"),
            "source_url": (source or {}).get("url"),
        }
        self.store._request(
            "/rest/v1/documents?on_conflict=doc_id",
            payload=[payload],
        )

    def replace_chunks(self, doc_id: str, chunks: list[Chunk], vectors: list) -> None:
        """Delete then insert, rather than upsert.

        A re-chunked document may produce *fewer* sections than before, and an
        upsert keyed on `chunk_id` would leave the orphans behind - stale
        passages that still match queries and still resolve to a parent
        document whose text no longer contains them.
        """
        self.store._request(
            f"/rest/v1/chunks?doc_id=eq.{doc_id}", method="DELETE"
        )
        if not chunks:
            return

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "text": chunk.text,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors or [None] * len(chunks))
        ]
        self.store._request("/rest/v1/chunks", payload=rows)

    def bump_version(self, documents: int, chunks: int, embedder: str) -> int:
        result = self.store.rpc(
            "bump_corpus_version",
            {"p_documents": documents, "p_chunks": chunks, "p_embedder": embedder},
        )
        return result if isinstance(result, int) else 0

    def record_job(self, public_id: str, status: str, **fields) -> None:
        """Job state in Postgres, which is this project's Redis Streams + Celery.

        It lives in the database rather than the worker's memory because
        ingestion runs in a GitHub Action or on somebody's laptop, and whatever
        started it is not around to be asked how it went.
        """
        try:
            self.store._request(
                "/rest/v1/ingestion_jobs",
                payload=[{"public_id": public_id, "status": status, **fields}],
            )
        except Exception:  # noqa: BLE001 - job bookkeeping must not fail an ingest
            pass


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def ingest_documents(
    docs: list[tuple[Doc, dict | None]],
    cfg: Settings,
    dry_run: bool = False,
    force: bool = False,
) -> IngestReport:
    """Embed and upsert, skipping anything whose content has not changed."""
    from agent.stages import current_recorder

    recorder = current_recorder()
    report = IngestReport(dry_run=dry_run)
    report.documents_seen = len(docs)

    writer = None if dry_run else SupabaseWriter(cfg)
    existing = {} if (dry_run or force) else writer.existing_hashes()

    # No embedding on a dry run. It is the expensive step, it is metered
    # against a free-tier daily quota, and a run that writes nothing has no use
    # for vectors it computed.
    embedder = None
    if not cfg.embedding.enabled:
        recorder.skip("embed_chunks", f"EMBEDDER={cfg.embedding.backend}")
    elif dry_run:
        recorder.skip("embed_chunks", "dry run: nothing is written, so nothing is embedded")
    else:
        from agent.embed import embedding_available, get_embedder

        if embedding_available(cfg):
            embedder = get_embedder(cfg)
        else:
            recorder.degrade(
                "embed_chunks",
                f"EMBEDDER={cfg.embedding.backend} is not usable; chunks written "
                "without vectors and the dense arm will be inert",
            )
            report.errors.append(
                f"EMBEDDER={cfg.embedding.backend} is not usable; chunks will be "
                "written without vectors and the dense arm will be inert."
            )

    if dry_run:
        recorder.skip("store_embeddings", "dry run: nothing is written")

    # The ledger is one line per stage per run, so the stages inside this loop
    # are timed and counted here and emitted once, after it. Twelve `chunk`
    # lines would be a worse log than one that says twelve documents were
    # chunked - and the per-document detail is already in `IngestReport`.
    total_chunks = 0
    chunk_ms = 0
    embed_ms = 0
    store_ms = 0
    for doc, source in docs:
        digest = content_hash(doc)

        started = time.perf_counter()
        chunks = split_document(doc)
        chunk_ms += int((time.perf_counter() - started) * 1000)
        total_chunks += len(chunks)

        if existing.get(doc.doc_id) == digest:
            report.documents_unchanged += 1
            continue

        vectors = []
        if embedder is not None:
            started = time.perf_counter()
            try:
                vectors = embedder.embed_documents([c.text for c in chunks])
                report.embedded += len(vectors)
            except Exception as exc:  # noqa: BLE001 - one document must not end the run
                report.errors.append(f"{doc.doc_id}: embedding failed - {exc}")
                vectors = []
            embed_ms += int((time.perf_counter() - started) * 1000)

        if dry_run:
            report.documents_written += 1
            report.chunks_written += len(chunks)
            continue

        started = time.perf_counter()
        try:
            writer.upsert_document(doc, digest, source)
            writer.replace_chunks(doc.doc_id, chunks, vectors)
            store_ms += int((time.perf_counter() - started) * 1000)
            report.documents_written += 1
            report.chunks_written += len(chunks)
            if source:
                writer.record_job(
                    source.get("public_id", doc.doc_id),
                    "done",
                    doc_id=doc.doc_id,
                    chunks=len(chunks),
                )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{doc.doc_id}: {type(exc).__name__}: {exc}")
            if source:
                writer.record_job(
                    source.get("public_id", doc.doc_id), "failed", error=str(exc)[:500]
                )

    recorder.ran(
        "chunk",
        detail=f"{len(docs)} document(s) -> {total_chunks} chunk(s) [{chunk_policy()}]",
        ms=chunk_ms,
    )

    if embedder is not None:
        recorder.ran(
            "embed_chunks",
            detail=f"{report.embedded} vector(s), {cfg.embedding.signature}",
            ms=embed_ms,
        )

    if not dry_run:
        if report.documents_written:
            recorder.ran(
                "store_embeddings",
                detail=(
                    f"supabase: {report.documents_written} document(s), "
                    f"{report.chunks_written} chunk(s) upserted, "
                    f"{report.documents_unchanged} unchanged"
                ),
                ms=store_ms,
            )
        else:
            recorder.skip(
                "store_embeddings",
                f"nothing to write: all {report.documents_unchanged} document(s) "
                "unchanged by content hash",
            )

    if not dry_run and report.documents_written:
        writer.bump_version(
            len(docs), total_chunks, cfg.embedding.signature
        )

    return report


def collect_from_repository(cfg: Settings) -> list[tuple[Doc, dict | None]]:
    """The twelve runbooks, straight from the repository.

    This is what makes the database populatable without a Cloudinary account -
    and it is the path the deployment actually uses, since the corpus is
    checked in.
    """
    from agent.stages import current_recorder

    recorder = current_recorder()

    # Recorded here rather than inside `agent.core.corpus`, which stays a set of
    # functions with no opinion about logging. The three stages collapse into
    # one call on this path because the corpus is already markdown on disk:
    # there is nothing to extract from a PDF and nothing to clean.
    with recorder.stage("extract_text") as ledger:
        corpus = load_corpus(cfg.corpus_dir)
        ledger.detail(
            f"{len(corpus)} markdown file(s) read from {cfg.corpus_dir}/ - "
            "no parser needed, the corpus is checked in"
        )

    recorder.ran(
        "clean",
        detail="front-matter split and body trimmed by the corpus loader",
    )
    recorder.ran(
        "extract_metadata",
        detail=(
            "front-matter on every document: "
            + ", ".join(
                f"{d.doc_id}(service={d.service or '-'})" for d in corpus[:3]
            )
            + (", ..." if len(corpus) > 3 else "")
        ),
    )

    return [(doc, None) for doc in corpus]


def collect_from_cloudinary(cfg: Settings) -> list[tuple[Doc, dict | None]]:
    """Whatever has been uploaded, parsed into Docs."""
    from ingest.cloudinary_client import delivery_url, fetch_document, list_documents

    collected: list[tuple[Doc, dict | None]] = []
    for resource in list_documents(cfg):
        public_id = resource["public_id"]
        filename = resource.get("filename") or public_id
        raw = fetch_document(public_id, cfg)
        doc = parse_bytes(raw, f"{filename}.{resource.get('format', 'md')}")
        collected.append(
            (doc, {"public_id": public_id, "url": delivery_url(public_id, cfg)})
        )
    return collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingest.pipeline",
        description="Parse, chunk, embed and upsert documents into Supabase.",
    )
    parser.add_argument(
        "--seed", action="store_true", help="ingest runbooks/ from the repository"
    )
    parser.add_argument(
        "--from-cloudinary", action="store_true", help="ingest uploaded documents"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, chunk and report, but write nothing and embed nothing",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-embed even if the content is unchanged"
    )
    parser.add_argument("--profile", help="config profile (default: APP_ENV, or 'local')")
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args(argv)

    if not (args.seed or args.from_cloudinary):
        parser.error("choose --seed and/or --from-cloudinary")

    cfg = load_settings(args.profile) if args.profile else load_settings()

    if not args.dry_run and not cfg.supabase.configured:
        print(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY are not set, so there is "
            "nowhere to write. Set them, or use --dry-run to parse and chunk only.",
            file=sys.stderr,
        )
        return 2

    # The ingest half of the ledger (stages 1-8), scoped here for the same
    # reason `answer_question` scopes the query half: one run, one ledger.
    from agent.stages import INGEST, new_recorder, using_recorder

    with using_recorder(new_recorder(INGEST, cfg=cfg, dry_run=args.dry_run)):
        docs: list[tuple[Doc, dict | None]] = []
        if args.seed:
            docs += collect_from_repository(cfg)
        if args.from_cloudinary:
            docs += collect_from_cloudinary(cfg)

        report = ingest_documents(docs, cfg, dry_run=args.dry_run, force=args.force)

    print(json.dumps(report.as_dict(), indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)

    return 1 if report.errors and not report.documents_written else 0


if __name__ == "__main__":
    sys.exit(main())
