"""The corpus in Postgres, searched with one SQL statement.

Talks to Supabase through PostgREST over plain `urllib`, not the `supabase`
Python package. That package pulls in `gotrue`, `storage3`, `realtime`,
`postgrest` and their transitive trees; we need to POST JSON to two RPC
endpoints. Vercel's Python bundler does no tree-shaking, so every dependency is
weighed against the bundle limit, and this one loses to forty lines of stdlib.

The interesting work is not here - it is in `supabase/migrations/0002`, where
dense similarity, full-text ranking, RRF fusion and the metadata pre-filter
happen in a single query. This module marshals arguments to it and turns rows
back into `Candidate`s.

**Python remains authoritative on the metadata rules.** The SQL pre-filters
because filtering after ranking means asking for the top 8 and keeping three;
but `agent.core.retrieve.metadata_filter` runs afterwards over what comes back,
and since SQL has already dropped everything it would drop, it should drop
nothing. When it does drop something, the two have diverged, and
`_warn_on_divergence` says so rather than letting the SQL quietly win.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from agent.config import Settings, current_settings
from agent.core.corpus import known_failure_modes  # noqa: F401 - re-exported for tests
from agent.core.models import Candidate, Doc, QuerySpec
from agent.core.retrieve import LexicalIndex, build_index
from agent.store import StoreUnavailable

TIMEOUT_S = 15.0


def filter_value(value: object) -> str:
    """Percent-encode a value before it goes into a PostgREST filter.

    PostgREST filters are query parameters, so an unescaped value is not data -
    it is syntax. A `,` splits an `in.()` list, a `&` starts another filter, a
    `*` is a `like` wildcard, and a `.` separates the operator from its
    argument. A value carrying any of them changes which rows the request
    matches.

    That mattered in one place in particular. `ingest.pipeline.replace_chunks`
    interpolated a `doc_id` straight into a **DELETE** filter, and that `doc_id`
    comes from the front-matter of an uploaded document - so it was
    attacker-controlled input steering a delete. Everything that builds a
    filter path goes through here now, including the answer cache, whose key is
    a hex digest and was never exploitable but has no reason to be the
    exception.
    """
    return urllib.parse.quote(str(value), safe="")


class SupabaseStore:
    name = "supabase"

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or current_settings()
        if not self.cfg.supabase.configured:
            raise StoreUnavailable(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY are not set. "
                "Set them, or use STORE_BACKEND=files."
            )
        self.base = self.cfg.supabase.url.rstrip("/")
        self._documents: tuple[Doc, ...] | None = None
        self._index: LexicalIndex | None = None
        self.divergences: list[str] = []

    # ----------------------------------------------------------------------
    # Transport
    # ----------------------------------------------------------------------

    def _headers(self) -> dict:
        key = self.cfg.supabase.service_key
        return {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # The service key bypasses row-level security, which is why it is
            # only ever used server-side and never reaches the browser.
            "Prefer": "return=representation",
        }

    def _request(self, path: str, payload: dict | None = None, method: str = "POST"):
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers=self._headers(), method=method
        )

        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise StoreUnavailable(
                f"Supabase returned {exc.code} for {path}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise StoreUnavailable(f"Cannot reach Supabase at {self.base}: {exc}") from exc

        return json.loads(body) if body.strip() else []

    def rpc(self, function: str, params: dict):
        return self._request(f"/rest/v1/rpc/{function}", params)

    # ----------------------------------------------------------------------
    # Documents
    # ----------------------------------------------------------------------

    @staticmethod
    def _to_doc(row: dict) -> Doc:
        """One row to a Doc.

        `service` and `failure_mode` come back as JSON null for general
        documents, which maps to Python None - the same value the markdown
        loader produces for an absent front-matter field. That equivalence is
        what lets one metadata filter serve both backends.
        """
        return Doc(
            doc_id=row["doc_id"],
            title=row["title"],
            service=row.get("service"),
            failure_mode=row.get("failure_mode"),
            doc_type=row["doc_type"],
            date=str(row["date"]) if row.get("date") else None,
            text=row.get("content", ""),
        )

    def documents(self) -> tuple[Doc, ...]:
        if self._documents is not None:
            return self._documents

        rows = self._request(
            "/rest/v1/documents?select=*&order=doc_id.asc", method="GET"
        )
        if not rows:
            raise StoreUnavailable(
                "The `documents` table is empty. Run the ingestion pipeline "
                "(`python -m ingest.pipeline --seed`) before querying."
            )

        self._documents = tuple(self._to_doc(row) for row in rows)
        return self._documents

    def lexical_index(self) -> LexicalIndex:
        """Built locally, from the documents Postgres holds.

        Used only for the gate's coverage measure - deliberately not delegated
        to SQL. Coverage asks whether the question's words appear anywhere in
        the corpus, and if the two backends answered that differently they
        would gate differently, which would make the equivalence test measure
        nothing.
        """
        if self._index is None:
            self._index = build_index(self.documents())
        return self._index

    # ----------------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------------

    def retrieve(
        self, spec: QuerySpec, query: str, cfg: Settings | None = None
    ) -> tuple[list[Candidate], list[str]]:
        from agent.stages import current_recorder

        cfg = cfg or self.cfg
        retrieval = cfg.retrieval
        recorder = current_recorder()
        trace: list[str] = []

        vector = None
        if not retrieval.is_hybrid:
            recorder.skip("embed_query", f"RETRIEVAL_MODE={retrieval.mode}")

        if retrieval.is_hybrid:
            from agent.embed import embedding_available, get_embedder

            if embedding_available(cfg):
                try:
                    with recorder.stage("embed_query") as ledger:
                        vector = get_embedder(cfg).embed_query(query)
                        ledger.detail(f"{cfg.embedding.signature} dims={len(vector)}")
                        ledger.io(
                            input={"query": query},
                            output={"dims": len(vector)},
                            embedder=cfg.embedding.signature,
                        )
                except Exception as exc:  # noqa: BLE001 - degrade to lexical
                    recorder.degrade(
                        "dense_retrieve",
                        f"query embedding failed ({type(exc).__name__}); lexical-only",
                    )
                    vector = None
            else:
                recorder.skip("embed_query", "no usable embedder")
            if vector is None:
                trace.append(
                    "retrieve: hybrid requested but no embedder is available - "
                    "falling back to lexical-only"
                )

        params = {
            "query_text": query,
            "query_embedding": vector,
            "p_service": spec.service,
            "p_failure_mode": spec.failure_mode,
            "p_date": spec.date,
            "match_count": max(retrieval.bm25_top_k, retrieval.dense_top_k),
            "rrf_k": retrieval.rrf_k,
        }

        # One statement does the sparse rank, the dense rank and the fusion, so
        # all three stages resolve to the same call. Recorded as three lines
        # anyway: the ledger describes the pipeline, not the SQL, and a reader
        # comparing this backend against `FileStore` needs the same twelve rows.
        with recorder.stage("sparse_retrieve") as ledger:
            rows = self.rpc("hybrid_search", params)
            ledger.detail(f"hybrid_search rpc, pre-filtered, {len(rows)} rows")
            # The vector is left out deliberately - it is 768 floats that
            # render as noise - but everything the SQL was actually given is
            # here, because "why did the database return these rows" is not
            # answerable from the result set alone.
            ledger.io(
                input={
                    "query_text": query,
                    "service": spec.service,
                    "failure_mode": spec.failure_mode,
                    "date": spec.date,
                    "match_count": params["match_count"],
                    "has_query_vector": vector is not None,
                },
                output=[
                    {
                        "doc_id": row.get("doc_id"),
                        "fused": round(float(row.get("fused_score") or 0.0), 5),
                    }
                    for row in rows
                ],
                method="hybrid_search rpc (pgvector)",
            )

        if vector is not None:
            recorder.ran("dense_retrieve", detail="fused in-database (pgvector)")
            recorder.ran("rrf_fuse", detail=f"in-database, k={retrieval.rrf_k}")
        else:
            recorder.degrade("dense_retrieve", "no query vector; SQL ran lexical-only")
            recorder.skip("rrf_fuse", "only one ranked list to fuse")

        candidates = [
            Candidate(
                doc=self._to_doc(row),
                lexical_score=float(row.get("lexical_score") or 0.0),
                dense_score=float(row.get("dense_score") or 0.0),
                fused_score=float(row.get("fused_score") or 0.0),
                reason=(
                    f"rrf={float(row.get('fused_score') or 0.0):.4f} "
                    f"(fts={float(row.get('lexical_score') or 0.0):.3f}, "
                    f"cos={float(row.get('dense_score') or 0.0):.2f})"
                ),
            )
            for row in rows
        ]

        trace.insert(0, f"retrieve: supabase returned {len(candidates)} pre-filtered rows")
        if vector is not None:
            trace.append(
                "retrieve: dense top "
                + ", ".join(
                    f"{c.doc_id}({c.dense_score:.2f})"
                    for c in sorted(candidates, key=lambda c: -c.dense_score)[:3]
                )
            )

        self._warn_on_divergence(spec, candidates, trace)
        return candidates, trace

    def _warn_on_divergence(
        self, spec: QuerySpec, candidates: list[Candidate], trace: list[str]
    ) -> None:
        """Check the SQL pre-filter against the Python rule that defines it.

        The Python filter runs after this anyway, so a disagreement costs
        correctness nothing - Python wins, as it should. What it costs is
        trust: if SQL is admitting rows Python drops, the two implementations
        of the rule this exercise turns on have drifted, and that should be
        visible in the trace and in `/health` rather than discovered later.
        """
        offenders = []
        for candidate in candidates:
            doc = candidate.doc
            if spec.service and doc.service and doc.service != spec.service:
                offenders.append(f"{doc.doc_id} (service {doc.service})")
            elif (
                spec.failure_mode
                and doc.failure_mode
                and doc.failure_mode != spec.failure_mode
            ):
                offenders.append(f"{doc.doc_id} (failure_mode {doc.failure_mode})")

        if offenders:
            message = (
                "SQL pre-filter disagrees with the Python metadata filter on "
                f"{offenders} - hybrid_search and metadata_filter have drifted. "
                "Python is authoritative and has dropped them."
            )
            self.divergences.append(message)
            trace.append(f"WARNING: {message}")

    # ----------------------------------------------------------------------

    def health(self) -> dict:
        """Whether this store is usable, and whether it matches the schema.

        The dimension check is the one worth having. `chunks.embedding` is
        `vector(768)` because pgvector needs a concrete width in the column
        type, which ties the schema to the profile's EMBEDDING_DIMS. Getting
        that wrong surfaces as a cast error inside a query on a live request;
        checking it here turns it into a line on `/health`.
        """
        expected = self.cfg.embedding.dims
        report = {
            "store": self.name,
            "url": self.base,
            "documents": 0,
            "reachable": False,
            "embedding_dims_expected": expected,
            "error": "",
        }

        try:
            report["documents"] = len(self.documents())
            report["reachable"] = True
        except StoreUnavailable as exc:
            report["error"] = str(exc)
            return report

        if expected != 768:
            report["error"] = (
                f"EMBEDDING_DIMS is {expected} but the schema declares "
                "vector(768). Either set EMBEDDING_DIMS=768 or alter the column "
                "- they cannot disagree."
            )

        if self.divergences:
            report["filter_divergences"] = self.divergences[-3:]

        return report
