"""The corpus, read off disk and indexed in memory.

This is the store Phases 0-5 used implicitly, now behind the protocol so the
SQL backend can take its place without the graph noticing. It is also the
reference implementation: the Phase 6 equivalence test asserts that Supabase
scores identically to this, so whatever this does is the definition of correct.

It stays the default, and stays supported rather than being a stepping stone.
The corpus is twelve markdown files in the repository, so this backend needs no
credentials, no network and no migration - which is what makes the test suite
and the evaluation harness runnable by anyone who clones this.
"""

from __future__ import annotations

from agent.config import Settings, current_settings
from agent.core.corpus import load_corpus
from agent.core.models import Candidate, Doc, QuerySpec
from agent.core.retrieve import (
    LexicalIndex,
    bm25_search,
    build_dense_index,
    build_index,
    dense_search,
    rrf_fuse,
)

# Dense indexes, memoised on (corpus directory, embedder signature). Embedding
# seventy chunks is a second locally and an HTTP round trip against Gemini;
# rebuilding per request would dominate every response.
_dense_indexes: dict[tuple[str, str], object] = {}


def reset_dense_cache() -> None:
    """Drop the memoised indexes. For tests that switch profiles or corpora."""
    _dense_indexes.clear()


class FileStore:
    name = "files"

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or current_settings()

    def documents(self) -> tuple[Doc, ...]:
        return load_corpus(self.cfg.corpus_dir)

    def lexical_index(self) -> LexicalIndex:
        return build_index(self.documents())

    # ----------------------------------------------------------------------

    def _dense_index(self, cfg: Settings):
        """The dense index, or None if dense retrieval cannot run.

        Returning None rather than raising is deliberate: a missing key or an
        uninstalled embedder must degrade the system to lexical-only - a
        complete, working retriever - rather than fail the request.
        """
        from agent.embed import embedding_available, get_embedder
        from agent.stages import current_recorder

        if not embedding_available(cfg):
            # The degradation this codebase is most likely to suffer in silence:
            # `EMBEDDER=local` with fastembed uninstalled looks exactly like a
            # working hybrid run from the outside.
            current_recorder().degrade(
                "dense_retrieve",
                f"EMBEDDER={cfg.embedding.backend} is not usable "
                "(missing package or key); serving lexical-only",
            )
            return None

        key = (cfg.corpus_dir, cfg.embedding.signature)
        if key in _dense_indexes:
            self._index_built = False  # served from the memo, nothing recomputed
            return _dense_indexes[key]

        self._index_built = True

        try:
            index = build_dense_index(self.documents(), get_embedder(cfg))
        except Exception as exc:  # noqa: BLE001 - degrade to lexical, see docstring
            current_recorder().degrade(
                "dense_retrieve",
                f"index build failed ({type(exc).__name__}); serving lexical-only",
            )
            return None

        _dense_indexes[key] = index
        return index

    def retrieve(
        self, spec: QuerySpec, query: str, cfg: Settings | None = None
    ) -> tuple[list[Candidate], list[str]]:
        """Rank lexically, rank densely, fuse. No filtering, no gating."""
        from agent.stages import current_recorder

        cfg = cfg or self.cfg
        retrieval = cfg.retrieval
        recorder = current_recorder()
        index = self.lexical_index()

        with recorder.stage("sparse_retrieve") as ledger:
            lexical = bm25_search(index, spec, retrieval.bm25_top_k)
            ledger.detail(
                "bm25 top="
                + (", ".join(f"{c.doc_id}({c.lexical_score:.1f})" for c in lexical[:3])
                   or "nothing")
            )
            # The ranked list, as the observation's output. A retrieval step
            # whose trace shows a duration and nothing retrieved cannot answer
            # the only question anybody opens it to ask.
            ledger.io(
                input={"query": query, "top_k": retrieval.bm25_top_k},
                output=[
                    {"doc_id": c.doc_id, "score": round(c.lexical_score, 4)}
                    for c in lexical
                ],
                method="bm25",
            )

        trace: list[str] = []

        if not retrieval.is_hybrid:
            recorder.skip("dense_retrieve", f"RETRIEVAL_MODE={retrieval.mode}")
            recorder.skip("embed_query", f"RETRIEVAL_MODE={retrieval.mode}")
            recorder.skip("rrf_fuse", "only one ranked list to fuse")

        dense_index = self._dense_index(cfg) if retrieval.is_hybrid else None
        if retrieval.is_hybrid and dense_index is None:
            # Say so rather than reporting a hybrid run that was quietly
            # lexical - that would corrupt any comparison between the arms.
            trace.append(
                "retrieve: hybrid requested but no embedder is available - "
                "falling back to lexical-only"
            )

        if dense_index is None:
            # `_dense_index` already recorded *why* as a degradation; these two
            # never got the chance to run at all.
            recorder.skip("embed_query", "no dense index")
            recorder.skip("rrf_fuse", "only one ranked list to fuse")
            return lexical, trace

        from agent.embed import get_embedder

        with recorder.stage("embed_query") as ledger:
            vector = get_embedder(cfg).embed_query(query)
            ledger.detail(f"{cfg.embedding.signature} dims={len(vector)}")
            # The text, not the vector: 384 floats render as noise and cost
            # payload. What a reader needs is which query was embedded, and by
            # which model - the signature is what makes an index built with one
            # embedder and queried by another findable after the fact.
            ledger.io(
                input={"query": query},
                output={"dims": len(vector)},
                embedder=cfg.embedding.signature,
            )

        with recorder.stage("dense_retrieve") as ledger:
            dense = dense_search(dense_index, vector, retrieval.dense_top_k)
            # The model and the splitting rules belong on this line because on
            # this backend the index is built lazily, here, at query time -
            # stage 06 never runs, so nothing else in a query ledger says how
            # these chunks were made or what embedded them.
            from agent.core.chunk import policy

            ledger.detail(
                f"model={cfg.embedding.signature} "
                f"index={'built' if getattr(self, '_index_built', False) else 'memoised'} "
                f"chunks={len(dense_index.chunks)} [{policy()}] "
                f"scored by best chunk per document; top="
                + (", ".join(f"{c.doc_id}({c.dense_score:.2f})" for c in dense[:3])
                   or "nothing")
            )
            ledger.io(
                input={"query": query, "top_k": retrieval.dense_top_k},
                output=[
                    {"doc_id": c.doc_id, "score": round(c.dense_score, 4)}
                    for c in dense
                ],
                embedder=cfg.embedding.signature,
                chunks=len(dense_index.chunks),
            )

        trace.append(
            "retrieve: dense top "
            + ", ".join(f"{c.doc_id}({c.dense_score:.2f})" for c in dense[:3])
        )

        with recorder.stage("rrf_fuse") as ledger:
            fused = rrf_fuse(lexical, dense, retrieval.rrf_k)
            ledger.detail(
                f"lexical={len(lexical)} dense={len(dense)} -> {len(fused)} "
                f"(k={retrieval.rrf_k})"
            )
            # Both inputs and the fused order, because the interesting failure
            # here is a document that either arm ranked well and fusion buried.
            ledger.io(
                input={
                    "lexical": [c.doc_id for c in lexical],
                    "dense": [c.doc_id for c in dense],
                },
                output=[
                    {"doc_id": c.doc_id, "rrf": round(c.fused_score, 5)} for c in fused
                ],
                rrf_k=retrieval.rrf_k,
            )

        return fused, trace

    def health(self) -> dict:
        try:
            count = len(self.documents())
            error = ""
        except Exception as exc:  # noqa: BLE001 - health must never 500
            count, error = 0, f"{type(exc).__name__}: {exc}"

        return {
            "store": self.name,
            "corpus_dir": self.cfg.corpus_dir,
            "documents": count,
            "reachable": not error,
            "error": error,
        }
