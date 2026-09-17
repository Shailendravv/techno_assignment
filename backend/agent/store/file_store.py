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

from agent.config import Settings, settings as default_settings
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
        self.cfg = cfg or default_settings

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

        if not embedding_available(cfg):
            return None

        key = (cfg.corpus_dir, cfg.embedding.signature)
        if key in _dense_indexes:
            return _dense_indexes[key]

        try:
            index = build_dense_index(self.documents(), get_embedder(cfg))
        except Exception:  # noqa: BLE001 - degrade to lexical, see docstring
            return None

        _dense_indexes[key] = index
        return index

    def retrieve(
        self, spec: QuerySpec, query: str, cfg: Settings | None = None
    ) -> tuple[list[Candidate], list[str]]:
        """Rank lexically, rank densely, fuse. No filtering, no gating."""
        cfg = cfg or self.cfg
        retrieval = cfg.retrieval
        index = self.lexical_index()

        lexical = bm25_search(index, spec, retrieval.bm25_top_k)
        trace: list[str] = []

        dense_index = self._dense_index(cfg) if retrieval.is_hybrid else None
        if retrieval.is_hybrid and dense_index is None:
            # Say so rather than reporting a hybrid run that was quietly
            # lexical - that would corrupt any comparison between the arms.
            trace.append(
                "retrieve: hybrid requested but no embedder is available - "
                "falling back to lexical-only"
            )

        if dense_index is None:
            return lexical, trace

        from agent.embed import get_embedder

        vector = get_embedder(cfg).embed_query(query)
        dense = dense_search(dense_index, vector, retrieval.dense_top_k)
        trace.append(
            "retrieve: dense top "
            + ", ".join(f"{c.doc_id}({c.dense_score:.2f})" for c in dense[:3])
        )
        return rrf_fuse(lexical, dense, retrieval.rrf_k), trace

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
