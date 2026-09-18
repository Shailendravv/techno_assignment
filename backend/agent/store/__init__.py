"""Where the corpus and the index live: on disk, or in Postgres.

The `Store` protocol is drawn at a very specific line, and the line is the
point of the whole phase.

    the store RANKS          - lexical, dense, and the fusion of the two
    Python FILTERS and GATES - the metadata rules, and whether to answer at all

So a store is responsible for "which documents look relevant, in what order",
which is the part that genuinely differs between an in-memory BM25 index and a
SQL statement over pgvector. It is *not* responsible for the metadata filter or
the gate, which stay in `agent.core` as pure functions with one implementation
and one test suite.

That matters because the metadata filter is the component this exercise turns
on. Letting each backend own a copy of it would mean the rule that wins the
exercise existed twice, drifting apart, with the tests only ever covering one.

The SQL backend *does* also apply the filter - as a pre-filter, inside the
query, because filtering after ranking means asking for the top 8 and keeping
three. But Python re-applies it afterwards and remains authoritative. Since SQL
has already dropped everything Python would, that second pass should drop
nothing, and `SupabaseStore` reports it loudly when it does. The SQL is an
optimisation whose divergence from the source of truth is detectable rather
than silent.
"""

from __future__ import annotations

from typing import Protocol

from agent.config import Settings, current_settings
from agent.core.models import Candidate, Doc, QuerySpec
from agent.core.retrieve import LexicalIndex


class StoreUnavailable(RuntimeError):
    """The configured store cannot be reached or is not configured."""


class Store(Protocol):
    """What the graph is allowed to assume about where documents live."""

    name: str

    def documents(self) -> tuple[Doc, ...]:
        """Every document, for the baseline arm and for `/health`."""

    def lexical_index(self) -> LexicalIndex:
        """A local index, used only for the gate's corpus-coverage measure.

        Deliberately local even for the SQL backend. Coverage asks "do this
        question's words appear anywhere in the corpus at all", and the answer
        must not depend on which backend is mounted or the two would gate
        differently - which would make the equivalence test meaningless.
        Twelve documents is a cheap thing to hold.
        """

    def retrieve(
        self, spec: QuerySpec, query: str, cfg: Settings
    ) -> tuple[list[Candidate], list[str]]:
        """Ranked candidates and trace lines. Fused, not yet filtered or gated."""

    def health(self) -> dict:
        """Enough to tell, from `/health`, whether this store is actually usable."""


# Stores are memoised per configuration.
#
# `get_store()` used to construct a new one on every call, and it is called at
# least twice per request - once in `analyze_node` for the vocabularies, once
# in `retrieve_node` - plus once more per turn of the corrective loop. On the
# file backend that is invisible, because `load_corpus` is itself cached. On
# Supabase every instance re-fetched the whole corpus over HTTP.
#
# The correctness half matters more than the latency. `SupabaseStore` keeps
# `self.divergences` - the record of the SQL pre-filter disagreeing with the
# Python metadata filter, which is the designed safeguard against the two
# implementations of this system's central rule drifting apart. With a fresh
# instance per call, the one that recorded a divergence was discarded and
# `/health` read a different object, so the safeguard could not fire. A safety
# net that structurally cannot catch anything is worse than none, because it is
# believed.
_stores: dict[tuple, Store] = {}


def _store_key(cfg: Settings) -> tuple:
    """What actually distinguishes one store from another."""
    return (
        cfg.store,
        cfg.corpus_dir,
        cfg.supabase.url,
        cfg.supabase.service_key,
        cfg.embedding.signature,
    )


def reset_stores() -> None:
    """Drop the memoised stores. For tests, and for a changed configuration."""
    _stores.clear()


def get_store(cfg: Settings | None = None) -> Store:
    """The store the active profile asks for.

    Falls back to files when Supabase is selected but not configured. A missing
    credential should leave a working system reading the repository, not a 500
    from every request - the corpus is checked in, so files is always available.
    """
    cfg = cfg or current_settings()

    key = _store_key(cfg)
    store = _stores.get(key)
    if store is not None:
        return store

    if cfg.store == "supabase":
        from agent.store.supabase_store import SupabaseStore

        if cfg.supabase.configured:
            store = SupabaseStore(cfg)

    if store is None:
        from agent.store.file_store import FileStore

        store = FileStore(cfg)

    _stores[key] = store
    return store


def store_kind(cfg: Settings | None = None) -> str:
    """What is actually mounted, which may not be what was asked for."""
    cfg = cfg or current_settings()
    if cfg.store == "supabase" and not cfg.supabase.configured:
        return "files (supabase requested but not configured)"
    return cfg.store


__all__ = [
    "Candidate",
    "Doc",
    "QuerySpec",
    "Store",
    "StoreUnavailable",
    "get_store",
    "reset_stores",
    "store_kind",
]
