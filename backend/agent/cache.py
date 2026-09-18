"""The exact-answer cache - and why there is no semantic one.

A repeated question should not cost a model call. On a tier allowing roughly two
questions a minute, during a live demo where the same four questions get asked
repeatedly, that is the difference between a responsive system and a 429.

The key is a normalised question, so trailing whitespace and capitalisation do
not each get their own row. `no_match` results are cached too, deliberately: a
refusal is a real, considered result - the one this whole design exists to
produce - and re-deriving it costs exactly what deriving it did. Declining to
cache it would make the cheapest outcome the slowest.

--------------------------------------------------------------------------
**We are not building the semantic cache, and this is the reason.**

The obvious next step - embed the question, and serve a cached answer when a
previous question lands within some cosine threshold - is actively dangerous on
*this* corpus, and dangerous in exactly the way the rest of the system is built
to prevent.

The corpus is deliberately full of near-duplicates. "checkout-api is running
hot on CPU" and "payments-api is running hot on CPU" differ by one token, are
answered by different documents, and embed extremely close together - closer
than many genuine paraphrases of the same question. We measured the analogous
distributions when calibrating the retrieval gate and found them overlapping
(answerable 0.62-0.89, unanswerable 0.58-0.73). A semantic cache is the same
measurement with worse consequences: a retrieval mistake is caught downstream
by the metadata filter and by the grounding model, whereas a cache hit
*bypasses the entire pipeline* - filter, gate, grader and all - and serves an
answer citing the wrong document with no stage left to catch it.

So a semantic cache here would reintroduce the precise failure the brief warns
about, in the one place where none of our three defences can see it. The
honest engineering answer is that this corpus cannot support one safely, and a
cache that must not be trusted is not worth the table.

If it were needed, the mitigation would be to key on the *QuerySpec* rather
than the question text - same service, same failure mode, same intent - which
is exact matching on the structured fields rather than similarity on the prose.
That would be safe, because it is the same discriminator the filter uses. It is
noted here rather than built, because at four repeated demo questions the exact
cache already gets the win.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone

from agent.config import Settings, current_settings

# Answers are cached per configuration. A lexical-arm answer served to a hybrid
# request would make an A/B comparison read its own cache.
#
# Bump this when a change alters what the pipeline produces for an unchanged
# question. Every prior row keeps a key nothing will compute again, so it is a
# purge that needs no write to the database and cannot half-apply.
#
#   1 -> 2  The grader was being shown `doc.text[:900]`, so a policy document
#           that answered the question below that offset was dropped. Answers
#           cached before that fix are wrong, not merely old: Q13 of the
#           evaluation set was cached citing RB-005 instead of RB-010.
_CACHE_VERSION = 2


def normalise_question(question: str) -> str:
    """Collapse the spellings of the same question.

    Whitespace, case, smart quotes and trailing punctuation only. Deliberately
    *not* stemming or stopword removal: "how do I roll back checkout-api" and
    "how do I roll back payments-api" must never normalise together, and the
    more aggressive the normalisation, the closer that gets.
    """
    text = unicodedata.normalize("NFKC", question).strip().lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?!. ")


def cache_key(question: str, cfg: Settings) -> str:
    """Identity for a cached answer: the question *and* how it was answered."""
    material = "|".join(
        [
            str(_CACHE_VERSION),
            normalise_question(question),
            cfg.retrieval.mode,
            cfg.models.generator,
            str(cfg.retrieval.grader_enabled),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:48]


class NullCache:
    """No caching. The default locally, so the harness measures the pipeline."""

    enabled = False

    def get(self, question: str) -> dict | None:
        return None

    def put(self, question: str, result: dict) -> None:
        return None


class SupabaseCache:
    """Exact-match cache in Postgres.

    Every operation fails soft. A cache is an optimisation, and an optimisation
    that can fail a request is a liability - so a broken cache costs latency,
    never an answer.

    **Invalidation is by corpus version, and it is now actually wired up.** The
    schema has carried a `corpus_version` column and a `bump_corpus_version()`
    function since the table was created, and `0003_cache_and_jobs.sql` says
    "the answer cache reads it, so a re-ingest invalidates every cached answer".
    It did not. `put()` never wrote the column, so every row took the default of
    1, and `get()` never filtered on it. Against the live database the corpus
    had been re-ingested - `corpus_meta.corpus_version` was 2 - while every
    cached answer still said 1 and was still being served. A documented
    mechanism that is inert is worse than an absent one, because it is believed.
    """

    enabled = True

    def __init__(self, cfg: Settings):
        from agent.store.supabase_store import SupabaseStore

        self.cfg = cfg
        self.store = SupabaseStore(cfg)
        self._corpus_version: int | None = None

    def corpus_version(self) -> int:
        """The corpus generation this process should read and write.

        Read once per process. A re-ingest bumps it, so a long-lived instance
        keeps serving the previous generation until it restarts - acceptable
        for a cache, and the alternative is a round trip on every request to
        save a round trip.

        Falls back to 1 when the row cannot be read, which matches the column
        default: a cache that cannot determine its generation should behave
        like the old one rather than start writing rows nothing will ever match.
        """
        if self._corpus_version is None:
            try:
                rows = self.store._request(
                    "/rest/v1/corpus_meta?select=corpus_version&limit=1", method="GET"
                )
                self._corpus_version = int(rows[0]["corpus_version"]) if rows else 1
            except Exception:  # noqa: BLE001 - never fail a request over a cache
                self._corpus_version = 1
        return self._corpus_version

    def _freshness_filter(self) -> str:
        """The `created_at` bound, as a PostgREST filter fragment.

        Empty when expiry is switched off, so the query is unchanged for anyone
        running without a TTL.

        Age is measured from `created_at`, not `last_used_at`: the question is
        how old the *answer* is, and a row that is read often is not thereby
        more current. Touching `last_used_at` on read would otherwise keep a
        popular wrong answer alive forever, which is the opposite of the point.
        """
        from agent.store.supabase_store import filter_value

        ttl = self.cfg.answer_cache_ttl_s
        if ttl <= 0:
            return ""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl)
        return f"&created_at=gte.{filter_value(cutoff.isoformat())}"

    def get(self, question: str) -> dict | None:
        from agent.store.supabase_store import filter_value

        key = cache_key(question, self.cfg)
        try:
            rows = self.store._request(
                f"/rest/v1/answer_cache?question_key=eq.{filter_value(key)}"
                f"&corpus_version=eq.{self.corpus_version()}"
                f"{self._freshness_filter()}&select=*",
                method="GET",
            )
        except Exception:  # noqa: BLE001 - a cache miss is always an option
            return None

        if not rows:
            return None

        row = rows[0]
        self._touch(key)
        return {
            "answer": row["answer"],
            "cited_doc_ids": list(row.get("cited_doc_ids") or []),
            "confidence": row["confidence"],
            "cached": True,
        }

    def _touch(self, key: str) -> None:
        """Record that this row was served.

        `hits` and `last_used_at` were on the table from the start and nothing
        ever wrote them - every live row read 0 and `last_used_at == created_at`.
        They are worth having rather than dropping, because hit rate read
        against `no_match` rate is the signature of a poisoned cache: refusals
        being served faster and faster is what a stored outage looks like from
        the outside.

        Fire and forget. A cache that fails a request to update a counter has
        the priorities backwards.
        """
        try:
            from agent.store.supabase_store import filter_value

            self.store._request(
                f"/rest/v1/answer_cache?question_key=eq.{filter_value(key)}",
                payload={"last_used_at": "now()"},
                method="PATCH",
            )
        except Exception:  # noqa: BLE001
            return None

    def put(self, question: str, result: dict) -> None:
        try:
            self.store._request(
                "/rest/v1/answer_cache?on_conflict=question_key",
                payload=[
                    {
                        "question_key": cache_key(question, self.cfg),
                        "question": question[:2000],
                        "answer": result.get("answer", ""),
                        "cited_doc_ids": result.get("cited_doc_ids") or [],
                        "confidence": result.get("confidence", "no_match"),
                        "arm": self.cfg.retrieval.mode,
                        "model": self.cfg.models.generator,
                        # The column that makes a re-ingest invalidate this row.
                        "corpus_version": self.corpus_version(),
                        # Written explicitly because this is an upsert. On a
                        # conflict PostgREST updates only the columns supplied,
                        # so omitting `created_at` would leave the original
                        # value in place - and once a TTL reads that column,
                        # every refreshed answer would be born already expired
                        # and that question would never be cached again.
                        "created_at": "now()",
                    }
                ],
            )
        except Exception:  # noqa: BLE001 - never fail a request to write a cache
            return None


def get_cache(cfg: Settings | None = None):
    """The cache the active profile asks for.

    Off unless `ANSWER_CACHE` is on *and* Supabase is configured - there is
    nowhere else to put it, and a cache that silently does nothing is worse
    than one that is plainly absent.
    """
    cfg = cfg or current_settings()

    if cfg.answer_cache and cfg.supabase.configured:
        try:
            return SupabaseCache(cfg)
        except Exception:  # noqa: BLE001
            return NullCache()

    return NullCache()
