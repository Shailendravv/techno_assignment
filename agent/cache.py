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

from agent.config import Settings, settings as default_settings

# Answers are cached per configuration. A lexical-arm answer served to a hybrid
# request would make an A/B comparison read its own cache.
_CACHE_VERSION = 1


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
    """

    enabled = True

    def __init__(self, cfg: Settings):
        from agent.store.supabase_store import SupabaseStore

        self.cfg = cfg
        self.store = SupabaseStore(cfg)

    def get(self, question: str) -> dict | None:
        key = cache_key(question, self.cfg)
        try:
            rows = self.store._request(
                f"/rest/v1/answer_cache?question_key=eq.{key}&select=*", method="GET"
            )
        except Exception:  # noqa: BLE001 - a cache miss is always an option
            return None

        if not rows:
            return None

        row = rows[0]
        return {
            "answer": row["answer"],
            "cited_doc_ids": list(row.get("cited_doc_ids") or []),
            "confidence": row["confidence"],
            "cached": True,
        }

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
    cfg = cfg or default_settings

    if cfg.answer_cache and cfg.supabase.configured:
        try:
            return SupabaseCache(cfg)
        except Exception:  # noqa: BLE001
            return NullCache()

    return NullCache()
