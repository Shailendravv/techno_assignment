"""Every tuned constant and model ID in the system lives here.

Two reasons this file exists rather than scattering values through the code:

1. `qwen/qwen3.8-27b` is a *preview* model on Groq. Preview models change and
   disappear. When that happens we want one line to edit, not a grep.
2. The gate thresholds are fitted to our own corpus. Phase 4 sweeps them, and a
   sweep is only practical if there is a single place to set them from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Models:
    """Groq model IDs, by the role each one plays.

    Roles, not names, are what the code refers to - so swapping a model is a
    config change and never a code change.
    """

    # Production model. 8k TPM / 1k RPD on the free tier, which is roughly two
    # questions a minute sustained. This is the default generator.
    generator: str = _env("GROQ_GENERATOR_MODEL", "openai/gpt-oss-120b")

    # Preview model, 450 tok/s. The one named to the assessor, kept as the
    # quality arm and selectable per-request.
    reasoner: str = _env("GROQ_REASONER_MODEL", "qwen/qwen3.8-27b")

    # Cheapest and fastest. Does relevance grading and query rewriting, which is
    # our free substitute for a cross-encoder reranker.
    grader: str = _env("GROQ_GRADER_MODEL", "openai/gpt-oss-20b")


@dataclass(frozen=True)
class Retrieval:
    """Shortlist sizes and the thresholds that decide `no_match`."""

    # The corpus is 12 documents. Recall is nearly free at this size, so we take
    # a generous shortlist and let the metadata filter do the cutting.
    bm25_top_k: int = _env_int("BM25_TOP_K", 8)
    dense_top_k: int = _env_int("DENSE_TOP_K", 8)
    final_top_k: int = _env_int("FINAL_TOP_K", 4)

    # RRF's smoothing constant. 60 is the value from the original paper and we
    # have no corpus-specific reason to move it.
    rrf_k: int = _env_int("RRF_K", 60)

    # The relevance floor, as a fraction of the best possible score for this
    # query. Raw BM25 scores are not comparable across queries, so we normalise
    # per-query before comparing - see core/retrieve.py.
    #
    # Too high and we reject real questions (MISSED). Too low and we cite
    # rubbish (FALSE_CITATION). Phase 4 sweeps this; the default is a starting
    # point, not a measured optimum.
    lexical_floor: float = _env_float("LEXICAL_FLOOR", 0.35)

    # Absolute cosine floor for the dense arm. Unrelated text still lands around
    # 0.6-0.7 with most embedding models, which is exactly why dense retrieval
    # makes `no_match` harder rather than easier.
    cosine_floor: float = _env_float("COSINE_FLOOR", 0.72)

    # "lexical" = BM25 + metadata filter (the arm that must work on its own).
    # "hybrid"  = adds dense retrieval and RRF fusion.
    mode: str = _env("RETRIEVAL_MODE", "lexical")


@dataclass(frozen=True)
class Settings:
    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)

    corpus_dir: str = _env("CORPUS_DIR", "runbooks")

    # "files" reads the corpus off disk and indexes in memory (Phases 0-5).
    # "supabase" reads from Postgres (Phase 6). `answer_question()` is identical
    # either way; only what gets injected changes.
    store: str = _env("STORE_BACKEND", "files")

    groq_api_key: str = _env("GROQ_API_KEY", "")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")

    # Bound on the corrective-retrieval loop. An unbounded rewrite cycle on a
    # rate-limited free tier is a real hazard, not a theoretical one.
    max_rewrites: int = _env_int("MAX_REWRITES", 1)

    # Groq free tier returns 429 readily. Retry with exponential backoff.
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 3)
    llm_timeout_s: float = _env_float("LLM_TIMEOUT_S", 45.0)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)


def load_settings() -> Settings:
    """Build a Settings from the current environment.

    Called rather than imported as a singleton so tests can monkeypatch the
    environment and get a fresh object.
    """
    return Settings()


settings = load_settings()

NO_MATCH_MESSAGE = (
    "I couldn't find anything in the runbooks that answers this. "
    "Rather than cite the closest-sounding document, I'd rather say so."
)
