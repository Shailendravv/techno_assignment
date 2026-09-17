"""Every tuned constant and model ID in the system lives here.

Three reasons this file exists rather than scattering values through the code:

1. `qwen/qwen3.8-27b` is a *preview* model on Groq. Preview models change and
   disappear. When that happens we want one line to edit, not a grep.
2. The gate thresholds are fitted to our own corpus. Phase 4 sweeps them, and a
   sweep is only practical if there is a single place to set them from.
3. Local and deployed want genuinely different components - a local ONNX
   embedder against a hosted one, files against Postgres - and that difference
   should be data, not a branch in the code.

Settings resolve through three layers, highest priority first:

    environment variable  >  config/<APP_ENV>.json  >  the defaults below

`APP_ENV` picks the profile and defaults to `local`. Profile files use the same
key names as the environment variables, so there is one namespace rather than a
mapping table, and they are committed - which is why they never contain
secrets. API keys are read from the environment only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Read `.env` from the repository root.

    Deliberately hand-rolled rather than a dependency: it is fifteen lines, and
    everything in `requirements.txt` has to be justified against Vercel's
    bundle limit. On a cloud platform (Vercel, Lambda) real environment
    variables win, so the deployed function uses the platform's configuration
    and ignores any `.env` that gets bundled by mistake. Locally the reverse:
    `.env` takes precedence, so editing it takes effect on the next reload
    instead of being shadowed by a stale process environment.
    """
    path = ROOT / ".env"
    if not path.is_file():
        return
    is_cloud = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key and (not is_cloud or key not in os.environ):
            os.environ[key] = value


_load_dotenv()


def active_profile_name() -> str:
    return os.getenv("APP_ENV", "local").strip() or "local"


def load_profile(name: str | None = None) -> dict:
    """Read `config/<name>.json`.

    A missing profile is not an error. The defaults in this module are a
    complete, working configuration on their own; a profile only shifts them.
    That matters because the deployed function may be built without the config
    directory, and "no profile" must degrade to "the defaults" rather than to a
    crash inside a request handler.
    """
    path = ROOT / "config" / f"{name or active_profile_name()}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    # `_comment` keys document the file for whoever opens it next. JSON has no
    # comment syntax, and a profile nobody can annotate gets stale fast.
    return {k: v for k, v in data.items() if not k.startswith("_")}


_profile = load_profile()


def _raw(name: str):
    """One setting, from the environment or the active profile."""
    if name in os.environ:
        return os.environ[name]
    return _profile.get(name)


def _env(name: str, default: str) -> str:
    value = _raw(name)
    return default if value is None else str(value)


def _env_float(name: str, default: float) -> float:
    value = _raw(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = _raw(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Models:
    """Groq model IDs, by the role each one plays.

    Roles, not names, are what the code refers to - so swapping a model is a
    config change and never a code change.
    """

    # Production model. 8k TPM / 1k RPD on the free tier, which is roughly two
    # questions a minute sustained. This is the default generator.
    generator: str = field(default_factory=lambda: _env("GROQ_GENERATOR_MODEL", "openai/gpt-oss-120b"))

    # Preview model, 450 tok/s. The one named to the assessor, kept as the
    # quality arm and selectable per-request.
    reasoner: str = field(default_factory=lambda: _env("GROQ_REASONER_MODEL", "qwen/qwen3.8-27b"))

    # Cheapest and fastest. Does relevance grading and query rewriting, which is
    # our free substitute for a cross-encoder reranker.
    grader: str = field(default_factory=lambda: _env("GROQ_GRADER_MODEL", "openai/gpt-oss-20b"))


@dataclass(frozen=True)
class Embedding:
    """Which embedder to use, and what shape its vectors are.

    The two backends are not interchangeable at runtime - they produce
    different vectors of different widths - so an index built with one cannot
    be queried with the other. `signature` is what makes that failure loud
    rather than silent: it is written into the cache key and checked when an
    index is loaded.
    """

    # "local"  - fastembed / bge-small-en-v1.5. Deterministic, offline,
    #            un-rate-limited: what the evaluation harness uses.
    # "gemini" - gemini-embedding-001 over HTTP. What the deployed function
    #            uses, because ~200MB of onnxruntime has no business in a
    #            serverless bundle.
    # "none"   - dense retrieval disabled; the lexical arm still works.
    backend: str = field(default_factory=lambda: _env("EMBEDDER", "local"))

    model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # bge-small is 384. Gemini is asked for 768 via output_dimensionality -
    # chosen for quality and speed, not to save space; 60 chunks is nothing.
    dims: int = field(default_factory=lambda: _env_int("EMBEDDING_DIMS", 384))

    # Embedding the same twelve documents on every harness run is wasted time
    # locally and wasted quota against Gemini. Cached to disk, keyed by
    # signature plus a hash of the text.
    cache_dir: str = field(default_factory=lambda: _env("EMBEDDING_CACHE_DIR", ".embeddings_cache"))

    # Gemini distinguishes document-side from query-side embeddings. That
    # asymmetry is most of why it is worth using, so it is not optional.
    doc_task_type: str = "RETRIEVAL_DOCUMENT"
    query_task_type: str = "RETRIEVAL_QUERY"

    @property
    def signature(self) -> str:
        return f"{self.backend}:{self.model}:{self.dims}"

    @property
    def enabled(self) -> bool:
        return self.backend != "none"


@dataclass(frozen=True)
class Retrieval:
    """Shortlist sizes and the thresholds that decide `no_match`."""

    # The corpus is 12 documents. Recall is nearly free at this size, so we take
    # a generous shortlist and let the metadata filter do the cutting.
    bm25_top_k: int = field(default_factory=lambda: _env_int("BM25_TOP_K", 8))
    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 8))
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 4))

    # RRF's smoothing constant. 60 is the value from the original paper and we
    # have no corpus-specific reason to move it.
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # Best BM25 score per content term. Raw BM25 scores are not comparable
    # across queries - a longer question scores higher just by having more
    # words - so we divide by query length before comparing.
    #
    # Too high and we reject real questions (MISSED). Too low and we cite
    # rubbish (FALSE_CITATION). Phase 4 sweeps this; the default is a starting
    # point, not a measured optimum.
    lexical_floor: float = field(default_factory=lambda: _env_float("LEXICAL_FLOOR", 0.35))

    # Fraction of the question's content words that must appear somewhere in
    # the corpus. This catches the off-topic question that BM25 still ranks
    # confidently: "refund" appears in no document, so however good the nearest
    # match looks, the corpus is not about that.
    coverage_floor: float = field(default_factory=lambda: _env_float("COVERAGE_FLOOR", 0.60))

    # Absolute cosine floor for the dense arm, calibrated in Phase 5 against
    # the known-negative questions rather than guessed.
    #
    # The measurement is worth stating because it is not the happy result. On
    # this corpus the cosine distributions *overlap*: answerable questions span
    # 0.62-0.89 and known-negative ones span 0.58-0.73, so no threshold
    # separates them. That is the concrete form of "dense retrieval makes
    # no_match harder, not easier" - a vector search always returns its nearest
    # neighbours, and unrelated text still looks respectably close.
    #
    # So this floor is set *above* the highest negative we measured. It is a
    # guard rail, not a discriminator: it can admit a question the lexical
    # floor rejected, but only on dense evidence stronger than anything an
    # unanswerable question produced. On our twenty questions it never fires,
    # because nothing that clears the coverage floor falls below the lexical
    # one. It is carried for the corpus we do not have rather than the one we
    # do, and the honest report of it is "inert, deliberately conservative".
    cosine_floor: float = field(default_factory=lambda: _env_float("COSINE_FLOOR", 0.75))

    # "lexical" = BM25 + metadata filter (the arm that must work on its own).
    # "hybrid"  = adds dense retrieval and RRF fusion.
    mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))

    # The CRAG relevance grader. Off locally because it costs an extra LLM call
    # per question against a tier that allows about two a minute, and the
    # harness runs twenty questions; on in the deployed profile, where a single
    # question has budget to spare.
    grader_enabled: bool = field(default_factory=lambda: _env_bool("GRADER_ENABLED", False))

    @property
    def is_hybrid(self) -> bool:
        return self.mode == "hybrid"


@dataclass(frozen=True)
class Supabase:
    url: str = field(default_factory=lambda: _env("SUPABASE_URL", ""))
    service_key: str = field(default_factory=lambda: _env("SUPABASE_SERVICE_KEY", ""))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.service_key)


@dataclass(frozen=True)
class Cloudinary:
    cloud_name: str = field(default_factory=lambda: _env("CLOUDINARY_CLOUD_NAME", ""))
    api_key: str = field(default_factory=lambda: _env("CLOUDINARY_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: _env("CLOUDINARY_API_SECRET", ""))

    # Raw delivery rather than PDF delivery: free Cloudinary accounts block PDF
    # delivery by default for security, and discovering that on demo day is a
    # bad afternoon.
    folder: str = field(default_factory=lambda: _env("CLOUDINARY_FOLDER", "runbooks"))
    resource_type: str = "raw"

    @property
    def configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)


@dataclass(frozen=True)
class Settings:
    profile: str = field(default_factory=active_profile_name)

    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)
    embedding: Embedding = field(default_factory=Embedding)
    supabase: Supabase = field(default_factory=Supabase)
    cloudinary: Cloudinary = field(default_factory=Cloudinary)

    corpus_dir: str = field(default_factory=lambda: _env("CORPUS_DIR", "runbooks"))

    # "files" reads the corpus off disk and indexes in memory (Phases 0-5).
    # "supabase" reads from Postgres (Phase 6). `answer_question()` is identical
    # either way; only what gets injected changes.
    store: str = field(default_factory=lambda: _env("STORE_BACKEND", "files"))

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    # Bound on the corrective-retrieval loop. An unbounded rewrite cycle on a
    # rate-limited free tier is a real hazard, not a theoretical one.
    max_rewrites: int = field(default_factory=lambda: _env_int("MAX_REWRITES", 1))

    # Exact-answer cache, keyed on the normalised question. Off locally so the
    # harness measures the pipeline rather than the cache.
    answer_cache: bool = field(default_factory=lambda: _env_bool("ANSWER_CACHE", False))

    # Groq free tier returns 429 readily. Retry with exponential backoff.
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 45.0))

    # Origins allowed to call the API from a browser. Comma-separated. Defaults
    # cover the Vite dev server so `npm run dev` works against a local API with
    # no extra setup; a deployed frontend origin goes in the environment.
    cors_origins: str = field(
        default_factory=lambda: _env(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        )
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    def describe(self) -> dict:
        """What this instance is configured as - for `/health` and the CLI.

        Reports whether each credential is present, never the credential. The
        health endpoint is public on a deployed app.
        """
        return {
            "profile": self.profile,
            "store": self.store,
            "retrieval_mode": self.retrieval.mode,
            "grader_enabled": self.retrieval.grader_enabled,
            "embedder": self.embedding.backend if self.embedding.enabled else "disabled",
            "embedding_dims": self.embedding.dims,
            "groq_configured": self.has_groq,
            "gemini_configured": self.has_gemini,
            "supabase_configured": self.supabase.configured,
            "cloudinary_configured": self.cloudinary.configured,
        }


def load_settings(profile: str | None = None) -> Settings:
    """Build a Settings from the current environment and profile.

    Called rather than imported as a singleton so tests can monkeypatch the
    environment, or name a profile, and get a fresh object. Passing `profile`
    re-reads the JSON, which is what makes "does dev.json actually parse" a
    testable question rather than a deployment-time surprise.
    """
    global _profile
    _load_dotenv()
    if profile is not None:
        previous = _profile
        _profile = load_profile(profile)
        try:
            return Settings(profile=profile)
        finally:
            _profile = previous
    return Settings()


settings = load_settings()

NO_MATCH_MESSAGE = (
    "I couldn't find anything in the runbooks that answers this. "
    "Rather than cite the closest-sounding document, I'd rather say so."
)
