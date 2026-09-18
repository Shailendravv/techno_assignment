"""Embeddings: one interface, two real backends, and a disk cache.

The split exists because local and deployed want genuinely different things.

- The **evaluation harness** needs determinism and no rate limit. Measuring
  whether hybrid retrieval beats lexical is worthless if the numbers move
  between runs, and it cannot be done at all against a tier that allows 90
  requests a minute when a sweep makes thousands. So locally we run
  `bge-small-en-v1.5` through `fastembed` - a pinned ONNX model, offline, the
  same vectors every time.
- The **deployed function** cannot carry that model. `fastembed` pulls in
  `onnxruntime` plus a model download, and Vercel's Python bundler does no
  tree-shaking. So in the cloud, embeddings are an HTTP call to Gemini.

Which one runs is `EMBEDDER` in the active profile - `local` in `config/local.json`,
`gemini` in `config/dev.json`. Nothing else in the system knows the difference.

**The two are not interchangeable.** They produce vectors of different widths
from different models; an index built by one cannot be queried by the other.
That failure is silent - cosine similarity between a 384-dim and a 768-dim
space is not an error, it is a crash or, worse, nonsense - so every cached
vector carries the embedder's signature and loading checks it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol, Sequence

from agent.config import Embedding, Settings, current_settings


class EmbedderUnavailable(RuntimeError):
    """The configured backend cannot run: no key, or the package is missing."""


def l2_normalise(vector: Sequence[float]) -> list[float]:
    """Scale a vector to unit length, so cosine similarity is a dot product.

    This is not a micro-optimisation, it is a correctness requirement for
    Gemini. Gemini normalises its output at the full 3072 dimensions; ask for
    768 via `output_dimensionality` and you get a *truncation* of that vector,
    which is no longer unit length. Skipping this step does not raise - it
    silently degrades every similarity score in the system, which is the worst
    kind of bug to have in a retriever.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, tolerant of vectors that are not pre-normalised."""
    if len(a) != len(b):
        raise ValueError(
            f"dimension mismatch: {len(a)} vs {len(b)}. An index built with one "
            "embedder is being queried with another - check EMBEDDER/EMBEDDING_DIMS."
        )
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Embedder(Protocol):
    """What the retrieval code is allowed to assume about an embedder."""

    signature: str
    dims: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

class _DiskCache:
    """Cache vectors by (signature, text hash).

    Locally this turns a 12-second corpus embed into a file read. Against
    Gemini it is the difference between a free-tier daily quota that lasts and
    one that does not. The signature is in the key, so switching backends
    cannot read the other backend's vectors back.
    """

    def __init__(self, directory: str, signature: str):
        self.signature = signature
        self.path = Path(directory) / f"{_safe(signature)}.json"
        self._data: dict[str, list[float]] | None = None
        self._dirty = False

    def _load(self) -> dict[str, list[float]]:
        if self._data is not None:
            return self._data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            # A signature mismatch means these vectors came from a different
            # model or width. Discard them rather than mixing two spaces.
            stored = raw.get("vectors", {}) if raw.get("signature") == self.signature else {}
            self._data = stored if isinstance(stored, dict) else {}
        except (OSError, json.JSONDecodeError, AttributeError):
            # A corrupt or absent cache costs time, never correctness. Embed
            # again rather than failing, and never let a cache read raise into
            # a request handler.
            self._data = {}
        return self._data

    def get(self, text: str) -> list[float] | None:
        return self._load().get(_key(text))

    def put(self, text: str, vector: list[float]) -> None:
        self._load()[_key(text)] = vector
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty or self._data is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"signature": self.signature, "vectors": self._data}),
                encoding="utf-8",
            )
            self._dirty = False
        except OSError:
            # Read-only filesystem, which is the normal state of a serverless
            # function. Losing the cache is fine; failing the request is not.
            pass


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _safe(signature: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in signature)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class LocalEmbedder:
    """`fastembed` running `bge-small-en-v1.5` as ONNX, on the CPU.

    Chosen over a larger model for a reason that is about the experiment, not
    about quality: it is small enough to run the whole corpus in a second, so a
    threshold sweep across twenty questions is practical. A retriever whose
    calibration cannot be re-run is a retriever whose thresholds are guesses.

    bge models want an instruction prefix on the *query* side only, which is
    the closest this model has to Gemini's task-type asymmetry. Applying it to
    documents as well would defeat the point.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, cfg: Embedding):
        self.cfg = cfg
        self.signature = cfg.signature
        self.dims = cfg.dims
        self._model = None
        self._cache = _DiskCache(cfg.cache_dir, cfg.signature)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbedderUnavailable(
                "`fastembed` is not installed. It is a dev dependency only "
                "(requirements-dev.txt) - the deployed profile uses Gemini. "
                "Install it, or set EMBEDDER=none to run lexical-only."
            ) from exc
        self._model = TextEmbedding(model_name=self.cfg.model)
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed, consulting the cache first and only computing the misses."""
        vectors: list[list[float] | None] = [self._cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        if missing:
            model = self._load_model()
            computed = list(model.embed([texts[i] for i in missing]))
            for index, raw in zip(missing, computed):
                vector = l2_normalise([float(x) for x in raw])
                vectors[index] = vector
                self._cache.put(texts[index], vector)
            self._cache.flush()

        return [v for v in vectors if v is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.QUERY_PREFIX + text])[0]


class GeminiEmbedder:
    """`gemini-embedding-001`, truncated to 768 dimensions and re-normalised.

    The reason to prefer this over any other free hosted embedder is the true
    `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` asymmetry: a question and the
    passage answering it are not the same kind of text, and a model that knows
    which it is given produces better matches than one that treats both alike.

    Free tier: 90 requests/minute, 950/day, 250 texts per request. Our corpus is
    about 60 chunks, so a full re-index is one request.
    """

    BATCH = 250

    def __init__(self, cfg: Embedding, api_key: str):
        if not api_key:
            raise EmbedderUnavailable(
                "GEMINI_API_KEY is not set, and EMBEDDER=gemini. Set the key, "
                "switch to EMBEDDER=local, or set EMBEDDER=none for lexical-only."
            )
        self.cfg = cfg
        self.api_key = api_key
        self.signature = cfg.signature
        self.dims = cfg.dims
        self._client = None
        self._cache = _DiskCache(cfg.cache_dir, cfg.signature)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise EmbedderUnavailable(
                "The `google-genai` package is not installed."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        client = self._get_client()
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start : start + self.BATCH]
            response = client.models.embed_content(
                model=self.cfg.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dims,
                ),
            )
            # Re-normalise: Gemini normalises at 3072 dims, so a vector
            # truncated to 768 is not unit length. See l2_normalise.
            out.extend(l2_normalise(e.values) for e in response.embeddings)
        return out

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float] | None] = [self._cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        if missing:
            computed = self._call([texts[i] for i in missing], task_type)
            for index, vector in zip(missing, computed):
                vectors[index] = vector
                self._cache.put(texts[index], vector)
            self._cache.flush()

        return [v for v in vectors if v is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts), self.cfg.doc_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.cfg.query_task_type)[0]


class NullEmbedder:
    """`EMBEDDER=none`. Refuses to embed, rather than returning zero vectors.

    Returning zeros would be worse than failing: the dense arm would silently
    score everything at 0.0, hybrid would quietly degrade to lexical, and the
    measurement comparing the two would report a difference of nothing without
    anyone noticing the arm was switched off.
    """

    signature = "none"
    dims = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbedderUnavailable("EMBEDDER=none: dense retrieval is disabled.")

    def embed_query(self, text: str) -> list[float]:
        raise EmbedderUnavailable("EMBEDDER=none: dense retrieval is disabled.")


_cached: dict[str, Embedder] = {}


def get_embedder(cfg: Settings | None = None) -> Embedder:
    """The embedder the active profile asks for.

    Cached per signature: constructing a `LocalEmbedder` is cheap, but loading
    the ONNX session it wraps is not, and doing it per request would dominate
    every response time.
    """
    cfg = cfg or current_settings()
    embedding = cfg.embedding

    if embedding.signature in _cached:
        return _cached[embedding.signature]

    backend = embedding.backend.lower()
    if backend == "local":
        embedder: Embedder = LocalEmbedder(embedding)
    elif backend == "gemini":
        embedder = GeminiEmbedder(embedding, cfg.gemini_api_key)
    elif backend in ("none", "null", ""):
        embedder = NullEmbedder()
    else:
        raise EmbedderUnavailable(
            f"Unknown EMBEDDER {embedding.backend!r}. Use local, gemini, or none."
        )

    _cached[embedding.signature] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Drop the memoised embedders. For tests that switch profiles."""
    _cached.clear()


def embedding_available(cfg: Settings | None = None) -> bool:
    """Can dense retrieval actually run right now.

    Checked before routing into the dense arm so that a missing key degrades to
    lexical-only - which is a working system - rather than to a 500.
    """
    cfg = cfg or current_settings()
    if not cfg.embedding.enabled:
        return False
    backend = cfg.embedding.backend.lower()
    if backend == "gemini":
        return bool(cfg.gemini_api_key)
    if backend == "local":
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False
        return True
    return False
