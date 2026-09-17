# Configuration

Every tuned constant and model ID lives in one place —
[`agent/config.py`](../architecture/components.md) — for three reasons:

1. `qwen/qwen3.8-27b` is a *preview* model on Groq. Preview models change and
   disappear. When that happens, there should be one line to edit, not a
   grep across the codebase.
2. The gate thresholds are fitted to this corpus, and a sweep (see
   [Gate Calibration](../evaluation/gate-calibration.md)) is only practical
   with a single place to set them from.
3. Local and deployed want genuinely different components — a local ONNX
   embedder against a hosted one, files against Postgres — and that
   difference should be data, not a branch in the code.

## The three layers

Settings resolve highest-priority first:

```
environment variable  >  config/<APP_ENV>.json  >  the code default
```

`APP_ENV` picks the profile and defaults to `local`. Profile files use the
**same key names as the environment variables**, so there is one namespace
rather than a mapping table.

| | `local` (default) | `dev` |
|---|---|---|
| `STORE_BACKEND` | `files` | `supabase` |
| `EMBEDDER` | `local` (fastembed, 384d) | `gemini` (768d) |
| `GRADER_ENABLED` | `false` | `true` |
| `ANSWER_CACHE` | `false` | `true` |

!!! warning "Profiles are committed, so they carry no secrets"
    `test_no_profile_contains_a_secret` enforces this. API keys are read from
    the environment only. A missing profile is not an error — the code
    defaults are a complete working configuration, so a function built without
    `config/` degrades to them rather than crashing in a request handler.

## Models — by role, not by name

```python
generator: str = "openai/gpt-oss-120b"   # production model, 8k TPM free
reasoner:  str = "qwen/qwen3.8-27b"      # preview, selectable per-request
grader:    str = "openai/gpt-oss-20b"    # cheapest and fastest
```

Code refers to `settings.models.generator`, never to a model string
directly — swapping a model is a config change, never a code change.

## Retrieval thresholds

| Setting | Default | Env var | Meaning |
|---|---|---|---|
| `bm25_top_k` | `8` | `BM25_TOP_K` | Shortlist size before filtering. Recall is nearly free at 12 documents, so this is generous. |
| `final_top_k` | `4` | `FINAL_TOP_K` | Survivors after the metadata filter. |
| `lexical_floor` | `0.35` | `LEXICAL_FLOOR` | Best BM25 score per content term, normalised for question length. |
| `coverage_floor` | `0.60` | `COVERAGE_FLOOR` | Fraction of the question's content words that must appear anywhere in the corpus. See [Gate Calibration](../evaluation/gate-calibration.md) for how this value was chosen. |
| `dense_top_k` | `8` | `DENSE_TOP_K` | Shortlist size for the dense arm, before fusion. |
| `rrf_k` | `60` | `RRF_K` | RRF's smoothing constant. The value from the original paper; no corpus-specific reason to move it. |
| `cosine_floor` | `0.75` | `COSINE_FLOOR` | A guard rail, **not a discriminator**. Measured, the cosine distributions overlap (answerable 0.62–0.89, unanswerable 0.58–0.73), so no threshold separates them. Set above the highest negative observed; inert on these twenty questions. See [Hybrid Retrieval](../evaluation/hybrid-retrieval.md). |
| `mode` | `"hybrid"` | `RETRIEVAL_MODE` | `"lexical"` = BM25 + metadata filter only. `"hybrid"` adds dense retrieval and RRF fusion, and degrades to lexical automatically when no embedder is available. |
| `grader_enabled` | `false` local, `true` dev | `GRADER_ENABLED` | The CRAG relevance grader. One extra model call per question, which is why it is off for the twenty-question harness. |

## Embedding

| Setting | Default | Env var | Meaning |
|---|---|---|---|
| `backend` | `local` | `EMBEDDER` | `local` (fastembed/bge-small) · `gemini` · `none` (refuses rather than returning zero vectors, so a disabled arm cannot masquerade as a weak one). |
| `model` | `BAAI/bge-small-en-v1.5` | `EMBEDDING_MODEL` | |
| `dims` | `384` local, `768` dev | `EMBEDDING_DIMS` | **Coupled to the SQL schema**, which declares `vector(768)`. `/health` checks the two agree rather than letting it surface as a cast error mid-query. |
| `cache_dir` | `.embeddings_cache` | `EMBEDDING_CACHE_DIR` | Vectors are cached by embedder *signature*, so two vector spaces can never be mixed. |

## Everything else

| Setting | Default | Meaning |
|---|---|---|
| `corpus_dir` | `"runbooks"` | Where the corpus loader reads `*.md` from. |
| `store` | `"files"` | `"files"` reads the corpus off disk. `"supabase"` reads from Postgres, falling back to files when unconfigured. `answer_question()` is identical either way; only what gets injected changes. |
| `max_rewrites` | `1` | Bound on the corrective-retrieval loop, checked in a routing edge — an unbounded rewrite cycle on a rate-limited free tier is a real hazard, not a theoretical one. |
| `answer_cache` | `false` | Exact-match cache keyed on the normalised question *and* the arm. Off locally so the harness measures the pipeline rather than the cache. |
| `llm_max_retries` | `3` | Retries with exponential backoff on Groq's 429. |
| `llm_timeout_s` | `45.0` | Per-call timeout. |

## Secrets

`GROQ_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_*`, `CLOUDINARY_*` and
`LANGFUSE_*` are read from the environment (or from a hand-rolled `.env`
loader — deliberately not a dependency, since everything in
`requirements.txt` has to be justified against Vercel's bundle limit).
Real environment variables always win over both `.env` and the profile, so a
deployed function uses its platform configuration and ignores any `.env`
bundled by mistake.

`settings.describe()`, which backs `/health`, reports whether each credential
is **present** and never what it is — that endpoint is public on a deployed
app.

Copy `.env.example` to `.env` and set `GROQ_API_KEY` — see
[Getting Started](../getting-started.md).
