# Configuration

Every tuned constant and model ID lives in one place —
[`agent/config.py`](../architecture/components.md) — for two reasons:

1. `qwen/qwen3.8-27b` is a *preview* model on Groq. Preview models change and
   disappear. When that happens, there should be one line to edit, not a
   grep across the codebase.
2. The gate thresholds are fitted to this corpus, and a sweep (see
   [Gate Calibration](../evaluation/gate-calibration.md)) is only practical
   with a single place to set them from.

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
| `cosine_floor` | `0.72` | `COSINE_FLOOR` | Reserved for a dense-retrieval arm; unrelated text still scores 0.6–0.7 on most embedding models. |
| `mode` | `"lexical"` | `RETRIEVAL_MODE` | `"lexical"` = BM25 + metadata filter only. `"hybrid"` adds dense retrieval and RRF fusion, when enabled. |

## Everything else

| Setting | Default | Meaning |
|---|---|---|
| `corpus_dir` | `"runbooks"` | Where the corpus loader reads `*.md` from. |
| `store` | `"files"` | `"files"` reads the corpus off disk (current). `"supabase"` would read from Postgres. `answer_question()` is identical either way; only what gets injected changes. |
| `max_rewrites` | `1` | Bound on any future corrective-retrieval loop — an unbounded rewrite cycle on a rate-limited free tier is a real hazard. |
| `llm_max_retries` | `3` | Retries with exponential backoff on Groq's 429. |
| `llm_timeout_s` | `45.0` | Per-call timeout. |

## Secrets

`GROQ_API_KEY` and `GEMINI_API_KEY` are read from the environment (or from a
hand-rolled `.env` loader — deliberately not a dependency, since everything
in `requirements.txt` has to be justified against Vercel's bundle limit).
Real environment variables always win over `.env`, so a deployed function
uses its platform configuration and ignores any `.env` bundled by mistake.

Copy `.env.example` to `.env` and set `GROQ_API_KEY` — see
[Getting Started](../getting-started.md).
