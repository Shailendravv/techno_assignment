# Deployment

The agent runs three ways from one codebase: a CLI, an evaluation harness, and a
FastAPI app on Vercel. All three call the same `answer_question()`, so a harness
score is evidence about what the deployed API will do.

## Configuration profiles

Settings resolve in three layers, highest priority first:

```
environment variable  >  config/<APP_ENV>.json  >  code default
```

`APP_ENV` picks the profile and defaults to `local`.

| | `local` (default) | `dev` |
|---|---|---|
| store | files (`runbooks/*.md`) | Supabase Postgres |
| embedder | `fastembed` / bge-small, 384d | `gemini-embedding-001`, 768d |
| relevance grader | off | on |
| answer cache | off | on |
| for | tests, the harness, a laptop | the deployed function |

Profile files use the **same key names as the environment variables**, so there
is one namespace rather than a mapping table. They are committed, which is why
they contain **no secrets** — `test_no_profile_contains_a_secret` enforces that.
Credentials come from the environment only.

Two of these differences carry the weight:

- **`fastembed` does not belong in a serverless bundle.** It is ~200MB of
  `onnxruntime` plus a model download, and Vercel's Python bundler does no
  tree-shaking. Gemini is an HTTP call.
- **The deployed app is stateless.** The index lives in Postgres rather than
  being rebuilt from disk on every cold start.

A missing profile is not an error. The code defaults are a complete working
configuration, so a function built without `config/` degrades to them rather
than crashing inside a request handler.

## Setting it up

```bash
# 1. Run the migrations in order, in the Supabase SQL editor.
#    0001_schema.sql  0002_hybrid_search.sql  0003_cache_and_jobs.sql

# 2. Populate it. Offline - never inside a request handler.
python -m ingest.pipeline --seed --profile dev

# 3. Check what the pipeline would do first, if you prefer:
python -m ingest.pipeline --seed --dry-run

# 4. Deploy. Set APP_ENV=dev plus the keys from .env.example in the Vercel
#    dashboard, then check what actually shipped.
vercel build && du -sh .vercel/output/functions/*.func
vercel deploy --prod
```

## What the bundle includes, and why

`vercel.json` excludes `tests/`, `eval/`, `ingest/`, `baseline/`, `docs/`,
`site/`, `supabase/`, the virtualenv and the embedding cache.

It deliberately **does not exclude `runbooks/` or `web/`**, and an earlier
version that excluded all markdown was wrong. If Supabase is not configured the
app falls back to `FileStore`, which reads the checked-in corpus — so excluding
those twelve small markdown files would have broken the fallback silently, in
production, in a way no test would catch.

The practical consequence: **the deployed app serves correctly with only a
`GROQ_API_KEY` set.** Supabase and Cloudinary are upgrades, not requirements.

## Graceful degradation

Every external dependency has a defined behaviour when absent, and none of them
is a 500:

| Missing | Behaviour |
|---|---|
| Supabase | falls back to `FileStore` and the checked-in corpus |
| embedder (no Gemini key) | falls back to lexical-only retrieval, and says so in the trace |
| Groq key | retrieval and every `no_match` path still work; `/ask` returns 503 with the reason |
| Langfuse | no client is constructed and no network call is made; the per-stage trace is still returned ([Observability](architecture/observability.md)) |
| answer cache | every request is a miss |

`/health` reports the store **actually mounted** alongside the one requested.
They differ when Supabase is selected but unconfigured, and a health endpoint
that reported only the request would let a misconfigured deployment look correct.

## The keep-alive cron

A Supabase free project **pauses after seven days of inactivity**, and unpausing
it is a manual click in a dashboard. Discovering that the night before a demo is
the failure `.github/workflows/keepalive.yml` exists to prevent.

It runs weekly and does two things: asserts `store_reachable` is true, then asks
the off-topic question and asserts the answer is `no_match`. That second check is
deliberate — it costs no model call and it exercises the behaviour this system is
actually built around, so a regression in the thing that matters most fails the
cron rather than waiting for a demo.

Set the `APP_URL` repository secret to arm it. Without it, the job exits cleanly
rather than failing.

## Free-tier limits worth knowing before demo day

| Limit | Consequence | Mitigation |
|---|---|---|
| Groq 8k TPM | about two questions a minute | backoff and retry, answer cache, committed report as a fallback demo |
| `qwen/qwen3.8-27b` is preview | can change or vanish | model IDs live in config; `gpt-oss-120b` is the production default |
| Supabase pauses after 7 idle days | demo dead, manual unpause | the weekly cron above |
| Vercel 4.5 MB body cap | uploads would fail | browser uploads straight to Cloudinary; the API only signs |
| Cloudinary blocks PDF delivery on free | source links 404 | everything stored as `resource_type: raw` |
| Gemini 950 requests/day | a re-index is one request; sweeps are not | vectors are disk-cached by signature |
