# Observability

This agent produces three records of the same run, and they are deliberately
not the same thing:

| Record | Where | Audience |
|---|---|---|
| The per-stage **trace** | `--trace`, or `{"explain": true}` | whoever asked the question and wants to know why it was answered that way |
| The stage **ledger** | `logs/app.log`, one line per declared stage | whoever is debugging this instance, including the stages that did *not* run |
| The Langfuse **trace tree** | `cloud.langfuse.com`, when configured | whoever is looking across runs: cost, latency, refusal rate, regressions |

The first two are described in [Pipeline Components](components.md). This page
is the third.

## Off unless configured

Tracing needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. Without them
nothing is constructed, no network call is made, and every call site gets a
no-op handle — so the default configuration, the CLI and the evaluation harness
all run exactly as they did before Langfuse existed.

`LANGFUSE_TRACING_ENABLED=false` turns it off without deleting the keys. The
test suite sets it for every test (`tests/conftest.py`), which is what makes
"the suite sends nothing to Langfuse" true on a developer's machine and not
only on a clean one.

`/health` reports `langfuse_configured` alongside the other credentials — as a
boolean, never a value.

## What one trace looks like

One question is one trace. Multi-turn grouping is `session_id`'s job.

```
answer-question                 SPAN        input: the question   output: answer + citations
├── analyze-query               SPAN        the extracted service / failure mode / intent
├── retrieve-documents          RETRIEVER   the ranked shortlist
│   ├── search-lexical          RETRIEVER   BM25 scores per document
│   ├── embed-query             EMBEDDING   which embedder, how many dimensions
│   ├── search-dense            RETRIEVER   cosine scores per document
│   ├── fuse-rankings           CHAIN       both input rankings and the fused order
│   └── filter-and-gate         GUARDRAIL   what was dropped, and the gate's verdict
├── grade-relevance             EVALUATOR   the CRAG filter's keep/drop decision
│   └── call-grader             GENERATION  model, prompt, tokens
├── rewrite-query               SPAN        only on the corrective loop
│   └── call-grader             GENERATION
└── generate-answer             SPAN        the answer, and any invented citations discarded
    ├── build-prompt            SPAN        which defensive clauses the prompt carried
    └── call-generator          GENERATION  model, prompt, tokens, finish reason
```

The observation **types** are chosen rather than defaulted, because that is
what Langfuse filters, evaluators and dashboards are built on. A `retriever`
only looks something up; a `generation` carries model and token usage; the
metadata filter and gate is a `guardrail` because refusing is its entire job;
the CRAG grader is an `evaluator`.

The tree is built in one place. `StageRecorder.stage()` in `agent/stages.py`
already wraps every declared stage of both pipelines, so it opens the Langfuse
observation too — the log ledger and the trace tree are the same events, and
nesting follows the call stack without anything passing a parent around. The
only observation with no stage behind it is `retrieve-documents`, which groups
the four ranking stages and the gate; they are separately skippable, but in a
trace tree they are one retrieval step.

## What is on a trace, and why

| Attribute | Value | What it buys |
|---|---|---|
| Trace input/output | the question; the answer, citations and confidence | The tracing table is readable at a glance, and dataset experiments compare like with like |
| `environment` | the profile (`local`, `dev`) | Local experiments do not pollute deployed dashboards |
| Tags | `mode:hybrid`, `store:files`, `role:generator` | Compare arms — these are known before the run, which is what tags require |
| `session_id` | client-supplied; the harness sets one per arm | Twenty harness traces read as *the run that produced a score* |
| `user_id` | client-supplied; the harness sets `harness:<arm>` | Cost and quality per caller |
| Scores | `confidence` (categorical), `answered` (boolean) | "Show me every question we declined" — the query this design exists to make askable |
| Generations | model, messages, token usage, attempt number | Cost and latency per model, and per retry |

**Why the outcome is a score and not a tag.** Langfuse fixes tags when an
observation is created, and confidence is only known once the pipeline has run.
Recording the outcome as a score is not a workaround — it is what scores are
for, and it means a refusal rate can be charted over time.

**Cost may show as zero.** Langfuse derives cost from its model pricing table,
and Groq-hosted open models (`openai/gpt-oss-120b`) are not always in it. Token
counts are always captured, so cost can be attributed after the fact; the
tokens are the measurement, the price is a lookup.

## Retries are traced individually

`agent/llm.py` opens one `generation` per round trip to Groq, not per logical
call. A 429 that was retried is two requests that both counted against the
quota, and a trace showing one would disagree with the bill. A failed attempt
is marked `ERROR` with the exception on it.

## Flushing

Langfuse batches on a background thread, which is the wrong default for both
places this code runs:

- **The API.** A Vercel function can be frozen the moment it returns, so
  "flush later" means "do not flush". `/ask` registers the flush as a Starlette
  background task: it runs after the response body is sent and before the ASGI
  cycle completes, so the trace is delivered without the caller paying for the
  round trip. The 503 path flushes inline, because an `HTTPException` leaves
  the normal response path and takes its background tasks with it.
- **The CLI and the harness.** Both are short-lived processes that would
  otherwise exit first. The CLI flushes in a `finally`; the harness flushes
  once per arm rather than once per question, since the queue is shared.

## Masking

Everything on its way to Langfuse passes a masking pass that redacts
credential-shaped strings (`sk-…`, `gsk_…`, `pk-lf-…`, `AIza…`, JWTs) and email
addresses. The question itself is **not** masked and is not meant to be: it is
the trace input, and a trace whose input cannot be read answers nothing. The
masker catches the accident — a key pasted into a question — rather than
serving as a compliance control. A corpus with real personal data in it needs a
decision about whether to trace at all, not a wider regular expression.

## Failure is never the caller's problem

Every public function in `agent/observability.py` swallows its own exceptions
and degrades to a no-op handle. The rule is the same one the cache and the
stage ledger follow: **the pipeline is the product, and the instrumentation
around it is not.** Losing a trace is an acceptable cost; losing an answer to
an on-call engineer is not.

The one thing deliberately *not* swallowed is an exception from the pipeline
itself. A stage that raises is recorded as `failed` in the ledger, marked
`ERROR` with its status message in Langfuse, and re-raised unchanged.
