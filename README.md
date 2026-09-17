# Runbook Agent

A grounded question-answering agent over twelve operational runbooks. Ask it a
question in plain English; it finds the right document and answers **while
telling you which document the answer came from** — or tells you plainly that
nothing in the corpus applies.

That last part is the hard part, and it is the feature this system is built
around. The corpus deliberately contains near-duplicate documents: a runbook for
`payments-api` high CPU that shares 69% of its words with the one for
`checkout-api` high CPU, and a `payments-api` rollback procedure that shares 77%
with the `checkout-api` one. Finding a document that *looks* relevant is easy.
Refusing to answer when nothing truly applies, and not being fooled by a
near-identical document about the wrong service, is the actual problem.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt

cp .env.example .env            # then add your GROQ_API_KEY
```

A free Groq key takes about a minute to get at
[console.groq.com/keys](https://console.groq.com/keys). No card, no paid tier.

```bash
# Ask a question
python -m agent "checkout-api is running hot on CPU - what should I check first?"

# Watch it decline, and see why
python -m agent "What is our refund policy for orders over $500?" --trace

# Run the evaluation harness (this is the scored deliverable)
python -m eval.harness --arm hybrid --out harness_output.json

# Compare against the no-retrieval control
python -m eval.harness --compare lexical,hybrid,baseline --delay 30

# The API
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I safely roll back checkout-api?"}'
```

**Most of the system needs no API key.** Retrieval, filtering, the gate, and the
entire test suite run offline:

```bash
pytest -q                       # 260 tests, no network, no key

# Gate calibration, and the lexical-vs-hybrid measurement. Both retrieval-only:
# free, deterministic, reproducible.
python -m eval.harness --sweep
python -m eval.harness --retrieval-only --compare lexical,hybrid
```

### Configuration

Settings resolve in three layers, highest first:

```
environment variable  >  config/<APP_ENV>.json  >  code default
```

| profile | store | embedder | for |
|---|---|---|---|
| `local` (default) | files | `fastembed` / bge-small, offline | tests, the harness, a laptop |
| `dev` | Supabase | `gemini-embedding-001` | the deployed function |

The profiles are committed and contain **no secrets** — a test enforces that.
Credentials come from the environment only; see [.env.example](.env.example).

---

## The entry point

Everything is reachable through one function, as the brief requires:

```python
from agent.api import answer_question

answer_question("checkout-api is running hot on CPU")
# {
#   "answer": "Start by checking the deploy log for a recent release...",
#   "cited_doc_ids": ["RB-001"],
#   "confidence": "high"
# }
```

`confidence` is one of `high` | `medium` | `low` | `no_match`. **An empty
`cited_doc_ids` with `confidence: "no_match"` is a real result, not an error** —
it means nothing in the corpus answers the question, and saying so is the
correct behaviour.

The CLI, the HTTP API, and the evaluation harness all call this same function,
so a harness score is evidence about what the deployed API will actually do.

---

## How it works

```
question
   |
   v
[analyze]      what is this question about?
   |           service? failure mode? intent? date?
   v
[retrieve]     BM25 + dense  ->  RRF fusion  ->  metadata filter  ->  gate
   |                                            drops contradictions
   |
   +---- gate rejects ------------------------------> [finalize]  no_match
   |                                                   zero LLM calls
   v
[grade]        a small model: does each document ACTUALLY apply?
   |
   +---- nothing relevant --> [rewrite] --> back to [analyze]   (once, bounded)
   |
   v
[ground]       "here are 4 documents. Answer only from these,
   |            or say none of them apply."
   v
[finalize]     verify citations, score confidence
   |
   v
{answer, cited_doc_ids, confidence}
```

Four signals, each doing a different job:

| Signal | Job | Why |
|---|---|---|
| **BM25** | Ranks, precisely | Scores literal tokens, so it cannot blur `checkout-api` into `payments-api` |
| **Dense** | Ranks, for recall | Catches questions phrased in words the runbooks never use. Added for one measured failure — see below |
| **Metadata filter** | Discriminates | A wrong service is a *contradiction*, not a weak signal to be outvoted by 400 words of similar prose. It drops, it does not penalise |
| **The gate** | Decides whether to answer | If nothing credible survives, the model is never called |

Dense retrieval is added for **recall only**, on the explicit understanding that
it is bad at what BM25 is good at: it is trained to map near-duplicates close
together, which is exactly wrong for this corpus. So the labour is divided —
**dense retrieval finds candidates, the metadata filter discriminates.**

The gate is a **conditional edge in the LangGraph state graph**, not an `if`
inside a function. That makes "we do not call the model when nothing survived" a
declared property of the structure rather than a branch buried in a call stack —
and it is a safety guarantee, not an optimisation. A model that is never shown a
document cannot invent a citation for one.

There are **three independent chances to decline**, each catching what it is
actually good at:

1. The **gate** catches questions whose vocabulary is nowhere in the corpus.
2. The **relevance grader** catches retrieved documents that look right but do
   not apply. If it rejects everything, the query is rewritten in the corpus's
   vocabulary and retrieval runs once more — the loop is bounded at one retry by
   a counter checked in a routing edge.
3. The **grounding prompt** states plainly that returning no citations is a
   correct and expected answer.

One threshold contorted to catch everything would be worse than three
mechanisms each aimed at a different failure.

---

## Layout

```
agent/
  api.py            answer_question()  <- the entry point
  graph.py          LangGraph StateGraph; the gate as a conditional edge
  state.py          what flows between nodes
  core/             PURE functions - no LangGraph, no network, no I/O
    corpus.py       parse runbooks/*.md front-matter into Doc records
    query.py        question -> QuerySpec (service, failure mode, intent, date)
    retrieve.py     BM25 + dense + RRF + metadata filter + gate  <- the important one
    chunk.py        split on ## headings; cite the parent doc_id
    confidence.py   high | medium | low | no_match
  nodes/            thin adapters: unpack state, call a core fn, write back
    ground.py       the grounding prompt, and citation verification
    grade.py        the CRAG relevance grader and query rewriter
  store/            Store protocol -> FileStore | SupabaseStore
  embed.py          one Embedder interface: fastembed | Gemini | none
  cache.py          exact-answer cache (and why there is no semantic one)
  observability.py  Langfuse export, off unless configured
  llm.py            Groq client: 429 backoff, defensive JSON parsing
  config.py         every tuned constant and model ID

config/             local.json, dev.json - committed, no secrets
app/                FastAPI surface (thin by design)
web/                single-page UI
ingest/             offline pipeline: Cloudinary -> parse -> chunk -> embed -> Supabase
supabase/migrations single-query hybrid search, in SQL
baseline/           the no-retrieval control arm
eval/               questions, harness, scorer, retrieval-only scorer
runbooks/           RB-001.md .. RB-012.md
tests/              260 tests, all offline
```

`agent/core/` imports nothing heavy on purpose. That is what keeps the test
suite for the component that wins this exercise fast, and runnable on a machine
with no keys and no internet.

---

## The stack, and why

| Layer | Choice | Why this one |
|---|---|---|
| Generation | Groq `openai/gpt-oss-120b` | Production model, 8k TPM free — about two questions a minute |
| Reasoning arm | Groq `qwen/qwen3.8-27b` | Selectable with `--reasoner`. Preview model, so its ID lives in config |
| Grading | Groq `openai/gpt-oss-20b` | Cheapest and fastest |
| Lexical | `rank_bm25` locally, Postgres `ts_rank_cd` deployed | Same interface, two backends |
| Embeddings (local) | `fastembed` / `bge-small-en-v1.5`, 384d ONNX | Deterministic, offline, un-rate-limited — so the harness is reproducible |
| Embeddings (deployed) | `gemini-embedding-001` @ 768d | Real `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` asymmetry, and no 200MB of onnxruntime in the bundle |
| Store | Supabase Postgres + pgvector | Dense similarity, full-text and the metadata filter in **one SQL statement** |
| Object store | Cloudinary (`resource_type: raw`) | Free accounts block PDF delivery by default; raw is not subject to that rule |
| Hosting | Vercel Python runtime | 500MB bundle, generous max duration |
| Orchestration | LangGraph | Earns its place at the conditional edges, not the happy path |

Everything runs on free tiers. Groq has **no embeddings endpoint**, which is why
embeddings come from Google instead.

Supabase, Cloudinary and Langfuse are talked to over stdlib `urllib` rather than
their SDKs. Those packages drag in large dependency trees for what amounts to a
few JSON POSTs, Vercel's bundler does no tree-shaking, and the bundle limit is
real.

---

## Evaluation

Twenty questions: the five from the brief, plus fifteen held out — ten
answerable and five where the correct answer is `no_match`. **Both the corpus and
the questions were written and committed before any retrieval code existed.**
The corpus is self-authored, so that ordering is the only real defence against
tuning a retriever to documents that were tuned to suit it.

The harness reports the four outcomes the brief names, plus `PARTIAL`:

|  | agent cited something | agent said `no_match` |
|---|---|---|
| **a doc applies** | `CORRECT_CITATION` / `PARTIAL` / `WRONG_CITATION` | `MISSED` |
| **nothing applies** | `FALSE_CITATION` | `CORRECT_NO_MATCH` |

A single accuracy number would let a lazy agent hide. An agent that never says
`no_match` and one that says it too readily can score identically while needing
opposite fixes — lower the gate floor versus raise it. Four outcomes tell them
apart; one number cannot.

### Does hybrid retrieval actually help?

Measured directly at the retrieval stage — no model calls, deterministic, and
reproducible for free with
`python -m eval.harness --retrieval-only --compare lexical,hybrid`. Results are
committed in [`retrieval_comparison.json`](retrieval_comparison.json):

| | lexical | hybrid |
|---|---|---|
| retrieval recall (answerable questions whose document reached the model) | **93%** | **100%** |
| ranked first | 93% | 93% |
| unanswerable questions stopped before any model call | 80% | 80% |

The arms differ on **exactly one question** — Q14, the vocabulary-mismatch
question written in Phase 1 to predict this, which moves from `NOT_RETRIEVED` to
`RETRIEVED`. Nothing regresses. So hybrid is the default, degrading to lexical
automatically when no embedder is available.

That is a small win, honestly reported: the dense arm earns its place on one
question in twenty.

**A planned idea the measurement killed.** The design called for an absolute
cosine floor calibrated against known negatives. Measured, the distributions
overlap — answerable questions reach 0.62–0.89 and unanswerable ones reach
0.58–0.73. No threshold separates them. That is the concrete form of *dense
retrieval makes `no_match` harder, not easier*. The floor is therefore set above
the highest negative as a guard rail, never fires on these twenty questions, and
a test pins the overlap so the finding is not quietly forgotten.

**The end-to-end four-outcome scores are not published here, because they have
not been run** — scoring needs a `GROQ_API_KEY`. The harness is built and tested;
running it is one command. See [WRITEUP.md](WRITEUP.md) §3.

### Gate calibration

`python -m eval.harness --sweep` sweeps the gate floor and shows the trade-off
directly. Results are in [`eval_sweep.json`](eval_sweep.json):

| floor | gated | `MISSED` | unanswerable admitted |
|---|---|---|---|
| 0.30 | 2 | 0 | 3 |
| 0.50 | 3 | 0 | 2 |
| **0.60** | **4** | **0** | **1** |
| 0.65 | 5 | 1 | 1 |
| 0.80 | 12 | 7 | 0 |

0.60 sits at the knee. Above 0.65 we start refusing questions that have answers,
and a refusal is unrecoverable, whereas an unanswerable question that reaches the
model can still be caught by the second gate.

---

## What this does not handle well

Written up front, because the brief asks for it and because it is easier to
design against a known weakness than to discover it in review.

- **Vocabulary mismatch — reduced, not solved.** Dense retrieval fixed the one
  question we predicted it would ("the checkout service is dragging its feet").
  The failure mode is not gone: the synonym table in the query analyser is
  hand-maintained, so its coverage is exactly as good as our imagination, and
  the questions we are graded on were written by someone else. **This is still
  the most likely source of lost marks.**
- **Services we have not documented.** A question about a service outside the
  corpus yields `service=None` and the filter cannot help. We detect
  service-shaped names we do not cover and decline, which handles the common
  case, but it is pattern matching and will miss unusual namings.
- **Multi-hop questions.** "Did the 2026-08-10 incident follow our escalation
  policy?" needs two documents read together. We retrieve both and the model may
  well manage it, but nothing in the design is built for synthesis and nothing
  verifies it happened.
- **The filter is strict by construction.** A question that legitimately spans
  two services gets pruned too hard. Precision over recall was a deliberate
  choice, because the brief punishes confident wrong citations more than misses.
  This is the cost of that choice.
- **Two tuned constants** — the coverage floor (0.60) and the cosine floor
  (0.75) — fitted to twenty questions over a corpus we wrote ourselves. Freezing
  both in Phase 1 before any retrieval code existed reduces the overfitting but
  cannot eliminate it. Twenty questions is a small sample.
- **Three things are built but never executed against the real service**, for
  want of credentials: the SQL migrations, the FileStore/SupabaseStore
  equivalence test (skipped, and a skip is not a pass), and the Cloudinary
  upload round-trip. [WRITEUP.md](WRITEUP.md) §7 says exactly what that leaves
  unverified.
- **Free-tier fragility.** 8k tokens a minute, and `qwen/qwen3.8-27b` is a
  preview model. Both are someone else's capacity, not ours.

---

## Deployment

```bash
# 1. Run the migrations, in order, in the Supabase SQL editor:
#    supabase/migrations/0001_schema.sql
#    supabase/migrations/0002_hybrid_search.sql
#    supabase/migrations/0003_cache_and_jobs.sql

# 2. Populate it. Runs offline - never inside a request handler.
python -m ingest.pipeline --seed --profile dev

# 3. Deploy. Set APP_ENV=dev plus the keys from .env.example in the Vercel
#    dashboard, then check what actually shipped in the bundle.
vercel build && du -sh .vercel/output/functions/*.func
vercel deploy --prod
```

A Supabase free project **pauses after seven days of inactivity** and unpausing
is a manual click. [`.github/workflows/keepalive.yml`](.github/workflows/keepalive.yml)
runs weekly, asserts the store is reachable, and asks the off-topic question to
check the gate still declines it. Set the `APP_URL` repository secret to arm it.

If Supabase is not configured, the deployed app falls back to reading the
checked-in `runbooks/` — so it serves correctly with only a `GROQ_API_KEY` set.

---

## Design notes

The full write-up — retrieval choice, the measurements, and what it does not
handle well — is in **[WRITEUP.md](WRITEUP.md)**. Longer-form architecture notes
are in [`docs/`](docs/) (`mkdocs serve` to read them as a site).

Three decisions worth knowing about, because each was measured rather than
assumed, and each is recorded next to the code it explains:

- **Why embeddings are not the *primary* retrieval signal.** An embedding model
  is trained to map paraphrases to nearby points — precisely what makes it bad
  at a corpus whose documents differ by one or two tokens. It was added as a
  second signal for recall, and it moved exactly one question. See
  `agent/core/retrieve.py`.
- **Why IDF-weighted gate coverage was tried and reverted.** It gated six
  correct answers, including "how do I safely roll back checkout-api". With
  twelve documents the vocabulary is small, so ordinary words are absent too and
  absence stops being evidence. Recorded in `agent/core/retrieve.py` so it is
  not attempted again.
- **Why there is no semantic answer cache.** On a corpus of near-duplicates it
  would serve the wrong document with the filter, the grader and the grounding
  model all bypassed. Recorded in `agent/cache.py`.
