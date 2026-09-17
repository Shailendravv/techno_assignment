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
python -m eval.harness --arm lexical --out harness_output.json

# Compare against the no-retrieval control
python -m eval.harness --compare lexical,baseline --delay 8

# The API
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I safely roll back checkout-api?"}'
```

**Most of the system needs no API key.** Retrieval, filtering, the gate, and the
entire test suite run offline:

```bash
pytest -q                       # 112 tests, no network, no key
python -m eval.harness --sweep  # gate calibration, retrieval only
```

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
[retrieve]     BM25 over 12 docs  ->  metadata filter  ->  gate
   |                                   drops contradictions
   |
   +---- gate rejects ------------------------------> [finalize]  no_match
   |                                                   zero LLM calls
   v
[ground]       "here are 4 documents. Answer only from these,
   |            or say none of them apply."
   v
[finalize]     verify citations, score confidence
   |
   v
{answer, cited_doc_ids, confidence}
```

Three signals, each doing a different job:

| Signal | Job | Why |
|---|---|---|
| **BM25** | Ranks | Scores literal tokens, so it cannot blur `checkout-api` into `payments-api` |
| **Metadata filter** | Discriminates | A wrong service is a *contradiction*, not a weak signal to be outvoted by 400 words of similar prose. It drops, it does not penalise |
| **The gate** | Decides whether to answer | If nothing credible survives, the model is never called |

The gate is a **conditional edge in the LangGraph state graph**, not an `if`
inside a function. That makes "we do not call the model when nothing survived" a
declared property of the structure rather than a branch buried in a call stack —
and it is a safety guarantee, not an optimisation. A model that is never shown a
document cannot invent a citation for one.

There are **two independent chances to decline**. The gate catches questions
whose vocabulary is nowhere in the corpus. The model catches questions that
retrieved something plausible that does not actually apply, because its prompt
states plainly that returning no citations is a correct and expected answer.

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
    retrieve.py     BM25 + metadata filter + gate      <- the important one
    confidence.py   high | medium | low | no_match
  nodes/            thin adapters: unpack state, call a core fn, write back
  llm.py            Groq client: 429 backoff, defensive JSON parsing
  config.py         every tuned constant and model ID

app/                FastAPI surface (thin by design)
baseline/           the no-retrieval control arm
eval/               questions, harness, scorer
runbooks/           RB-001.md .. RB-012.md
tests/              112 tests, all offline
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
| Lexical | `rank_bm25` | Twelve documents; an in-memory index is the right size |
| Orchestration | LangGraph | Earns its place at the conditional edges, not the happy path |

Everything runs on free tiers. Groq has **no embeddings endpoint**, which is why
Phase 5 takes embeddings from Google's `gemini-embedding-001` instead.

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

**Retrieval alone, with no model involved, puts the right document at rank 1 for
18 of the 20 questions.** The two it misses are both understood: one is a
deliberate vocabulary-mismatch question that lexical matching cannot reach, and
one is the refund question, which is left for the model's second gate.

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

- **Vocabulary mismatch.** BM25 matches words, not meanings. "The checkout
  service is dragging its feet" shares no tokens with the CPU runbook and will
  be refused. The synonym table in the query analyser patches common cases, but
  it is hand-maintained — its coverage is exactly as good as our imagination,
  and the questions we are graded on were written by someone else. **This is the
  most likely source of lost marks**, and it is what Phase 5's dense retrieval
  is for.
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
- **Two tuned constants**, fitted to our own corpus and swept against our own
  questions.
- **Free-tier fragility.** 8k tokens a minute, and `qwen/qwen3.8-27b` is a
  preview model. Both are someone else's capacity, not ours.

---

## Design notes

The reasoning behind the architecture is in
[UNDERSTANDING.md](UNDERSTANDING.md); the build sequence is in
[IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md). Two decisions worth reading
about there, because both were measured rather than assumed:

- **Why embeddings are not the primary retrieval signal.** An embedding model is
  trained to map paraphrases to nearby points — which is precisely what makes it
  bad at a corpus whose documents differ by one or two tokens.
- **Why IDF-weighted gate coverage was tried and reverted.** It gated six
  correct answers. The reasoning is recorded in `agent/core/retrieve.py` so it
  is not attempted again.
