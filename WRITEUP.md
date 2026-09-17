# What this agent does, why it retrieves the way it does, and where it breaks

A grounded question-answering agent over twelve operational runbooks, exposed as
one function:

```python
answer_question(question) -> {"answer": str, "cited_doc_ids": [str], "confidence": str}
```

`confidence` is `high | medium | low | no_match`. An empty `cited_doc_ids` is a
real result, not an error.

---

## 1. The problem, as I understood it

The exercise is not really about retrieval quality. It is about **refusal**.

A corpus of twelve runbooks contains deliberate near-duplicates: the same
failure on a different service, the same service with a different failure. So
for any question, there is always a document that *looks* like the answer. An
agent optimising for helpfulness will find it, cite it confidently, and be
wrong — and a confident wrong citation is worse than no answer, because an
engineer mid-incident cannot cheaply tell the difference.

Two of the four outcomes the brief asks the harness to distinguish are about
exactly this. So the system is built around one question asked three times, at
three different stages, with three different mechanisms: **is this document
actually the answer, or does it merely resemble it?**

---

## 2. The retrieval design, and why

```
question
   ↓
analyse            rule-based: extract service, failure mode, intent, date
   ↓
rank               BM25  +  dense (bge-small / Gemini)  →  RRF fusion
   ↓
FILTER             hard-drop any document whose metadata contradicts the question
   ↓
GATE ──────────────────────────────────→ no_match, zero model calls
   ↓
grade              a small model judges each candidate's relevance (CRAG)
   ↓ ← rewrite ────┘ (bounded: one retry)
ground             answer only from the supplied documents, or decline
   ↓
verify             discard any cited ID we did not actually supply
```

### The metadata filter is the load-bearing component

The runbooks are near-duplicates as *prose*. RB-001 (`checkout-api` / CPU) and
RB-003 (`payments-api` / CPU) share roughly 90% of their words. No amount of
similarity scoring reliably separates them, because they genuinely are similar —
the 1–2 tokens that decide the answer are exactly what a similarity score
averages away.

But they are not similar as *data*. One says `service: checkout-api` and the
other says `service: payments-api`. So the design's central move is to stop
treating the distinguishing feature as text and start treating it as a field:

| Question says | Document says | Action |
|---|---|---|
| service X | service Y | **drop** |
| service X | no service (a policy doc) | keep |
| failure mode A | failure mode B | **drop** |
| failure mode A | no failure mode | keep |
| a date | postmortem, other date | **drop** |
| nothing specific | anything | keep |

This is a **hard drop, not a score penalty**. A wrong service is not weak
evidence to be out-voted by four hundred words of similar prose; it is a
contradiction.

Rows two and four are the ones that are easy to get wrong, and they are why
question 5 works. A general policy document has `service: null`, and dropping it
for "mismatching" a named service would break every question a policy answers.
The same clause exists in SQL (`service is null or service = :service`) and has
a test guarding it from being "simplified" away.

### Why a rule-based query analyser

Deterministic, instant, free, and every decision can be explained by pointing at
a line. It is also the component I am least confident in — see §6.

### Why dense retrieval was added, and what it cost

BM25 cannot match words that are not there. The evaluation set contains two
questions written *before any retrieval code existed* specifically to fail this
way — question 14 is "the checkout service is dragging its feet and the boxes
are working too hard", a question about CPU with no CPU vocabulary in it.

Embeddings are trained on precisely that mapping, so dense retrieval was added
for **recall** — on the explicit understanding that it is bad at the thing BM25
is good at. It is trained to map near-duplicates close together, which is
exactly wrong for this corpus. Hence the division of labour: **dense retrieval
finds candidates; the metadata filter discriminates.** Fusion happens first, the
filter runs on the fused list, the gate runs last.

Fusion is RRF over ranks rather than a weighted score blend, because a BM25
score and a cosine similarity are not on the same scale and any weighting I
picked would be a constant fitted to my own twenty questions.

---

## 3. Measurements

### Lexical vs hybrid retrieval

Twenty questions — the five from the brief plus fifteen held out, frozen before
the retriever was written. Retrieval stage only: no model calls, deterministic,
reproducible with `python -m eval.harness --retrieval-only --compare lexical,hybrid`.

| | lexical | hybrid |
|---|---|---|
| retrieval recall (answerable questions whose document reached the model) | **93%** | **100%** |
| ranked first | 93% | 93% |
| unanswerable questions stopped before any model call | 80% | 80% |

The arms differ on **exactly one question** — Q14, the vocabulary-mismatch
question, which moves from `NOT_RETRIEVED` to `RETRIEVED`. Nothing regresses and
`no_match` behaviour is unchanged. So hybrid is the default, and it degrades to
lexical automatically when no embedder is available.

That is a small win, honestly reported: the dense arm earns its place on one
question in twenty, and that question was written in advance to need it.

### Calibrating the gate

Sweeping the corpus-coverage floor (`python -m eval.harness --sweep`):

| floor | gated | MISSED (real questions refused) | unanswerable admitted |
|---|---|---|---|
| 0.30 | 2 | 0 | 3 |
| 0.35 | 3 | 0 | 2 |
| 0.55 | 4 | **0** | 1 |
| **0.60** | **4** | **0** | **1** |
| 0.65 | 5 | 1 | 1 |
| 0.80 | 12 | 7 | 0 |

0.60 is the knee. Above it, real questions start being refused; below it, more
unanswerable questions reach the model. The asymmetry is deliberate: a question
admitted here still faces two more gates, whereas a question refused here is
refused permanently. **Admitting one is recoverable; refusing one is not.**

### A planned idea that the measurement killed

The design called for an absolute cosine floor calibrated against known-negative
questions. Measured, the two distributions **overlap**:

- answerable questions: best cosine **0.62 – 0.89**
- unanswerable questions: best cosine **0.58 – 0.73**

No threshold separates them. This is the concrete form of *dense retrieval makes
`no_match` harder, not easier*: a vector search always returns its nearest
neighbours, so there is no such thing as "nothing matched".

So the cosine floor is not a discriminator. It is set to 0.75 — above the highest
negative observed — as a guard rail that can admit a question the lexical floor
rejected only on dense evidence stronger than any unanswerable question produced.
On these twenty questions **it never fires**, and a test pins the overlap so the
finding cannot be quietly forgotten.

The corpus-coverage check binds regardless of dense score, and that is what stops
hybrid trading `no_match` recall for citation recall.

### What is NOT measured, and why

**The end-to-end four-outcome scores are not in this document, because I have not
run them.** The harness that produces them is built, tested (`tests/test_scoring.py`)
and ready — but scoring requires a `GROQ_API_KEY`, and no key was available while
this was written. Running it is one command:

```bash
python -m eval.harness --compare lexical,hybrid,baseline --delay 30 --out harness_output.json
```

I would rather report the gap than publish numbers I did not measure. Everything
in §3 above is reproducible offline, today, with no credentials.

Also unmeasured for the same reason: **arm C**, the no-retrieval control that
stuffs all twelve documents into one prompt. Its purpose is to establish whether
the retrieval pipeline earns anything at this corpus size — and I would expect
the honest answer to be "less than it looks", since twelve documents fit in a
context window comfortably. That asymmetry is the finding, not a flaw: the
baseline is being handed an advantage it loses immediately at any realistic size.

---

## 4. Three gates, because one is not enough

| | stage | catches | cost |
|---|---|---|---|
| 1 | retrieval gate | questions whose vocabulary is absent from the corpus | free |
| 2 | relevance grader | retrieved documents that look right but do not apply | one cheap model call |
| 3 | grounding prompt | everything else — the model is told refusing is correct | free (same call) |

Gate 1 is a conditional edge in the LangGraph state graph, not an `if` inside a
function. That is deliberate: when it rejects, the grounding node is **never
entered**, so no request is made and no document is ever shown to a model. "We
do not hallucinate citations when nothing survived" becomes a structural
property rather than a promise. A test asserts `llm_calls == 0` on that path.

Gate 1 catches question 19 ("how many vacation days do engineers get") on 50%
coverage. It does **not** catch question 16 ("what's our refund policy for orders
over $500"), which scores 75% because *policy*, *orders* and *500* all appear —
the last only because 500 is an HTTP status code in three runbooks. That question
is gate 2 and 3's job. Two independent chances to decline, each catching what it
is actually good at, is a better design than one threshold contorted to catch
everything.

Finally, `verify_citations()` discards any ID the model returned that was not in
the pack we sent. `RB-007` is an easy string to invent after reading RB-001
through RB-006, and a citation to a document that was never read is worse than no
citation — it looks authoritative and cannot be falsified without checking by hand.

---

## 5. Why LangGraph, honestly

For most of this pipeline, LangGraph is ceremony. `analyse → retrieve → ground →
finalize` is a straight line and would be perfectly happy as four function calls.

It earns its place in two spots:

1. **The gate as a conditional edge**, described above — the bypass is part of the
   graph's declared structure rather than a branch buried in a call stack.
2. **The corrective-retrieval loop.** `retrieve → grade → rewrite → retrieve` is a
   cycle with a bounded counter and two exits, which is genuinely awkward in a
   linear pipeline. The bound lives in a routing edge and in `AgentState`, so
   "this cannot run away" is something you read rather than trust — which matters
   on a tier allowing roughly two questions a minute. A test asserts the loop
   stops after exactly one rewrite.

If those two things were removed, I would remove LangGraph too.

---

## 6. What it does not handle well

Written in advance where possible, because designing against a known weakness is
easier than discovering it in review.

**Vocabulary mismatch — reduced, not solved.** Hybrid retrieval fixed Q14. The
failure mode is not gone: the synonym table in `agent/core/query.py` is
hand-written, so its coverage is exactly as good as my imagination, and the
questions this will be graded on were written by someone else. *This is the most
likely source of lost marks.* The natural upgrade is a small LLM extraction call
behind the same signature — nothing downstream would change.

**Services outside the corpus degrade in two different ways, and one is wrong.**
A question naming a service-shaped name we have no documents for (`search-api`)
is correctly short-circuited to `no_match`. But a question about a *real* service
we happen not to document, phrased without that shape, yields `service: None` —
the filter cannot help, and we fall back to lexical similarity alone.

**Multi-hop questions are not answered, they are approximated.** We retrieve
several documents and hand them over. Nothing synthesises across them and nothing
verifies that the model did.

**The filter is strict by construction.** A question legitimately spanning two
services gets pruned too hard. I chose precision over recall deliberately; this is
the bill for that choice, and on a different corpus it would be the wrong call.

**Two constants are fitted to a corpus I wrote myself.** The coverage floor
(0.60) and the cosine floor (0.75) come from twenty questions over twelve
documents. The mitigation — freezing corpus and questions in Phase 1 before any
retrieval code existed — reduces the overfitting but cannot eliminate it. Twenty
questions is a small sample and these numbers should be treated as provisional.

**The self-authored corpus is the deepest caveat.** I wrote the documents *and*
the questions. There is an unavoidable risk of having unconsciously written
documents my retriever happens to be good at.

**`ts_rank_cd` and `rank_bm25` are not the same scorer.** The gate's lexical
floor is calibrated against BM25. The equivalence test between the two backends
exists to catch drift, and it has **not been run** — see §7.

**Free-tier fragility.** `qwen/qwen3.8-27b` is a preview model that can vanish;
Groq's 8k TPM allows about two questions a minute; a Supabase free project pauses
after seven idle days. None of that is capacity I control. Mitigations: model IDs
in config, backoff and retry, an answer cache, and a weekly keep-alive cron.

---

## 7. Production concerns: implemented, or declined in writing

| Concern | Status |
|---|---|
| Hybrid retrieval (dense + lexical) | Implemented, measured |
| Metadata / ACL pre-filtering | Implemented, in a single SQL statement |
| Reranking | Substituted: an LLM relevance grader in place of a cross-encoder |
| Relevance grading + corrective retry (CRAG) | Implemented, bounded at one retry |
| Exact-answer cache | Implemented (Postgres, keyed on normalised question + arm) |
| **Semantic cache** | **Declined — see below** |
| Tracing / observability | Implemented (Langfuse over HTTP, off unless configured) |
| Offline/online ingestion split | Implemented; ingestion never runs in a request |
| Job state | A Postgres table, replacing Redis Streams + Celery |
| ACL / multi-tenant isolation | Declined: single-tenant demo. RLS is on with no policy, so a leaked anon key reads nothing |
| OCR | Declined: no scanned documents in this corpus |
| Table / multimodal extraction | Declined: no tables in this corpus |
| Kubernetes, HA/DR | Declined: serverless, zero budget |

**Why there is no semantic cache.** Embedding the question and serving a cached
answer when a previous question lands within some cosine threshold is actively
dangerous *on this corpus*, and dangerous in exactly the way the rest of the
system is built to prevent. "checkout-api is running hot on CPU" and "payments-api
is running hot on CPU" differ by one token, are answered by different documents,
and embed closer together than many genuine paraphrases — the overlapping
distributions in §3 are the same measurement. A retrieval mistake is caught
downstream by the filter, the grader and the grounding model; **a cache hit
bypasses all three.** It would reintroduce the precise failure the brief warns
about in the one place where none of my defences can see it. If it were needed,
the safe key would be the extracted `QuerySpec` — same service, same failure
mode, same intent — which is exact matching on the fields the filter already
uses, not similarity on prose.

### What has not been verified, stated plainly

Three things are built and unit-tested but have **never executed against the real
service**, because no credentials were available:

1. **The SQL migrations have not been run.** They need Postgres with pgvector.
   Covered instead by tests asserting the filter clauses textually — aimed
   squarely at deletion of the `service is null` disjunct, which would silently
   break every policy question.
2. **The `FileStore` / `SupabaseStore` equivalence test is skipped**, not passing.
   It is the real check that the migration is behaviour-preserving, and a skip is
   not a pass.
3. **Cloudinary upload and ingestion have not round-tripped.** The signature
   algorithm is pinned against a hand-computed SHA-1 vector, because Cloudinary
   answers a wrong signature with a bare 401 and no hint which of parameter set,
   sort order or empty-value handling was wrong.

260 tests pass offline with no credentials and no network.

---

## 8. Running it

```bash
pip install -r requirements-dev.txt

# The tests that matter most: filter and gate, no network, no model, no keys.
pytest -q

# The measurement in §3. Free, deterministic, no key needed.
python -m eval.harness --retrieval-only --compare lexical,hybrid
python -m eval.harness --sweep

# The agent. The second question must decline.
python -m agent --trace "checkout-api is running hot on CPU - what should I check first?"
python -m agent --trace "How many vacation days do engineers get?"

# The API and the UI.
uvicorn app.main:app --reload
```

`--trace` prints what each stage decided. For the questions the agent declines,
that trace is the most interesting output the system produces.

Configuration is layered: environment variable → `config/<APP_ENV>.json` →
code default. `config/local.json` runs offline against files with a local ONNX
embedder; `config/dev.json` runs against Supabase and Gemini. Profiles are
committed and contain no secrets; credentials come from the environment only.
