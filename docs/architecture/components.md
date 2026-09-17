# Pipeline Components

Each component takes one type and returns another (see
[Data Contracts](data-contracts.md)), so each is testable in isolation.

## The graph — `agent/graph.py`

```python
analyze -> retrieve -> +-- (gate fails) ------------> finalize
                       |                              no_match, LLM never called
                       +-- (candidates survive) ----> ground -> finalize
```

Four nodes, compiled once at import (`COMPILED = build_graph()`), because
compilation is not free and the process reuses warm instances between
requests:

| Node | Does |
|---|---|
| `analyze_node` | Loads the corpus, runs the query analyzer, records what it found in the trace. |
| `retrieve_node` | BM25 shortlist → metadata filter → gate. Records which documents were dropped and *why*. |
| `route_after_retrieval` | The gate as a routing decision — returns `"ground"` or `"finalize"`. This conditional edge is why the graph exists at all. |
| `ground_node` | Sends the survivors to the LLM, discards any invented citation IDs. |
| `finalize_node` | The single place the output shape is decided, reached from both branches. |

Nodes are thin adapters: unpack state, call a pure function from `agent.core`,
write the result back. The logic lives in `agent.core`, where it is tested
without a graph, a network, or a key.

## The entry point — `agent/api.py`

```python
def answer_question(
    question: str,
    model_role: str = "generator",
    cfg: Settings | None = None,
    with_trace: bool = False,
) -> dict:
    ...
```

Everything — CLI, HTTP API, evaluation harness — calls this one function, so
a harness score is evidence about what the deployed API will actually do,
rather than about a separate code path that happens to resemble it.

An empty or blank question short-circuits before the graph even runs, and
returns `no_match` immediately.

## Corpus Loader — `agent/core/corpus.py`

Reads `runbooks/*.md` once at startup. Each file carries YAML front-matter —
`doc_id`, `service`, `failure_mode`, `doc_type`, `date` — which becomes
structured fields on a `Doc`. Without this, metadata lives only in prose and
the filter has nothing structured to compare, which puts the whole design
back to hoping a similarity score notices the service name.

## Query Analyzer — `agent/core/query.py`

Extracts the same structured fields *from the question* that the loader
extracted from the documents, so the two can be compared like with like:

- **service** — matched against the known service list, including loose
  forms (`"checkout api"`, `"the checkout service"`).
- **failure_mode** — matched via a hand-maintained synonym table (`cpu` ←
  *hot, pegged, throttling, high load*; `connections` ← *too many
  connections, pool exhausted, connection refused*).
- **intent** — `diagnose` / `rollback` / `policy` / `postmortem`.
- **date** — a `YYYY-MM-DD` regex, needed to tell a postmortem question from a
  runbook question about the same service and failure mode.

Rule-based rather than LLM-based, deliberately: deterministic, instant, free,
and debuggable. The interface is the seam where an LLM extraction call could
be substituted later without touching anything downstream.

## Lexical Search & Metadata Filter — `agent/core/retrieve.py`

This is the component that wins or loses the exercise.

**BM25** ranks all twelve documents and returns the top ~8 as `Candidate`
objects with scores. Rare tokens (`checkout`, `payments`, `cpu`) dominate the
score, which is exactly the property that keeps `checkout-api` and
`payments-api` from blurring together.

**The metadata filter** then compares `QuerySpec` fields against each
candidate's `Doc` fields:

| Situation | Action |
|---|---|
| Question names service X, doc is service Y | **Drop** |
| Question names failure mode A, doc is mode B | **Drop** |
| Question names service X, doc has no service (a general policy doc) | **Keep** |
| Question has a date, doc is a postmortem with a different date | **Drop** |
| Question names nothing specific | **Keep all**, rely on the score floor |

Two rules are easy to get wrong and are pinned by tests:

!!! warning "Never drop general docs"
    Documents with `service=None` (the policy docs) are never dropped for
    "mismatching" a named service, and documents with `failure_mode=None`
    (the rollback runbooks) are never dropped for mismatching a failure mode.
    Skip either rule and a real question breaks.

A tie-break on specificity also matters: general docs were outranking
service-specific ones on pure BM25 score, because a company-wide policy
document can share more words with a question than the one document that is
actually specific to it. The filter preserves BM25's ordering *within* each
specificity group rather than letting a general doc win outright.

## Relevance Gate — `agent/core/retrieve.py`

If the candidate list is empty, or the best score is under a floor, the
pipeline returns `no_match` immediately — no LLM call. Two floors, both in
[`agent/config.py`](../reference/configuration.md):

- **`lexical_floor`** — best BM25 score per content term, normalised for
  question length.
- **`coverage_floor`** — the fraction of the question's content words that
  appear *anywhere* in the corpus. This catches the off-topic question that
  BM25 still ranks confidently: "refund" appears in no document, so however
  good the nearest match looks, the corpus is not about that.

An IDF-weighted version of coverage was tried and reverted — see
[Known Limitations](../limitations.md) and the docstring in
`agent/core/retrieve.py` for why, so it doesn't get re-attempted.

## LLM Grounding — `agent/nodes/ground.py` + `agent/llm.py`

Sends the surviving documents plus the question to the generator model, with
a prompt that:

1. supplies each document with its ID clearly labelled,
2. requires the answer to use only the supplied text,
3. requires a JSON reply with `answer` and `cited_doc_ids`,
4. **states plainly that `no_match` is a correct answer when none of the
   documents apply.**

`agent/llm.py` handles the free-tier realities: retry with backoff on HTTP
429, and defensive JSON extraction that survives code fences, surrounding
prose, `<think>` blocks, and braces inside strings.

## Citation Verifier — inside `ground_node`

Discards any cited ID that wasn't in the pack actually sent to the model.
`RB-007` is an easy string to invent after reading RB-001 through RB-005, and
a citation to a document that was never shown to the model is worse than no
citation — it looks authoritative and is unfalsifiable without checking by
hand.

## Confidence Scorer — `agent/core/confidence.py`

Grades an answer that has already passed the gate and been verified, based on
*how much of the question could be pinned to structured fields* — not on how
fluent the prose reads:

| Label | When |
|---|---|
| `high` | Two or more of {service, failure mode, date} matched, and a single clear citation. |
| `medium` | Matched on one dimension; or answered by a general doc for a policy-shaped question. |
| `low` | Survived the floor but barely; matched nothing structural. |
| `no_match` | Gate short-circuited, the model declined, or verification emptied the citation list. |

A policy question (nothing structural named) answered by a policy document
scores `medium`, not `low` — that's the expected shape of a *good* answer to
a question like "what's our incident communication policy?", not a weak one.

## Baseline — `baseline/stuff_all.py`

The control arm. Puts all twelve documents in one prompt and asks the model
to answer and cite, with no retrieval at all. It exists so the write-up can
cite a measurement instead of an opinion — see
[Evaluation](../evaluation/scoring.md#the-baseline-comparison).
