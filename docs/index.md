---
title: Home
---

# Runbook Agent

A grounded question-answering agent over twelve operational runbooks. Ask it a
question in plain English; it finds the right document and answers **while
telling you which document the answer came from** — or tells you plainly that
nothing in the corpus applies.

That last part is the hard part, and it is the feature this system is built
around.

## The trap in the corpus

The corpus deliberately contains near-duplicate documents: a runbook for
`payments-api` high CPU shares **69%** of its words with the one for
`checkout-api` high CPU, and a `payments-api` rollback procedure shares **77%**
with the `checkout-api` one. Finding a document that *looks* relevant is easy.
Refusing to answer when nothing truly applies — and not being fooled by a
near-identical document about the wrong service — is the actual problem.

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install, add a free Groq key, and ask your first question.

    [:octicons-arrow-right-24: Quick start](getting-started.md)

- :material-graph-outline:{ .lg .middle } **Architecture**

    ---

    Why BM25 + a metadata filter beats embeddings on *this* corpus, and how the
    LangGraph state machine makes the gate a structural guarantee.

    [:octicons-arrow-right-24: Read the design](architecture/overview.md)

- :material-clipboard-check-outline:{ .lg .middle } **Evaluation**

    ---

    Four outcomes, not one accuracy number — and the sweep that calibrated the
    relevance gate.

    [:octicons-arrow-right-24: Scoring model](evaluation/scoring.md)

- :material-source-branch:{ .lg .middle } **Development History**

    ---

    What each phase (frozen corpus → retrieval → agent → harness) actually
    committed, and why the order matters.

    [:octicons-arrow-right-24: Phase by phase](development-history.md)

</div>

## The one function

Everything is reachable through a single entry point, as the brief requires:

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
`cited_doc_ids` with `confidence: "no_match"` is a real result, not an error**
— it means nothing in the corpus answers the question, and saying so is the
correct behaviour.

The CLI, the HTTP API, and the evaluation harness all call this same function,
so a harness score is evidence about what the deployed API will actually do.

!!! info "Zero-budget constraint"
    Everything here runs on free tiers: Groq for generation (`openai/gpt-oss-120b`
    as the default, `qwen/qwen3.8-27b` as a selectable reasoning arm), and an
    in-memory BM25 index instead of a paid vector database. There is no
    embeddings dependency in the current pipeline — see
    [why](architecture/overview.md#why-not-embeddings).
