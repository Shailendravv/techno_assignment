# The Corpus

Twelve self-authored documents under `runbooks/`. Every document carries YAML
front-matter (`doc_id`, `service`, `failure_mode`, `doc_type`, `date`) that
the [corpus loader](../architecture/components.md#corpus-loader-agentcorecorpuspy)
turns into structured `Doc` fields — this is what lets the metadata filter
compare a question against a document as data instead of prose.

| ID | Title | service | failure_mode | type | Role |
|---|---|---|---|---|---|
| **RB-001** | checkout-api — High CPU | checkout-api | cpu | runbook | Answers Q1 |
| **RB-002** | checkout-api — Too many connections | checkout-api | connections | runbook | Answers Q2 |
| RB-003 | payments-api — High CPU | payments-api | cpu | runbook | near-dup of RB-001 (service) |
| RB-004 | checkout-api — Memory / OOM | checkout-api | memory | runbook | near-dup of RB-001 (failure mode) |
| **RB-005** | checkout-api — Rollback procedure | checkout-api | — | runbook | Answers Q3 |
| RB-006 | payments-api — Rollback procedure | payments-api | — | runbook | near-dup of RB-005 (service) |
| RB-007 | inventory-api — Too many connections | inventory-api | connections | runbook | near-dup of RB-002 (service) |
| RB-008 | inventory-api — Stock sync lag | inventory-api | sync_lag | runbook | unique |
| RB-009 | On-call escalation policy | — | — | policy | general |
| RB-010 | Deploy & release process | — | — | policy | general |
| **RB-011** | Customer incident communication policy | — | — | policy | Answers Q5 |
| **RB-012** | Postmortem: checkout-api, 2026-08-10 | checkout-api | connections | postmortem | Answers Q4 |

## The traps, at a glance

```
same failure, different service          same service, different failure
────────────────────────────────         ───────────────────────────────
RB-001 <-> RB-003   (cpu)                 RB-001 <-> RB-004   (checkout-api)
RB-002 <-> RB-007   (connections)
RB-005 <-> RB-006   (rollback)

same service + same failure, different document type
──────────────────────────────────────────────────────
RB-002 <-> RB-012   (checkout-api / connections; runbook vs postmortem)
```

RB-002 vs. RB-012 is the sharpest test in the corpus: same service, same
failure mode, overlapping words — separated only by `doc_type` and `date`.
It exercises whether the design's `intent` and `date` fields actually do
their job, because BM25 and the service/failure-mode filter alone cannot tell
these two apart.

- **Q2** ("what's the likely cause?") wants the **runbook** → RB-002, with
  RB-012 as optional supporting evidence.
- **Q4** ("what happened on 2026-08-10?") wants the **incident record** →
  RB-012, *not* RB-002.

## Why RB-012 is written the way it is

The brief says Q2 may optionally also cite RB-012. That only makes sense if
the 2026-08-10 incident *was* a connection-exhaustion incident — so RB-012's
root cause is written as a connection pool exhausted by a bad deploy, the
same failure RB-002 is the runbook for.

## Measured overlap

From the [Phase 0–1 commit](../development-history.md#phase-01-foundations-corpus-and-the-frozen-evaluation-set):

| Pair | Token overlap | Distinguished by |
|---|---|---|
| RB-001 / RB-003 | 69% | `service` |
| RB-002 / RB-007 | 73% | `service` |
| RB-005 / RB-006 | 77% | `service` |
| RB-001 / RB-004 | 41% | `failure_mode` |

`tests/test_corpus_integrity.py` asserts that every heavily-overlapping pair
is separated by at least one metadata field — that assertion is the claim
the whole design rests on.
