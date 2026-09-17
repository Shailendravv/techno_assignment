# Gate Calibration

The relevance gate has two failure directions, and they trade off against
each other as the `coverage_floor` threshold moves:

- **Raise the floor** → fewer `FALSE_CITATION`s (citing rubbish on an
  unanswerable question), but more `MISSED` (refusing a question that
  actually had an answer).
- **Lower the floor** → the opposite.

`python -m eval.harness --sweep` runs retrieval only — no model, no key, no
cost — because the gate is a retrieval-stage decision, which makes the sweep
free and fully reproducible. Results are recorded in `eval_sweep.json` at the
repo root; the calibration table:

| floor | gated | `MISSED` | unanswerable admitted |
|---|---|---|---|
| 0.30 | 2 | 0 | 3 |
| 0.50 | 3 | 0 | 2 |
| **0.60** | **4** | **0** | **1** |
| 0.65 | 5 | 1 | 1 |
| 0.80 | 12 | 7 | 0 |

**0.60 is the knee**, and it's what `Retrieval.coverage_floor` defaults to in
[`agent/config.py`](../reference/configuration.md). Above 0.65 the gate
starts refusing questions that have real answers.

## Why the asymmetry is deliberate

A refusal is unrecoverable — once the gate rejects a question, the model
never sees it and the answer is `no_match`, full stop. An unanswerable
question that slips past the gate is still recoverable: it reaches the model,
whose prompt explicitly permits declining, so the *second* gate can still
catch it. That asymmetry is why the floor is set to let the model arbitrate
rather than to be maximally strict at the retrieval stage — a confident wrong
citation is treated as the worse failure than a question that reaches the
model and gets declined there instead.

## An idea that was tried and reverted

An IDF-weighted coverage signal was tested on the theory that a word absent
from the corpus names a subject with no matching document, so it should count
as strongly informative. Measured against the evaluation set, it gated **six
correct answers** — the phrase "how do I safely roll back checkout-api" fell
to 27% coverage, below every genuine `no_match` question in the set.

The flaw: with only twelve documents, ordinary words like *"safely"* are also
absent from the corpus. Absence from a small corpus is not strong evidence of
anything. The reasoning is recorded directly in the docstring of
`agent/core/retrieve.py` so the idea is not re-attempted without re-deriving
why it failed.

The gate therefore stays deliberately blunt: it rejects unknown-service
questions outright and clearly off-topic ones on coverage, but does not catch
subtler cases like "what is our refund policy for orders over $500" — which
scores 75% coverage because *policy*, *orders*, and *500* all occur in the
corpus somewhere (the last only as an HTTP status code). That one is left to
the model's second chance to decline.
