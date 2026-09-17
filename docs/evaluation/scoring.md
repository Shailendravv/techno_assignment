# The Scoring Model

## Why not just report accuracy

A single accuracy number lets a lazy agent hide. Two agents can score 70% in
completely opposite ways:

- **Agent A** never says `no_match`. It scores perfectly on answerable
  questions and fails every unanswerable one — the exact failure the brief
  warns against, and the most dangerous output the system can produce,
  because a confident wrong citation looks right.
- **Agent B** says `no_match` too readily. It is never wrong, just frequently
  useless.

These need opposite fixes — lower the gate floor versus raise it — and one
number cannot tell you which problem an agent has. Four outcomes can.
`tests/test_scoring.py::test_the_two_failure_modes_are_visible_separately`
pins exactly this: it asserts that a never-decline agent and an
always-decline agent score identically on a naive accuracy metric but
differently once broken into outcomes — so a future simplification back to
one number would fail loudly.

## The four outcomes, plus one refinement

|  | agent cited something | agent said `no_match` |
|---|---|---|
| **a doc applies** | `CORRECT_CITATION` / `PARTIAL` / `WRONG_CITATION` | `MISSED` |
| **nothing applies** | `FALSE_CITATION` | `CORRECT_NO_MATCH` |

`PARTIAL` is the fifth outcome, added because "cited RB-001 and RB-003 when
only RB-001 was expected" is a different failure from "cited RB-003 alone" —
and because the brief's own note that citing RB-012 alongside RB-002 on
question 2 is *"good, not required"* means an extra citation is sometimes
correct rather than noise. Each question can declare `acceptable_extra_ids`
so that sanctioned extra citations score as `CORRECT_CITATION`, not `PARTIAL`.

## Metrics reported

From `eval/report.py::summarise`:

| Metric | Definition |
|---|---|
| `citation_precision` | Of the times the agent cited something, how often was it exactly right. |
| `citation_recall` | Of the questions a document does answer, how many did the agent get. |
| `no_match_precision` | Of the times the agent declined, how often was it right to. |
| `no_match_recall` | Of the questions nothing answers, how many did the agent decline. |
| `overall_score` | `PARTIAL` counts as half credit: `(2·(CORRECT_CITATION + CORRECT_NO_MATCH) + PARTIAL) / (2·total)`. |

`by_tag` breaks the score down by question type (near-duplicate traps vs.
vocabulary-mismatch questions vs. clean no-match questions), because a single
number also hides *which kind* of question an arm is bad at.

## The evaluation set

Twenty questions: the five given in the brief, plus fifteen held out — ten
answerable and five expected `no_match`. **Both the corpus and the questions
were written and committed before any retrieval code existed** (see
[Development History](../development-history.md)). The corpus is
self-authored, so that ordering is the only real defence against tuning a
retriever to documents that were tuned to suit it.

Two of the held-out questions are deliberate vocabulary mismatches, written
specifically to fail under lexical-only retrieval — they exist to justify
adding a dense-retrieval arm later, and writing them up front stops that
decision being made only after seeing scores.

## What retrieval alone achieves

**Retrieval alone, with no model involved, puts the right document at rank 1
for 18 of the 20 questions.** The two misses are both understood:

1. A deliberate vocabulary-mismatch question, which lexical matching cannot
   reach by construction.
2. The refund-policy question, which is left for the model's second gate
   rather than the retrieval-stage gate — see
   [Gate Calibration](gate-calibration.md).

## The baseline comparison { #the-baseline-comparison }

`baseline/stuff_all.py` stuffs all twelve documents into one prompt with no
retrieval, and is deliberately given an advantage: twelve documents fit
comfortably in a context window, which they would not at any realistic
corpus size. Running

```bash
python -m eval.harness --compare lexical,baseline --delay 8
```

prints both arms side by side — citation precision/recall, no-match
precision/recall, overall score, and total LLM calls — which becomes the
comparison table for the write-up. If the no-retrieval baseline ever won on
this corpus, that would be a genuine finding worth reporting rather than
hiding.

## Running it

```bash
python -m eval.harness --arm lexical --out harness_output.json
python -m eval.harness --given-only          # just the 5 questions from the brief
python -m eval.harness --sweep               # gate calibration, retrieval only, no key needed
```

The harness calls `answer_question()` directly — no HTTP server involved —
so the thing being scored is exactly the thing that gets deployed.
