# Does hybrid retrieval actually help?

Dense retrieval was added to fix one predicted failure. This page is that
prediction being checked rather than asserted.

## How this is measured

At the **retrieval stage only** — no model calls at all:

```bash
python -m eval.harness --retrieval-only --compare lexical,hybrid
```

Free, deterministic, and reproducible with no API key. Results are committed in
`retrieval_comparison.json`.

Measuring here rather than end to end is deliberate. "Does hybrid beat lexical"
is a question about retrieval, and running it through the full pipeline would
cost forty grounding calls against a tier allowing about two a minute — and then
report a difference partly caused by model variance.

!!! warning "These are not the brief's four outcomes"
    No model is called, so nothing here is a citation. A question counted
    `ADMITTED` is not a false citation — it is a question the retrieval gate
    passed to the *second* gate, which may well refuse it. See
    [The Scoring Model](scoring.md) for the outcomes that are scored.

## The result

| | lexical | hybrid |
|---|---|---|
| retrieval recall | **93%** | **100%** |
| ranked first | 93% | 93% |
| unanswerable questions stopped before any model call | 80% | 80% |

The arms differ on **exactly one question**:

| | pack the model would see | outcome |
|---|---|---|
| lexical | `RB-002, RB-012, RB-004, RB-009` | `NOT_RETRIEVED` |
| hybrid | `RB-002, RB-012, RB-004, RB-001` | `RETRIEVED` (rank 4) |

Question 14 is *"The checkout service is dragging its feet and the boxes are
working too hard"* — a question about CPU containing no CPU vocabulary. It was
written in Phase 1, before any retrieval code existed, specifically to fail
under lexical-only retrieval.

Nothing regresses, and `no_match` behaviour is unchanged. So hybrid is the
default, degrading to lexical automatically when no embedder is available.

**That is a small win, honestly reported.** The dense arm earns its place on one
question in twenty, and that question was written in advance to need it.

## The division of labour

Dense retrieval is added for **recall**, on the explicit understanding that it is
bad at what BM25 is good at. An embedding model is *trained* to map
near-duplicates close together — which is precisely wrong for a corpus whose
whole difficulty is telling near-duplicates apart.

```
dense retrieval finds candidates  ->  the metadata filter discriminates
```

Fusion first, then the filter as a hard drop, then the gate. The filter acts on
document metadata, which no amount of embedding similarity can blur: RB-003 says
`service: payments-api` whatever its prose resembles.

Fusion is RRF over **ranks**, not a weighted blend of scores. A BM25 score and a
cosine similarity are not on the same scale, are not comparable across queries,
and have no principled conversion between them — any weighting would be a
constant fitted to these twenty questions.

## A planned idea the measurement killed

The design called for an absolute cosine floor, calibrated against the
known-negative questions. Measured, the two distributions **overlap**:

| | best cosine |
|---|---|
| answerable questions | 0.62 – 0.89 |
| unanswerable questions | 0.58 – 0.73 |

No threshold separates them. Q14 — a question that *does* have an answer —
scores 0.62, below Q18, which does not, at 0.73.

This is the concrete form of **dense retrieval makes `no_match` harder, not
easier**. A vector search always returns its k nearest neighbours; there is no
such thing as "nothing matched", and unrelated text still lands at a
respectable-looking similarity.

So the cosine floor is not a discriminator. It is set to **0.75** — above the
highest negative observed — as a guard rail: it can admit a question the lexical
floor rejected, but only on dense evidence stronger than anything an unanswerable
question produced. On these twenty questions **it never fires**, because nothing
that clears the coverage floor falls below the lexical one.

`test_dense_scores_do_not_separate_answerable_from_unanswerable` pins the overlap,
so if it ever stops being true, that is noticed rather than assumed.

## Why the coverage check binds regardless

The hybrid gate lets a question clear on *either* signal — but the
corpus-coverage check applies either way. A question whose vocabulary is largely
absent from the corpus is rejected however confident the vector space looks about
it, because coverage measures something the embedder cannot see: whether this
corpus is about this subject at all, rather than how similar two strings are.

Without that, hybrid would trade `no_match` recall for citation recall, and the
brief is explicit that a confident wrong citation is the worse mistake.
