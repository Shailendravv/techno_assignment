# Known Limitations

Written up front, on the theory that it is easier to design against a known
weakness than to discover it in review.

## Vocabulary mismatch

BM25 matches words, not meanings. *"The checkout service is dragging its
feet"* shares no tokens with the CPU runbook and will be refused. The synonym
table in the query analyser patches common cases, but it is hand-maintained —
its coverage is exactly as good as the imagination that built it, and the
questions this system is graded on were written by someone else.

**This is the most likely source of lost marks.** It is also the reason the
evaluation set deliberately includes vocabulary-mismatch questions rather
than only easy ones — see [Development History → Phase 0–1](development-history.md#phase-01-foundations-corpus-and-the-frozen-evaluation-set).

## Services that aren't documented

A question about a service outside the corpus yields `service=None`, and the
metadata filter has nothing to filter on. The system detects service-shaped
names it doesn't cover and declines, which handles the common case — but
that's pattern matching, and it will miss unusual namings.

## Multi-hop questions

*"Did the 2026-08-10 incident follow our escalation policy?"* needs two
documents read together (a postmortem and a policy doc). The pipeline
retrieves both, and the model may well manage the synthesis — but nothing in
the design is built for combining evidence across documents, and nothing
verifies that it happened correctly.

## The filter is strict by construction

Dropping on contradiction is what defeats the near-duplicate trap — but a
question that legitimately spans two services gets pruned too hard.
Precision was chosen over recall deliberately, because the brief punishes a
confident wrong citation more than it punishes a miss. This is the cost of
that choice, not an oversight.

## No semantic answer cache, deliberately

Embedding the question and serving a cached answer when a previous question
lands within some cosine threshold would be actively dangerous *on this
corpus*, and dangerous in exactly the way the rest of the system is built to
prevent.

"checkout-api is running hot on CPU" and "payments-api is running hot on CPU"
differ by one token, are answered by different documents, and embed closer
together than many genuine paraphrases — the overlapping distributions in
[Hybrid Retrieval](evaluation/hybrid-retrieval.md) are the same measurement.

A retrieval mistake is caught downstream by the metadata filter, the relevance
grader and the grounding model. **A cache hit bypasses all three.** It would
reintroduce the precise failure the brief warns about, in the one place none of
the defences can see it.

If it were needed, the safe key would be the extracted `QuerySpec` — same
service, same failure mode, same intent — which is exact matching on the
fields the filter already uses, not similarity on prose.

## Two tuned constants

`coverage_floor` (0.60) and `cosine_floor` (0.75) are fitted to this corpus and
swept against these questions. Twenty questions over twelve documents is a small
sample, and these numbers should be treated as provisional. See [Gate Calibration](evaluation/gate-calibration.md)
for the sweep and the reasoning, but treat both as starting points rather
than universal constants — a different corpus size or a different question
style would need its own sweep.

## Free-tier fragility

8,000 tokens a minute on Groq's free tier, and `qwen/qwen3.8-27b` is a
preview model that can change or disappear without notice — which is exactly
why its ID lives in [`agent/config.py`](reference/configuration.md) rather
than scattered through the code. Both constraints are someone else's
capacity, not this system's.

## The honest one

This design is tuned against a corpus it wrote itself. There is a real risk
of unconsciously authoring documents the retriever happens to be good at.
The mitigation was procedural, not architectural: write all twelve documents
first, finalise them, and only then write retrieval code — and write the
held-out questions before looking at any scores. See
[Development History](development-history.md) for how that ordering shows
up in the commits.

## What has never been executed

Three things are built and unit-tested but have never run against the real
service, for want of credentials — and a skipped test is not a passing one:

1. **The SQL migrations have not been run.** They need Postgres with pgvector.
   Covered instead by tests asserting the filter clauses textually, aimed at
   deletion of the `service is null` disjunct — which would silently break
   every question a general policy document answers.
2. **The `FileStore`/`SupabaseStore` equivalence test is skipped**, not passing.
   It is the real check that the migration is behaviour-preserving.
3. **The Cloudinary upload has not round-tripped.** The signature algorithm is
   pinned against a hand-computed SHA-1 vector, because Cloudinary answers a
   wrong signature with a bare 401 and no diagnosis.

The end-to-end four-outcome scores are likewise not published, because running
the harness needs a `GROQ_API_KEY`. The retrieval-stage measurements are
reproducible offline today.
