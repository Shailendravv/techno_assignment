"""Turn the accumulated evidence into one of four labels.

The brief requires this field, and it carries real information: it is how a
caller tells a solid answer from a guess. `no_match` shares the channel with
the confidence levels, which means "how sure are we" and "did we find anything"
are answered together - so a caller can never read a confident-looking answer
that is not actually grounded in anything.
"""

from __future__ import annotations

from agent.core.models import Candidate, Confidence, QuerySpec


def score_confidence(
    spec: QuerySpec,
    candidates: list[Candidate],
    cited_doc_ids: list[str],
) -> Confidence:
    """Grade an answer that has already passed the gate and been verified.

    The signal is *how much of the question we could pin to structured fields*,
    not how fluent the answer reads. A question that named a service and a
    failure mode, and was answered by the one document matching both, is a
    different kind of result from one that matched on nothing in particular and
    landed on a general policy doc - even when both produce good prose.
    """
    if not cited_doc_ids:
        return "no_match"

    cited = [c for c in candidates if c.doc_id in cited_doc_ids]
    if not cited:
        # The model cited something we did not supply, and verification stripped
        # it. Nothing is grounded, so nothing is claimed.
        return "no_match"

    matched_dimensions = sum(
        (
            bool(spec.service) and any(c.doc.service == spec.service for c in cited),
            bool(spec.failure_mode)
            and any(c.doc.failure_mode == spec.failure_mode for c in cited),
            bool(spec.date) and any(c.doc.date == spec.date for c in cited),
        )
    )

    single_clear_citation = len(cited_doc_ids) == 1
    answered_by_general_doc = all(c.doc.is_general for c in cited)

    if matched_dimensions >= 2 and single_clear_citation:
        return "high"

    if matched_dimensions >= 1:
        # Matched on something structural, but either the answer spans several
        # documents or only one dimension was pinned down.
        return "medium" if not answered_by_general_doc or single_clear_citation else "low"

    if answered_by_general_doc and single_clear_citation:
        # A policy question answered by a policy document. Nothing structural
        # to match on, because the question named nothing structural - this is
        # the expected shape of a good answer to question 5, not a weak one.
        return "medium" if spec.intent == "policy" else "low"

    return "low"
