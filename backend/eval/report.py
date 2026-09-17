"""Scoring: the four outcomes the brief names, plus one refinement.

The brief asks for a runner that reports a score, "not just printed output for
a human to eyeball", and it specifies four outcomes to distinguish:

  1. cited the right doc(s)          4. incorrectly returned no_match
  2. cited the wrong doc(s)             when a doc did apply
  3. correctly returned no_match

That list is a 2x2, and it exists because a single accuracy number lets a lazy
agent hide. Two agents can both score 70% in opposite ways: one never says
no_match and fails every unanswerable question; one says it too readily and is
never wrong, just useless. Those need opposite fixes - lower the gate floor
versus raise it - and one number cannot tell you which you have.

We add a fifth outcome, PARTIAL, because "cited RB-001 and RB-003 when only
RB-001 was expected" is a different failure from "cited RB-003 alone", and the
brief's own note that citing RB-012 on question 2 is "good, not required"
means an extra citation is sometimes correct rather than noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from eval.questions import EvalQuestion

# Truth applies    | agent cited      | agent said no_match
# -----------------+------------------+--------------------
# a doc applies    | CORRECT_CITATION | MISSED
#                  | PARTIAL          |
#                  | WRONG_CITATION   |
# nothing applies  | FALSE_CITATION   | CORRECT_NO_MATCH

CORRECT_CITATION = "CORRECT_CITATION"
PARTIAL = "PARTIAL"
WRONG_CITATION = "WRONG_CITATION"
CORRECT_NO_MATCH = "CORRECT_NO_MATCH"
MISSED = "MISSED"
FALSE_CITATION = "FALSE_CITATION"

OUTCOMES = (
    CORRECT_CITATION, PARTIAL, WRONG_CITATION,
    CORRECT_NO_MATCH, MISSED, FALSE_CITATION,
)

GOOD = {CORRECT_CITATION, CORRECT_NO_MATCH}


@dataclass
class QuestionResult:
    id: int
    question: str
    expected: list[str]
    got: list[str]
    outcome: str
    confidence: str
    tags: list[str] = field(default_factory=list)
    answer: str = ""
    elapsed_ms: int = 0
    llm_calls: int = 0
    error: str = ""


def classify(question: EvalQuestion, cited: list[str]) -> str:
    """Place one result in the 2x2."""
    cited_set = set(cited)

    if question.is_no_match:
        return CORRECT_NO_MATCH if not cited_set else FALSE_CITATION

    if not cited_set:
        return MISSED

    expected = set(question.expected_doc_ids)
    allowed = expected | set(question.acceptable_extra_ids)

    if expected <= cited_set and cited_set <= allowed:
        # Everything required, nothing that was not permitted. Citing an
        # explicitly acceptable extra counts here, not as noise.
        return CORRECT_CITATION

    if cited_set & expected:
        # Found at least one right document, but brought something unsanctioned
        # with it, or missed part of a multi-document answer.
        return PARTIAL

    return WRONG_CITATION


def summarise(results: list[QuestionResult]) -> dict:
    """Compute the reported metrics.

    Each is defined against the outcome table above rather than against a
    generic notion of accuracy, so that a number moving can always be traced to
    a specific kind of mistake.
    """
    counts = {outcome: 0 for outcome in OUTCOMES}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1

    answerable = counts[CORRECT_CITATION] + counts[PARTIAL] + counts[WRONG_CITATION] + counts[MISSED]
    unanswerable = counts[CORRECT_NO_MATCH] + counts[FALSE_CITATION]
    cited_anything = counts[CORRECT_CITATION] + counts[PARTIAL] + counts[WRONG_CITATION] + counts[FALSE_CITATION]
    said_no_match = counts[CORRECT_NO_MATCH] + counts[MISSED]
    total = len(results) or 1

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 3) if denominator else 0.0

    return {
        "outcomes": counts,
        # Of the times we cited something, how often was it exactly right.
        "citation_precision": ratio(counts[CORRECT_CITATION], cited_anything),
        # Of the questions a document does answer, how often did we get it.
        "citation_recall": ratio(counts[CORRECT_CITATION], answerable),
        # Of the times we declined, how often were we right to.
        "no_match_precision": ratio(counts[CORRECT_NO_MATCH], said_no_match),
        # Of the questions nothing answers, how often did we decline.
        "no_match_recall": ratio(counts[CORRECT_NO_MATCH], unanswerable),
        # PARTIAL is half credit: the right document was found, with noise.
        "overall_score": ratio(
            2 * (counts[CORRECT_CITATION] + counts[CORRECT_NO_MATCH]) + counts[PARTIAL],
            2 * total,
        ),
        "counts": {
            "total": len(results),
            "answerable": answerable,
            "unanswerable": unanswerable,
        },
        "total_llm_calls": sum(r.llm_calls for r in results),
        "mean_elapsed_ms": round(sum(r.elapsed_ms for r in results) / total),
    }


def by_tag(results: list[QuestionResult]) -> dict:
    """Break the score down by kind of question.

    A single number hides which *sort* of question an arm is bad at. The
    near-duplicate traps and the vocabulary-mismatch questions are testing
    different weaknesses, and they should be readable separately.
    """
    grouped: dict[str, dict] = {}
    for result in results:
        for tag in result.tags:
            bucket = grouped.setdefault(tag, {"total": 0, "good": 0, "ids": []})
            bucket["total"] += 1
            bucket["good"] += result.outcome in GOOD
            if result.outcome not in GOOD:
                bucket["ids"].append(result.id)
    for bucket in grouped.values():
        bucket["rate"] = round(bucket["good"] / bucket["total"], 3)
    return grouped


def build_report(arm: str, results: list[QuestionResult]) -> dict:
    return {
        "arm": arm,
        **summarise(results),
        "by_tag": by_tag(results),
        "per_question": [asdict(r) for r in results],
    }


def format_human(report: dict) -> str:
    """A short readable summary, printed alongside the JSON.

    The JSON is the deliverable; this is so a person can see at a glance
    whether the two error types are balanced.
    """
    lines = [
        f"arm: {report['arm']}",
        "",
        f"  {'outcome':20} count",
        f"  {'-' * 26}",
    ]
    for outcome in OUTCOMES:
        mark = "ok " if outcome in GOOD else "   "
        lines.append(f"  {mark}{outcome:17} {report['outcomes'][outcome]:3d}")

    lines += [
        "",
        f"  citation precision  {report['citation_precision']:.0%}"
        f"   (of what we cited, how much was right)",
        f"  citation recall     {report['citation_recall']:.0%}"
        f"   (of answerable questions, how many we got)",
        f"  no_match precision  {report['no_match_precision']:.0%}"
        f"   (of our refusals, how many were right)",
        f"  no_match recall     {report['no_match_recall']:.0%}"
        f"   (of unanswerable questions, how many we refused)",
        "",
        f"  overall             {report['overall_score']:.0%}",
        f"  llm calls           {report['total_llm_calls']}",
    ]

    if report.get("by_tag"):
        lines += ["", "  by question type:"]
        for tag, bucket in sorted(report["by_tag"].items()):
            failed = f"  failed: {bucket['ids']}" if bucket["ids"] else ""
            lines.append(
                f"    {tag:22} {bucket['good']}/{bucket['total']}"
                f"  {bucket['rate']:.0%}{failed}"
            )

    return "\n".join(lines)
