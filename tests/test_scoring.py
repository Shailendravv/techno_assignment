"""Phase 4: the scorer.

Scoring logic is easy to get subtly wrong and hard to notice when you do - a
lenient scorer flatters every arm equally, so the comparison still looks
plausible while being meaningless. These tests pin the 2x2 from the brief and,
in particular, the two cases the brief calls out by name: that an extra
citation is sometimes correct rather than noise, and that refusing to answer is
a right answer rather than a failure.
"""

from __future__ import annotations

from eval.questions import EvalQuestion
from eval.report import (
    CORRECT_CITATION,
    CORRECT_NO_MATCH,
    FALSE_CITATION,
    MISSED,
    PARTIAL,
    WRONG_CITATION,
    QuestionResult,
    build_report,
    classify,
    summarise,
)


def q(expected, acceptable=None, qid=1) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question="...",
        expected_doc_ids=expected,
        acceptable_extra_ids=acceptable or [],
    )


# --------------------------------------------------------------------------
# The row where a document applies.
# --------------------------------------------------------------------------

def test_exactly_the_expected_document_is_correct():
    assert classify(q(["RB-001"]), ["RB-001"]) == CORRECT_CITATION


def test_a_sanctioned_extra_citation_is_still_correct():
    """The brief says citing RB-012 on question 2 is 'good, not required'. An
    extra the brief permits must not be scored as noise, or we would punish an
    answer the brief calls good."""
    question = q(["RB-002"], acceptable=["RB-012"])

    assert classify(question, ["RB-002"]) == CORRECT_CITATION
    assert classify(question, ["RB-002", "RB-012"]) == CORRECT_CITATION


def test_an_unsanctioned_extra_is_partial():
    """Right document found, noise included. A different failure from citing
    the wrong document alone, and worth separating."""
    assert classify(q(["RB-001"]), ["RB-001", "RB-003"]) == PARTIAL


def test_citing_only_the_wrong_document_is_wrong():
    """The near-duplicate trap, failed. This is the most dangerous output the
    system can produce: confident, well-written, and wrong."""
    assert classify(q(["RB-001"]), ["RB-003"]) == WRONG_CITATION


def test_refusing_an_answerable_question_is_missed():
    assert classify(q(["RB-001"]), []) == MISSED


def test_an_incomplete_multi_document_answer_is_partial():
    assert classify(q(["RB-001", "RB-002"]), ["RB-001"]) == PARTIAL


# --------------------------------------------------------------------------
# The row where nothing applies.
# --------------------------------------------------------------------------

def test_declining_an_unanswerable_question_is_correct():
    """no_match is a legitimate expected return value, not an error."""
    assert classify(q([]), []) == CORRECT_NO_MATCH


def test_citing_anything_for_an_unanswerable_question_is_a_false_citation():
    """Forcing a citation to the closest-sounding document is the specific
    failure the brief warns against."""
    assert classify(q([]), ["RB-011"]) == FALSE_CITATION


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

def _results(*outcomes) -> list[QuestionResult]:
    return [
        QuestionResult(
            id=i, question="...", expected=[], got=[], outcome=o, confidence="low"
        )
        for i, o in enumerate(outcomes, 1)
    ]


def test_a_perfect_run_scores_one():
    summary = summarise(_results(CORRECT_CITATION, CORRECT_CITATION, CORRECT_NO_MATCH))

    assert summary["overall_score"] == 1.0
    assert summary["citation_precision"] == 1.0
    assert summary["no_match_recall"] == 1.0


def test_partial_earns_half_credit():
    summary = summarise(_results(PARTIAL, PARTIAL))
    assert summary["overall_score"] == 0.5


def test_the_two_failure_modes_are_visible_separately():
    """The whole reason for reporting four outcomes. An agent that never
    declines and one that declines too readily must be distinguishable, because
    they need opposite fixes."""
    never_declines = summarise(_results(CORRECT_CITATION, FALSE_CITATION))
    declines_too_readily = summarise(_results(MISSED, CORRECT_NO_MATCH))

    assert never_declines["no_match_recall"] == 0.0
    assert never_declines["citation_recall"] == 1.0

    assert declines_too_readily["no_match_recall"] == 1.0
    assert declines_too_readily["citation_recall"] == 0.0

    # Same overall score, opposite problems - which is exactly why one number
    # is not enough.
    assert never_declines["overall_score"] == declines_too_readily["overall_score"]


def test_an_agent_that_always_declines_is_caught():
    """It is never wrong, just useless. Precision on refusals collapses even
    though it makes no false citations at all."""
    summary = summarise(_results(MISSED, MISSED, MISSED, CORRECT_NO_MATCH))

    assert summary["citation_recall"] == 0.0
    assert summary["no_match_precision"] == 0.25
    assert summary["overall_score"] == 0.25


def test_the_report_breaks_results_down_by_question_type():
    """A single number hides which *sort* of question an arm is bad at."""
    results = [
        QuestionResult(id=1, question="x", expected=["RB-001"], got=["RB-001"],
                       outcome=CORRECT_CITATION, confidence="high",
                       tags=["near-duplicate-trap"]),
        QuestionResult(id=2, question="y", expected=["RB-001"], got=[],
                       outcome=MISSED, confidence="no_match",
                       tags=["vocabulary-mismatch"]),
    ]
    report = build_report("lexical", results)

    assert report["by_tag"]["near-duplicate-trap"]["rate"] == 1.0
    assert report["by_tag"]["vocabulary-mismatch"]["rate"] == 0.0
    assert report["by_tag"]["vocabulary-mismatch"]["ids"] == [2]


def test_the_report_is_json_serialisable():
    """It is the deliverable, so it has to survive json.dump."""
    import json

    report = build_report("lexical", _results(CORRECT_CITATION, CORRECT_NO_MATCH))
    assert json.loads(json.dumps(report))["arm"] == "lexical"
