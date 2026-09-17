"""Scoring the retrieval stage on its own, with no model and no API key.

This exists because the question "does hybrid retrieval beat lexical" is a
question about *retrieval*, and answering it through the full pipeline means
paying for twenty grounding calls per arm against a tier that allows about two
a minute - and then reading a difference partly caused by model variance.

So this measures the thing directly: given a question, does the document that
answers it end up in the pack the model would have been shown, and does an
unanswerable question get stopped before a model is involved at all.

**These are not the brief's four outcomes and must not be read as them.** The
brief scores what the agent finally cites, which depends on the grounding model
declining when it should. This scores the stage before that. A question counted
`ADMITTED` here is not a false citation - it is a question the retrieval gate
passed to the second gate, which is where it may well be refused. The two
reports answer different questions and the JSON labels which is which.

What this *is* good for: comparing two retrieval configurations on identical
inputs, deterministically, for free, as many times as you like.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from agent.config import Settings
from agent.core.query import analyze_query
from agent.core.retrieve import (
    best_cosine,
    coverage,
    metadata_filter,
    normalised_top_score,
    passes_gate,
    passes_hybrid_gate,
)
from eval.questions import EvalQuestion

# Answerable questions
RETRIEVED = "RETRIEVED"          # the expected document is in the pack
NOT_RETRIEVED = "NOT_RETRIEVED"  # the gate passed, but the pack is missing it
GATED = "GATED"                  # the gate refused a question that had an answer

# Unanswerable questions
STOPPED = "STOPPED"              # the gate refused it - no model call at all
ADMITTED = "ADMITTED"            # passed to the grounding model's judgement

RETRIEVAL_OUTCOMES = (RETRIEVED, NOT_RETRIEVED, GATED, STOPPED, ADMITTED)


@dataclass
class RetrievalResult:
    id: int
    question: str
    expected: list[str]
    pack: list[str]
    outcome: str
    rank: int  # 1-based position of the expected document, 0 if absent
    gate_passed: bool
    gate_reason: str
    lexical_score: float
    cosine_score: float
    coverage: float
    tags: list[str] = field(default_factory=list)


def run_retrieval(
    questions: list[EvalQuestion], cfg: Settings
) -> tuple[list[RetrievalResult], bool]:
    """Score one arm. Returns (results, dense_was_actually_used).

    Goes through the configured store rather than reading the corpus directly,
    so `--store supabase` actually exercises Postgres. That is what makes the
    Phase 6 equivalence check a real comparison rather than two runs of the
    same code path.

    The second return value matters: an arm configured as hybrid but running
    without an embedder is silently lexical, and reporting it as hybrid would
    make the comparison meaningless.
    """
    from agent.store import get_store

    store = get_store(cfg)
    docs = store.documents()
    lexical_index = store.lexical_index()
    retrieval = cfg.retrieval
    results: list[RetrievalResult] = []
    dense_used = False

    for question in questions:
        spec = analyze_query(question.question, docs)
        ranked, _ = store.retrieve(spec, question.question, cfg)

        # Which gate applies depends on whether the dense arm actually
        # contributed, not on what the profile asked for.
        dense_ran = any(c.dense_score > 0.0 for c in ranked)
        dense_used = dense_used or dense_ran
        gate = passes_hybrid_gate if dense_ran else passes_gate

        pack = metadata_filter(spec, ranked, retrieval.final_top_k)
        passed, why = gate(spec, pack, lexical_index, retrieval)

        pack_ids = [c.doc_id for c in pack]
        rank = next(
            (pack_ids.index(e) + 1 for e in question.expected_doc_ids if e in pack_ids),
            0,
        )

        if question.is_no_match:
            outcome = STOPPED if not passed else ADMITTED
        elif not passed:
            outcome = GATED
        else:
            outcome = RETRIEVED if rank else NOT_RETRIEVED

        results.append(
            RetrievalResult(
                id=question.id,
                question=question.question,
                expected=list(question.expected_doc_ids),
                pack=pack_ids if passed else [],
                outcome=outcome,
                rank=rank,
                gate_passed=passed,
                gate_reason=why,
                lexical_score=round(normalised_top_score(pack, question.question), 3),
                cosine_score=round(best_cosine(pack), 3),
                coverage=round(coverage(question.question, lexical_index), 3),
                tags=list(question.tags),
            )
        )

    return results, dense_used


def summarise_retrieval(results: list[RetrievalResult]) -> dict:
    counts = {outcome: 0 for outcome in RETRIEVAL_OUTCOMES}
    for result in results:
        counts[result.outcome] += 1

    answerable = counts[RETRIEVED] + counts[NOT_RETRIEVED] + counts[GATED]
    unanswerable = counts[STOPPED] + counts[ADMITTED]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 3) if denominator else 0.0

    found = [r for r in results if r.rank]
    return {
        "outcomes": counts,
        # Of the questions a document answers, how often it reached the model.
        "retrieval_recall": ratio(counts[RETRIEVED], answerable),
        # Of those, how often it was ranked first rather than merely present.
        "top1_rate": ratio(sum(1 for r in found if r.rank == 1), answerable),
        # Of the unanswerable questions, how many never reached a model at all.
        "gate_stop_rate": ratio(counts[STOPPED], unanswerable),
        "mean_rank_when_found": (
            round(sum(r.rank for r in found) / len(found), 2) if found else 0.0
        ),
        "counts": {
            "total": len(results),
            "answerable": answerable,
            "unanswerable": unanswerable,
        },
    }


def build_retrieval_report(
    arm: str, results: list[RetrievalResult], dense_used: bool
) -> dict:
    return {
        "stage": "retrieval",
        "arm": arm,
        "dense_arm_active": dense_used,
        **summarise_retrieval(results),
        "per_question": [asdict(r) for r in results],
        "note": (
            "Retrieval-stage only. These are NOT the brief's four outcomes: no "
            "model was called, so nothing here is a citation. ADMITTED means "
            "the retrieval gate passed a question to the grounding model, which "
            "is the second gate and may still refuse it."
        ),
    }


def format_retrieval_human(report: dict) -> str:
    lines = [
        f"arm: {report['arm']}  (retrieval stage only, no model calls)",
        f"  dense arm active: {report['dense_arm_active']}",
        "",
        f"  {'outcome':16} count",
        f"  {'-' * 22}",
    ]
    good = {RETRIEVED, STOPPED}
    for outcome in RETRIEVAL_OUTCOMES:
        mark = "ok " if outcome in good else "   "
        lines.append(f"  {mark}{outcome:13} {report['outcomes'][outcome]:3d}")

    lines += [
        "",
        f"  retrieval recall    {report['retrieval_recall']:.0%}"
        "   (answerable questions whose document reached the pack)",
        f"  ranked first        {report['top1_rate']:.0%}"
        "   (and was the top candidate)",
        f"  gate stop rate      {report['gate_stop_rate']:.0%}"
        "   (unanswerable questions stopped before any model call)",
        f"  mean rank when found  {report['mean_rank_when_found']}",
    ]
    return "\n".join(lines)


def compare_arms(reports: dict[str, dict]) -> dict:
    """Per-question, which arm did better - the table the write-up needs."""
    arms = list(reports)
    by_question: dict[int, dict] = {}

    for arm in arms:
        for result in reports[arm]["per_question"]:
            entry = by_question.setdefault(
                result["id"],
                {"id": result["id"], "question": result["question"],
                 "expected": result["expected"], "arms": {}},
            )
            entry["arms"][arm] = {
                "outcome": result["outcome"],
                "rank": result["rank"],
                "pack": result["pack"],
            }

    good = {RETRIEVED, STOPPED}
    differences = []
    for entry in by_question.values():
        outcomes = {a: entry["arms"][a]["outcome"] for a in arms if a in entry["arms"]}
        ranks = {a: entry["arms"][a]["rank"] for a in arms if a in entry["arms"]}
        if len(set(outcomes.values())) > 1 or len(set(ranks.values())) > 1:
            differences.append(
                {**entry, "verdict": {a: outcomes[a] in good for a in outcomes}}
            )

    return {
        "arms": arms,
        "questions_where_arms_differ": differences,
        "summary": {
            arm: {
                "retrieval_recall": reports[arm]["retrieval_recall"],
                "top1_rate": reports[arm]["top1_rate"],
                "gate_stop_rate": reports[arm]["gate_stop_rate"],
            }
            for arm in arms
        },
    }


def write(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
