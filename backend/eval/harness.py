"""The evaluation runner.

    python -m eval.harness --arm lexical --out harness_output.json
    python -m eval.harness --compare lexical,baseline
    python -m eval.harness --sweep

Takes a list of (question, expected_doc_ids or none) pairs, runs them, and
reports a machine-readable score - which is what the brief asks for, as opposed
to printed output for a human to eyeball. A short human summary is printed
alongside, because the JSON is for the record and the summary is for the person
watching it run.

It calls `answer_question()` directly rather than over HTTP, so there is no
server to start and the thing being measured is exactly the thing that gets
deployed.

A note on cost, because it constrains how this can be used. Groq's free tier
allows roughly 8,000 tokens a minute. The pipeline arm sends about 1,500 tokens
per question; the baseline arm stuffs all twelve documents and sends about
6,000. So a full pipeline run is a couple of minutes and a full baseline run is
closer to twenty. `--delay` paces requests to stay inside the limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

from agent.api import answer_question
from agent.config import Settings, load_settings
from agent.llm import LLMUnavailable
from eval.questions import ALL_QUESTIONS, GIVEN_QUESTIONS, EvalQuestion
from eval.report import QuestionResult, build_report, classify, format_human

ARMS = ("lexical", "hybrid", "baseline")


def _run_one(
    question: EvalQuestion, arm: str, cfg: Settings
) -> QuestionResult:
    started = time.perf_counter()
    error = ""

    try:
        if arm == "baseline":
            from baseline.stuff_all import answer_question_baseline

            result = answer_question_baseline(question.question, cfg=cfg)
            result.setdefault("llm_calls", 1)
        else:
            result = answer_question(question.question, cfg=cfg, with_trace=True)
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        error = f"{type(exc).__name__}: {exc}"
        result = {"answer": "", "cited_doc_ids": [], "confidence": "no_match"}

    cited = result.get("cited_doc_ids", [])
    return QuestionResult(
        id=question.id,
        question=question.question,
        expected=list(question.expected_doc_ids),
        got=list(cited),
        outcome=classify(question, cited),
        confidence=result.get("confidence", "no_match"),
        tags=list(question.tags),
        answer=result.get("answer", "")[:400],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        llm_calls=result.get("llm_calls", 0),
        error=error,
    )


def run_arm(
    arm: str,
    questions: list[EvalQuestion],
    cfg: Settings,
    delay: float = 0.0,
    quiet: bool = False,
) -> dict:
    """Run every question through one arm and score the results."""
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, 1):
        result = _run_one(question, arm, cfg)
        results.append(result)

        if not quiet:
            mark = "ok  " if result.outcome in {"CORRECT_CITATION", "CORRECT_NO_MATCH"} else "FAIL"
            print(
                f"  [{index:2d}/{len(questions)}] {mark} Q{result.id:<3} "
                f"{result.outcome:17} got={result.got or '[]'} "
                f"want={result.expected or 'no_match'}",
                flush=True,
            )

        # Pace requests to stay inside the free tier. Skipped after the last
        # question and whenever the answer cost nothing.
        if delay and index < len(questions) and result.llm_calls:
            time.sleep(delay)

    return build_report(arm, results)


def _apply_mode(cfg: Settings, arm: str) -> Settings:
    """The hybrid arm differs from the lexical one only by configuration."""
    if arm == "hybrid":
        return replace(cfg, retrieval=replace(cfg.retrieval, mode="hybrid"))
    if arm == "lexical":
        return replace(cfg, retrieval=replace(cfg.retrieval, mode="lexical"))
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.harness",
        description="Score the agent against the evaluation set.",
    )
    parser.add_argument("--arm", choices=ARMS, default="lexical")
    parser.add_argument(
        "--compare",
        help="comma-separated arms to run side by side, e.g. lexical,baseline",
    )
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument(
        "--given-only",
        action="store_true",
        help="run only the five questions from the brief",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to wait between questions, to stay inside the free tier",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="sweep the gate floors and report how the two error types trade off",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "score the retrieval stage without calling any model - free, "
            "deterministic, and needs no API key. This is how the lexical vs "
            "hybrid comparison is measured."
        ),
    )
    parser.add_argument(
        "--profile",
        help="config profile to load (default: APP_ENV, or 'local')",
    )
    parser.add_argument(
        "--store",
        choices=("files", "supabase"),
        help=(
            "override the storage backend. Running the same questions through "
            "both and diffing the outcomes is how the Phase 6 migration is "
            "shown to be behaviour-preserving."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_settings(args.profile) if args.profile else load_settings()
    if args.store:
        cfg = replace(cfg, store=args.store)
    questions = GIVEN_QUESTIONS if args.given_only else ALL_QUESTIONS

    if args.sweep:
        return _sweep(cfg, questions, args.out)

    if args.retrieval_only:
        arms = args.compare.split(",") if args.compare else [args.arm]
        return _retrieval_only(cfg, questions, arms, args.out)

    arms = args.compare.split(",") if args.compare else [args.arm]
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; choose from {ARMS}")

    reports = {}
    try:
        for arm in arms:
            print(f"\nrunning arm: {arm}  ({len(questions)} questions)", flush=True)
            reports[arm] = run_arm(
                arm, questions, _apply_mode(cfg, arm), delay=args.delay
            )
    except LLMUnavailable as exc:
        print(f"\nCannot run the harness: {exc}", file=sys.stderr)
        print(
            "Set GROQ_API_KEY in .env (see .env.example). Retrieval is testable "
            "without a key via `pytest`, but scoring needs the grounding step.",
            file=sys.stderr,
        )
        return 2

    print()
    for arm, report in reports.items():
        print(format_human(report))
        print()

    if len(reports) > 1:
        _print_comparison(reports)

    payload = reports[arms[0]] if len(reports) == 1 else {"arms": reports}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def _print_comparison(reports: dict) -> None:
    """The table that becomes the write-up."""
    arms = list(reports)
    rows = [
        ("overall", "overall_score"),
        ("citation precision", "citation_precision"),
        ("citation recall", "citation_recall"),
        ("no_match precision", "no_match_precision"),
        ("no_match recall", "no_match_recall"),
    ]

    print("comparison")
    print(f"  {'metric':22}" + "".join(f"{a:>12}" for a in arms))
    print("  " + "-" * (22 + 12 * len(arms)))
    for label, key in rows:
        print(f"  {label:22}" + "".join(f"{reports[a][key]:>11.0%} " for a in arms))
    print(f"  {'llm calls':22}" + "".join(f"{reports[a]['total_llm_calls']:>12}" for a in arms))
    print()


def _retrieval_only(
    cfg: Settings, questions: list[EvalQuestion], arms: list[str], out: str | None
) -> int:
    """Score retrieval without a model, and compare the arms side by side.

    This is the answer to "does hybrid beat lexical". It is measured here
    rather than through the full pipeline because the question is about
    retrieval, and running it end to end would cost forty grounding calls and
    then report a difference partly caused by model variance.
    """
    from eval.retrieval_eval import (
        build_retrieval_report,
        compare_arms,
        format_retrieval_human,
        run_retrieval,
    )

    reports: dict[str, dict] = {}
    for arm in arms:
        if arm == "baseline":
            print("skipping 'baseline': it has no retrieval stage to score", flush=True)
            continue

        arm_cfg = _apply_mode(cfg, arm)
        print(f"\nrunning arm: {arm}  ({len(questions)} questions, retrieval only)",
              flush=True)
        results, dense_used = run_retrieval(questions, arm_cfg)
        reports[arm] = build_retrieval_report(arm, results, dense_used)

        if arm == "hybrid" and not dense_used:
            print(
                "  WARNING: hybrid was requested but no embedder is available, "
                "so this arm ran lexical-only. Install fastembed or set "
                "GEMINI_API_KEY; the comparison is meaningless otherwise.",
                flush=True,
            )

    if not reports:
        print("nothing to score", file=sys.stderr)
        return 2

    print()
    for report in reports.values():
        print(format_retrieval_human(report))
        print()

    payload: dict
    if len(reports) > 1:
        comparison = compare_arms(reports)
        payload = {"stage": "retrieval", "arms": reports, "comparison": comparison}

        print("where the arms differ")
        if not comparison["questions_where_arms_differ"]:
            print("  nowhere - the arms are identical on this question set.")
        for entry in comparison["questions_where_arms_differ"]:
            print(f"  Q{entry['id']}  want={entry['expected'] or 'no_match'}")
            for arm, detail in entry["arms"].items():
                print(
                    f"    {arm:9} {detail['outcome']:14} "
                    f"rank={detail['rank']}  pack={detail['pack'] or '[]'}"
                )
        print()
    else:
        payload = next(iter(reports.values()))

    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {out}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


def _sweep(cfg: Settings, questions: list[EvalQuestion], out: str | None) -> int:
    """Sweep the gate's coverage floor and watch the two error types move.

    This is how the threshold gets chosen. Raising it trades FALSE_CITATION
    (citing rubbish) for MISSED (refusing a question that had an answer), and
    the right setting depends on which mistake costs more. For this exercise a
    confident wrong citation is the worse failure, so we bias towards refusing.

    The sweep runs retrieval only - no model, no key, no cost - because the
    gate is a retrieval-stage decision and this way the sweep is reproducible.
    """
    from agent.core.corpus import load_corpus
    from agent.core.query import analyze_query
    from agent.core.retrieve import bm25_search, build_index, metadata_filter, passes_gate

    docs = load_corpus(cfg.corpus_dir)
    index = build_index(docs)

    print("gate coverage floor sweep (retrieval only, no model calls)\n")
    print(f"  {'floor':>6} {'gated':>6} {'MISSED':>7} {'admitted no_match':>18}  note")
    print("  " + "-" * 62)

    rows = []
    for floor in [round(0.05 * i, 2) for i in range(0, 17)]:
        retrieval = replace(cfg.retrieval, coverage_floor=floor)
        missed = admitted = gated = 0

        for question in questions:
            spec = analyze_query(question.question, docs)
            kept = metadata_filter(
                spec, bm25_search(index, spec, retrieval.bm25_top_k), retrieval.final_top_k
            )
            passed, _ = passes_gate(spec, kept, index, retrieval)
            gated += not passed
            if not passed and not question.is_no_match:
                missed += 1  # a question that had an answer, refused
            if passed and question.is_no_match:
                admitted += 1  # an unanswerable question, sent to the model

        rows.append({"floor": floor, "gated": gated, "missed": missed,
                     "admitted_no_match": admitted})
        note = ""
        if missed == 0 and admitted <= 1:
            note = "<- good balance"
        print(f"  {floor:>6.2f} {gated:>6} {missed:>7} {admitted:>18}  {note}")

    print(
        "\n  MISSED rises as the floor rises: real questions get refused.\n"
        "  'admitted no_match' falls: fewer unanswerable questions reach the model.\n"
        "  Whatever reaches the model is still subject to the second gate, so\n"
        "  admitting one is recoverable while refusing one is not."
    )

    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"sweep": rows}, handle, indent=2)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
