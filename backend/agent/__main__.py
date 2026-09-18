"""Command-line interface: `python -m agent "your question"`.

Exists so the working prototype can be demonstrated live without starting a
server. `--trace` prints what each stage decided, which is the fastest way to
show *why* the agent declined to answer something - the behaviour the whole
design is built around and the one that is hardest to believe without seeing
the reasoning.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.api import answer_question
from agent.config import load_settings
from agent.llm import LLMUnavailable

CONFIDENCE_MARK = {
    "high": "++",
    "medium": " +",
    "low": " ~",
    "no_match": " -",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="Ask a question about the operational runbooks.",
    )
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument(
        "--trace", action="store_true", help="show what each stage decided"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the raw result dict"
    )
    parser.add_argument(
        "--reasoner",
        action="store_true",
        help="use the reasoning model instead of the default generator",
    )
    parser.add_argument(
        "--profile",
        help="config profile to load: local (default) or dev",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question)
    cfg = load_settings(args.profile) if args.profile else None
    try:
        result = answer_question(
            question,
            model_role="reasoner" if args.reasoner else "generator",
            cfg=cfg,
            with_trace=args.trace or args.json,
            # `--trace` prints the counter and `--json` serialises it, so it
            # has to be asked for. Without this the CLI reports 0 for a run
            # that did call the model - the one wrong value that looks right.
            with_metrics=args.trace or args.json,
        )
    except LLMUnavailable as exc:
        # Missing configuration, not a failure of the agent. Say so plainly
        # rather than printing a traceback, and point at the fix.
        print(f"Cannot answer this question: {exc}", file=sys.stderr)
        print(
            "\nRetrieval reached the grounding step, which needs a model.\n"
            "Set GROQ_API_KEY in .env (see .env.example) and try again.\n"
            "Questions the agent declines need no key - try:\n"
            '  python -m agent "How many vacation days do engineers get?"',
            file=sys.stderr,
        )
        return 2
    finally:
        # A CLI run is the short-lived process the Langfuse docs warn about:
        # the SDK batches on a background thread, and `python -m agent` exits
        # long before that thread would have got its turn. In the `finally` so
        # the run that raised - the one worth looking at - is also sent.
        from agent.observability import flush

        flush()

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if args.trace:
        print("trace")
        for line in result.get("trace", []):
            print(f"  {line}")
        print(f"  llm calls: {result.get('llm_calls', 0)}")
        print()

    print(result["answer"])
    print()
    cited = ", ".join(result["cited_doc_ids"]) or "none"
    print(f"  cited:      {cited}")
    print(f"  confidence: {CONFIDENCE_MARK.get(result['confidence'], '  ')} "
          f"{result['confidence']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
