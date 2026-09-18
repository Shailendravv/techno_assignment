"""The entry point the brief specifies.

    answer_question(question) -> {answer, cited_doc_ids, confidence}

Everything the agent does is reachable through this one function. The CLI, the
evaluation harness, and `POST /ask` all call it, so all three exercise
identical code - which is what makes a harness score evidence about what the
deployed API will actually do, rather than about a separate code path that
happens to resemble it.

LangGraph lives inside this function, not around it. The graph is an
implementation detail; the signature is the contract.

The cache and the trace exporter wrap the graph here rather than inside it, for
the same reason: they are concerns of the entry point, and a node that knew
about either would be a node the harness could not run in isolation.
"""

from __future__ import annotations

import time

from agent.config import NO_MATCH_MESSAGE, Settings, settings as default_settings
from agent.graph import COMPILED
from agent.state import new_state


def answer_question(
    question: str,
    model_role: str = "generator",
    cfg: Settings | None = None,
    with_trace: bool = False,
    with_metrics: bool = False,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Answer a question from the runbooks, or decline to.

    Returns exactly the three fields the brief requires:

        answer         a natural-language answer, derived only from the cited
                       documents - or an explanation of why nothing applies
        cited_doc_ids  the documents the answer is actually grounded in. May be
                       empty, and an empty list is a real result, not an error
        confidence     "high" | "medium" | "low" | "no_match"

    `with_trace=True` adds a `trace` key listing what each stage decided. It is
    off by default because it is not part of the contract, and on in the CLI
    and the harness because that is where you need to know *why* an answer came
    out the way it did.

    `with_metrics=True` adds `llm_calls`. It is a separate switch from
    `with_trace` because the two differ in both cost and audience: the trace is
    a verbose per-stage list for a human debugging one decision, while the
    counter is a single integer worth reporting on every request. Keeping them
    apart means a caller that reports the counter has to ask for it, so the
    number it prints was measured rather than defaulted.

    `session_id` and `user_id` are for Langfuse and nothing else - they change
    no behaviour here. A session groups the traces of one conversation or one
    harness run, which is the difference between reading twenty separate
    questions and reading the run that asked them.
    """
    cfg = cfg or default_settings

    if not question or not question.strip():
        result = {
            "answer": "Ask me something about the runbooks.",
            "cited_doc_ids": [],
            "confidence": "no_match",
        }
        if with_metrics:
            result["llm_calls"] = 0
        return result

    question = question.strip()
    started = time.perf_counter()

    from agent import observability
    from agent.cache import get_cache

    cache = get_cache(cfg)

    # The Langfuse trace is scoped here, around the cache and the graph both,
    # for the same reason they are: this is the entry point, and one question
    # is one trace. Scoping it inside the graph would lose the cache hits - the
    # cheapest and most misleading runs to have no record of.
    with observability.trace_run(
        question,
        cfg,
        session_id=session_id,
        user_id=user_id,
        model_role=model_role,
    ) as root:
        hit = cache.get(question)
        if hit is not None:
            # A cache hit still gets a ledger, all of it skipped for one stated
            # reason. Otherwise the fastest runs are the ones that log nothing,
            # and "no stage lines" would mean both "cached" and "logging is
            # broken".
            from agent.stages import QUERY, new_recorder

            cached_ledger = new_recorder(QUERY, cfg=cfg)
            cached_ledger.skip_remaining("answer cache: exact hit, pipeline not run")
            cached_ledger.flush()

            result = {
                "answer": hit["answer"],
                "cited_doc_ids": hit["cited_doc_ids"],
                "confidence": hit["confidence"],
            }
            observability.finish(
                root,
                result,
                llm_calls=0,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                cached=True,
            )
            if with_trace:
                result["trace"] = ["cache: exact hit, pipeline not run"]
            if with_metrics:
                result["llm_calls"] = 0
            return result

        state = new_state(question, model_role=model_role)
        state["settings"] = cfg

        # The stage ledger is scoped here, around the graph, for the same
        # reason the cache and the trace are: it is a concern of the entry
        # point. A node that constructed its own recorder would emit a separate
        # ledger per rewrite of the corrective loop, and the loop is one run.
        from agent.stages import QUERY, new_recorder, using_recorder

        recorder = new_recorder(QUERY, cfg=cfg)
        with using_recorder(recorder):
            final = COMPILED.invoke(state)

        result = {
            "answer": final.get("answer", NO_MATCH_MESSAGE),
            "cited_doc_ids": final.get("cited_doc_ids", []),
            "confidence": final.get("confidence", "no_match"),
        }

        # Cache the refusal too. A `no_match` is a considered result, and
        # re-deriving it costs exactly what deriving it did.
        cache.put(question, result)

        observability.finish(
            root,
            result,
            llm_calls=final.get("llm_calls", 0),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

        if with_trace:
            result["trace"] = final.get("trace", [])
        if with_metrics:
            result["llm_calls"] = final.get("llm_calls", 0)
        return result
