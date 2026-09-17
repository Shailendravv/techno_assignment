"""The entry point the brief specifies.

    answer_question(question) -> {answer, cited_doc_ids, confidence}

Everything the agent does is reachable through this one function. The CLI, the
evaluation harness, and `POST /ask` all call it, so all three exercise
identical code - which is what makes a harness score evidence about what the
deployed API will actually do, rather than about a separate code path that
happens to resemble it.

LangGraph lives inside this function, not around it. The graph is an
implementation detail; the signature is the contract.
"""

from __future__ import annotations

from agent.config import NO_MATCH_MESSAGE, Settings, settings as default_settings
from agent.graph import COMPILED
from agent.state import new_state


def answer_question(
    question: str,
    model_role: str = "generator",
    cfg: Settings | None = None,
    with_trace: bool = False,
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
    """
    if not question or not question.strip():
        return {
            "answer": "Ask me something about the runbooks.",
            "cited_doc_ids": [],
            "confidence": "no_match",
        }

    state = new_state(question.strip(), model_role=model_role)
    state["settings"] = cfg or default_settings

    final = COMPILED.invoke(state)

    result = {
        "answer": final.get("answer", NO_MATCH_MESSAGE),
        "cited_doc_ids": final.get("cited_doc_ids", []),
        "confidence": final.get("confidence", "no_match"),
    }
    if with_trace:
        result["trace"] = final.get("trace", [])
        result["llm_calls"] = final.get("llm_calls", 0)
    return result
