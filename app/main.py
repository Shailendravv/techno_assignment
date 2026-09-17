"""The HTTP surface. Deliberately thin.

Every route here delegates to `agent.answer_question()` - the same function the
CLI and the evaluation harness call. That is the point: all three entry points
exercise identical code, so a harness score is evidence about what the API will
actually do.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from agent.api import answer_question
from agent.config import settings
from agent.llm import LLMUnavailable
from app.schemas import AskRequest, AskResponse, HealthResponse

app = FastAPI(
    title="Runbook Agent",
    description="Grounded question answering over operational runbooks.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness, and a readable summary of how this instance is configured.

    Also doubles as the endpoint the weekly keep-alive cron hits, since a
    Supabase free project pauses after seven idle days.
    """
    try:
        from agent.core.corpus import load_corpus

        corpus_docs = len(load_corpus(settings.corpus_dir))
    except Exception:  # noqa: BLE001 - health must never 500
        corpus_docs = 0

    return HealthResponse(
        status="ok",
        corpus_docs=corpus_docs,
        store=settings.store,
        retrieval_mode=settings.retrieval.mode,
        groq_configured=settings.has_groq,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the runbooks, or decline to.

    A `no_match` response is a 200, not a 404. The agent declining to answer is
    a successful outcome - arguably the most valuable one it produces - and
    signalling it as an error would invite clients to treat it as a fault and
    retry, or to hide it.
    """
    started = time.perf_counter()
    try:
        result = answer_question(request.question, model_role=request.model_role)
    except LLMUnavailable as exc:
        # Configuration, not a bug: no key, so the grounding step cannot run.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        cited_doc_ids=result["cited_doc_ids"],
        confidence=result["confidence"],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
