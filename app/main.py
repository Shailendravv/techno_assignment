"""The HTTP surface. Deliberately thin.

Every route here delegates to `agent.answer_question()` - the same function the
CLI and the evaluation harness call. That is the point: all three entry points
exercise identical code, so a harness score is evidence about what the API will
actually do.
"""

from __future__ import annotations

import time

from fastapi import FastAPI

from agent.config import settings
from app.schemas import HealthResponse

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
