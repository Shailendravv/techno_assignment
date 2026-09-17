"""Tracing, exported to Langfuse when it is configured and nowhere otherwise.

The agent already produces a trace: every node appends a line saying what it
decided, and `AgentState.trace` accumulates them. That is the substance, and it
is available locally through `python -m agent --explain` and over HTTP through
`{"explain": true}`. This module only ships it somewhere durable.

Two decisions worth stating.

**Over HTTP, not the SDK.** `langfuse` pulls in a dependency tree for what is
one authenticated POST of a JSON batch. Vercel's Python bundler does no
tree-shaking and the bundle limit is real, so this follows the same pattern as
the Supabase store and the Cloudinary client: stdlib `urllib`, forty lines.

**Fire-and-forget, and silent on failure.** An observability backend that can
fail a request is worse than no observability. Every path here swallows its
exceptions, and the only thing a broken Langfuse costs is the trace.

Off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, which means
the default configuration - and the whole test suite - makes no network calls.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from agent.config import Settings, settings as default_settings

TIMEOUT_S = 3.0
DEFAULT_HOST = "https://cloud.langfuse.com"


def _credentials() -> tuple[str, str, str] | None:
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not (public and secret):
        return None
    return public, secret, os.getenv("LANGFUSE_HOST", DEFAULT_HOST).rstrip("/")


def enabled() -> bool:
    return _credentials() is not None


def _stage(line: str) -> str:
    """The node a trace line came from: "retrieve: kept RB-001" -> "retrieve"."""
    head = line.split(":", 1)[0].strip()
    return head if head and " " not in head else "step"


def build_payload(
    question: str,
    result: dict,
    trace: list[str],
    llm_calls: int,
    elapsed_ms: int,
    cfg: Settings,
) -> dict:
    """One Langfuse trace, with each node's decision as a span.

    Built as a pure function so it can be asserted on without a network call -
    and so that the thing tested is the thing sent.
    """
    trace_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    events: list[dict] = [
        {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now,
            "body": {
                "id": trace_id,
                "name": "answer_question",
                "input": {"question": question},
                "output": result,
                "metadata": {
                    "profile": cfg.profile,
                    "retrieval_mode": cfg.retrieval.mode,
                    "store": cfg.store,
                    "embedder": cfg.embedding.backend,
                    "llm_calls": llm_calls,
                    "elapsed_ms": elapsed_ms,
                },
                # Tagged by outcome, because "show me every question we
                # declined" is the query worth having. A refusal is the result
                # this design exists to produce, so it should be the easiest
                # thing to go and look at.
                "tags": [
                    f"confidence:{result.get('confidence', 'unknown')}",
                    f"mode:{cfg.retrieval.mode}",
                    "declined" if not result.get("cited_doc_ids") else "answered",
                ],
            },
        }
    ]

    for index, line in enumerate(trace):
        events.append(
            {
                "id": str(uuid.uuid4()),
                "type": "span-create",
                "timestamp": now,
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": f"{index:02d} {_stage(line)}",
                    "output": {"decision": line},
                },
            }
        )

    return {"batch": events}


def export_trace(
    question: str,
    result: dict,
    trace: list[str],
    llm_calls: int,
    elapsed_ms: int,
    cfg: Settings | None = None,
) -> bool:
    """Send one trace. Returns whether it went. Never raises."""
    cfg = cfg or default_settings
    credentials = _credentials()
    if credentials is None:
        return False

    public, secret, host = credentials
    payload = build_payload(question, result, trace, llm_calls, elapsed_ms, cfg)
    token = base64.b64encode(f"{public}:{secret}".encode("utf-8")).decode("ascii")

    request = urllib.request.Request(
        f"{host}/api/public/ingestion",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        # Observability must never be able to fail a request. Losing a trace is
        # an acceptable cost; losing an answer to an on-call engineer is not.
        return False
