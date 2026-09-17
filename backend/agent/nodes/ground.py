"""Ask the model to answer from the supplied documents, or to decline.

The model's job here is emphatically *not* to search. Retrieval has already
happened; it is handed a short list and asked two things: does any of this
actually answer the question, and if so, what is the answer using only what is
written here.

That framing is what makes this the second gate. The first gate is lexical and
catches questions whose vocabulary is nowhere in the corpus. This one catches
the question that retrieved something plausible-looking that does not actually
apply - which is the failure the brief warns about most directly, because a
confident wrong citation looks right.
"""

from __future__ import annotations

from agent.config import Settings, settings as default_settings
from agent.core.models import Candidate
from agent.llm import LLMBadJSON, LLMUnavailable, chat_json

SYSTEM_PROMPT = """\
You answer questions about a company's operational runbooks, for an engineer \
who is probably in the middle of an incident.

You will be given a question and a short list of documents. Follow these rules \
exactly.

1. Answer ONLY from the documents provided. Do not use anything you know about \
how systems generally work. If a document says to check the deploy log first, \
say that; do not add steps it does not mention.

2. Cite every document you actually used, by its exact ID. Do not cite a \
document you did not draw on, and never invent an ID.

3. If none of these documents actually answers the question, say so and return \
an empty citation list. This is a CORRECT and EXPECTED response, not a \
failure. The documents you have been given were selected because they looked \
similar to the question, and looking similar is not the same as applying. A \
runbook for a different service, or for a different failure, does not answer \
the question - do not stretch it to fit.

4. Be specific and brief. An engineer reading this is busy.

Reply with a JSON object and nothing else:

{"answer": "<your answer, or why nothing here applies>", "cited_doc_ids": ["RB-001"]}
"""


def _format_documents(candidates: list[Candidate]) -> str:
    blocks = []
    for candidate in candidates:
        doc = candidate.doc
        facts = [f"service: {doc.service or 'applies to all services'}"]
        if doc.failure_mode:
            facts.append(f"failure mode: {doc.failure_mode}")
        facts.append(f"type: {doc.doc_type}")
        if doc.date:
            facts.append(f"date: {doc.date}")

        blocks.append(
            f"--- BEGIN {doc.doc_id} ---\n"
            f"Title: {doc.title}\n"
            f"({'; '.join(facts)})\n\n"
            f"{doc.text}\n"
            f"--- END {doc.doc_id} ---"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, candidates: list[Candidate]) -> list[dict]:
    ids = ", ".join(c.doc_id for c in candidates)
    user = (
        f"Question: {question}\n\n"
        f"You have been given {len(candidates)} document(s): {ids}\n"
        f"You may cite only these IDs, and only the ones you actually used.\n\n"
        f"{_format_documents(candidates)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def verify_citations(
    cited: list[str], candidates: list[Candidate]
) -> tuple[list[str], list[str]]:
    """Keep only IDs that were in the pack we actually sent.

    Models occasionally produce a plausible-looking ID they were never shown -
    "RB-007" is an easy string to invent when you have just read RB-001 through
    RB-006. A citation to a document that was never read is worse than no
    citation, because it looks authoritative and cannot be falsified without
    checking by hand.

    Returns (verified, invented) so the invented ones can be recorded rather
    than silently discarded.
    """
    supplied = {c.doc_id for c in candidates}
    verified, invented = [], []
    for doc_id in cited:
        normalised = str(doc_id).strip().upper()
        target = verified if normalised in supplied else invented
        if normalised not in target:
            target.append(normalised)
    return verified, invented


def ground(
    question: str,
    candidates: list[Candidate],
    role: str = "generator",
    cfg: Settings | None = None,
) -> tuple[str, list[str], list[str], int]:
    """Run the grounding call.

    Returns (answer, verified_ids, invented_ids, llm_calls). Degrades rather
    than raising: a model that is unavailable or will not produce JSON should
    cost us an answer, not a crash in a request handler.
    """
    cfg = cfg or default_settings

    try:
        parsed, calls = chat_json(
            build_messages(question, candidates), role=role, cfg=cfg
        )
    except LLMUnavailable:
        raise
    except LLMBadJSON:
        return (
            "The model did not return a usable answer for this question.",
            [],
            [],
            2,
        )

    answer = str(parsed.get("answer", "")).strip()
    # Groq's smaller free-tier models sometimes over-escape newlines inside the
    # JSON string (emitting `\\n` instead of `\n`), which json.loads then hands
    # back as a literal two-character "\n" rather than a line break.
    answer = answer.replace("\\n", "\n")
    raw_cited = parsed.get("cited_doc_ids") or []
    if isinstance(raw_cited, str):
        raw_cited = [raw_cited]

    verified, invented = verify_citations(list(raw_cited), candidates)
    return answer, verified, invented, calls
