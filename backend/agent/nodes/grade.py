"""Corrective retrieval: grade the shortlist, and rewrite the query if it is junk.

This is the third gate, and it is the one that needs a cycle.

  gate 1  lexical    - is this question's vocabulary in the corpus at all
  gate 2  this node  - does each retrieved document actually apply
  gate 3  grounding  - shown the survivors, does the model still decline

The pattern is CRAG (corrective retrieval-augmented generation). A small, cheap
model looks at each candidate and answers one narrow yes/no question: *does this
document actually answer this question?* Not "is it similar" - retrieval has
already established similarity, and similarity is exactly what fails on a corpus
of near-duplicates. If every candidate is graded irrelevant, we rewrite the
question in the corpus's own vocabulary and retrieve once more.

**Why this is the part that justifies LangGraph.** Everything else in this agent
is a straight line and would be perfectly happy as four function calls. This is
not: `retrieve -> grade -> rewrite -> retrieve -> grade` is a loop with a
bounded counter and two exits, and expressing it as a conditional edge with a
retry count in state is genuinely clearer than the nested control flow it would
otherwise require. It is also where the loop's bound becomes visible: the cap
lives in `AgentState` and is checked by an edge, so "this cannot run away" is a
property you can read off the graph rather than trust a function to honour.

It doubles as our free substitute for the cross-encoder reranker a production
system would use here (`bge-reranker-v2-m3` and friends). A 20B model asked one
binary question per document is not as good as a purpose-built reranker, but it
costs nothing and it is aimed at the same failure.

**It is off by default locally.** Grading costs one extra LLM call per question
against a tier that allows roughly two a minute, and the harness runs twenty
questions. `GRADER_ENABLED` is false in `config/local.json` and true in
`config/dev.json`, where a single request has budget to spare.
"""

from __future__ import annotations

import re

from agent.config import Settings, current_settings
from agent.core.models import Candidate
from agent.core.retrieve import content_terms
from agent.llm import LLMBadJSON, LLMUnavailable, chat_json

GRADER_PROMPT = """\
You are a relevance filter for an operational runbook search system. You are \
not answering the question - you are deciding which documents are worth \
reading.

For each document, decide one thing: would an engineer with this question find \
their answer in this document?

Be strict, and be strict in one specific direction. These documents were \
retrieved because they look similar to the question, and this corpus contains \
deliberate near-duplicates: the same failure on a different service, the same \
service with a different failure. A document about the right kind of problem on \
the wrong service does NOT answer the question. Neither does a document about \
the wrong kind of problem on the right service. Similar is not the same as \
applicable.

A general policy document that genuinely covers the question is relevant even \
though it names no service.

Reply with a JSON object and nothing else:

{"relevant": ["RB-001"], "reason": "one short sentence"}

List only the IDs that pass. An empty list is a valid and useful answer.
"""

REWRITE_PROMPT = """\
You rewrite search queries for a corpus of operational runbooks.

The user's question retrieved nothing relevant, which usually means they \
described a symptom in their own words rather than in the words the runbooks \
use. Rewrite it using the vocabulary an on-call engineer would write in a \
runbook: name the service if one is implied, and name the failure mode in \
technical terms.

For example, "the boxes are working too hard" means high CPU utilisation; \
"dragging its feet" means elevated latency.

Keep it short - a search query, not a sentence. Change only the vocabulary, \
never the meaning, and never invent a service that was not implied.

Reply with a JSON object and nothing else:

{"query": "<the rewritten query>"}
"""


def _sections(text: str) -> list[str]:
    """Split a document at its markdown section headings, keeping each heading.

    Element 0 is whatever precedes the first `##` - the title and the opening
    paragraph - which is why it is always kept below.
    """
    parts = (part.strip() for part in re.split(r"\n(?=#{2,3} )", text))
    return [part for part in parts if part]


def _relevance(section: str, terms: list[str]) -> int:
    """How many distinct question terms this section mentions.

    Distinct rather than total, so a section that repeats one word does not
    outrank one that covers several.
    """
    lowered = section.lower()
    return sum(1 for term in set(terms) if term in lowered)


def _excerpt(doc_text: str, terms: list[str], head_chars: int, extra_chars: int) -> str:
    """The opening, plus the section that best answers *this* question.

    This function exists because of a wrong citation, and the shape of that bug
    is worth keeping written down. The grader used to be shown `doc.text[:900]`.
    That is a sound economy for a runbook, whose subject is its first paragraph,
    and wrong for a policy document that covers six topics in sequence.

    Q13 asks what expand-and-contract is. RB-010 defines it - "always in two
    separate releases" - 1258 characters in, under `## Database migrations`, so
    the grader was handed a copy of RB-010 that never mentions the term. RB-005,
    a rollback runbook, happens to mention it at character 800, inside the
    window. The grader kept RB-005 and dropped RB-010, and it was right to, on
    the evidence it was given. The truncation was what was wrong.

    So: keep the opening, and if the question is answered somewhere further
    down, show that part too. Still bounded - a grader that reads whole
    documents costs more than the grounding call it is supposed to protect.
    """
    head = doc_text[:head_chars]
    if len(doc_text) <= head_chars or not terms:
        return head if len(doc_text) <= head_chars else head + "\n[...truncated]"

    # Only sections that begin past the head are candidates; anything inside it
    # has already been shown.
    tail = doc_text[head_chars:]
    best = max(_sections(tail), key=lambda s: _relevance(s, terms), default="")

    if not best or _relevance(best, terms) == 0:
        return head + "\n[...truncated]"

    excerpt = best[:extra_chars]
    if len(best) > extra_chars:
        excerpt += "\n[...truncated]"
    return f"{head}\n[...]\n{excerpt}"


def _format_for_grading(
    candidates: list[Candidate],
    question: str = "",
    head_chars: int = 600,
    extra_chars: int = 700,
) -> str:
    """Show the grader enough to judge, and no more.

    Bounded deliberately: sending four full runbooks to a grader would cost more
    tokens than the grounding call it is supposed to protect. What is *shown*
    within that budget is chosen by the question rather than by byte offset -
    see `_excerpt`.
    """
    terms = content_terms(question) if question else []
    blocks = []
    for candidate in candidates:
        doc = candidate.doc
        blocks.append(
            f"--- {doc.doc_id} ---\n"
            f"Title: {doc.title}\n"
            f"service: {doc.service or 'applies to all services'}; "
            f"failure mode: {doc.failure_mode or 'not specific to one'}; "
            f"type: {doc.doc_type}\n\n"
            f"{_excerpt(doc.text, terms, head_chars, extra_chars)}"
        )
    return "\n\n".join(blocks)


def grade_candidates(
    question: str,
    candidates: list[Candidate],
    cfg: Settings | None = None,
) -> tuple[list[Candidate], str, int, bool]:
    """Keep only the candidates the grader judges relevant.

    Returns (kept, reason, llm_calls, degraded). The last element matters
    because failing open is invisible in the result: the candidates pass
    through unchanged and the answer looks ordinary. The answer cache needs to
    know, since it has no TTL and would otherwise store an ungraded answer as
    though the grader had approved it.

    Degrades towards *keeping* things. If the grader is unavailable or returns
    nonsense, we pass the candidates through untouched rather than dropping
    them, because the grader is one of three gates and the two either side of
    it still run. Failing open here costs a possible wrong citation that the
    grounding step can still refuse; failing closed would silently turn every
    answerable question into `no_match` the moment the grader had a bad day.
    """
    cfg = cfg or current_settings()

    if not candidates:
        return [], "nothing to grade", 0, False

    messages = [
        {"role": "system", "content": GRADER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"{_format_for_grading(candidates, question)}"
            ),
        },
    ]

    try:
        parsed, calls = chat_json(messages, role="grader", cfg=cfg, max_tokens=300)
    except (LLMUnavailable, LLMBadJSON):
        return candidates, "grader unavailable - candidates passed through", 0, True

    relevant = parsed.get("relevant") or []
    if isinstance(relevant, str):
        relevant = [relevant]
    # Only IDs we actually supplied: the grader is as capable of inventing one
    # as any other model, and an invented ID here would resurrect a document
    # the metadata filter had already dropped.
    supplied = {c.doc_id for c in candidates}
    keep = {str(r).strip().upper() for r in relevant} & supplied

    reason = str(parsed.get("reason", "")).strip()[:160]
    kept = [c for c in candidates if c.doc_id in keep]
    for candidate in candidates:
        if candidate.doc_id not in keep:
            candidate.drop(f"grader: not relevant ({reason or 'no reason given'})")

    return kept, reason or f"{len(kept)}/{len(candidates)} judged relevant", calls, False


def rewrite_query(
    question: str,
    cfg: Settings | None = None,
) -> tuple[str, int]:
    """Restate the question in the corpus's vocabulary.

    Returns (rewritten, llm_calls). On any failure the original question comes
    back unchanged, so a broken rewrite costs one retrieval attempt rather than
    the answer.
    """
    cfg = cfg or current_settings()

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        parsed, calls = chat_json(messages, role="grader", cfg=cfg, max_tokens=120)
    except (LLMUnavailable, LLMBadJSON):
        return question, 0

    rewritten = str(parsed.get("query", "")).strip()
    # A rewrite that came back empty, or vastly longer than the question, is a
    # model that has misunderstood the task rather than a better query.
    if not rewritten or len(rewritten) > 4 * len(question) + 80:
        return question, calls
    return rewritten, calls
