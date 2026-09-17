"""The control experiment: no retrieval at all.

Every document goes into one prompt and the model is asked to answer and cite.
This exists so the write-up can say *why* we chose the retrieval design using a
measurement rather than an opinion - and so that, if it turns out the baseline
wins on a corpus this small, we find that out and report it honestly instead of
defending a pipeline that earns nothing.

It is a fair comparison on one axis and an unfair one on another, and both are
worth stating. Fair: same corpus, same questions, same model, same scoring.
Unfair to the baseline: twelve documents is small enough to fit in a context
window, so this arm is being given an advantage it would lose immediately at
any realistic corpus size. That asymmetry is the finding, not a flaw in the
experiment.

Token cost is the practical catch. Stuffing twelve runbooks is roughly 6,000
tokens per question against a free-tier budget of 8,000 per minute, so this arm
runs about one question a minute and consumes most of a day's allowance over a
full sweep.
"""

from __future__ import annotations

from agent.config import Settings, settings as default_settings
from agent.core.corpus import load_corpus
from agent.core.models import Candidate
from agent.llm import LLMBadJSON, chat_json
from agent.nodes.ground import SYSTEM_PROMPT, _format_documents, verify_citations


def answer_question_baseline(
    question: str,
    role: str = "generator",
    cfg: Settings | None = None,
) -> dict:
    """Answer using the whole corpus, with no retrieval and no filtering.

    Returns the same three fields as the real agent so the harness can score
    both arms with identical code. Confidence is necessarily cruder here: with
    no retrieval there are no scores, no metadata agreement, and no gate, so
    there is simply less evidence to grade with. That is itself part of what
    the comparison shows.
    """
    cfg = cfg or default_settings
    docs = load_corpus(cfg.corpus_dir)
    candidates = [Candidate(doc=d) for d in docs]

    user = (
        f"Question: {question}\n\n"
        f"You have been given all {len(candidates)} documents in the corpus. "
        f"You may cite only these IDs, and only the ones you actually used.\n\n"
        f"{_format_documents(candidates)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    try:
        parsed, _ = chat_json(messages, role=role, cfg=cfg, max_tokens=1200)
    except LLMBadJSON:
        return {
            "answer": "The model did not return a usable answer.",
            "cited_doc_ids": [],
            "confidence": "no_match",
        }

    raw_cited = parsed.get("cited_doc_ids") or []
    if isinstance(raw_cited, str):
        raw_cited = [raw_cited]
    verified, _ = verify_citations(list(raw_cited), candidates)

    return {
        "answer": str(parsed.get("answer", "")).strip(),
        "cited_doc_ids": verified,
        # No retrieval means no relevance signal to grade with. We can only
        # say whether the model cited anything, so we do not pretend to more.
        "confidence": "medium" if verified else "no_match",
    }
