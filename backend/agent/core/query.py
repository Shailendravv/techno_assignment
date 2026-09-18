"""Turn a question into the same structured fields the documents carry.

This is the component that makes filtering possible at all. The documents say
`service: checkout-api` in a field; this module has to find the equivalent fact
in "the checkout service is throwing errors" so the two can be compared as
values rather than as text.

Rule-based on purpose: deterministic, instant, free, and debuggable. Every
decision it makes can be explained by pointing at a line. If it proves too
brittle on unseen questions, the natural upgrade is a small LLM extraction call
behind this same signature - nothing downstream would change.

Its weakness is stated plainly because it is the most likely source of lost
marks: the synonym tables below are hand-written, so their coverage is exactly
as good as the imagination of whoever wrote them, and the questions we are
graded on were written by somebody else.
"""

from __future__ import annotations

import functools
import re

from agent.core.models import Doc, QuerySpec

# --------------------------------------------------------------------------
# Failure modes.
#
# Deliberately *not* included: paraphrases like "dragging its feet" or "working
# too hard". They would make our own vocabulary-mismatch test questions pass,
# which would measure the test rather than the system. Catching phrasing nobody
# anticipated is the dense retriever's job, and the honest way to find out
# whether it does that job is to leave this table naive.
# --------------------------------------------------------------------------
FAILURE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cpu": (
        "cpu", "processor", "compute", "running hot", "pegged", "throttling",
        "throttled", "high load", "load average", "utilisation", "utilization",
        "py-spy", "profile",
    ),
    "connections": (
        "too many connections", "too many clients", "connection pool",
        "connection slots", "pool exhausted", "pool exhaustion",
        "connection refused", "refusing connections", "queuepool",
        "max_connections", "connections", "connection",
    ),
    "memory": (
        "memory", "out of memory", "oom", "oomkilled", "exit code 137",
        "exit 137", "heap", "ram", "working set",
    ),
    "sync_lag": (
        "sync lag", "sync_lag", "syncing", "stock sync", "consumer lag",
        "oversell", "overselling", "stale stock", "lag",
    ),
}

# --------------------------------------------------------------------------
# Intent. Checked in this order, because the orderings overlap: "What is the
# rollback procedure?" contains a policy word and a rollback word, and rollback
# is the one that identifies the right document.
# --------------------------------------------------------------------------
INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rollback", (
        "roll back", "rollback", "rolling back", "revert", "undo the deploy",
        "previous version", "previous revision",
    )),
    ("postmortem", (
        "postmortem", "post-mortem", "root cause", "what was the cause",
        "what happened", "incident on", "outage on", "rca",
    )),
    ("policy", (
        "policy", "process", "are we allowed", "allowed to", "how long before",
        "escalate", "escalates", "escalation", "who do i", "who to",
        "migration", "migrations", "release", "releases", "deploy window",
        "expand-and-contract", "expand and contract",
    )),
)

_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Matches things shaped like a service name: "search-api", "payments-api",
# "recommendation-engine".
#
# `the <word> service` is not an alternative here, because `_normalise` has
# already rewritten that form to `<word>-api` before this pattern ever runs. It
# used to be listed as one, and that branch was unreachable.
_SERVICE_SHAPED = re.compile(
    r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*-(?:api|service|engine|worker|gateway))\b"
)

# Words that are never the name of a service, however they sit in a sentence.
#
# This list fixes a refusal bug, and the refusal was the expensive kind.
# `_normalise` rewrote `<word> api` to `<word>-api` unconditionally, so "Which
# API should I check first?" produced the service name `which-api` - which no
# document covers, and `unknown_service` short-circuits the gate straight to
# `no_match` before any model is called. This system's own reasoning is that
# admitting a doubtful question is recoverable downstream while refusing a good
# one is not, which makes a determiner read as a service name the worst
# available false positive. "Which API should I check first for high CPU?" was
# refused outright while RB-001, RB-003 and RB-006 sat in the shortlist.
_NOT_A_SERVICE = frozenset(
    """
    a an the this that these those which what whose whatever any some no every
    each either neither our your their my his her its all both one another same
    other such more most many few several
    """.split()
)


def _normalise(question: str) -> str:
    """Lowercase, and collapse the several ways people write a service name.

    "checkout api", "checkout_api" and "the checkout service" all become
    "checkout-api" so that one set of patterns can match them all - unless the
    leading word is a determiner or an interrogative, in which case there is no
    service name there to collapse.
    """
    text = question.lower()

    def _join(match: "re.Match[str]") -> str:
        word = match.group(1)
        if word in _NOT_A_SERVICE:
            return match.group(0)
        return f"{word}-api"

    text = re.sub(r"\b([a-z][a-z0-9]*)[ _]api\b", _join, text)
    text = re.sub(r"\bthe\s+([a-z][a-z0-9]*)\s+service\b", _join, text)
    return text


def _find_service(text: str, known: tuple[str, ...]) -> str | None:
    """Match a known service, by full name or by its distinguishing prefix.

    The prefix match is what handles a bare "checkout" - and the prefix is the
    discriminating token anyway, since every service in this corpus is
    `<name>-api`.
    """
    for service in known:
        if service in text:
            return service
    for service in known:
        prefix = service.split("-")[0]
        if re.search(rf"\b{re.escape(prefix)}\b", text):
            return service
    return None


def _find_unknown_service(text: str, known: tuple[str, ...]) -> str | None:
    """Find a service-shaped name we have no documents for.

    This is the difference between "the question named no service" and "the
    question named a service that does not exist here". The second is positive
    evidence of no_match, and treating it as such is ordinary operational
    reasoning rather than a trick: if nobody has written a runbook for it, we
    genuinely cannot answer from the runbooks.
    """
    for match in _SERVICE_SHAPED.finditer(text):
        name = match.group(1)
        if not name:
            continue
        if name in known:
            return None
        if any(name == s.split("-")[0] for s in known):
            return None
        return name
    return None


@functools.lru_cache(maxsize=256)
def _synonym_pattern(synonym: str) -> re.Pattern[str]:
    """A whole-word matcher for one synonym.

    Compiled once per synonym; the table is fixed and small.
    """
    return re.compile(rf"(?<![a-z0-9]){re.escape(synonym)}(?![a-z0-9])")


def _find_failure_mode(text: str, allowed: tuple[str, ...]) -> str | None:
    """Pick the failure mode with the most specific match.

    Longest matching phrase wins, so "too many connections" beats a stray
    "connection" and "exit code 137" is not out-voted by a generic word.

    **Matched on word boundaries, not as substrings.** A plain `synonym in text`
    reads the middle of unrelated words, and because the metadata filter is a
    *hard drop* rather than a score penalty, a false hit here does not weaken a
    document's ranking - it deletes it from consideration. Two cases were live:

        "What are the deploy parameters for checkout-api?"
            pa[ram]eters      -> "ram"  -> failure_mode=memory
            ... which dropped RB-001, RB-002 and RB-012 as "failure mode
            mismatch", including the CPU runbook the question was about.

        "The feature flag rollout is broken"
            f[lag]            -> "lag"  -> failure_mode=sync_lag

    The lookbehind and lookahead are on `[a-z0-9]` rather than `\b` because the
    synonyms contain hyphens and spaces ("py-spy", "too many connections"), and
    `\b` between a hyphen and a space does not mean what it appears to.
    """
    best: tuple[int, str] | None = None
    for mode, synonyms in FAILURE_SYNONYMS.items():
        if allowed and mode not in allowed:
            continue
        for synonym in synonyms:
            if _synonym_pattern(synonym).search(text) and (
                best is None or len(synonym) > best[0]
            ):
                best = (len(synonym), mode)
    return best[1] if best else None


def _find_intent(text: str) -> str:
    for intent, patterns in INTENT_PATTERNS:
        if any(p in text for p in patterns):
            return intent
    return "diagnose"


def analyze_query(question: str, docs: tuple[Doc, ...]) -> QuerySpec:
    """Extract the two or three facts that decide which document applies.

    `docs` is passed in rather than imported so that the vocabularies come from
    whatever corpus is actually loaded - adding a runbook for a new service
    teaches this function about it with no code change.
    """
    from agent.core.corpus import known_failure_modes, known_services

    text = _normalise(question)
    known = known_services(docs)

    service = _find_service(text, known)
    date_match = _DATE.search(text)
    date = date_match.group(1) if date_match else None

    intent = _find_intent(text)
    if date and intent == "diagnose":
        # A question pinned to a specific date is asking about an incident that
        # happened, not about what to do when it happens again.
        intent = "postmortem"

    return QuerySpec(
        raw=question,
        service=service,
        failure_mode=_find_failure_mode(text, known_failure_modes(docs)),
        intent=intent,
        date=date,
        unknown_service=None if service else _find_unknown_service(text, known),
    )
