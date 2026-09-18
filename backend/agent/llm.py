"""Groq client, wrapped for the two things that actually go wrong on a free tier.

1. **429s.** The free tier allows roughly two questions a minute. We retry with
   exponential backoff and honour `retry-after` when Groq sends it.
2. **JSON that isn't quite JSON.** Models fence it, prefix it with prose, or - if
   the model is a reasoning model like `qwen/qwen3.8-27b` - wrap the whole thing
   in a `<think>` block. We parse defensively, ask once for a repair, and then
   degrade gracefully rather than raising into the request path.

The client is constructed lazily so that importing this module costs nothing and
works with no API key. Every stage that does not call an LLM must stay runnable
offline, which is most of the test suite.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

from agent.config import Settings, current_settings
from logger.zap import create_logger

log = create_logger()


class LLMUnavailable(RuntimeError):
    """No API key, or the `groq` package is not installed."""


class LLMBadJSON(ValueError):
    """The model would not produce parseable JSON, even after a repair attempt."""


@dataclass
class LLMResult:
    text: str
    model: str
    calls: int  # how many round trips this took, retries included


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

_client = None
# FastAPI runs sync endpoints in a threadpool, so two requests genuinely race
# here. Without the lock one thread can set `_client = None` while another is
# reading it - benign in that it only costs a rebuilt client, but the window
# where a caller sees a client built for a different key is not.
_client_lock = threading.Lock()


def _get_client(cfg: Settings):
    global _client
    with _client_lock:
        if _client is not None:
            # Keyed on the api key, not just "is it built": `_load_dotenv` lets
            # a local `.env` edit change the key inside a live process, and a
            # client cached under the old one would keep using it silently.
            if getattr(_client, "api_key", None) == cfg.groq_api_key:
                return _client
            _client = None
        if not cfg.groq_api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set. Retrieval-only paths still work; "
                "grounding does not."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - environment problem
            raise LLMUnavailable("The `groq` package is not installed.") from exc
        _client = Groq(api_key=cfg.groq_api_key, timeout=cfg.llm_timeout_s)
        return _client


def status_code_of(exc: Exception) -> int | None:
    """The HTTP status an exception carries, whichever client raised it.

    Two shapes reach this codebase and they do not agree. The `groq` SDK
    exposes `status_code` on the exception and headers under
    `exc.response.headers`; `urllib.error.HTTPError` - which is what the
    Supabase store and the Cloudinary client raise, because both talk HTTP over
    the standard library rather than pulling in an SDK - exposes `code` and
    `headers` directly.

    Reading only the first shape meant every Supabase and Cloudinary 429 was
    classified as non-retryable.
    """
    for attribute in ("status_code", "code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_upstream_rate_limit(exc: Exception) -> bool:
    """Whether this exception is a 429 from someone we depend on.

    Public because the API layer needs it too: a rate limit that survived the
    retry budget should become a 503 with a `Retry-After`, not an opaque 500.
    """
    if status_code_of(exc) == 429:
        return True
    return type(exc).__name__ == "RateLimitError"


def _is_rate_limit(exc: Exception) -> bool:
    """Identify a 429 without importing groq's exception classes.

    Keeping this structural rather than importing `groq.RateLimitError` means
    this module still imports when the package is absent.
    """
    if type(exc).__name__ in {"RateLimitError", "APIStatusError"}:
        return status_code_of(exc) in (429, None)
    return status_code_of(exc) == 429


def _retry_after(exc: Exception, attempt: int) -> float:
    """Prefer the server's own backoff hint; fall back to exponential."""
    headers = (
        getattr(getattr(exc, "response", None), "headers", None)
        or getattr(exc, "headers", None)
        or {}
    )
    hinted = headers.get("retry-after") or headers.get("Retry-After")
    if hinted:
        try:
            return min(float(hinted), 30.0)
        except (TypeError, ValueError):
            pass
    return min(2.0**attempt, 16.0)


def chat(
    messages: list[dict],
    role: str = "generator",
    cfg: Settings | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> LLMResult:
    """One chat completion, retried on rate limits.

    `role` is "generator" | "reasoner" | "grader" - never a model name. The
    mapping lives in config, so swapping models is never a code change.
    """
    from agent.stages import current_recorder

    cfg = cfg or current_settings()
    model = getattr(cfg.models, role)
    client = _get_client(cfg)
    recorder = current_recorder()

    from agent import observability

    last: Exception | None = None
    for attempt in range(cfg.llm_max_retries):
        started = time.perf_counter()
        # One Langfuse `generation` per round trip, opened here because this is
        # the only place that knows the model, the messages as sent, and the
        # token usage that comes back - which is everything Langfuse needs to
        # attribute cost. A span further up could time the call and nothing
        # else.
        with observability.generation(
            role=role,
            model=model,
            messages=messages,
            attempt=attempt + 1,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as generation:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""

                # One line per request that reached Groq. This is the only
                # place that knows a call actually went out - every count
                # further up is derived from what this function returns - so a
                # run's real API usage is reconstructable from the log alone.
                #
                # Tokens are logged because the free tier's binding limit is 8k
                # *per minute*, not requests per day: a 429 is predictable from
                # token spend and from nothing else.
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                log.info(
                    "llm_call",
                    model=model,
                    role=role,
                    attempt=attempt + 1,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                # The same three numbers, in the shape Langfuse prices from.
                # They are the whole reason cost shows up in a dashboard at
                # all: without usage on the generation, a trace reports latency
                # and nothing about what the run spent.
                generation.update(
                    output=text,
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    },
                    metadata={
                        "finish_reason": getattr(
                            response.choices[0], "finish_reason", ""
                        )
                    },
                )

                # The gateway stage records the *first* call of a run, which is
                # the one that says a request reached Groq at all. Per-call
                # detail is the `llm_call` line above; this is the ledger's
                # single row.
                recorder.ran(
                    "llm_gateway",
                    detail=(
                        f"{model} role={role} attempt={attempt + 1} "
                        f"tokens={total_tokens}"
                    ),
                    ms=int((time.perf_counter() - started) * 1000),
                )

                return LLMResult(
                    text=_THINK_BLOCK.sub("", text).strip(),
                    model=model,
                    calls=attempt + 1,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last = exc
                # On the generation as well as in the log. A 429 that was
                # retried is a real request, and a cost or latency view that
                # only counted the successful attempt would under-report both.
                generation.update(
                    level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                )
                if not _is_rate_limit(exc) or attempt == cfg.llm_max_retries - 1:
                    # A request that failed still reached Groq and still
                    # counted against the quota. Logging only successes would
                    # produce a local record that cannot be reconciled with
                    # their dashboard.
                    log.error(
                        "llm_call_failed",
                        model=model,
                        role=role,
                        attempt=attempt + 1,
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    recorder.degrade(
                        "llm_gateway",
                        f"{model} failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}",
                    )
                    raise
                wait_s = _retry_after(exc, attempt)
                log.warning(
                    "rate_limited",
                    model=model,
                    role=role,
                    attempt=attempt + 1,
                    wait_s=wait_s,
                )
        # Outside the `with`, so the generation is closed before the backoff
        # and its duration is the request rather than the request plus the wait.
        time.sleep(wait_s)

    raise last  # type: ignore[misc]  # unreachable; loop either returns or raises


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Tries, in order: the whole string, the string with code fences stripped, and
    the first balanced `{...}` span. Raises LLMBadJSON if none parse.
    """
    candidates = [text, _FENCE.sub("", text).strip()]

    start = text.find("{")
    if start != -1:
        depth, in_string, escaped = 0, False, False
        for i, char in enumerate(text[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMBadJSON(f"No JSON object found in model output: {text[:200]!r}")


def chat_json(
    messages: list[dict],
    role: str = "generator",
    cfg: Settings | None = None,
    max_tokens: int = 1024,
) -> tuple[dict, int]:
    """A chat completion that must return a JSON object.

    Returns (parsed, llm_calls). On unparseable output we make exactly one
    repair attempt - handing the model its own bad output and asking for JSON
    only - and then give up. One repair, not a loop: on a tier this
    rate-limited, an unbounded retry is worse than a degraded answer.
    """
    cfg = cfg or current_settings()
    first = chat(messages, role=role, cfg=cfg, max_tokens=max_tokens)
    try:
        return extract_json(first.text), first.calls
    except LLMBadJSON:
        pass

    repair = messages + [
        {"role": "assistant", "content": first.text},
        {
            "role": "user",
            "content": (
                "That was not valid JSON. Reply with the JSON object only - "
                "no prose, no markdown fences, no explanation."
            ),
        },
    ]
    second = chat(repair, role=role, cfg=cfg, max_tokens=max_tokens)
    return extract_json(second.text), first.calls + second.calls
