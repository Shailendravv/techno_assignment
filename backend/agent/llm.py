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
import time
from dataclasses import dataclass

from agent.config import Settings, settings as default_settings
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


def _get_client(cfg: Settings):
    global _client
    if _client is not None:
        return _client
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


def _is_rate_limit(exc: Exception) -> bool:
    """Identify a 429 without importing groq's exception classes.

    Keeping this structural rather than importing `groq.RateLimitError` means
    this module still imports when the package is absent.
    """
    if type(exc).__name__ in {"RateLimitError", "APIStatusError"}:
        return getattr(exc, "status_code", None) in (429, None)
    return getattr(exc, "status_code", None) == 429


def _retry_after(exc: Exception, attempt: int) -> float:
    """Prefer the server's own backoff hint; fall back to exponential."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    hinted = headers.get("retry-after")
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
    cfg = cfg or default_settings
    model = getattr(cfg.models, role)
    client = _get_client(cfg)

    last: Exception | None = None
    for attempt in range(cfg.llm_max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            return LLMResult(
                text=_THINK_BLOCK.sub("", text).strip(),
                model=model,
                calls=attempt + 1,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not _is_rate_limit(exc) or attempt == cfg.llm_max_retries - 1:
                raise
            wait_s = _retry_after(exc, attempt)
            log.warning(
                "rate_limited",
                model=model,
                role=role,
                attempt=attempt + 1,
                wait_s=wait_s,
            )
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
    cfg = cfg or default_settings
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
