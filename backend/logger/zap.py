"""Application logging.

Two things this module guarantees:

1. **Locally, log lines never reach the terminal.** They go to `logs/app.log`
   instead, so `python -m agent "..."` and `uvicorn app.main:app --reload`
   keep the terminal for their own output - the answer, `--trace`, the
   uvicorn access log - and `logs/` is where a run's history actually lives.
2. **Deployed, there is no `logs/` to write to.** A Vercel function's
   filesystem is read-only outside `/tmp`, and its log viewer reads stdout,
   not a file nobody can open. So the "dev" profile logs one JSON object per
   line to stdout instead.

Which of the two applies is decided by `APP_ENV` (see `agent.config`) -
nothing here guesses at a container platform.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = "logs"
LOG_FILE = "app.log"

# 5 MB x 5 backups per run profile. Bounded log growth without reimplementing
# rotation by hand - `RotatingFileHandler` already does this correctly.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

_LOGGER_NAME = "app_logger"

# Built-in LogRecord attributes that cannot be used as extra= keys.
# logging.Logger.makeRecord() raises KeyError if an extra key shadows one of
# these.
_LOGRECORD_RESERVED = frozenset(
    {
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message",
        "pathname", "process", "processName", "relativeCreated",
        "thread", "threadName", "exc_info", "exc_text", "stack_info",
        "asctime", "taskName",
    }
)


def _safe_extra(kwargs: dict) -> dict:
    """Prefix any kwarg whose name collides with a reserved LogRecord field."""
    return {(f"_{k}" if k in _LOGRECORD_RESERVED else k): v for k, v in kwargs.items()}


class Logger:
    """A thin wrapper over `logging.Logger` for structured, leveled calls.

        log.warning("rate_limited", attempt=attempt, wait_s=wait)

    rather than string-formatting fields into the message by hand.
    """

    def __init__(self, logger: "logging.Logger | logging.LoggerAdapter"):
        self.logger = logger

    def debug(self, msg: str, **kwargs) -> None:
        self.logger.debug(msg, extra=_safe_extra(kwargs))

    def info(self, msg: str, **kwargs) -> None:
        self.logger.info(msg, extra=_safe_extra(kwargs))

    def warning(self, msg: str, **kwargs) -> None:
        self.logger.warning(msg, extra=_safe_extra(kwargs))

    warn = warning  # alias for call sites written against the stdlib API

    def error(self, msg: str, **kwargs) -> None:
        self.logger.error(msg, extra=_safe_extra(kwargs))

    def with_fields(self, **kwargs) -> "Logger":
        """A logger that merges `kwargs` into every record it emits."""
        return Logger(logging.LoggerAdapter(self.logger, kwargs))


class _BaseFormatter(logging.Formatter):
    """Common timestamp handling: the machine's own clock, with its offset.

    Local time, because the first thing anyone does with a log line is compare
    it against something they just did - a request they sent, a command they
    ran, a clock on the wall. A UTC line forces that comparison through mental
    arithmetic, and on a machine at +05:30 it silently reads as "this happened
    five hours ago" to anyone skimming.

    The offset is always written out, which is what makes this safe to change.
    The original reason for UTC was that a log should not be ambiguous when
    read on a different machine later - and a timestamp carrying `+05:30` is
    not ambiguous, it is fully qualified. Nothing is lost.

    On Vercel this needs no special case: those containers run UTC, so
    `astimezone()` resolves to `+00:00` there and the deployed lines stay UTC
    by fact rather than by force.
    """

    _SKIP = _LOGRECORD_RESERVED

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        ct = datetime.fromtimestamp(record.created).astimezone()
        return ct.strftime(datefmt) if datefmt else ct.isoformat()

    def _extra_fields(self, record: logging.LogRecord) -> dict:
        extra: dict = {}
        for key, value in record.__dict__.items():
            if key in self._SKIP:
                continue
            try:
                json.dumps(value)
                extra[key] = value
            except (TypeError, ValueError):
                extra[key] = str(value)
        return extra


class TextFormatter(_BaseFormatter):
    """`timestamp LEVEL logger: message - {"extra": "fields"}` - what `logs/app.log` holds.

    The timestamp is the writing machine's local time with its UTC offset, so
    it lines up with the clock of whoever is reading the file.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = self._extra_fields(record)
        return f"{base} - {json.dumps(extra)}" if extra else base


class JsonFormatter(_BaseFormatter):
    """One JSON object per line - what the deployed profile emits to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            **self._extra_fields(record),
        }
        return json.dumps(payload)


def _is_local(profile: str) -> bool:
    return profile.strip().lower() == "local"


def _build_handler(profile: str) -> logging.Handler:
    if _is_local(profile):
        # Best-effort: a logging setup must never be the reason a run fails.
        # If `logs/` cannot be created (permissions, a read-only checkout),
        # fall back to stdout rather than raising out of module import.
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            handler: logging.Handler = RotatingFileHandler(
                os.path.join(LOG_DIR, LOG_FILE),
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
        except OSError:
            handler = logging.StreamHandler()
        else:
            handler.setFormatter(
                TextFormatter(
                    fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S%z",
                )
            )
            return handler

    handler = logging.StreamHandler()
    # No `datefmt`, so `formatTime` falls through to `isoformat()` - full ISO
    # 8601 with the offset, which every log aggregator parses without being
    # told a pattern.
    handler.setFormatter(JsonFormatter())
    return handler


# Third-party libraries that log at INFO/DEBUG by default and would otherwise
# drown out application log lines with connection-pool chatter.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "groq", "google_genai")

_configured_profile: Optional[str] = None


def create_logger(profile: Optional[str] = None) -> Logger:
    """The application's logger, configured for the active profile.

    "local": file-only, `logs/app.log` (gitignored) - the terminal stays free
    for the CLI's own output. Any other profile ("dev", deployed): one JSON
    line per record to stdout, because that is what Vercel's log viewer reads
    and the filesystem cannot be written to.

    Idempotent - every entry point (the CLI, the FastAPI app, the ingestion
    pipeline) calls this, and it must not pile up a new handler on every call.
    Re-running it with a different `profile` (as the test suite does) swaps
    the handler instead of stacking one.
    """
    from agent.config import active_profile_name

    resolved = profile or active_profile_name()
    logger = logging.getLogger(_LOGGER_NAME)

    global _configured_profile
    if _configured_profile != resolved:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
        logger.addHandler(_build_handler(resolved))
        _configured_profile = resolved

    logger.setLevel(logging.DEBUG if _is_local(resolved) else logging.INFO)
    # Never hand records to the root logger too - that is the second path by
    # which they could still end up on the terminal (a library, or a test
    # runner, that has called `logging.basicConfig()`).
    logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return Logger(logger)


# ---------------------------------------------------------------------------
# Context-scoped logger
# ---------------------------------------------------------------------------
# Optional: lets code that does not want to import `create_logger` directly -
# a deeply nested helper, say - reach the current logger instead of threading
# it through every call. Nothing in this codebase sets this yet; it exists for
# whichever entry point first needs a request-scoped logger (e.g. a FastAPI
# middleware attaching a request ID).

_log_context: ContextVar[Optional[Logger]] = ContextVar("log_context", default=None)


class LoggerContextError(Exception):
    """Raised by `get_logger_from_context` when nothing has set one."""


def get_logger_from_context() -> Logger:
    logger = _log_context.get()
    if logger is None:
        raise LoggerContextError("No logger has been set in this context")
    return logger


def set_logger_in_context(logger: Logger) -> None:
    _log_context.set(logger)
