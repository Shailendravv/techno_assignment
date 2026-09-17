"""Vercel entrypoint.

Vercel's Python runtime looks for a module-level ASGI `app`. This file exists
only to expose one; all routing lives in `app/main.py`.
"""

from app.main import app  # noqa: F401
