"""Cloudinary, as the canonical document store.

Two things happen here and they are deliberately on opposite sides of the
network boundary:

1. **Signing** an upload, which happens in the API. The signature is a hash over
   the upload parameters and the API secret, so the secret never leaves the
   server while the browser still gets permission to upload one specific thing.
2. **Fetching** a document, which happens in the offline ingestion pipeline,
   never in a request handler.

**Why uploads do not pass through our API.** Vercel caps request *and* response
bodies at 4.5 MB. A runbook is small, but a PDF of one is not reliably small,
and an upload path that works until someone attaches something big is worse
than one that never worked. So the browser posts the file straight to
Cloudinary and sends us only the resulting `public_id`. Our API handles a few
hundred bytes either way.

**Two free-tier gotchas, handled at setup rather than at demo time:**

- Free Cloudinary accounts block PDF *delivery* by default, for security. The
  fix is either enabling it under Settings -> Security -> PDF and ZIP files
  delivery, or - what we do - storing everything as `resource_type: raw`, which
  is not subject to that rule.
- Raw file uploads cap out around 10 MB on the free plan.

Written against Cloudinary's REST API over stdlib `urllib` rather than the
`cloudinary` package, for the same reason the Supabase store is: this is three
HTTP calls and a SHA-1, and Vercel's bundler does no tree-shaking.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from agent.config import Cloudinary, Settings, settings as default_settings

TIMEOUT_S = 30.0
API_BASE = "https://api.cloudinary.com/v1_1"


class CloudinaryUnavailable(RuntimeError):
    """Cloudinary is not configured, or would not answer."""


def sign_params(params: dict, api_secret: str) -> str:
    """Cloudinary's upload signature.

    The algorithm is fixed by Cloudinary and the details matter:

    - Parameters sorted by key, joined as `k=v` with `&`, then the API secret
      appended directly - not as another parameter, and with no separator.
    - `file`, `api_key`, `resource_type` and `cloud_name` are excluded. They are
      sent with the upload but are not part of what is signed.
    - Empty values are dropped. Signing `k=` and then omitting `k` from the
      upload produces a mismatch, which Cloudinary reports only as a generic
      401 - so this is worth getting right rather than debugging live.
    """
    signable = {
        k: v
        for k, v in params.items()
        if k not in ("file", "api_key", "resource_type", "cloud_name")
        and v not in (None, "")
    }
    payload = "&".join(f"{k}={signable[k]}" for k in sorted(signable))
    return hashlib.sha1(f"{payload}{api_secret}".encode("utf-8")).hexdigest()


def build_upload_signature(
    cfg: Settings | None = None,
    public_id: str | None = None,
    timestamp: int | None = None,
) -> dict:
    """Everything the browser needs to upload directly, and nothing more.

    Note what is *not* returned: the API secret. The browser gets a signature
    over specific parameters, valid for one upload into one folder. It cannot
    mint another for a different folder or a different `public_id`, which is the
    whole point of signing server-side rather than shipping an unsigned preset.
    """
    cfg = cfg or default_settings
    cloudinary = cfg.cloudinary
    if not cloudinary.configured:
        raise CloudinaryUnavailable(
            "CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET "
            "are not all set."
        )

    params: dict = {
        "timestamp": timestamp or int(time.time()),
        "folder": cloudinary.folder,
    }
    if public_id:
        params["public_id"] = public_id

    return {
        "cloud_name": cloudinary.cloud_name,
        "api_key": cloudinary.api_key,
        "resource_type": cloudinary.resource_type,
        "upload_url": (
            f"{API_BASE}/{cloudinary.cloud_name}/{cloudinary.resource_type}/upload"
        ),
        "signature": sign_params(params, cloudinary.api_secret),
        **params,
    }


def _basic_auth(cloudinary: Cloudinary) -> str:
    import base64

    raw = f"{cloudinary.api_key}:{cloudinary.api_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _get(url: str, headers: dict, binary: bool = False):
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise CloudinaryUnavailable(f"Cloudinary returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise CloudinaryUnavailable(f"Cannot reach Cloudinary: {exc}") from exc

    return body if binary else json.loads(body.decode("utf-8"))


def list_documents(cfg: Settings | None = None) -> list[dict]:
    """Every raw asset in our folder.

    Used by the ingestion pipeline to discover what to ingest, so that dropping
    a file into Cloudinary is enough to make it a candidate - no manifest to
    keep in step.
    """
    cfg = cfg or default_settings
    cloudinary = cfg.cloudinary
    if not cloudinary.configured:
        raise CloudinaryUnavailable("Cloudinary is not configured.")

    url = (
        f"{API_BASE}/{cloudinary.cloud_name}/resources/{cloudinary.resource_type}"
        f"?prefix={urllib.parse.quote(cloudinary.folder)}&max_results=100"
    )
    payload = _get(url, {"Authorization": _basic_auth(cloudinary)})
    return payload.get("resources", [])


def fetch_document(public_id: str, cfg: Settings | None = None) -> bytes:
    """Download one raw asset.

    Called only from the offline pipeline. Parsing and embedding are too slow
    and too memory-hungry for a serverless invocation, and keeping them out of
    the request path is also what keeps `pymupdf` and `fastembed` out of the
    deployed bundle entirely.
    """
    cfg = cfg or default_settings
    cloudinary = cfg.cloudinary
    if not cloudinary.configured:
        raise CloudinaryUnavailable("Cloudinary is not configured.")

    url = (
        f"https://res.cloudinary.com/{cloudinary.cloud_name}/"
        f"{cloudinary.resource_type}/upload/{public_id}"
    )
    return _get(url, {"Authorization": _basic_auth(cloudinary)}, binary=True)


def delivery_url(public_id: str, cfg: Settings | None = None) -> str:
    """The public URL a citation links to."""
    cfg = cfg or default_settings
    cloudinary = cfg.cloudinary
    return (
        f"https://res.cloudinary.com/{cloudinary.cloud_name}/"
        f"{cloudinary.resource_type}/upload/{public_id}"
    )
