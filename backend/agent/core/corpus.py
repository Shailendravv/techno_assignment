"""Load `runbooks/*.md` into structured records.

The whole design rests on the distinguishing features of a document being
*structured fields* rather than prose. A service name buried in four hundred
words of near-identical text is a weak signal that a similarity score can
out-vote. The same service name in a `service:` field is a fact the filter can
act on. This module is what turns the former into the latter.

Parsing happens once and is cached, because the corpus does not change while
the process is running.
"""

from __future__ import annotations

import functools
import glob
import os

import frontmatter

from agent.core.models import Doc


class CorpusError(RuntimeError):
    """The corpus is missing, empty, or a document is malformed."""


REQUIRED_FIELDS = ("doc_id", "title", "service", "failure_mode", "doc_type")


def _as_optional_str(value: object) -> str | None:
    """Normalise YAML's several spellings of 'absent' to None.

    `null`, an empty string, and the literal string "None" all mean the same
    thing here, and a document that says `service: ""` must behave exactly like
    one that says `service: null` - otherwise it would fail every comparison
    instead of matching everything.
    """
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "null", "None", "~") else text


def parse_doc(path: str) -> Doc:
    post = frontmatter.load(path)
    meta = post.metadata

    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise CorpusError(f"{path}: front-matter is missing {missing}")
    if not post.content.strip():
        raise CorpusError(f"{path}: document body is empty")

    return Doc(
        doc_id=str(meta["doc_id"]).strip(),
        title=str(meta["title"]).strip(),
        service=_as_optional_str(meta["service"]),
        failure_mode=_as_optional_str(meta["failure_mode"]),
        doc_type=str(meta["doc_type"]).strip(),
        date=_as_optional_str(meta.get("date")),
        text=post.content.strip(),
    )


@functools.lru_cache(maxsize=4)
def load_corpus(directory: str = "runbooks") -> tuple[Doc, ...]:
    """Parse every markdown file in `directory`, sorted by doc_id.

    Returns a tuple rather than a list so the result is hashable and safe to
    cache - callers must not mutate the corpus.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.md")))
    if not paths:
        raise CorpusError(
            f"No documents found in {directory!r}. "
            "Run from the repository root, or set CORPUS_DIR."
        )

    docs = [parse_doc(p) for p in paths]

    seen: dict[str, str] = {}
    for doc, path in zip(docs, paths):
        if doc.doc_id in seen:
            raise CorpusError(
                f"Duplicate doc_id {doc.doc_id!r} in {path} and {seen[doc.doc_id]}"
            )
        seen[doc.doc_id] = path

    return tuple(sorted(docs, key=lambda d: d.doc_id))


def known_services(docs: tuple[Doc, ...]) -> tuple[str, ...]:
    """The services the corpus actually documents.

    Derived from the corpus rather than hard-coded, so adding a runbook for a
    new service teaches the query analyser about it automatically. A question
    about a service outside this set yields `service=None`, which means the
    filter cannot help - that limitation is real and is reported in the
    write-up.
    """
    return tuple(sorted({d.service for d in docs if d.service}))


def known_failure_modes(docs: tuple[Doc, ...]) -> tuple[str, ...]:
    return tuple(sorted({d.failure_mode for d in docs if d.failure_mode}))


def by_id(docs: tuple[Doc, ...]) -> dict[str, Doc]:
    return {d.doc_id: d for d in docs}
