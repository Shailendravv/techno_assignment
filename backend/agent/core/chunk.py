"""Split documents into `##` sections, for the dense arm.

Two reasons retrieval happens at section level rather than document level:

1. **Input limits.** `gemini-embedding-001` takes 2048 tokens. Our longest
   runbook is comfortably inside that, but an embedding of a whole document is
   an average of everything in it, and averaging "Symptoms", "Diagnosis" and
   "Escalation" together produces a vector that is a good match for nothing in
   particular. A question about escalation should match the escalation section.
2. **The passages are genuinely about different things.** "Rolling back" and
   "Database migrations" are separate topics that happen to share a file.

And one reason citation does *not* happen at section level: the brief's
contract is `cited_doc_ids`, plural documents. So chunks carry their parent
`doc_id` and every chunk-level hit is collapsed back to its document before
anything downstream sees it. This is parent-document retrieval - fine-grained
matching, coarse-grained citation.

**The second-order effect is the important one, and it cuts against us.**
Section-level chunks are *more* alike across near-duplicate documents than
whole documents are: RB-001's "First checks" and RB-003's "First checks" are
nearly the same text with one service name changed. Chunking therefore
increases the load on the metadata filter rather than reducing it. That is
survivable only because the filter runs *after* fusion and drops on document
metadata, which chunking cannot blur - the parent's `service` field is the same
fact whichever section matched.
"""

from __future__ import annotations

import re

from agent.core.models import Chunk, Doc

# A `##` heading, but not `###`. Sub-sections stay with their parent section:
# splitting further would produce passages too short to embed meaningfully.
_HEADING = re.compile(r"^##\s+(?!#)(.+?)\s*$", re.MULTILINE)

# Text before the first `##`, which is the `# Title` line and any preamble.
_PREAMBLE = "preamble"

# Below this, a section is a heading and a sentence fragment - too little to
# produce a useful vector, and it dilutes the parent document's ranking.
MIN_SECTION_CHARS = 40


def policy() -> str:
    """The splitting rules in one line, for the stage ledger.

    Built from the constants rather than written out, so a log line cannot
    claim a policy the code stopped following. `no overlap` is stated
    explicitly because its absence is a deliberate choice and an unstated one
    reads as an oversight: these are structural boundaries, not a fixed-size
    window, so there is no span for neighbouring chunks to share.
    """
    return (
        f"'##' sections (not '###'), no overlap, "
        f"min {MIN_SECTION_CHARS} chars, title+heading prefixed"
    )


def _slug(heading: str) -> str:
    """A stable, readable chunk id fragment: "First checks" -> "first-checks"."""
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return slug or "section"


def split_document(doc: Doc) -> list[Chunk]:
    """Split one document on its `##` headings.

    Every chunk's text is prefixed with the document title and the section
    heading. That prefix is not decoration: it is the only place the service
    name reliably appears in a section like "Escalation", whose body is generic
    prose. Without it, a whole class of sections embed identically across
    documents, and the dense arm - which exists to catch what BM25 misses -
    would contribute nothing but confusion for them.

    A document with no `##` headings yields exactly one chunk covering all of
    it, so this is total: every document is represented whatever its shape.
    """
    text = doc.text
    matches = list(_HEADING.finditer(text))

    spans: list[tuple[str, str]] = []
    if not matches:
        spans.append((_PREAMBLE, text))
    else:
        head = text[: matches[0].start()].strip()
        if len(head) >= MIN_SECTION_CHARS:
            spans.append((_PREAMBLE, head))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            spans.append((match.group(1).strip(), text[match.end() : end].strip()))

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading, body in spans:
        if len(body.strip()) < MIN_SECTION_CHARS:
            continue

        slug = _slug(heading)
        seen[slug] = seen.get(slug, 0) + 1
        # Two `## Verifying` headings in one file would otherwise collide.
        suffix = "" if seen[slug] == 1 else f"-{seen[slug]}"

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{slug}{suffix}",
                doc_id=doc.doc_id,
                section=heading,
                text=f"{doc.title}\n{heading}\n\n{body.strip()}",
            )
        )

    if not chunks:
        # Every document must be represented, however terse. A document that
        # vanished from the dense index would be invisible to the arm that
        # exists to catch what the lexical arm misses.
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{_PREAMBLE}",
                doc_id=doc.doc_id,
                section=_PREAMBLE,
                text=f"{doc.title}\n\n{text.strip()}",
            )
        )

    return chunks


def chunk_corpus(docs: tuple[Doc, ...]) -> list[Chunk]:
    """Split every document, preserving corpus order."""
    return [chunk for doc in docs for chunk in split_document(doc)]
