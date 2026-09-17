"""Phase 1: the corpus is shaped the way the exercise requires.

These tests guard the *data*, not the retriever. They exist because the traps
are the whole point of the corpus, and a well-meaning later edit - tidying up
RB-003 so it reads less like RB-001, say - would quietly destroy the thing the
exercise is testing. If one of these fails, the corpus has drifted, not the code.
"""

from __future__ import annotations

import glob
import re

import frontmatter
import pytest

from eval.questions import ALL_QUESTIONS

REQUIRED_FIELDS = {"doc_id", "title", "service", "failure_mode", "doc_type", "date"}
KNOWN_SERVICES = {"checkout-api", "payments-api", "inventory-api"}
KNOWN_DOC_TYPES = {"runbook", "policy", "postmortem"}


def _load_all() -> dict[str, frontmatter.Post]:
    posts = {}
    for path in sorted(glob.glob("runbooks/*.md")):
        post = frontmatter.load(path)
        posts[post.metadata["doc_id"]] = post
    return posts


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a: frontmatter.Post, b: frontmatter.Post) -> float:
    ta, tb = _tokens(a.content), _tokens(b.content)
    return len(ta & tb) / len(ta | tb)


@pytest.fixture(scope="module")
def docs() -> dict[str, frontmatter.Post]:
    return _load_all()


def test_corpus_has_twelve_documents(docs):
    assert len(docs) == 12
    assert set(docs) == {f"RB-{n:03d}" for n in range(1, 13)}


def test_every_document_has_complete_front_matter(docs):
    for doc_id, post in docs.items():
        assert REQUIRED_FIELDS <= set(post.metadata), f"{doc_id} is missing fields"
        assert post.content.strip(), f"{doc_id} has no body"


def test_metadata_values_come_from_the_known_vocabularies(docs):
    """The filter compares these values literally, so a typo in one of them
    silently disables filtering for that document."""
    for doc_id, post in docs.items():
        service = post.metadata["service"]
        assert service is None or service in KNOWN_SERVICES, f"{doc_id}: {service}"
        assert post.metadata["doc_type"] in KNOWN_DOC_TYPES, doc_id


def test_policy_documents_have_no_service(docs):
    """General docs apply to every service. Giving one a service field would
    make the filter drop it whenever a question named a different service -
    which is how the incident-communication question would break."""
    for doc_id, post in docs.items():
        if post.metadata["doc_type"] == "policy":
            assert post.metadata["service"] is None, doc_id
            assert post.metadata["failure_mode"] is None, doc_id


def test_the_postmortem_carries_a_date(docs):
    """Date is what separates the postmortem from the runbook covering the
    same service and the same failure mode."""
    postmortems = [d for d, p in docs.items() if p.metadata["doc_type"] == "postmortem"]
    assert postmortems == ["RB-012"]
    assert docs["RB-012"].metadata["date"] == "2026-08-10"
    assert isinstance(docs["RB-012"].metadata["date"], str)


@pytest.mark.parametrize(
    "a,b,why",
    [
        ("RB-001", "RB-003", "same failure (cpu), different service"),
        ("RB-002", "RB-007", "same failure (connections), different service"),
        ("RB-005", "RB-006", "same procedure (rollback), different service"),
    ],
)
def test_service_swap_pairs_are_genuine_near_duplicates(docs, a, b, why):
    """These are the traps. If the overlap drops, the corpus has stopped
    testing what it was built to test."""
    overlap = _overlap(docs[a], docs[b])
    assert overlap > 0.60, f"{a} vs {b} overlap fell to {overlap:.0%} ({why})"
    assert docs[a].metadata["service"] != docs[b].metadata["service"]


def test_same_service_different_failure_pair_is_a_near_duplicate(docs):
    overlap = _overlap(docs["RB-001"], docs["RB-004"])
    assert overlap > 0.35, f"RB-001 vs RB-004 overlap fell to {overlap:.0%}"
    assert docs["RB-001"].metadata["service"] == docs["RB-004"].metadata["service"]
    assert docs["RB-001"].metadata["failure_mode"] != docs["RB-004"].metadata["failure_mode"]


def test_every_trap_pair_is_separable_by_at_least_one_metadata_field(docs):
    """The design claim in one assertion: for any two documents that overlap
    heavily in wording, some structured field tells them apart. If this ever
    fails, no metadata filter can rescue that pair."""
    ids = sorted(docs)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _overlap(docs[a], docs[b]) <= 0.35:
                continue
            ma, mb = docs[a].metadata, docs[b].metadata
            differs = (
                ma["service"] != mb["service"]
                or ma["failure_mode"] != mb["failure_mode"]
                or ma["doc_type"] != mb["doc_type"]
            )
            assert differs, f"{a} and {b} overlap heavily and are indistinguishable"


def test_the_postmortem_shares_service_and_failure_with_its_runbook(docs):
    """Q2 may optionally also cite RB-012, which only makes sense if the
    2026-08-10 incident really was a connection-exhaustion incident."""
    rb2, rb12 = docs["RB-002"].metadata, docs["RB-012"].metadata
    assert rb2["service"] == rb12["service"] == "checkout-api"
    assert rb2["failure_mode"] == rb12["failure_mode"] == "connections"
    assert rb2["doc_type"] != rb12["doc_type"]


def test_no_match_questions_have_no_answer_in_the_corpus(docs):
    """A no_match question is only a fair test if the corpus genuinely cannot
    answer it. 'refund' in particular must appear nowhere."""
    body = " ".join(p.content.lower() for p in docs.values())
    for absent in ("refund", "vacation", "ssl certificate", "recommendation-engine",
                   "search-api"):
        assert absent not in body, f"{absent!r} appears in the corpus"


def test_question_set_is_twenty_with_five_no_match():
    assert len(ALL_QUESTIONS) == 20
    assert sum(1 for q in ALL_QUESTIONS if q.is_no_match) == 5
    assert len({q.id for q in ALL_QUESTIONS}) == 20


def test_every_expected_doc_id_exists(docs):
    referenced = {
        d for q in ALL_QUESTIONS for d in q.expected_doc_ids + q.acceptable_extra_ids
    }
    assert referenced <= set(docs), f"unknown doc ids: {referenced - set(docs)}"


def test_every_document_is_the_answer_to_some_question(docs):
    """If a document answers nothing, it is padding, and padding that nobody
    asks about cannot be scored."""
    answered = {d for q in ALL_QUESTIONS for d in q.expected_doc_ids}
    assert answered == set(docs), f"never the expected answer: {set(docs) - answered}"
