"""Phase 2: extracting structured facts from a question.

The filter can only compare fields it is given, so everything the filter does
depends on this component reading the question correctly. These tests pin the
behaviour that matters and, just as importantly, pin the behaviour we have
deliberately *not* implemented.
"""

from __future__ import annotations

import pytest

from agent.core.corpus import load_corpus
from agent.core.query import analyze_query


@pytest.fixture(scope="module")
def docs():
    return load_corpus("runbooks")


# --------------------------------------------------------------------------
# Services, in the several ways people write them.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "checkout-api is running hot",
        "checkout api is running hot",
        "the checkout service is running hot",
        "checkout is running hot",
        "CHECKOUT-API is running hot",
    ],
)
def test_service_is_recognised_however_it_is_written(docs, question):
    assert analyze_query(question, docs).service == "checkout-api"


def test_services_are_not_confused_with_each_other(docs):
    assert analyze_query("payments-api is slow", docs).service == "payments-api"
    assert analyze_query("inventory-api is slow", docs).service == "inventory-api"


def test_a_question_naming_no_service_yields_none(docs):
    spec = analyze_query("What is our incident communication policy?", docs)
    assert spec.service is None
    assert spec.unknown_service is None


# --------------------------------------------------------------------------
# The distinction the gate depends on: "named no service" is not the same as
# "named a service we have never heard of".
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the runbook for search-api high latency?", "search-api"),
        ("How do I restart the recommendation-engine service?", "recommendation-engine"),
    ],
)
def test_service_shaped_names_we_do_not_document_are_flagged(docs, question, expected):
    spec = analyze_query(question, docs)
    assert spec.service is None
    assert spec.unknown_service == expected


def test_a_known_service_is_never_flagged_as_unknown(docs):
    for question in ("checkout-api is down", "the payments service is down"):
        spec = analyze_query(question, docs)
        assert spec.unknown_service is None, question


# --------------------------------------------------------------------------
# Failure modes.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("checkout-api is running hot on CPU", "cpu"),
        ("payments-api CPU is pegged above 85%", "cpu"),
        ("we are seeing 'too many connections' errors", "connections"),
        ("inventory-api is refusing connections with too many clients", "connections"),
        ("containers keep restarting with exit code 137", "memory"),
        ("the pods are getting OOMKilled", "memory"),
        ("the inventory sync lag alert is firing", "sync_lag"),
        ("we are overselling stock", "sync_lag"),
    ],
)
def test_failure_modes_are_recognised_from_symptoms(docs, question, expected):
    assert analyze_query(question, docs).failure_mode == expected


def test_the_most_specific_phrase_wins(docs):
    """'too many connections' must beat a stray 'connection', so that a longer
    and more precise phrase is not out-voted by a generic one."""
    spec = analyze_query(
        "we are seeing too many connections on checkout-api", docs
    )
    assert spec.failure_mode == "connections"


def test_questions_about_no_particular_failure_yield_none(docs):
    """A rollback question is about a service but not about any one failure.
    Inventing a failure mode here would make the filter drop documents that
    legitimately apply."""
    for question in (
        "How do I safely roll back checkout-api to a previous version?",
        "When are we allowed to deploy to production?",
    ):
        assert analyze_query(question, docs).failure_mode is None, question


# --------------------------------------------------------------------------
# Intent, where the orderings overlap.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("How do I safely roll back checkout-api?", "rollback"),
        ("What is the rollback procedure for payments-api?", "rollback"),
        ("What is our policy for communicating an incident?", "policy"),
        ("When are we allowed to deploy to production?", "policy"),
        ("checkout-api is running hot on CPU", "diagnose"),
    ],
)
def test_intent_is_classified(docs, question, expected):
    assert analyze_query(question, docs).intent == expected


def test_rollback_beats_policy_when_both_words_appear(docs):
    """'What is the rollback procedure' contains a policy word and a rollback
    word. Rollback is the one that identifies the right document."""
    assert analyze_query(
        "What is the rollback procedure for payments-api?", docs
    ).intent == "rollback"


# --------------------------------------------------------------------------
# Dates.
# --------------------------------------------------------------------------

def test_a_date_is_extracted_and_implies_a_postmortem(docs):
    spec = analyze_query(
        "We had a checkout-api incident on 2026-08-10 - what was the root cause?",
        docs,
    )
    assert spec.date == "2026-08-10"
    assert spec.intent == "postmortem"
    assert spec.service == "checkout-api"


def test_a_date_alone_switches_intent_away_from_diagnose(docs):
    """A question pinned to a specific date is asking what happened, not what
    to do when it happens again. That is what separates question 4 from
    question 2, which share a service and a failure mode."""
    spec = analyze_query("what went on with checkout-api on 2026-08-10", docs)
    assert spec.intent == "postmortem"


def test_questions_without_dates_have_none(docs):
    assert analyze_query("checkout-api is running hot", docs).date is None


# --------------------------------------------------------------------------
# What we have deliberately not done.
# --------------------------------------------------------------------------

def test_paraphrases_are_not_in_the_synonym_table(docs):
    """This is an assertion about scope, not a bug.

    The synonym table is hand-written. Adding "dragging its feet" to it would
    make our own vocabulary-mismatch test question pass, which would measure
    the test rather than the system. Catching unanticipated phrasing is the
    dense retriever's job, and leaving this table naive is what makes the
    lexical-versus-hybrid comparison in Phase 5 mean anything.
    """
    spec = analyze_query(
        "The checkout service is dragging its feet and the boxes are "
        "working too hard.",
        docs,
    )
    assert spec.service == "checkout-api"  # the service is still found
    assert spec.failure_mode is None  # but the symptom is not


def test_the_analyzer_is_deterministic(docs):
    """Rule-based on purpose: same input, same output, no network, no cost."""
    question = "checkout-api is running hot on CPU - what should I check first?"
    assert analyze_query(question, docs) == analyze_query(question, docs)
