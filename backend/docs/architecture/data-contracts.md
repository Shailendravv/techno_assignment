# Data Contracts

Components talk through four dataclasses and nothing else (`agent/core/models.py`).
Each stage takes one type and returns another, which is what lets every stage
be tested in isolation — `metadata_filter` can be unit-tested with no network
access at all, which matters because it's the component that most needs it.

```python
@dataclass
class Doc:                       # <- Corpus Loader
    doc_id: str                  # "RB-001"
    title: str
    service: str | None          # None for general/policy docs
    failure_mode: str | None
    doc_type: str                # "runbook" | "policy" | "postmortem"
    date: str | None             # postmortems only
    text: str

@dataclass
class QuerySpec:                 # <- Query Analyzer
    raw: str
    service: str | None
    failure_mode: str | None
    intent: str                  # diagnose | rollback | policy | postmortem
    date: str | None

@dataclass
class Candidate:                 # <- Lexical Search / Metadata Filter
    doc: Doc
    lexical_score: float
    verdict: str                 # "kept" | "dropped"
    reason: str                  # "service mismatch: payments-api != checkout-api"

@dataclass
class Answer:                    # <- what answer_question() serialises to dict
    answer: str
    cited_doc_ids: list[str]
    confidence: str
```

`Candidate.reason` exists purely so that when a question fails, it's possible
to see *which stage* threw the right document away instead of guessing. On a
12-document corpus that turns debugging from archaeology into reading a line
of text — and it's exactly what `--trace` on the CLI surfaces.

## The wiring

The whole system is a short, readable pipeline, which is the point:

```python
# conceptually, what agent/graph.py wires together
_DOCS  = load_corpus("runbooks/")     # module-level: parsed once
_INDEX = build_bm25(_DOCS)

def answer_question(question: str) -> dict:
    spec       = analyze_query(question, _DOCS)
    candidates = bm25_search(_INDEX, spec, top_k=8)
    candidates = metadata_filter(spec, candidates, top_k=4)

    if not passes_gate(candidates):
        return Answer(answer=NO_MATCH_MESSAGE, cited_doc_ids=[], confidence="no_match").as_dict()

    raw   = ground_with_llm(spec, candidates)
    cited = verify_citations(raw.cited_doc_ids, candidates)

    return Answer(
        answer=raw.answer,
        cited_doc_ids=cited,
        confidence=score_confidence(spec, candidates, cited),
    ).as_dict()
```

Three properties worth noting:

- **The corpus is parsed once**, at import. Per-question work is only BM25 +
  filter + at most one LLM call.
- **There is exactly one LLM call per question**, on the happy path.
  `no_match` questions cost zero calls — this matters on a free tier.
- **Every stage is independently testable**, with the metadata filter needing
  it the most, since it's the one that decides the near-duplicate trap.
