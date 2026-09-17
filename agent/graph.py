"""The agent, as a LangGraph state graph.

    analyze -> retrieve -> +-- (gate fails) -----------------------> finalize
                           |                    no_match, LLM never called
                           |
                           +-- (candidates survive) --> grade --+
                                                                |
                    +-- (something is relevant) -----------------+--> ground -> finalize
                    |                                           |
                    +-- (nothing relevant, rewrites left) -------+--> rewrite -+
                    |                                           |             |
                    +-- (nothing relevant, budget spent) -------+--> finalize |
                                                                              |
                          ^-------------------- analyze <--------------------+

Two structural properties this buys, both of which are the reason the graph is
here rather than a function calling four other functions:

1. **The gate is an edge, not an `if`.** Returning "finalize" from
   `route_after_retrieval` means the grounding node is never entered, so no
   request is made and no document is ever shown to a model. "We do not call the
   model when nothing survived" becomes a declared property of the structure
   rather than a branch buried in a call stack - and it is a safety guarantee,
   not an optimisation. A model that is never shown a document cannot invent a
   citation for one.

2. **The corrective loop is bounded by construction.** `rewrite` routes back to
   `analyze`, and the only thing stopping that cycling forever is a counter in
   `AgentState` checked by an edge. On a free tier that allows roughly two
   questions a minute, "this cannot run away" needs to be legible, and here it
   is one function you can read.

Nodes are thin. Every one unpacks state, calls a pure function from
`agent.core`, and writes the result back. The logic lives in `agent.core`,
where it can be tested without a graph, without a network, and without keys.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.config import NO_MATCH_MESSAGE, Settings, settings as default_settings
from agent.core.confidence import score_confidence
from agent.core.corpus import load_corpus
from agent.core.query import analyze_query
from agent.core.retrieve import (
    best_cosine,
    bm25_search,
    build_dense_index,
    build_index,
    dense_search,
    metadata_filter,
    passes_gate,
    passes_hybrid_gate,
    rrf_fuse,
)
from agent.nodes.grade import grade_candidates, rewrite_query
from agent.nodes.ground import ground
from agent.state import AgentState


def _cfg(state: AgentState) -> Settings:
    return state.get("settings") or default_settings


def _query(state: AgentState) -> str:
    """The text we are actually searching for.

    Equal to the question, unless the corrective loop has rewritten it. Keeping
    the original in `question` matters: it is what gets shown to the grounding
    model, because the user asked that, not our paraphrase of it.
    """
    return state.get("search_query") or state["question"]


# --------------------------------------------------------------------------
# The dense index.
#
# Memoised on (corpus directory, embedder signature) rather than rebuilt per
# request. Embedding seventy chunks is a second locally and an HTTP round trip
# against Gemini; doing it per request would dominate every response, and on a
# warm serverless instance it would be pure waste.
#
# The signature is in the key because two embedders produce incompatible
# vectors - reusing one's index for the other is a dimension-mismatch crash at
# best and silent nonsense at worst.
# --------------------------------------------------------------------------
_dense_indexes: dict[tuple[str, str], object] = {}


def _get_dense_index(cfg: Settings):
    """Build or fetch the dense index, or return None if it cannot be built.

    Returning None rather than raising is deliberate. A missing API key or an
    uninstalled embedder must degrade the system to lexical-only - which is a
    complete, working retriever that answers nineteen of our twenty questions -
    rather than failing the request. Degrading loudly in the trace and quietly
    in the response is the right trade for an incident tool.
    """
    from agent.embed import embedding_available, get_embedder

    if not embedding_available(cfg):
        return None

    key = (cfg.corpus_dir, cfg.embedding.signature)
    if key in _dense_indexes:
        return _dense_indexes[key]

    try:
        embedder = get_embedder(cfg)
        index = build_dense_index(load_corpus(cfg.corpus_dir), embedder)
    except Exception:  # noqa: BLE001 - degrade to lexical, see docstring
        return None

    _dense_indexes[key] = index
    return index


def reset_dense_index_cache() -> None:
    """Drop the memoised indexes. For tests that switch profiles or corpora."""
    _dense_indexes.clear()


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def analyze_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    docs = load_corpus(cfg.corpus_dir)
    query = _query(state)
    spec = analyze_query(query, docs)

    detail = (
        f"service={spec.service}, failure_mode={spec.failure_mode}, "
        f"intent={spec.intent}, date={spec.date}"
    )
    if spec.unknown_service:
        detail += f", unknown_service={spec.unknown_service}"
    if query != state["question"]:
        detail += f" (on rewritten query {query!r})"

    return {"spec": spec, "trace": [f"analyze: {detail}"]}


def retrieve_node(state: AgentState) -> dict:
    """Rank, fuse, filter, gate.

    The order is the design and it is not interchangeable:

        rank (lexical + dense) -> fuse -> **filter** -> gate

    Fusion first, because the filter is a hard drop and dropping before fusing
    would mean the two arms fused over different populations. The filter after,
    because that is the stage that discriminates near-duplicates, and it acts on
    document metadata - a fact no similarity score can out-vote. The gate last,
    on the survivors, because whether to answer at all depends on what is left.
    """
    cfg = _cfg(state)
    retrieval = cfg.retrieval
    docs = load_corpus(cfg.corpus_dir)
    index = build_index(docs)
    spec = state["spec"]
    query = _query(state)

    lexical = bm25_search(index, spec, retrieval.bm25_top_k)
    trace: list[str] = []

    dense_index = _get_dense_index(cfg) if retrieval.is_hybrid else None
    if retrieval.is_hybrid and dense_index is None:
        # Asked for hybrid, cannot do hybrid. Say so rather than reporting a
        # hybrid run that was quietly lexical - that would corrupt any
        # comparison between the two arms.
        trace.append(
            "retrieve: hybrid requested but no embedder is available - "
            "falling back to lexical-only"
        )

    if dense_index is not None:
        from agent.embed import get_embedder

        query_vector = get_embedder(cfg).embed_query(query)
        dense = dense_search(dense_index, query_vector, retrieval.dense_top_k)
        ranked = rrf_fuse(lexical, dense, retrieval.rrf_k)
        trace.append(
            "retrieve: dense top "
            + ", ".join(f"{c.doc_id}({c.dense_score:.2f})" for c in dense[:3])
        )
    else:
        ranked = lexical

    kept = metadata_filter(spec, ranked, retrieval.final_top_k)
    gate = passes_hybrid_gate if dense_index is not None else passes_gate
    passed, why = gate(spec, kept, index, retrieval)

    trace.insert(
        0,
        "retrieve: kept "
        + (", ".join(f"{c.doc_id}({c.lexical_score:.1f})" for c in kept) or "nothing"),
    )
    dropped = [c for c in ranked if c.verdict == "dropped"]
    if dropped:
        trace.append(
            "filter dropped: " + "; ".join(f"{c.doc_id} - {c.reason}" for c in dropped)
        )
    trace.append(f"gate: {'pass' if passed else 'REJECT'} - {why}")

    return {"candidates": kept, "gate_passed": passed, "trace": trace}


def grade_node(state: AgentState) -> dict:
    """The CRAG relevance filter. Skipped entirely when disabled."""
    cfg = _cfg(state)
    candidates = state["candidates"]

    if not cfg.retrieval.grader_enabled:
        return {"graded": candidates, "trace": ["grade: disabled, candidates passed through"]}

    kept, reason, calls = grade_candidates(state["question"], candidates, cfg=cfg)

    dropped = [c.doc_id for c in candidates if c.doc_id not in {k.doc_id for k in kept}]
    trace = [
        f"grade: kept {[c.doc_id for c in kept] or 'nothing'} - {reason}"
    ]
    if dropped:
        trace.append(f"grade dropped: {dropped}")

    return {"graded": kept, "trace": trace, "llm_calls": calls}


def rewrite_node(state: AgentState) -> dict:
    """Restate the question in the corpus's vocabulary and try once more."""
    cfg = _cfg(state)
    rewritten, calls = rewrite_query(state["question"], cfg=cfg)

    return {
        "search_query": rewritten,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": [f"rewrite: retrying as {rewritten!r}"],
        "llm_calls": calls,
    }


def ground_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    candidates = state.get("graded") or state["candidates"]

    answer, cited, invented, calls = ground(
        state["question"],
        candidates,
        role=state.get("model_role", "generator"),
        cfg=cfg,
    )

    trace = [f"ground: model cited {cited or 'nothing'}"]
    if invented:
        trace.append(f"citation check: discarded invented IDs {invented}")
    if not cited:
        trace.append("ground: model declined - none of the documents applied")

    return {
        "raw_answer": answer,
        "raw_cited_ids": cited,
        "trace": trace,
        "llm_calls": calls,
    }


def finalize_node(state: AgentState) -> dict:
    """Assemble the three fields the brief specifies.

    Reached from every branch, so this is the single place the output shape is
    decided - whether we answered or declined, and whichever gate declined.
    """
    if not state.get("gate_passed"):
        return {
            "answer": NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (gate), zero grounding calls"],
        }

    if "raw_answer" not in state:
        # Finalize was reached without the generator ever running: the grader
        # judged every candidate irrelevant and the rewrite budget was spent.
        # Another no_match that cost no grounding call.
        return {
            "answer": NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (grader found nothing relevant)"],
        }

    cited = state.get("raw_cited_ids") or []
    candidates = state.get("graded") or state.get("candidates") or []

    if not cited:
        return {
            "answer": state.get("raw_answer") or NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (model declined)"],
        }

    confidence = score_confidence(state["spec"], candidates, cited)
    return {
        "answer": state.get("raw_answer", ""),
        "cited_doc_ids": cited,
        "confidence": confidence,
        "trace": [f"finalize: {confidence}, cited {cited}"],
    }


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

def route_after_retrieval(state: AgentState) -> str:
    """The gate, as a routing decision.

    Returning "finalize" here means neither the grader nor the generator is
    ever entered, so no request is made and no document is ever shown to a
    model. That is why this is an edge and not an `if`.
    """
    return "grade" if state.get("gate_passed") else "finalize"


def route_after_grading(state: AgentState) -> str:
    """Relevant candidates -> answer. None -> rewrite once, then give up.

    The bound is the whole point. `max_rewrites` is checked here, in the edge,
    so the loop's termination is a property of the graph rather than a promise
    made by a function. One rewrite, not a loop that converges eventually: on a
    tier this rate-limited, an unbounded retry is a worse failure than a
    `no_match`.
    """
    cfg = _cfg(state)
    if state.get("graded"):
        return "ground"
    if state.get("rewrites", 0) < cfg.max_rewrites:
        return "rewrite"
    return "finalize"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("ground", ground_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"grade": "grade", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"ground": "ground", "rewrite": "rewrite", "finalize": "finalize"},
    )
    # The cycle. Back to `analyze`, not to `retrieve`, because a rewritten
    # query may name a service the original only implied - and the metadata
    # filter can only act on a spec that has been re-extracted from it.
    graph.add_edge("rewrite", "analyze")
    graph.add_edge("ground", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import. Compilation is not free, and Vercel reuses warm
# instances, so doing this per request would pay the cost on every call.
COMPILED = build_graph()
