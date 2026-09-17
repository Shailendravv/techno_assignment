"""The agent, as a LangGraph state graph.

    analyze -> retrieve -> +-- (gate fails) ------------> finalize
                           |                              no_match, LLM never called
                           +-- (candidates survive) ----> ground -> finalize

The gate being a **conditional edge** rather than an `if` inside a function is
the point of using a graph here. It makes "we do not call the model when
nothing survived" a declared property of the structure instead of a branch
buried in a call stack - and that property is a safety guarantee, not an
optimisation. A model that is never shown a document cannot invent a citation
for one.

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
    bm25_search,
    build_index,
    metadata_filter,
    passes_gate,
)
from agent.nodes.ground import ground
from agent.state import AgentState


def _cfg(state: AgentState) -> Settings:
    return state.get("settings") or default_settings


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def analyze_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    docs = load_corpus(cfg.corpus_dir)
    spec = analyze_query(state["question"], docs)

    detail = (
        f"service={spec.service}, failure_mode={spec.failure_mode}, "
        f"intent={spec.intent}, date={spec.date}"
    )
    if spec.unknown_service:
        detail += f", unknown_service={spec.unknown_service}"

    return {"spec": spec, "trace": [f"analyze: {detail}"]}


def retrieve_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    docs = load_corpus(cfg.corpus_dir)
    index = build_index(docs)
    spec = state["spec"]

    shortlist = bm25_search(index, spec, cfg.retrieval.bm25_top_k)
    kept = metadata_filter(spec, shortlist, cfg.retrieval.final_top_k)
    passed, why = passes_gate(spec, kept, index, cfg.retrieval)

    dropped = [c for c in shortlist if c.verdict == "dropped"]
    trace = [
        "retrieve: kept "
        + (", ".join(f"{c.doc_id}({c.lexical_score:.1f})" for c in kept) or "nothing")
    ]
    if dropped:
        trace.append(
            "filter dropped: " + "; ".join(f"{c.doc_id} - {c.reason}" for c in dropped)
        )
    trace.append(f"gate: {'pass' if passed else 'REJECT'} - {why}")

    return {"candidates": kept, "gate_passed": passed, "trace": trace}


def ground_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    answer, cited, invented, calls = ground(
        state["question"],
        state["candidates"],
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

    Reached from both branches, so this is the single place the output shape is
    decided - whether we answered or declined.
    """
    if not state.get("gate_passed"):
        return {
            "answer": NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (gate), zero LLM calls"],
        }

    cited = state.get("raw_cited_ids") or []
    confidence = score_confidence(state["spec"], state["candidates"], cited)

    if not cited:
        return {
            "answer": state.get("raw_answer") or NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (model declined)"],
        }

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

    Returning "finalize" here means the grounding node is never entered, so no
    request is made and no document is ever shown to a model. That is why this
    is an edge and not an `if`.
    """
    return "ground" if state.get("gate_passed") else "finalize"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("ground", ground_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"ground": "ground", "finalize": "finalize"},
    )
    graph.add_edge("ground", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import. Compilation is not free, and Vercel reuses warm
# instances, so doing this per request would pay the cost on every call.
COMPILED = build_graph()
