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
from agent.core.query import analyze_query
from agent.core.retrieve import (
    metadata_filter,
    passes_gate,
    passes_hybrid_gate,
)
from agent.nodes.grade import grade_candidates, rewrite_query
from agent.nodes.ground import ground
from agent.stages import current_recorder
from agent.state import AgentState
from agent.store import get_store


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
# Nodes
# --------------------------------------------------------------------------

def analyze_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    recorder = current_recorder()

    with recorder.stage("query_enhance") as ledger:
        # From the store, not from disk: the service and failure-mode
        # vocabularies the analyser matches against are derived from whatever
        # corpus is actually mounted, so adding a runbook teaches it without a
        # code change - whichever backend that runbook lives in.
        docs = get_store(cfg).documents()
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

        # Named so the log does not overstate it: this is rule-based extraction
        # over a synonym table, not an LLM rewriting the question. The LLM
        # enhancement is stage 17, and only on the corrective loop.
        ledger.detail(f"rule_based; {detail}")

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

    Only the *ranking* step is delegated to the store, because that is the part
    that genuinely differs between an in-memory BM25 index and a SQL statement
    over pgvector. The filter and the gate stay here, in Python, with one
    implementation - so swapping the backend cannot change what gets dropped or
    what gets refused. That is what makes the Phase 6 equivalence test a
    meaningful check rather than a comparison of two different systems.
    """
    cfg = _cfg(state)
    retrieval = cfg.retrieval
    spec = state["spec"]
    store = get_store(cfg)
    recorder = current_recorder()

    # Stages 10 and 12-14 are recorded inside the store, because which of them
    # run is the store's decision - the SQL backend fuses in one statement.
    ranked, trace = store.retrieve(spec, _query(state), cfg)
    index = store.lexical_index()

    with recorder.stage("relevance_filter") as ledger:
        kept = metadata_filter(spec, ranked, retrieval.final_top_k)

        # Which gate depends on whether the dense arm actually contributed, not
        # on what the profile asked for. A hybrid run that silently fell back to
        # lexical must be gated as lexical, or it would be held to a floor no
        # candidate has a score for.
        dense_ran = any(c.dense_score > 0.0 for c in ranked)
        gate = passes_hybrid_gate if dense_ran else passes_gate
        passed, why = gate(spec, kept, index, retrieval)

        ledger.detail(
            f"kept={[c.doc_id for c in kept]} "
            f"dropped={len([c for c in ranked if c.verdict == 'dropped'])} "
            f"gate={'pass' if passed else 'REJECT'} ({why}) "
            f"via={'hybrid' if dense_ran else 'lexical'}"
        )

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
    recorder = current_recorder()

    if not cfg.retrieval.grader_enabled:
        # The silent skip this whole module was built for. Locally this is every
        # single run, and nothing in the answer reveals it.
        recorder.skip("relevance_grade", "GRADER_ENABLED=false in this profile")
        return {"graded": candidates, "trace": ["grade: disabled, candidates passed through"]}

    with recorder.stage("relevance_grade") as ledger:
        kept, reason, calls = grade_candidates(state["question"], candidates, cfg=cfg)
        ledger.detail(f"kept={[c.doc_id for c in kept]} llm_calls={calls} - {reason}")

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
    recorder = current_recorder()

    with recorder.stage("rewrite_query") as ledger:
        rewritten, calls = rewrite_query(state["question"], cfg=cfg)
        ledger.detail(
            f"attempt {state.get('rewrites', 0) + 1}/{cfg.max_rewrites} -> {rewritten!r}"
        )

    return {
        "search_query": rewritten,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": [f"rewrite: retrying as {rewritten!r}"],
        "llm_calls": calls,
    }


def ground_node(state: AgentState) -> dict:
    cfg = _cfg(state)
    candidates = state.get("graded") or state["candidates"]
    recorder = current_recorder()

    # Stages 18 and 19 are recorded inside `ground()` and `llm.chat()`, which
    # are the only places that know what the prompt contained and whether a
    # request actually reached Groq.
    with recorder.stage("generate") as ledger:
        answer, cited, invented, calls = ground(
            state["question"],
            candidates,
            role=state.get("model_role", "generator"),
            cfg=cfg,
        )
        ledger.detail(
            f"role={state.get('model_role', 'generator')} cited={cited or []} "
            f"invented={invented or []} llm_calls={calls}"
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
    recorder = current_recorder()

    if not state.get("gate_passed"):
        # Everything from grading onward was never entered - and that is the
        # design, not a bug. Saying so beats five blank `skipped` lines.
        recorder.skip_remaining("gate rejected; no document was shown to a model")
        return {
            "answer": NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (gate), zero grounding calls"],
        }

    if "raw_answer" not in state:
        recorder.skip_remaining("grader found nothing relevant; rewrite budget spent")
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
        recorder.skip_remaining("not reached; the model declined to cite anything")
        return {
            "answer": state.get("raw_answer") or NO_MATCH_MESSAGE,
            "cited_doc_ids": [],
            "confidence": "no_match",
            "trace": ["finalize: no_match (model declined)"],
        }

    # The answered path. Anything still unreported was not needed - most often
    # the rewrite, which only runs when grading empties the shortlist.
    recorder.skip_remaining("not reached on the answered path")

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
