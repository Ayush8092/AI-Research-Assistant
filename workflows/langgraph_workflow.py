"""
LangGraph Workflow — 8-node research pipeline.

START → planner → search →(conditional)→ reader → critic
      → summary → knowledge_graph → artifact → persist_memory → END
"""

import logging
import time
import uuid
from typing import Literal

from langgraph.graph import StateGraph, START, END

from agents.state import ResearchState, create_initial_state
from agents.planner_agent import planner_agent
from agents.search_agent import search_agent
from agents.reader_agent import reader_agent
from agents.critic_agent import critic_agent
from agents.summary_agent import summary_agent
from knowledge_graph.graph_builder import knowledge_graph_agent
from artifacts.artifact_agent import artifact_agent
from database.memory_store import MemoryStore

logger        = logging.getLogger(__name__)
_memory_store = None


def _get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def _safe_node(node_fn, node_name: str):
    """Wrap any node function with timing and error handling."""
    def wrapper(state: ResearchState) -> ResearchState:
        logger.info(f"[Workflow] ▶ {node_name}")
        t0 = time.time()
        try:
            result  = node_fn(state)
            elapsed = time.time() - t0
            logger.info(f"[Workflow] ✓ {node_name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[Workflow] ✗ {node_name} failed ({elapsed:.2f}s): {e}", exc_info=True)
            return {**state, "errors": state.get("errors",[]) + [f"{node_name}: {str(e)}"]}
    return wrapper


def persist_memory_node(state: ResearchState) -> ResearchState:
    """Final node: persist everything to SQLite."""
    memory     = _get_memory_store()
    session_id = state["session_id"]
    query      = state["query"]

    try:
        memory.create_session(session_id, query, state.get("subtopics",[]))
        if state.get("ranked_papers"):
            memory.save_papers(session_id, state["ranked_papers"])
        for paper in state.get("ranked_papers",[])[:10]:
            for itype, content in paper.get("insights",{}).items():
                if content:
                    memory.save_insight(session_id, paper["paper_id"], itype, str(content))
        for atype, content in state.get("artifacts",{}).items():
            if content and atype != "knowledge_graph_html":
                memory.save_artifact(session_id, atype, str(content))
        if state.get("metrics"):
            memory.save_metrics(session_id, state["metrics"])
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant",
            f"Research complete. Analyzed {len(state.get('ranked_papers',[]))} papers.")
        logger.info(f"[Workflow] Memory persisted for session {session_id[:8]}")
    except Exception as e:
        logger.error(f"[Workflow] Memory persistence failed: {e}")
    return state


def should_continue_after_search(state: ResearchState) -> Literal["reader","end_empty"]:
    if not state.get("raw_papers"):
        logger.warning("[Workflow] No papers found. Ending early.")
        return "end_empty"
    return "reader"


def end_empty_node(state: ResearchState) -> ResearchState:
    return {**state, "insights": {
        "topic":      state.get("query",""),
        "background": "No papers found. Try broadening your search terms.",
        "key_methods":"","common_datasets":"","evaluation_metrics":"",
        "limitations":"","research_gaps":"","future_directions":""
    }}


def build_workflow():
    """Build and compile the LangGraph research pipeline."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner",         _safe_node(planner_agent,         "PlannerAgent"))
    graph.add_node("search",          _safe_node(search_agent,          "SearchAgent"))
    graph.add_node("reader",          _safe_node(reader_agent,          "ReaderAgent"))
    graph.add_node("critic",          _safe_node(critic_agent,          "CriticAgent"))
    graph.add_node("summary",         _safe_node(summary_agent,         "SummaryAgent"))
    graph.add_node("knowledge_graph", _safe_node(knowledge_graph_agent, "KnowledgeGraphAgent"))
    graph.add_node("artifact",        _safe_node(artifact_agent,        "ArtifactAgent"))
    graph.add_node("persist_memory",  _safe_node(persist_memory_node,   "PersistMemory"))
    graph.add_node("end_empty",       end_empty_node)

    graph.add_edge(START,             "planner")
    graph.add_edge("planner",         "search")
    graph.add_conditional_edges("search", should_continue_after_search,
                                {"reader":"reader","end_empty":"end_empty"})
    graph.add_edge("end_empty",       "persist_memory")
    graph.add_edge("reader",          "critic")
    graph.add_edge("critic",          "summary")
    graph.add_edge("summary",         "knowledge_graph")
    graph.add_edge("knowledge_graph", "artifact")
    graph.add_edge("artifact",        "persist_memory")
    graph.add_edge("persist_memory",  END)

    return graph.compile()


_workflow = None

def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
        logger.info("[Workflow] Compiled.")
    return _workflow


def run_research_pipeline(query: str, filters: dict = None, session_id: str = None) -> ResearchState:
    """Execute the full research pipeline."""
    if not session_id:
        session_id = str(uuid.uuid4())

    t0 = time.time()
    logger.info(f"[Pipeline] Starting | session={session_id[:8]} | query='{query}'")

    initial_state = create_initial_state(query=query, session_id=session_id, filters=filters or {})
    final_state   = get_workflow().invoke(initial_state, config={"recursion_limit":20})

    final_state["metrics"]["query_time_sec"] = round(time.time() - t0, 2)
    logger.info(f"[Pipeline] ✓ {final_state['metrics']['query_time_sec']}s | "
                f"{final_state['metrics'].get('papers_retrieved',0)} papers")
    return final_state


def run_with_refinement(session_id: str, refinement_command: str,
                        previous_state: ResearchState) -> ResearchState:
    """Re-run pipeline with user-applied filters."""
    from agents.planner_agent import refine_plan
    updated = refine_plan(previous_state, refinement_command)
    return run_research_pipeline(
        query=updated["query"], filters=updated["filters"], session_id=session_id
    )