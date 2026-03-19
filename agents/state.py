"""
Shared LangGraph State Schema
Fixed: Removed anchor_terms field that was causing state validation errors.
anchor_terms is now computed locally in search_agent — not stored in state.
"""

from typing import TypedDict


class ResearchState(TypedDict):
    # ---- Input ----
    session_id: str
    query:      str
    filters:    dict

    # ---- Planner outputs ----
    subtopics:  list[str]
    agent_plan: list[str]

    # ---- Search outputs ----
    raw_papers: list[dict]

    # ---- Reader outputs ----
    processed_papers: list[dict]
    faiss_indexed:    bool
    cluster_info:     list[dict]
    research_trends:  dict

    # ---- Critic outputs ----
    ranked_papers: list[dict]

    # ---- Summary outputs ----
    insights: dict

    # ---- Knowledge graph ----
    knowledge_graph_entities: list[dict]
    knowledge_graph_edges:    list[dict]

    # ---- Artifacts ----
    artifacts: dict

    # ---- Metrics ----
    metrics: dict

    # ---- Errors ----
    errors: list[str]


def create_initial_state(
    query:      str,
    session_id: str,
    filters:    dict = None
) -> ResearchState:
    """Factory for a clean initial ResearchState."""
    return ResearchState(
        session_id        = session_id,
        query             = query,
        filters           = filters or {},
        subtopics         = [],
        agent_plan        = [],
        raw_papers        = [],
        processed_papers  = [],
        faiss_indexed     = False,
        cluster_info      = [],
        research_trends   = {},
        ranked_papers     = [],
        insights          = {},
        knowledge_graph_entities = [],
        knowledge_graph_edges    = [],
        artifacts         = {},
        metrics           = {},
        errors            = []
    )