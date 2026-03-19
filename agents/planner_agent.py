"""
PlannerAgent (Supervisor)
Decomposes user research queries into focused subtopics.

Fix: Detects query domain and generates domain-appropriate subtopics.
     No more hardcoded healthcare bias for non-medical queries.
"""

import logging
import re
from .state import ResearchState
from .llm_helper import llm_generate

logger = logging.getLogger(__name__)

DEFAULT_AGENT_PLAN = [
    "search_agent", "reader_agent", "critic_agent",
    "summary_agent", "knowledge_graph_agent", "artifact_agent"
]

HEALTHCARE_TERMS = {
    "clinical", "medical", "health", "patient", "hospital",
    "disease", "diagnosis", "treatment", "drug", "therapy",
    "ehr", "radiology", "pathology", "genomics", "biomedical",
    "healthcare", "surgery", "imaging", "cancer", "diabetes",
    "diabetic", "tumor", "lesion", "mammography", "retinopathy",
    "covid", "pandemic", "vaccine", "pharmaceutical"
}


def planner_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: PlannerAgent."""
    query   = state["query"]
    filters = state.get("filters", {})

    logger.info(f"[PlannerAgent] Planning for: '{query}'")

    is_healthcare = _is_healthcare_query(query)
    subtopics     = _generate_subtopics(query, is_healthcare)
    agent_plan    = list(DEFAULT_AGENT_PLAN)

    logger.info(f"[PlannerAgent] Domain: "
                f"{'healthcare' if is_healthcare else 'general ML/AI'}")
    logger.info(f"[PlannerAgent] Subtopics: {subtopics}")

    return {**state, "subtopics": subtopics, "agent_plan": agent_plan}


def _is_healthcare_query(query: str) -> bool:
    """Check if query is healthcare/medical domain."""
    q = query.lower()
    return any(term in q for term in HEALTHCARE_TERMS)


def _generate_subtopics(query: str, is_healthcare: bool) -> list[str]:
    """Generate focused, domain-appropriate subtopics."""

    if is_healthcare:
        areas = (
            "technical methods and architectures, "
            "clinical datasets and benchmarks, "
            "clinical applications, "
            "evaluation metrics, "
            "limitations and safety concerns"
        )
    else:
        areas = (
            "core methods and algorithms, "
            "benchmark datasets, "
            "applications and use cases, "
            "evaluation metrics, "
            "limitations and open problems"
        )

    prompt = f"""You are a research planning assistant.

Research query: "{query}"

Generate exactly 4 specific subtopics that are DIRECTLY related to this query.
Focus areas: {areas}

IMPORTANT RULES:
- Every subtopic MUST be about: "{query}"
- Do NOT add topics from unrelated domains
- Keep each subtopic under 7 words
- Be specific, not generic

Output ONLY a numbered list:
1. <subtopic>
2. <subtopic>
3. <subtopic>
4. <subtopic>"""

    response  = llm_generate(prompt, temperature=0.1, max_tokens=150)
    subtopics = _parse_list(response)

    if len(subtopics) >= 3:
        return subtopics[:4]
    return _fallback_subtopics(query)


def _parse_list(text: str) -> list[str]:
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        m = re.match(r"^\s*[\d\-\*\.]+\s*(.+)$", line.strip())
        if m:
            item = m.group(1).strip().strip(".")
            if len(item) > 3:
                items.append(item)
    return items


def _fallback_subtopics(query: str) -> list[str]:
    return [
        f"Methods for {query}",
        f"Datasets used in {query}",
        f"Evaluation of {query}",
        f"Limitations of {query}"
    ]


def refine_plan(state: ResearchState, cmd: str) -> ResearchState:
    """Update filters based on natural language refinement."""
    command = cmd.lower()
    filters = dict(state.get("filters", {}))

    if "after" in command:
        m = re.search(r"after\s+(\d{4})", command)
        if m:
            filters["year_after"] = int(m.group(1))

    if "exclude survey" in command or "no survey" in command:
        filters["exclude_surveys"] = True

    if "drill" in command or "deeper" in command:
        filters["max_papers"] = 12

    logger.info(f"[PlannerAgent] Updated filters: {filters}")
    return {**state, "filters": filters}