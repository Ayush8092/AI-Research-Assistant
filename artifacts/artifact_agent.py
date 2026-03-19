"""
Artifact Agent — LangGraph node wrapping all three generators.
"""

import logging
from agents.state import ResearchState
from .report_generator import generate_report
from .bibtex_generator import generate_bibtex
from .related_work_generator import generate_related_work

logger = logging.getLogger(__name__)


def artifact_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: ArtifactAgent."""
    logger.info("[ArtifactAgent] Generating artifacts...")
    artifacts = state.get("artifacts", {})

    try:
        artifacts["report"]      = generate_report(state)
        logger.info("[ArtifactAgent] ✓ report")
    except Exception as e:
        logger.error(f"[ArtifactAgent] Report failed: {e}")
        artifacts["report"] = "{}"

    try:
        artifacts["bibtex"]      = generate_bibtex(state)
        logger.info("[ArtifactAgent] ✓ bibtex")
    except Exception as e:
        logger.error(f"[ArtifactAgent] BibTeX failed: {e}")
        artifacts["bibtex"] = "% generation failed"

    try:
        artifacts["related_work"] = generate_related_work(state)
        logger.info("[ArtifactAgent] ✓ related_work")
    except Exception as e:
        logger.error(f"[ArtifactAgent] Related work failed: {e}")
        artifacts["related_work"] = "# Related Work\n\nGeneration failed."

    return {**state, "artifacts": artifacts}