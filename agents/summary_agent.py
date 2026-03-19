"""
SummaryAgent — Deep analytical research report.

Fixes:
  1. Strips ** from all LLM output
  2. Cross-paper comparison in methods section
  3. Contradiction detection prompt
  4. Research gap detector
  5. More specific, less generic content
  6. Robust multi-pattern parser
"""

import logging
import re
from .state import ResearchState
from .llm_helper import llm_generate

logger     = logging.getLogger(__name__)
MAX_PAPERS = 7


def summary_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: SummaryAgent."""
    ranked_papers = state.get("ranked_papers", [])
    query         = state["query"]

    if not ranked_papers:
        logger.warning("[SummaryAgent] No papers.")
        return {**state, "insights": _empty_insights(query)}

    top_papers = ranked_papers[:MAX_PAPERS]
    logger.info(
        f"[SummaryAgent] Generating analytical report for '{query}' "
        f"({len(top_papers)} papers)..."
    )

    context  = _build_rich_context(top_papers)
    insights = _generate_analytical_report(
        query, context, len(ranked_papers), top_papers
    )

    logger.info("[SummaryAgent] Done.")
    return {**state, "insights": insights}


def _build_rich_context(papers: list[dict]) -> str:
    """Build structured context for cross-paper analysis."""
    parts = []
    for i, p in enumerate(papers, 1):
        ins   = p.get("insights", {})
        year  = p.get("published","")[:4]
        title = p.get("title","")[:60]
        score = p.get("final_score", 0)
        cite  = p.get("citation_count", 0)
        conf  = ins.get("confidence", {})

        # Only include fields that were actually found
        method  = ins.get("methodology","")
        dataset = ins.get("datasets","")
        metrics = ins.get("evaluation_metrics", ins.get("metrics",""))
        contrib = ins.get("key_contributions","")
        problem = ins.get("problem_statement", ins.get("problem",""))
        limit   = ins.get("limitations","")

        part = f"PAPER {i}: {title} ({year})\n"
        part += f"  Citations: {cite} | Score: {score:.2f}\n"

        if problem and "Not" not in problem:
            part += f"  Problem: {problem[:120]}\n"
        if method and "Not" not in method:
            part += f"  Method: {method[:100]}\n"
        if dataset and "Not" not in dataset:
            part += f"  Data: {dataset[:80]}\n"
        if metrics and "Not" not in metrics:
            part += f"  Metrics: {metrics[:80]}\n"
        if contrib and "Not" not in contrib:
            part += f"  Contribution: {contrib[:120]}\n"
        if limit and "Not" not in limit:
            part += f"  Limitations: {limit[:100]}\n"

        parts.append(part)

    return "\n".join(parts)[:3000]


def _generate_analytical_report(
    query: str,
    context: str,
    total: int,
    papers: list[dict]
) -> dict:
    """
    Generate deep analytical report — not a generic summary.
    Focuses on comparisons, contradictions, and gaps.
    """
    # Build method comparison for the prompt
    methods_list = []
    for i, p in enumerate(papers, 1):
        ins    = p.get("insights",{})
        method = ins.get("methodology","")
        if method and "Not" not in method:
            methods_list.append(f"PAPER {i}: {method[:60]}")

    methods_str = "\n".join(methods_list) if methods_list else "See papers above"

    prompt = f"""You are a senior research analyst. Write a deep analytical report.

Topic: "{query}"
Papers analyzed: {total}

Paper details:
{context}

Methods used across papers:
{methods_str}

Write an analytical research report using EXACTLY these section markers.
Each section must be ANALYTICAL not descriptive — compare, contrast, critique.

SECTION 1 - BACKGROUND:
[3-4 sentences: What is the research problem? Why does it matter?
Reference specific papers. What gap in knowledge do they address?]

SECTION 2 - KEY METHODS:
[Compare methods across papers. How do they differ?
Which approach is most innovative? Are any methods better than others?
Use bullet points: - Method: Paper X uses Y, Paper Z uses W. Key difference: ...]

SECTION 3 - DATASETS:
[What data was used? Are datasets shared across papers?
Any notable gaps in data coverage?
Bullet points: - Dataset: used by Paper X for purpose Y]

SECTION 4 - EVALUATION METRICS:
[What metrics are used? Are they consistent across papers?
Are the evaluation approaches comparable?
Bullet points: - Metric: used by Paper X, measures Y]

SECTION 5 - LIMITATIONS:
[What are the common weaknesses?
Do any papers contradict each other?
What assumptions might be flawed?]

SECTION 6 - RESEARCH GAPS:
[What specific questions remain unanswered?
What did these papers NOT address?
Number each gap: 1. Gap: ... Evidence: Paper X did not...]

SECTION 7 - FUTURE DIRECTIONS:
[Concrete next steps grounded in the paper findings.
What would logically follow from this work?
Number each: 1. Direction: ... Based on: Paper X showed...]

Be specific. Name papers by number. Avoid generic statements.
Do NOT use ** markdown markers."""

    response = llm_generate(prompt, temperature=0.2, max_tokens=2000)

    if not response or "[LLM unavailable" in response:
        return _fallback_insights(query)

    # Clean ** markers from entire response
    response = _clean_response(response)

    return _parse_sections(response, query, total)


def _clean_response(text: str) -> str:
    """Strip all ** markers and clean formatting."""
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _parse_sections(
    response: str,
    query: str,
    total: int
) -> dict:
    """Multi-pattern robust parser."""
    sections = {
        "background":         "",
        "key_methods":        "",
        "common_datasets":    "",
        "evaluation_metrics": "",
        "limitations":        "",
        "research_gaps":      "",
        "future_directions":  ""
    }

    patterns = {
        "background": [
            r"SECTION\s*1\s*[-–:]\s*BACKGROUND[:\s]*(.*?)(?=SECTION\s*2|$)",
            r"BACKGROUND[:\s]+(.*?)(?=KEY\s*METHODS|SECTION\s*2|$)",
        ],
        "key_methods": [
            r"SECTION\s*2\s*[-–:]\s*KEY\s*METHODS[:\s]*(.*?)(?=SECTION\s*3|$)",
            r"KEY\s*METHODS[:\s]+(.*?)(?=DATASETS|SECTION\s*3|$)",
        ],
        "common_datasets": [
            r"SECTION\s*3\s*[-–:]\s*DATASETS[:\s]*(.*?)(?=SECTION\s*4|$)",
            r"DATASETS[:\s]+(.*?)(?=EVALUATION|SECTION\s*4|$)",
        ],
        "evaluation_metrics": [
            r"SECTION\s*4\s*[-–:]\s*EVALUATION\s*METRICS[:\s]*(.*?)(?=SECTION\s*5|$)",
            r"EVALUATION\s*METRICS[:\s]+(.*?)(?=LIMITATIONS|SECTION\s*5|$)",
        ],
        "limitations": [
            r"SECTION\s*5\s*[-–:]\s*LIMITATIONS[:\s]*(.*?)(?=SECTION\s*6|$)",
            r"LIMITATIONS[:\s]+(.*?)(?=RESEARCH\s*GAPS|SECTION\s*6|$)",
        ],
        "research_gaps": [
            r"SECTION\s*6\s*[-–:]\s*RESEARCH\s*GAPS[:\s]*(.*?)(?=SECTION\s*7|$)",
            r"RESEARCH\s*GAPS[:\s]+(.*?)(?=FUTURE|SECTION\s*7|$)",
        ],
        "future_directions": [
            r"SECTION\s*7\s*[-–:]\s*FUTURE\s*DIRECTIONS[:\s]*(.*?)(?=$)",
            r"FUTURE\s*DIRECTIONS[:\s]+(.*?)(?=$)",
        ]
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(
                pattern, response, re.DOTALL | re.IGNORECASE
            )
            if match:
                content = match.group(1).strip()
                # Clean markers again just in case
                content = re.sub(r'\*\*', '', content).strip()
                if len(content) > 20:
                    sections[key] = content
                    break

    filled = sum(1 for v in sections.values() if len(v) > 20)
    logger.info(f"[SummaryAgent] Parsed {filled}/7 sections.")

    if filled < 3:
        logger.warning("[SummaryAgent] Low parse success. Using split.")
        sections = _split_approach(response, sections)

    if not sections["background"] and len(response) > 50:
        sections["background"] = _clean_response(response[:600])

    return {
        "topic":                 query,
        "total_papers_analyzed": total,
        **sections
    }


def _split_approach(response: str, sections: dict) -> dict:
    """Fallback split parser."""
    parts = re.split(
        r'(?:SECTION\s*\d+\s*[-–:]?\s*|^\d+[\.\)]\s*)',
        response,
        flags=re.MULTILINE | re.IGNORECASE
    )
    key_order = [
        "background", "key_methods", "common_datasets",
        "evaluation_metrics", "limitations",
        "research_gaps", "future_directions"
    ]
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    for i, part in enumerate(parts[:7]):
        if i < len(key_order) and not sections[key_order[i]]:
            sections[key_order[i]] = re.sub(r'\*\*','',part).strip()
    return sections


def _fallback_insights(query: str) -> dict:
    return {
        "topic":                 query,
        "total_papers_analyzed": 0,
        "background":            f"Research on '{query}' is an active field.",
        "key_methods":           "See individual paper details.",
        "common_datasets":       "See individual paper details.",
        "evaluation_metrics":    "See individual paper details.",
        "limitations":           "See individual paper details.",
        "research_gaps":         "Insufficient data to identify gaps.",
        "future_directions":     "See individual paper details."
    }


def _empty_insights(query: str = "") -> dict:
    return {
        "topic":                 query,
        "total_papers_analyzed": 0,
        "background":            "No papers retrieved.",
        "key_methods":           "",
        "common_datasets":       "",
        "evaluation_metrics":    "",
        "limitations":           "",
        "research_gaps":         "",
        "future_directions":     ""
    }