"""
CriticAgent — Enhanced ranking with citation velocity and paper significance.

New features:
  1. Citation velocity (citations per year) — rewards recent impact
  2. Paper type classification (Core Systems vs Applications)
  3. "Why This Paper Matters" — 1-2 line significance statement
  4. Better venue tier detection
"""

import logging
import math
import re
from datetime import datetime
from .state import ResearchState
from utils.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)

W_CITATION  = 0.35
W_RECENCY   = 0.20
W_VENUE     = 0.10
W_RELEVANCE = 0.35

CURRENT_YEAR = datetime.now().year

VENUE_SCORES = {
    "high":    1.0,
    "medium":  0.6,
    "low":     0.3,
    "unknown": 0.15
}

# ------------------------------------------------------------------ #
#  Paper Type Classification                                           #
# ------------------------------------------------------------------ #

CORE_SYSTEMS_KEYWORDS = [
    "framework", "architecture", "coordination", "planning",
    "reasoning", "communication", "protocol", "orchestration",
    "agent design", "multi-agent system", "agent collaboration",
    "agent coordination", "llm agent", "autonomous agent",
    "agent framework", "tool use", "tool calling", "agent workflow",
    "hierarchical", "decentralized", "consensus", "negotiation",
    "emergent", "self-play", "role-playing", "reflection",
    "memory", "retrieval", "chain-of-thought", "prompt engineering",
    "benchmark", "evaluation framework", "task decomposition"
]

APPLICATION_KEYWORDS = [
    "clinical", "medical", "health", "patient", "hospital",
    "disease", "diagnosis", "drug", "protein", "genomics",
    "biology", "chemistry", "recommendation", "e-commerce",
    "finance", "legal", "education", "robotics", "manufacturing",
    "cybersecurity", "supply chain", "customer service",
    "alzheimer", "cancer", "covid", "diabetes", "radiology"
]


def classify_paper_type(paper: dict) -> str:
    """
    Classify paper as Core Systems research or Application.

    Core Systems: focuses on agent design, framework, architecture,
                  coordination methods, benchmarks
    Application:  uses multi-agent LLMs for a specific domain problem
    """
    title    = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    text     = title + " " + abstract

    core_score = sum(1 for kw in CORE_SYSTEMS_KEYWORDS if kw in text)
    app_score  = sum(1 for kw in APPLICATION_KEYWORDS if kw in text)

    # Title match is stronger signal — weight it 3x
    title_core = sum(3 for kw in CORE_SYSTEMS_KEYWORDS if kw in title)
    title_app  = sum(3 for kw in APPLICATION_KEYWORDS if kw in title)

    total_core = core_score + title_core
    total_app  = app_score  + title_app

    if total_core == 0 and total_app == 0:
        return "Core Systems"   # default for ambiguous papers
    if total_core >= total_app:
        return "Core Systems"
    return "Application"


# ------------------------------------------------------------------ #
#  Main Agent                                                          #
# ------------------------------------------------------------------ #

def critic_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: CriticAgent with velocity scoring and classification."""
    processed_papers = state.get("processed_papers", [])
    query            = state["query"]

    if not processed_papers:
        logger.warning("[CriticAgent] No papers.")
        return {**state, "ranked_papers": []}

    logger.info(
        f"[CriticAgent] Ranking {len(processed_papers)} papers..."
    )

    oa_client   = OpenAlexClient()
    paper_dicts = [
        {"paper_id": p["paper_id"], "doi": p.get("doi",""), "title": p["title"]}
        for p in processed_papers
    ]
    oa_metadata = oa_client.fetch_batch(paper_dicts)

    # Collect all citations for normalization
    all_citations = []
    for p in processed_papers:
        cite = _get_citation_count(p, oa_metadata)
        all_citations.append(cite)

    query_keywords = _extract_keywords(query)
    ranked         = []

    for paper in processed_papers:
        pid        = paper["paper_id"]
        oa_meta    = oa_metadata.get(pid)
        year       = _extract_year(paper.get("published",""))
        cite_count = _get_citation_count(paper, oa_metadata)

        # Scoring components
        citation_score  = _year_adjusted_citation_score(
            cite_count, year, all_citations
        )
        velocity_score  = _citation_velocity_score(cite_count, year)
        recency_score   = _recency_score(paper.get("published",""))
        venue_score     = _venue_score(oa_meta, paper)
        relevance_score = _relevance_score(paper, query_keywords)

        # Combined citation score: 70% absolute, 30% velocity
        combined_citation = (0.70 * citation_score + 0.30 * velocity_score)

        final_score = (
            W_CITATION  * combined_citation +
            W_RECENCY   * recency_score     +
            W_VENUE     * venue_score       +
            W_RELEVANCE * relevance_score
        )

        # Paper type classification
        paper_type = classify_paper_type(paper)

        # Why This Paper Matters
        significance = _generate_significance(
            paper, cite_count, year, venue_score, paper_type, query
        )

        score_breakdown = {
            "citation_score":   round(combined_citation, 3),
            "recency_score":    round(recency_score,     3),
            "venue_score":      round(venue_score,       3),
            "relevance_score":  round(relevance_score,   3),
            "final_score":      round(final_score,       3),
            "citation_count":   cite_count,
            "citation_velocity": round(
                cite_count / max(CURRENT_YEAR - year, 1), 1
            ),
            "venue":     oa_meta.venue if oa_meta else
                         paper.get("journal_ref",""),
            "venue_tier": oa_meta.venue_tier if oa_meta else "unknown"
        }

        strengths, weaknesses = _generate_sw(paper, score_breakdown)

        ranked.append({
            **paper,
            "final_score":     round(final_score, 4),
            "score_breakdown": score_breakdown,
            "strengths":       strengths,
            "weaknesses":      weaknesses,
            "citation_count":  cite_count,
            "venue":           score_breakdown["venue"],
            "paper_type":      paper_type,
            "significance":    significance
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    scores  = [p["final_score"] for p in ranked]
    metrics = state.get("metrics", {})
    metrics.update({
        "papers_selected": len(ranked),
        "score_mean": round(sum(scores)/len(scores), 4) if scores else 0,
        "score_std":  round(_std(scores), 4) if scores else 0,
        "core_papers": sum(1 for p in ranked if p.get("paper_type") == "Core Systems"),
        "app_papers":  sum(1 for p in ranked if p.get("paper_type") == "Application")
    })

    logger.info(
        f"[CriticAgent] Top: {ranked[0]['title'][:60] if ranked else 'N/A'}"
    )
    logger.info(
        f"[CriticAgent] Core: {metrics['core_papers']} | "
        f"Applications: {metrics['app_papers']}"
    )

    return {**state, "ranked_papers": ranked, "metrics": metrics}


# ------------------------------------------------------------------ #
#  Citation Velocity                                                   #
# ------------------------------------------------------------------ #

def _citation_velocity_score(citations: int, year: int) -> float:
    """
    Citations per year score — rewards recent high-impact papers.

    A 2024 paper with 30 citations (30 cites/year) scores higher than
    a 2018 paper with 50 citations (8 cites/year).
    """
    age      = max(CURRENT_YEAR - year, 1)
    velocity = citations / age   # citations per year

    # Normalize: 20 citations/year = perfect score
    return min(velocity / 20.0, 1.0)


def _get_citation_count(paper: dict, oa_metadata: dict) -> int:
    """Get citation count from S2 or OpenAlex."""
    pid = paper["paper_id"]
    if paper.get("source") == "semantic_scholar":
        return paper.get("citation_count", 0)
    oa_meta = oa_metadata.get(pid)
    if oa_meta:
        return oa_meta.citation_count
    return paper.get("citation_count", 0)


def _year_adjusted_citation_score(
    citations: int,
    year: int,
    all_citations: list
) -> float:
    """Year-adjusted citation score."""
    if citations <= 0:
        age = CURRENT_YEAR - year
        return 0.2 if age <= 1 else 0.05

    age = CURRENT_YEAR - year
    if age <= 1:
        return min(citations / 10.0, 1.0)
    elif age <= 2:
        return min(citations / 30.0, 1.0)
    else:
        max_cite = max(max(all_citations), 1)
        return math.log1p(citations) / math.log1p(max_cite)


def _recency_score(published: str) -> float:
    try:
        age = CURRENT_YEAR - int(published[:4])
        return max(0.0, math.exp(-0.25 * age))
    except (ValueError, TypeError):
        return 0.1


def _venue_score(oa_meta, paper: dict) -> float:
    if oa_meta and oa_meta.venue_tier:
        return VENUE_SCORES.get(oa_meta.venue_tier, 0.15)
    journal = paper.get("journal_ref","") or paper.get("venue","")
    if journal and len(journal.strip()) > 3:
        return VENUE_SCORES["low"]
    return VENUE_SCORES["unknown"]


def _relevance_score(paper: dict, query_keywords: set) -> float:
    title    = paper.get("title","").lower()
    abstract = paper.get("abstract","").lower()
    text     = title + " " + abstract

    if query_keywords:
        text_words    = set(re.findall(r'\b\w+\b', text))
        overlap       = len(query_keywords & text_words)
        keyword_score = min(overlap / max(len(query_keywords), 1), 1.0)
    else:
        keyword_score = 0.5

    title_words  = set(re.findall(r'\b\w+\b', title))
    title_score  = min(
        len(query_keywords & title_words) / max(len(query_keywords), 1),
        1.0
    )
    semantic_score = paper.get("semantic_similarity", 0.0)
    anchor_score   = paper.get("anchor_score", 0.0)

    return round(
        0.30 * keyword_score  +
        0.30 * semantic_score +
        0.20 * title_score    +
        0.20 * anchor_score,
        4
    )


# ------------------------------------------------------------------ #
#  Why This Paper Matters                                             #
# ------------------------------------------------------------------ #

def _generate_significance(
    paper: dict,
    citations: int,
    year: int,
    venue_score: float,
    paper_type: str,
    query: str
) -> str:
    """
    Generate a 1-2 sentence significance statement for each paper.
    Based on citations, recency, venue, type, and query relevance.
    No LLM — fast rule-based generation.
    """
    title    = paper.get("title","")
    age      = CURRENT_YEAR - year
    velocity = citations / max(age, 1)

    parts = []

    # Impact statement
    if citations > 100:
        parts.append(
            f"Highly influential work with {citations} citations, "
            f"indicating broad community adoption."
        )
    elif citations > 30:
        parts.append(
            f"Well-cited paper ({citations} citations) demonstrating "
            f"significant research impact."
        )
    elif velocity > 15:
        parts.append(
            f"Rapidly gaining traction with {velocity:.0f} citations/year, "
            f"suggesting strong recent relevance."
        )
    elif age <= 1 and citations > 0:
        parts.append(
            f"Very recent publication ({year}) with early citations, "
            f"representing cutting-edge work."
        )
    elif age <= 1:
        parts.append(
            f"Among the most recent papers ({year}) in this area, "
            f"representing the current state of research."
        )

    # Type-based significance
    if paper_type == "Core Systems":
        parts.append(
            f"Contributes foundational methods or frameworks "
            f"directly relevant to {query[:50]}."
        )
    else:
        domain = _detect_application_domain(
            paper.get("title","") + " " + paper.get("abstract","")
        )
        parts.append(
            f"Demonstrates real-world application of these methods "
            f"in {domain}."
        )

    return " ".join(parts[:2])


def _detect_application_domain(text: str) -> str:
    """Detect the application domain from paper text."""
    text = text.lower()
    domains = [
        ("protein",                   "computational biology"),
        ("drug",                      "drug discovery"),
        ("alzheimer",                 "neurology"),
        ("cancer",                    "oncology"),
        ("clinical",                  "clinical medicine"),
        ("medical",                   "medical research"),
        ("recommendation",            "recommendation systems"),
        ("finance",                   "financial technology"),
        ("robotics",                  "robotics"),
        ("education",                 "educational technology"),
        ("cybersecurity",             "cybersecurity"),
    ]
    for keyword, domain in domains:
        if keyword in text:
            return domain
    return "applied research"


def _generate_sw(paper, scores) -> tuple:
    strengths, weaknesses = [], []
    cite  = scores.get("citation_count", 0)
    vel   = scores.get("citation_velocity", 0)
    year  = _extract_year(paper.get("published",""))
    age   = CURRENT_YEAR - year
    ptype = paper.get("paper_type","")

    if cite > 50:
        strengths.append(f"Highly cited ({cite} citations)")
    elif cite > 10:
        strengths.append(f"Well cited ({cite} citations)")
    elif age <= 1 and cite > 0:
        strengths.append(f"New paper with {cite} early citations")

    if vel > 15:
        strengths.append(f"High citation velocity ({vel:.0f}/year)")

    if age <= 1:
        strengths.append("Very recent (2025)")
    elif age <= 2:
        strengths.append("Recent (2024)")

    tier = scores.get("venue_tier","unknown")
    if tier in ("high","medium"):
        strengths.append(
            f"Published in {scores.get('venue','')[:35]}"
        )

    if scores.get("relevance_score", 0) > 0.6:
        strengths.append("High relevance to query")

    if ptype == "Core Systems":
        strengths.append("Core systems/framework paper")

    if cite == 0 and age > 2:
        weaknesses.append("No citations found")
    if tier == "unknown":
        weaknesses.append("Venue not identified")
    if scores.get("relevance_score", 0) < 0.3:
        weaknesses.append("Lower relevance to query")
    if ptype == "Application":
        weaknesses.append("Application paper — not core systems research")

    return (
        strengths or ["Relevant to query"],
        weaknesses or ["No significant weaknesses"]
    )


def _extract_keywords(query: str) -> set:
    stop = {
        "in","for","of","the","a","an","and","or","on",
        "with","using","based","via","from","to","by","at",
        "is","are","was","be","it","its"
    }
    words = re.findall(r'\b\w+\b', query.lower())
    return {w for w in words if w not in stop and len(w) > 2}


def _extract_year(date_str: str) -> int:
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return 2020


def _std(values: list) -> float:
    if not values:
        return 0.0
    mean     = sum(values) / len(values)
    variance = sum((x - mean)**2 for x in values) / len(values)
    return math.sqrt(variance)