"""
Report Generator — research_report.json
Updated to include cluster themes and research timeline.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from agents.state import ResearchState
from analysis.research_trends import format_timeline_display

logger     = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/artifacts")


def generate_report(state: ResearchState) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    insights         = state.get("insights", {})
    ranked_papers    = state.get("ranked_papers", [])
    cluster_info     = state.get("cluster_info", [])
    research_trends  = state.get("research_trends", {})
    query            = state.get("query", "")
    session_id       = state.get("session_id", "")
    top_papers       = ranked_papers[:10]

    report = {
        "meta": {
            "session_id":            session_id,
            "query":                 query,
            "generated_at":          datetime.utcnow().isoformat(),
            "total_papers_analyzed": len(ranked_papers),
            "papers_in_report":      len(top_papers)
        },
        "topic":      insights.get("topic", query),
        "background": insights.get("background", ""),
        "key_methods":   _parse_list(insights.get("key_methods", "")),
        "datasets":      _parse_list(insights.get("common_datasets", "")),
        "metrics":       _parse_list(insights.get("evaluation_metrics", "")),
        "limitations":   insights.get("limitations", ""),
        "research_gaps": _parse_list(insights.get("research_gaps", "")),
        "future_work":   _parse_list(insights.get("future_directions", "")),

        # NEW: Research theme clusters
        "research_themes": {
            "total_clusters": len(cluster_info),
            "clusters": [
                {
                    "cluster_id":   c.get("cluster_id"),
                    "theme":        c.get("theme"),
                    "paper_count":  c.get("paper_count"),
                    "paper_titles": c.get("paper_titles", [])
                }
                for c in cluster_info
            ]
        },

        # NEW: Research timeline and trends
        "research_trends": {
            "timeline":         research_trends.get("timeline", {}),
            "timeline_display": format_timeline_display(
                                    research_trends.get("timeline", {})),
            "trend_summary":    research_trends.get("trend_summary", ""),
            "peak_year":        research_trends.get("peak_year", 0),
            "growth_rate":      research_trends.get("growth_rate", 0),
            "maturity_status":  research_trends.get("maturity_status", "unknown"),
            "year_range":       research_trends.get("year_range", (0, 0))
        },

        "top_papers": [
            {
                "rank":               i+1,
                "paper_id":           p.get("paper_id", ""),
                "title":              p.get("title", ""),
                "authors":            p.get("authors", [])[:3],
                "published":          p.get("published", ""),
                "final_score":        p.get("final_score", 0),
                "hybrid_score":       p.get("hybrid_score", 0),
                "citation_count":     p.get("citation_count", 0),
                "venue":              p.get("venue", ""),
                "arxiv_url":          p.get("arxiv_url", ""),
                "cluster_id":         p.get("cluster_id", 0),
                "cluster_theme":      p.get("cluster_theme", ""),
                "score_breakdown":    p.get("score_breakdown", {}),
                "strengths":          p.get("strengths", []),
                "weaknesses":         p.get("weaknesses", []),
                # Deep insights
                "insights": {
                    "problem_statement":  p.get("insights", {}).get("problem_statement", ""),
                    "methodology":        p.get("insights", {}).get("methodology", ""),
                    "datasets":           p.get("insights", {}).get("datasets", ""),
                    "evaluation_metrics": p.get("insights", {}).get("evaluation_metrics", ""),
                    "key_contributions":  p.get("insights", {}).get("key_contributions", ""),
                    "limitations":        p.get("insights", {}).get("limitations", ""),
                    "future_work":        p.get("insights", {}).get("future_work", "")
                }
            }
            for i, p in enumerate(top_papers)
        ],

        "knowledge_graph": {
            "total_entities": len(state.get("knowledge_graph_entities", [])),
            "total_edges":    len(state.get("knowledge_graph_edges", [])),
            "entities":       state.get("knowledge_graph_entities", [])[:30],
            "edges":          state.get("knowledge_graph_edges", [])[:50]
        }
    }

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    file_path   = OUTPUT_DIR / f"research_report_{session_id[:8]}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_json)

    logger.info(f"[ReportGenerator] Saved: {file_path}")
    return report_json


def _parse_list(text: str) -> list[str]:
    if not text:
        return []
    items = []
    for line in text.strip().split("\n"):
        clean = line.strip().lstrip("•-*0123456789.)> ").strip()
        if len(clean) > 3:
            items.append(clean)
    return items