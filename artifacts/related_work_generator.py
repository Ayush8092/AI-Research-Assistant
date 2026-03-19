"""
Related Work Generator — related_work.md

Fixes:
  - Detailed MD report with full academic narrative
  - Per-paper analysis section
  - Comparison table
  - No fake citations — only provided papers
  - Top 7 papers only
"""

import logging
import re
from pathlib import Path
from agents.state import ResearchState
from agents.llm_helper import llm_generate
from .bibtex_generator import _make_cite_key

logger     = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/artifacts")
MAX_PAPERS = 7


def generate_related_work(state: ResearchState) -> str:
    """Generate detailed related_work.md."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ranked_papers = state.get("ranked_papers", [])
    insights_data = state.get("insights", {})
    query         = state.get("query", "")
    session_id    = state.get("session_id", "")
    filters       = state.get("filters", {})

    if not filters.get("generate_related_work", False):
        return _generate_paper_list_md(query, ranked_papers)

    if not ranked_papers:
        return "# Related Work\n\nNo papers found.\n"

    top_papers = ranked_papers[:MAX_PAPERS]
    cite_map   = {p["paper_id"]: _make_cite_key(p) for p in top_papers}

    md_content = _build_detailed_md(query, top_papers, insights_data, cite_map)

    file_path = OUTPUT_DIR / f"related_work_{session_id[:8]}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"[RelatedWork] Saved: {file_path}")
    return md_content


def _generate_paper_list_md(query: str, papers: list[dict]) -> str:
    """
    Detailed paper list even when full generation is skipped.
    Much more useful than just titles.
    """
    lines = [
        f"# Related Work: {query}\n",
        f"> *Enable 'Generate Related Work' for full narrative. "
        f"Below is a structured paper summary.*\n",
        f"## Papers Analyzed ({len(papers[:MAX_PAPERS])})\n"
    ]

    for i, p in enumerate(papers[:MAX_PAPERS], 1):
        ins    = p.get("insights", {})
        year   = p.get("published", "")[:4]
        title  = p.get("title", "")
        authors = ", ".join(p.get("authors", [])[:2])
        if len(p.get("authors", [])) > 2:
            authors += " et al."
        score  = p.get("final_score", 0)
        cite   = p.get("citation_count", 0)
        url    = p.get("arxiv_url", "")
        theme  = p.get("cluster_theme", "")

        lines.append(f"### {i}. {title}")
        lines.append(f"**Authors:** {authors} | **Year:** {year} | "
                     f"**Citations:** {cite} | **Score:** {score:.3f}")
        if theme:
            lines.append(f"**Theme:** {theme}")
        if url:
            lines.append(f"**URL:** [{url}]({url})")
        lines.append("")

        problem = ins.get("problem_statement", ins.get("problem", ""))
        if problem and "Not" not in problem:
            lines.append(f"**Problem:** {problem}")

        method = ins.get("methodology", "")
        if method and "Not" not in method:
            lines.append(f"**Method:** {method}")

        datasets = ins.get("datasets", "")
        if datasets and "Not" not in datasets:
            lines.append(f"**Datasets:** {datasets}")

        metrics = ins.get("evaluation_metrics", ins.get("metrics", ""))
        if metrics and "Not" not in metrics:
            lines.append(f"**Metrics:** {metrics}")

        contrib = ins.get("key_contributions", "")
        if contrib and "Not" not in contrib:
            lines.append(f"**Contribution:** {contrib}")

        limit = ins.get("limitations", "")
        if limit and "Not" not in limit:
            lines.append(f"**Limitations:** {limit}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_detailed_md(
    query: str,
    papers: list[dict],
    insights_data: dict,
    cite_map: dict
) -> str:
    """Build the full detailed related work document."""

    # Generate narrative
    narrative = _generate_narrative(query, papers, cite_map)

    # Build comparison table
    comparison_table = _build_comparison_table(papers)

    # Build per-paper analysis
    per_paper = _build_per_paper_analysis(papers)

    # Reference list
    references = _build_references(papers)

    return f"""# Related Work: {query}

> *Auto-generated literature review from {len(papers)} papers.*
> *Verify all citations before submission.*

---

## 1. Overview

{insights_data.get('background', f'This section reviews existing literature on {query}.')}

---

## 2. Research Landscape

{narrative}

---

## 3. Methods Comparison

{comparison_table}

---

## 4. Per-Paper Analysis

{per_paper}

---

## 5. Research Gaps Identified

{_fmt_bullets(insights_data.get('research_gaps', ''))}

---

## 6. Future Directions

{_fmt_bullets(insights_data.get('future_directions', ''))}

---

## References

{references}
"""


def _generate_narrative(query: str, papers: list[dict], cite_map: dict) -> str:
    """Generate academic narrative paragraphs."""
    summaries = []
    for p in papers[:7]:
        key      = cite_map.get(p["paper_id"], "?")
        year     = p.get("published", "")[:4]
        ins      = p.get("insights", {})
        problem  = ins.get("problem_statement", ins.get("problem", ""))[:100]
        method   = ins.get("methodology", "")[:80]
        summaries.append(
            f"[{key}] ({year}): {problem} "
            f"Approach: {method}."
        )

    prompt = f"""Write a detailed 4-paragraph related work section for "{query}".

Available papers (use ONLY these):
{chr(10).join(summaries)}

RULES:
- Use \\cite{{key}} for citations — only use keys from the list above
- NO invented citations
- Each paragraph must have a distinct focus:
  Para 1: Foundational and early approaches
  Para 2: Recent advances (2022+) and state-of-the-art
  Para 3: Comparison of methods — what works, what doesn't
  Para 4: Limitations and open challenges from the literature
- Write 4-6 sentences per paragraph
- Academic tone, specific claims, no generic statements"""

    response = llm_generate(prompt, temperature=0.3, max_tokens=700)

    if "[LLM unavailable" in response or not response.strip():
        return _fallback_narrative(query, papers, cite_map)

    return response.strip()


def _clean_text(text: str) -> str:
    """Remove ** markers and clean text for MD output."""
    if not text:
        return ""
    text = re.sub(r'\*\*', '', str(text))
    text = re.sub(r'\*', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    return text


def _build_comparison_table(papers: list[dict]) -> str:
    """Clean comparison table — no ** markers, no overflow."""
    header = "| # | Paper | Year | Method | Dataset | Metrics | Citations |\n"
    sep    = "|---|-------|------|--------|---------|---------|----------|\n"
    rows   = []

    for i, p in enumerate(papers, 1):
        ins     = p.get("insights", {})
        title   = _clean_text(p.get("title",""))[:32] + "..."
        year    = p.get("published","")[:4]

        method  = _clean_text(ins.get("methodology",""))
        method  = "N/A" if "Not" in method or not method else method[:28]

        dataset = _clean_text(ins.get("datasets",""))
        dataset = "N/A" if "Not" in dataset or not dataset else dataset[:22]

        metrics = _clean_text(ins.get("evaluation_metrics", ins.get("metrics","")))
        metrics = "N/A" if "Not" in metrics or not metrics else metrics[:22]

        cite    = p.get("citation_count", 0)
        rows.append(
            f"| {i} | {title} | {year} | {method} | "
            f"{dataset} | {metrics} | {cite} |"
        )

    return header + sep + "\n".join(rows)
def _build_per_paper_analysis(papers: list[dict]) -> str:
    """Build per-paper detailed analysis section."""
    sections = []
    for i, p in enumerate(papers, 1):
        ins     = p.get("insights", {})
        title   = p.get("title", "")
        authors = ", ".join(p.get("authors", [])[:2])
        if len(p.get("authors", [])) > 2:
            authors += " et al."
        year    = p.get("published", "")[:4]
        url     = p.get("arxiv_url", "")
        score   = p.get("final_score", 0)
        cite    = p.get("citation_count", 0)
        theme   = p.get("cluster_theme", "")

        section = [
            f"### {i}. {title}",
            f"*{authors} ({year})* | Score: {score:.3f} | "
            f"Citations: {cite} | Theme: {theme}",
        ]
        if url:
            section.append(f"[{url}]({url})\n")

        for field, label in [
            ("problem_statement", "**Problem:**"),
            ("methodology",       "**Methodology:**"),
            ("datasets",          "**Datasets:**"),
            ("evaluation_metrics","**Metrics:**"),
            ("key_contributions", "**Key Contributions:**"),
            ("limitations",       "**Limitations:**"),
            ("future_work",       "**Future Work:**"),
        ]:
            val = ins.get(field, ins.get("problem" if field == "problem_statement"
                                         else "metrics" if field == "evaluation_metrics"
                                         else field, ""))
            if val and "Not" not in str(val):
                section.append(f"{label} {val}")

        sections.append("\n".join(section))

    return "\n\n---\n\n".join(sections)


def _build_references(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        year  = p.get("published", "")[:4]
        title = p.get("title", "")
        url   = p.get("arxiv_url", "")
        key   = _make_cite_key(p)
        lines.append(f"[{key}] {authors} ({year}). *{title}*. {url}")
    return "\n\n".join(lines)


def _fallback_narrative(query, papers, cite_map) -> str:
    recent = [p for p in papers if int(p.get("published","2020")[:4]) >= 2022]
    older  = [p for p in papers if int(p.get("published","2020")[:4]) < 2022]
    p1 = "Early work established foundational approaches " + " ".join(
        f"\\cite{{{cite_map.get(p['paper_id'],'')}}} " for p in older[:3]) + "."
    p2 = "Recent advances have shown significant improvements " + " ".join(
        f"\\cite{{{cite_map.get(p['paper_id'],'')}}} " for p in recent[:3]) + "."
    return f"{p1}\n\n{p2}\n\nDespite progress, important challenges remain open."


def _fmt_bullets(text: str) -> str:
    if not text:
        return "*Not identified.*"
    lines   = text.strip().split("\n")
    bullets = [
        f"- {line.strip().lstrip('•*-0123456789.)> ').strip()}"
        for line in lines
        if len(line.strip().lstrip('•*-0123456789.)> ').strip()) > 3
    ]
    return "\n".join(bullets) if bullets else "*Not identified.*"