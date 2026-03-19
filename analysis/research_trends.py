"""
Research Trends
Temporal analysis of research papers by publication year.

Fix: Added minimum paper threshold.
     If total_papers < 20 or year_range < 3,
     returns warning instead of misleading statistics.
"""

import logging
from collections import Counter
from agents.llm_helper import llm_generate

logger = logging.getLogger(__name__)

MIN_PAPERS_FOR_TRENDS = 20   # below this → show warning, not metrics
MIN_YEAR_RANGE        = 3    # need at least 3 years of data


def analyze_research_trends(papers: list[dict]) -> dict:
    """
    Perform temporal analysis on research papers.
    Returns warning dict if insufficient data.
    """
    if not papers:
        return _empty_trends()

    year_counts = _count_papers_by_year(papers)

    if not year_counts:
        return _empty_trends()

    years     = sorted(year_counts.keys())
    earliest  = years[0]
    latest    = years[-1]
    total     = sum(year_counts.values())
    year_span = latest - earliest

    # ------------------------------------------------------------------ #
    #  Fix: Minimum threshold check                                        #
    # ------------------------------------------------------------------ #
    if total < MIN_PAPERS_FOR_TRENDS or year_span < MIN_YEAR_RANGE:
        logger.info(f"[Trends] Insufficient data: {total} papers, "
                    f"{year_span} year span. Showing warning.")
        return {
            "timeline":        year_counts,
            "trend_summary":   (
                f"⚠️ **Insufficient data for reliable trend analysis.**\n\n"
                f"Only {total} papers across {year_span + 1} year(s) were analyzed. "
                f"A minimum of {MIN_PAPERS_FOR_TRENDS} papers spanning at least "
                f"{MIN_YEAR_RANGE} years is needed for statistically meaningful trends.\n\n"
                f"**What we can say:** Papers in this dataset were published between "
                f"{earliest} and {latest}, with the most papers in {max(year_counts, key=year_counts.get)}."
            ),
            "total_papers":    total,
            "year_range":      (earliest, latest),
            "peak_year":       max(year_counts, key=year_counts.get),
            "growth_rate":     None,           # explicitly None = insufficient data
            "maturity_status": "insufficient_data",
            "data_warning":    True
        }

    # Sufficient data — compute full statistics
    peak_year   = max(year_counts, key=year_counts.get)
    first_count = year_counts.get(earliest, 1)
    last_count  = year_counts.get(latest, 1)
    growth_rate = round(((last_count - first_count) / max(first_count, 1)) * 100, 1)
    maturity    = _classify_maturity(years, year_counts, total)

    # LLM trend summary only when we have enough data
    trend_summary = _generate_trend_summary(year_counts, peak_year, maturity, growth_rate)

    return {
        "timeline":        year_counts,
        "trend_summary":   trend_summary,
        "total_papers":    total,
        "year_range":      (earliest, latest),
        "peak_year":       peak_year,
        "growth_rate":     growth_rate,
        "maturity_status": maturity,
        "data_warning":    False
    }


def format_timeline_display(timeline: dict) -> str:
    """Format timeline as text bar chart."""
    if not timeline:
        return "No timeline data available."

    lines   = []
    max_val = max(timeline.values()) if timeline else 1

    for year in sorted(timeline.keys()):
        count     = timeline[year]
        bar_len   = int((count / max_val) * 20)
        bar       = "█" * bar_len
        paper_str = "paper" if count == 1 else "papers"
        lines.append(f"  {year} → {count:2d} {paper_str}  {bar}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Private helpers                                                      #
# ------------------------------------------------------------------ #

def _count_papers_by_year(papers: list[dict]) -> dict[int, int]:
    years = []
    for paper in papers:
        published = paper.get("published", "")
        try:
            year = int(str(published)[:4])
            if 2000 <= year <= 2030:
                years.append(year)
        except (ValueError, TypeError):
            continue
    return dict(sorted(Counter(years).items())) if years else {}


def _classify_maturity(years, year_counts, total) -> str:
    span = (years[-1] - years[0]) if len(years) > 1 else 0
    if span < 4 or total < 20:
        return "emerging"
    elif span > 10 and total > 50:
        return "mature"
    return "growing"


def _generate_trend_summary(year_counts, peak_year, maturity, growth_rate) -> str:
    timeline_text = "\n".join(
        f"  {year}: {count} papers"
        for year, count in sorted(year_counts.items())
    )

    prompt = f"""You are a research analyst. Below is a publication timeline:

{timeline_text}

Peak year: {peak_year}
Maturity: {maturity}
Growth rate: {growth_rate}%

Write 3-4 sentences covering:
1. When did this area start gaining attention?
2. When was the peak period?
3. Is the field emerging, growing, or mature?
4. What does the trend suggest about future activity?

Be specific. Academic style. No invented data."""

    response = llm_generate(prompt, temperature=0.3, max_tokens=200)

    if "[LLM unavailable" in response or not response.strip():
        return _fallback_trend_summary(year_counts, peak_year, maturity, growth_rate)

    return response.strip()


def _fallback_trend_summary(year_counts, peak_year, maturity, growth_rate) -> str:
    years = sorted(year_counts.keys())
    start = years[0] if years else "unknown"
    end   = years[-1] if years else "unknown"
    return (
        f"Research spans {start} to {end}, peaking in {peak_year}. "
        f"The field is {maturity} with {growth_rate}% growth. "
        f"{'Continued growth expected.' if maturity != 'mature' else 'Field has reached maturity.'}"
    )


def _empty_trends() -> dict:
    return {
        "timeline":        {},
        "trend_summary":   "Insufficient data for trend analysis.",
        "total_papers":    0,
        "year_range":      (0, 0),
        "peak_year":       0,
        "growth_rate":     0.0,
        "maturity_status": "unknown",
        "data_warning":    True
    }