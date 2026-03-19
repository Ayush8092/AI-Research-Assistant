"""
AI Research Assistant — Gradio Interface (9 tabs)
Updated: Paper type classification, citation velocity,
         Why This Paper Matters, bullet-point insights.
"""
from dotenv import load_dotenv
load_dotenv()

import logging
import sys
import time
import uuid
import json
import re
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from workflows.langgraph_workflow import run_research_pipeline, run_with_refinement
from database.memory_store import MemoryStore
from agents.llm_helper import check_ollama_available, list_available_models
from vectorstore.faiss_store import FAISSStore
from knowledge_graph.graph_builder import get_graph_stats
from analysis.research_trends import format_timeline_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger  = logging.getLogger(__name__)
memory  = MemoryStore()
_states = {}


# ------------------------------------------------------------------ #
#  Cache management                                                    #
# ------------------------------------------------------------------ #

def clear_search_cache() -> str:
    """Clear FAISS index and embed cache."""
    try:
        store = FAISSStore()
        count = store.index.ntotal
        store.clear()

        cache_path = Path("data/embed_cache.json")
        if cache_path.exists():
            with open(cache_path, "w") as f:
                json.dump({"embedded_ids": []}, f)

        logger.info(f"[App] Cache cleared. Removed {count} vectors.")
        return (
            f"✅ Search cache cleared successfully!\n"
            f"Removed {count} stored paper embeddings.\n"
            f"Your next search will start fresh."
        )
    except Exception as e:
        logger.error(f"[App] Cache clear failed: {e}")
        return f"❌ Cache clear failed: {str(e)}"


# ------------------------------------------------------------------ #
#  Global helper                                                       #
# ------------------------------------------------------------------ #

def _clean(text: str) -> str:
    """Strip ** markers and clean text for display."""
    if not text:
        return "Not available."
    text = re.sub(r'\*\*', '', str(text))
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() or "Not available."


# ------------------------------------------------------------------ #
#  Formatters                                                          #
# ------------------------------------------------------------------ #

def _papers_table(ranked_papers: list) -> list:
    """Papers table with paper type indicator in Theme column."""
    return [
        [
            i + 1,
            p.get("title", "")[:80],
            ", ".join(p.get("authors", [])[:2]) +
            (" et al." if len(p.get("authors", [])) > 2 else ""),
            p.get("published", "")[:4],
            p.get("citation_count", 0),
            p.get("venue", "arXiv")[:30],
            round(p.get("final_score", 0), 3),
            # Paper type badge + theme
            (
                f"{'🔵' if p.get('paper_type') == 'Core Systems' else '🟢'} "
                f"{p.get('cluster_theme', '')[:25]}"
            ),
            p.get("arxiv_url", "")
        ]
        for i, p in enumerate(ranked_papers[:20])
    ]


def _score_breakdown_md(paper: dict) -> str:
    """
    Display detailed paper insights:
    - Paper type badge (Core Systems / Application)
    - Why This Paper Matters
    - Citation velocity
    - Bullet-point insights
    - Paper comparison engine
    """
    sb           = paper.get("score_breakdown", {})
    insights     = paper.get("insights", {})
    comparison   = paper.get("comparison_data", {})
    paper_type   = paper.get("paper_type", "Unknown")
    significance = paper.get("significance", "")

    # Paper type badge and description
    type_badge = (
        "🔵 **Core Systems Paper**"
        if paper_type == "Core Systems"
        else "🟢 **Application Paper**"
    )
    type_desc = (
        "Focuses on multi-agent framework design, architecture, "
        "coordination methods, or benchmarks."
        if paper_type == "Core Systems"
        else "Applies multi-agent LLM methods to a specific domain problem."
    )

    # ── Title + Type + Significance ──────────────────────────────
    header_section = (
        f"### {_clean(paper.get('title', ''))[:80]}\n\n"
        f"{type_badge} — {type_desc}\n\n"
    )

    if significance:
        header_section += (
            f"> 💡 **Why This Paper Matters:** "
            f"{_clean(significance)}\n\n"
        )

    # ── Score table ──────────────────────────────────────────────
    score_section = (
        f"| Component | Weight | Score |\n"
        f"|-----------|--------|-------|\n"
        f"| Citation (velocity-adjusted) | 35% | "
        f"{sb.get('citation_score', 0):.3f} |\n"
        f"| Recency | 20% | {sb.get('recency_score', 0):.3f} |\n"
        f"| Venue | 10% | {sb.get('venue_score', 0):.3f} |\n"
        f"| Relevance | 35% | {sb.get('relevance_score', 0):.3f} |\n"
        f"| **Final** | **100%** | "
        f"**{sb.get('final_score', 0):.3f}** |\n\n"
        f"**Citations:** {sb.get('citation_count', 0)} | "
        f"**Velocity:** {sb.get('citation_velocity', 0):.1f}/year | "
        f"**Venue:** {_clean(sb.get('venue', 'Unknown'))} "
        f"({sb.get('venue_tier', 'unknown')} tier)\n\n"
        f"**Theme:** {_clean(paper.get('cluster_theme', 'N/A'))}\n\n"
        f"**Keywords:** "
        f"{', '.join(paper.get('cluster_keywords', []) or ['N/A'])}\n\n"
        f"**Strengths:** "
        f"{' | '.join(_clean(s) for s in paper.get('strengths', ['N/A']))}\n\n"
        f"**Weaknesses:** "
        f"{' | '.join(_clean(w) for w in paper.get('weaknesses', ['N/A']))}\n\n"
    )

    # ── Extracted Insights (bullet points) ───────────────────────
    insights_section = (
        f"---\n\n"
        f"### Extracted Insights\n\n"
        f"**Problem Statement**\n\n"
        f"{_clean(insights.get('problem_statement', insights.get('problem', '')))}\n\n"
        f"**Methodology**\n\n"
        f"{_clean(insights.get('methodology', ''))}\n\n"
        f"**Datasets Used**\n\n"
        f"{_clean(insights.get('datasets', ''))}\n\n"
        f"**Evaluation Metrics**\n\n"
        f"{_clean(insights.get('evaluation_metrics', insights.get('metrics', '')))}\n\n"
        f"**Key Contributions**\n\n"
        f"{_clean(insights.get('key_contributions', ''))}\n\n"
        f"**Limitations**\n\n"
        f"{_clean(insights.get('limitations', ''))}\n\n"
        f"**Future Work**\n\n"
        f"{_clean(insights.get('future_work', ''))}\n\n"
    )

    # ── Paper Comparison Engine ───────────────────────────────────
    comparison_section = ""
    if comparison and any(comparison.values()):
        comparison_section = (
            f"---\n\n"
            f"### Paper Comparison Engine\n\n"
            f"> *How this paper relates to others in the result set*\n\n"
            f"**Shared Methods Across Papers**\n\n"
            f"{_clean(comparison.get('shared_methods', ''))}\n\n"
            f"**Key Differences**\n\n"
            f"{_clean(comparison.get('key_differences', ''))}\n\n"
            f"**Contradictions Found**\n\n"
            f"{_clean(comparison.get('contradictions', ''))}\n\n"
            f"**Complementary Findings**\n\n"
            f"{_clean(comparison.get('complementary_findings', ''))}\n\n"
            f"**Strongest Paper Assessment**\n\n"
            f"{_clean(comparison.get('strongest_paper', ''))}\n\n"
        )

    return header_section + score_section + insights_section + comparison_section


def _report_md(insights: dict) -> str:
    """Display research report — clean all ** markers."""
    if not insights or not insights.get("background"):
        return "*Run a search to generate the report.*"

    return (
        f"# Research Report: {insights.get('topic', '')}\n"
        f"*Papers analyzed: {insights.get('total_papers_analyzed', 0)}*\n\n"
        f"---\n## Background\n"
        f"{_clean(insights.get('background', ''))}\n\n"
        f"---\n## Key Methods\n"
        f"{_clean(insights.get('key_methods', ''))}\n\n"
        f"---\n## Datasets\n"
        f"{_clean(insights.get('common_datasets', ''))}\n\n"
        f"---\n## Evaluation Metrics\n"
        f"{_clean(insights.get('evaluation_metrics', ''))}\n\n"
        f"---\n## Limitations\n"
        f"{_clean(insights.get('limitations', ''))}\n\n"
        f"---\n## Research Gaps\n"
        f"{_clean(insights.get('research_gaps', ''))}\n\n"
        f"---\n## Future Directions\n"
        f"{_clean(insights.get('future_directions', ''))}"
    )


def _kg_stats_md(entities: list, edges: list) -> str:
    """Display knowledge graph statistics."""
    stats     = get_graph_stats(entities, edges)
    tc        = stats.get("entity_types", {})
    rc        = stats.get("relation_types", {})
    relations = " | ".join(f"`{r}`: {c}" for r, c in rc.items())
    return (
        f"### Knowledge Graph\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Total Entities | {stats['total_entities']} |\n"
        f"| Total Edges | {stats['total_edges']} |\n"
        f"| Papers | {tc.get('paper', 0)} |\n"
        f"| Methods | {tc.get('method', 0)} |\n"
        f"| Datasets | {tc.get('dataset', 0)} |\n"
        f"| Tasks | {tc.get('task', 0)} |\n\n"
        f"**Relations:** {relations}"
    )


def _clusters_md(cluster_info: list) -> str:
    """Display research themes with keywords and top paper."""
    if not cluster_info:
        return "*Run a search to detect research themes.*"

    colors = ["🔵", "🟠", "🟢", "🟣", "🔴", "🟡"]
    lines  = [f"## Research Themes ({len(cluster_info)} clusters)\n"]

    for c in cluster_info:
        cid      = c.get("cluster_id", 0)
        theme    = c.get("theme", "Unknown")
        count    = c.get("paper_count", 0)
        titles   = c.get("paper_titles", [])
        keywords = c.get("keywords", [])
        top      = c.get("top_paper", "")
        icon     = colors[cid % len(colors)]

        lines.append(f"### {icon} {theme}")
        lines.append(f"*{count} paper{'s' if count != 1 else ''}*\n")

        if keywords:
            lines.append(f"**Keywords:** {', '.join(keywords)}\n")

        if top:
            lines.append(f"**Top paper:** {top}\n")

        for t in titles[:4]:
            lines.append(f"- {t}")
        lines.append("")

    return "\n".join(lines)


def _trends_md(research_trends: dict) -> str:
    """Display research timeline."""
    if not research_trends or not research_trends.get("timeline"):
        return "*Run a search to generate the timeline.*"

    timeline      = research_trends.get("timeline", {})
    trend_summary = research_trends.get("trend_summary", "")
    peak_year     = research_trends.get("peak_year", "N/A")
    growth_rate   = research_trends.get("growth_rate")
    maturity      = research_trends.get("maturity_status", "unknown")
    year_range    = research_trends.get("year_range", (0, 0))
    data_warning  = research_trends.get("data_warning", False)
    timeline_disp = format_timeline_display(timeline)

    icons = {
        "emerging":          "🌱",
        "growing":           "📈",
        "mature":            "🏛️",
        "unknown":           "❓",
        "insufficient_data": "⚠️"
    }

    result = "## Research Timeline\n\n"

    if data_warning:
        result += (
            "> ⚠️ **Limited data** — "
            "statistics may not be reliable.\n\n"
        )

    result += (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Year Range | {year_range[0]} – {year_range[1]} |\n"
        f"| Peak Year | {peak_year} |\n"
    )

    if growth_rate is not None:
        result += f"| Growth Rate | {growth_rate}% |\n"
    else:
        result += "| Growth Rate | ⚠️ Insufficient data |\n"

    result += (
        f"| Status | {icons.get(maturity, '❓')} "
        f"{maturity.replace('_', ' ').capitalize()} |\n\n"
        f"### Publication Timeline\n\n"
        f"```\n{timeline_disp}\n```\n\n"
        f"### Analysis\n\n{_clean(trend_summary)}"
    )
    return result


def _render_kg_html(kg_html_path: str) -> str:
    """Render knowledge graph as base64 iframe."""
    if not kg_html_path:
        return (
            "<p style='color:gray;padding:20px'>"
            "No graph yet.</p>"
        )

    path = Path(kg_html_path)
    if not path.exists():
        return (
            "<p style='color:gray;padding:20px'>"
            "Graph file not found.</p>"
        )

    try:
        import base64
        html_content = path.read_text(encoding="utf-8")
        encoded      = base64.b64encode(
            html_content.encode("utf-8")
        ).decode("utf-8")
        return (
            f'<iframe src="data:text/html;base64,{encoded}" '
            f'width="100%" height="550px" frameborder="0" '
            f'style="border-radius:8px;background:white;">'
            f'</iframe>'
        )
    except Exception as e:
        return f"<p style='color:orange'>Graph error: {e}</p>"


def save_file(content: str, filename: str):
    """Save content to file and return path."""
    if not content:
        return None
    p = Path(f"data/artifacts/{filename}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ #
#  Pipeline runners                                                    #
# ------------------------------------------------------------------ #

def run_pipeline(
    query,
    year_filter,
    exclude_surveys,
    max_papers,
    generate_rw_flag,
    progress=gr.Progress()
):
    if not query.strip():
        return (
            "⚠️ Please enter a query.",
            [], "", "", "", "", "", "", "", "", None, ""
        )

    session_id = str(uuid.uuid4())
    filters    = {
        "max_papers":            int(max_papers),
        "generate_related_work": generate_rw_flag
    }
    if year_filter > 2018:
        filters["year_after"] = int(year_filter)
    if exclude_surveys:
        filters["exclude_surveys"] = True

    progress(0.05, desc="🧠 Planning subtopics...")
    state = run_research_pipeline(
        query=query, filters=filters, session_id=session_id
    )
    _states[session_id] = state

    progress(0.90, desc="📊 Formatting results...")

    ranked          = state.get("ranked_papers", [])
    insights        = state.get("insights", {})
    arts            = state.get("artifacts", {})
    entities        = state.get("knowledge_graph_entities", [])
    edges           = state.get("knowledge_graph_edges", [])
    metrics         = state.get("metrics", {})
    errors          = state.get("errors", [])
    cluster_info    = state.get("cluster_info", [])
    research_trends = state.get("research_trends", {})

    # Cluster summary
    cluster_summary = ""
    if cluster_info:
        themes = [c.get("theme", "?") for c in cluster_info[:3]]
        cluster_summary = f"\n- Themes: **{', '.join(themes)}**"

    # Trend note
    trend_note = ""
    if research_trends.get("data_warning"):
        trend_note = "\n- ⚠️ Trend data insufficient for reliable stats"
    elif research_trends.get("maturity_status"):
        trend_note = (
            f"\n- Field: "
            f"**{research_trends.get('maturity_status','').capitalize()}**"
            f" | Peak: **{research_trends.get('peak_year','N/A')}**"
        )

    # Paper type breakdown
    core_count = metrics.get("core_papers", 0)
    app_count  = metrics.get("app_papers", 0)
    type_note  = ""
    if core_count or app_count:
        type_note = (
            f"\n- Types: 🔵 **{core_count} Core Systems** | "
            f"🟢 **{app_count} Applications**"
        )

    status = (
        f"✅ **Research Complete**\n"
        f"- Session: `{session_id[:8]}`\n"
        f"- Papers: **{metrics.get('papers_retrieved', 0)}** retrieved | "
        f"**{len(ranked)}** ranked\n"
        f"- Clusters: **{len(cluster_info)}** themes"
        f"{cluster_summary}"
        f"{type_note}\n"
        f"- Score mean: **{metrics.get('score_mean', 0):.3f}** | "
        f"Time: **{metrics.get('query_time_sec', 0):.1f}s**"
        f"{trend_note}\n"
        + (
            f"- ⚠️ {len(errors)} error(s): {errors[0][:50]}"
            if errors else ""
        )
    )

    kg_display = _render_kg_html(arts.get("knowledge_graph_html", ""))

    progress(1.0, desc="✅ Done!")
    return (
        status,
        _papers_table(ranked),
        _score_breakdown_md(ranked[0]) if ranked else "No papers found.",
        _report_md(insights),
        _clusters_md(cluster_info),
        _trends_md(research_trends),
        arts.get("report", "{}"),
        arts.get("bibtex", "% No BibTeX"),
        arts.get("related_work", "# No related work"),
        _kg_stats_md(entities, edges),
        kg_display,
        session_id
    )


def apply_refinement(
    refinement_cmd: str,
    session_id: str,
    progress=gr.Progress()
):
    if not session_id or not refinement_cmd.strip():
        return (
            "⚠️ Run a search first.",
            [], "", "", "", "", "", "", "", "", None, session_id
        )

    prev = _states.get(session_id)
    if not prev:
        return (
            "⚠️ Session not found. Run a new search.",
            [], "", "", "", "", "", "", "", "", None, session_id
        )

    progress(0.1, desc="🔄 Applying refinement...")
    state = run_with_refinement(session_id, refinement_cmd, prev)
    _states[session_id] = state

    ranked          = state.get("ranked_papers", [])
    insights        = state.get("insights", {})
    arts            = state.get("artifacts", {})
    entities        = state.get("knowledge_graph_entities", [])
    edges           = state.get("knowledge_graph_edges", [])
    cluster_info    = state.get("cluster_info", [])
    research_trends = state.get("research_trends", {})
    kg_display      = _render_kg_html(
        arts.get("knowledge_graph_html", "")
    )

    progress(1.0)
    return (
        f"✅ **Refinement applied:** *{refinement_cmd}* | "
        f"Papers: **{len(ranked)}**",
        _papers_table(ranked),
        _score_breakdown_md(ranked[0]) if ranked else "",
        _report_md(insights),
        _clusters_md(cluster_info),
        _trends_md(research_trends),
        arts.get("report", "{}"),
        arts.get("bibtex", ""),
        arts.get("related_work", ""),
        _kg_stats_md(entities, edges),
        kg_display,
        session_id
    )


def get_system_status() -> str:
    """Display system status including LLM provider and sources."""
    llm_ok  = check_ollama_available()
    models  = list_available_models() if llm_ok else []

    try:
        faiss_stats = FAISSStore().stats()
        faiss_ok    = True
    except Exception:
        faiss_stats = {}
        faiss_ok    = False

    sessions = memory.list_sessions()

    try:
        from agents.llm_helper import GROQ_API_KEY
        provider = (
            f"Groq ({models[0] if models else 'llama-3.1-8b-instant'})"
            if GROQ_API_KEY else "Ollama (local)"
        )
    except Exception:
        provider = "Unknown"

    return (
        "## System Status\n\n"
        "| Component | Status |\n|-----------|--------|\n"
        f"| LLM Provider | "
        f"{'✅ ' + provider if llm_ok else '❌ Not configured'} |\n"
        f"| FAISS Index | {'✅ Ready' if faiss_ok else '❌ Error'} |\n"
        f"| SQLite DB | ✅ Ready |\n"
        f"| ArXiv | ✅ Free (no key needed) |\n"
        f"| Semantic Scholar | ✅ Connected |\n"
        f"| PubMed | ✅ Free (no key needed) |\n"
        f"| CORE | ✅ Connected |\n\n"
        f"**FAISS:** "
        f"{faiss_stats.get('total_documents', 0)} / "
        f"{faiss_stats.get('max_documents', 500)} papers stored\n\n"
        f"**Sessions:** {len(sessions)}\n\n"
        "### Paper Classification\n"
        "- 🔵 **Core Systems** — framework, architecture, coordination\n"
        "- 🟢 **Application** — domain-specific implementations\n\n"
        "### Active Sources\n"
        "- ArXiv — latest preprints\n"
        "- Semantic Scholar — 200M papers + citations\n"
        "- PubMed — 36M peer-reviewed medical papers\n"
        "- CORE — 200M open access papers\n\n"
        "### Tips\n"
        "- Clear cache when switching to a different topic\n"
        "- Use 7-10 max papers for best quality\n"
        "- Enable Related Work only when needed\n"
        "- 🔵 Core Systems papers = foundational research\n"
        "- 🟢 Application papers = domain implementations\n"
    )


# ------------------------------------------------------------------ #
#  Gradio UI                                                           #
# ------------------------------------------------------------------ #

def build_ui():
    with gr.Blocks(title="AI Research Assistant") as demo:

        gr.HTML(
            "<div style='background:linear-gradient("
            "135deg,#1e3a5f,#2d6a9f);"
            "padding:20px;border-radius:12px;margin-bottom:16px'>"
            "<h1 style='color:white;margin:0'>"
            "🔬 AI Research Assistant</h1>"
            "<p style='color:#c8dff0;margin:4px 0 0 0'>"
            "Hierarchical Multi-Agent System • LangGraph • "
            "4-Source Search • Paper Classification • "
            "Comparison Engine"
            "</p>"
            "</div>"
        )

        session_state = gr.State("")

        with gr.Tabs():

            # ── Tab 1: Search & Rank ──────────────────────────────
            with gr.Tab("🔍 Search & Rank"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Research Query",
                            placeholder=(
                                "e.g. 'deep learning for diabetic "
                                "retinopathy detection'"
                            ),
                            lines=2
                        )
                    with gr.Column(scale=1):
                        run_btn = gr.Button(
                            "🚀 Run Research",
                            variant="primary",
                            size="lg"
                        )

                with gr.Row():
                    year_filter = gr.Slider(
                        2018, 2025, value=2020, step=1,
                        label="Papers after year"
                    )
                    max_papers = gr.Slider(
                        3, 15, value=7, step=1,
                        label="Max papers"
                    )
                    exclude_surveys = gr.Checkbox(
                        label="Exclude surveys",
                        value=False
                    )

                with gr.Row():
                    generate_rw_flag = gr.Checkbox(
                        label=(
                            "Generate Related Work "
                            "(slower — adds ~2 min)"
                        ),
                        value=False
                    )

                with gr.Accordion(
                    "⚠️ Search Cache — Clear when switching topics",
                    open=False
                ):
                    gr.Markdown(
                        "**When to clear:** Searched Topic A, "
                        "now searching Topic B — clear first.\n\n"
                        "**Keep cache:** Refining the same topic."
                    )
                    with gr.Row():
                        clear_btn    = gr.Button(
                            "🗑️ Clear Search Cache",
                            variant="secondary"
                        )
                        clear_status = gr.Markdown("")

                gr.Markdown(
                    "> 🔵 **Core Systems** = framework/architecture papers  "
                    "🟢 **Application** = domain-specific papers"
                )

                status_out = gr.Markdown(
                    "*Enter a query and click Run Research.*"
                )

                gr.Markdown("### 📊 Ranked Papers")
                papers_tbl = gr.Dataframe(
                    headers=[
                        "Rank", "Title", "Authors", "Year",
                        "Citations", "Venue", "Score",
                        "Type + Theme", "URL"
                    ],
                    datatype=[
                        "number", "str", "str", "str", "number",
                        "str", "number", "str", "str"
                    ],
                    column_count=(9, "fixed"),
                    wrap=True
                )

                gr.Markdown(
                    "### 🎯 Top Paper Details + Comparison Engine"
                )
                score_detail = gr.Markdown()

            # ── Tab 2: Research Report ────────────────────────────
            with gr.Tab("📄 Research Report"):
                report_md_out = gr.Markdown(
                    "*Run a search to generate the report.*"
                )

            # ── Tab 3: Research Themes ────────────────────────────
            with gr.Tab("🎨 Research Themes"):
                clusters_out = gr.Markdown(
                    "*Run a search to detect themes.*"
                )

            # ── Tab 4: Research Timeline ──────────────────────────
            with gr.Tab("📅 Research Timeline"):
                trends_out = gr.Markdown(
                    "*Run a search to generate timeline.*"
                )

            # ── Tab 5: Download Artifacts ─────────────────────────
            with gr.Tab("📦 Download Artifacts"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 research_report.json")
                        report_json_out = gr.Code(
                            language="json",
                            label="Report JSON",
                            lines=15
                        )
                        dl_report_btn = gr.DownloadButton(
                            label="💾 Download JSON"
                        )
                    with gr.Column():
                        gr.Markdown("### 📚 references.bib")
                        bibtex_out = gr.Code(
                            language="markdown",
                            label="BibTeX",
                            lines=15
                        )
                        dl_bib_btn = gr.DownloadButton(
                            label="💾 Download .bib"
                        )

                gr.Markdown("### 📝 related_work.md")
                rw_out    = gr.Markdown()
                dl_rw_btn = gr.DownloadButton(
                    label="💾 Download .md"
                )

            # ── Tab 6: Knowledge Graph ────────────────────────────
            with gr.Tab("🕸️ Knowledge Graph"):
                kg_stats_out = gr.Markdown(
                    "*Run a search to build the graph.*"
                )
                kg_viz_out = gr.HTML(
                    "<p style='color:gray;padding:20px'>"
                    "Knowledge graph will appear here after search."
                    "</p>"
                )

            # ── Tab 7: Refine Search ──────────────────────────────
            with gr.Tab("🔄 Refine Search"):
                gr.Markdown(
                    "### Conversational Refinement\n\n"
                    "Refine your current results without a full "
                    "new search.\n\n"
                    "**Examples:**\n"
                    "- `Focus on papers after 2022`\n"
                    "- `Exclude survey papers`\n"
                    "- `Drill deeper into clinical applications`"
                )
                refine_input = gr.Textbox(
                    label="Refinement Command",
                    lines=1
                )
                refine_btn = gr.Button(
                    "🔄 Apply Refinement",
                    variant="secondary"
                )

            # ── Tab 8: Session History ────────────────────────────
            with gr.Tab("🗂️ Session History"):
                refresh_btn = gr.Button("🔃 Refresh")
                history_tbl = gr.Dataframe(
                    headers=["Session ID", "Query", "Timestamp"],
                    datatype=["str", "str", "str"],
                    column_count=(3, "fixed")
                )

            # ── Tab 9: System Status ──────────────────────────────
            with gr.Tab("⚙️ System Status"):
                refresh_status_btn = gr.Button("🔃 Refresh Status")
                status_out_sys     = gr.Markdown(get_system_status())

        # ── All outputs (12) ──────────────────────────────────────
        all_outputs = [
            status_out,       # 1
            papers_tbl,       # 2
            score_detail,     # 3
            report_md_out,    # 4
            clusters_out,     # 5
            trends_out,       # 6
            report_json_out,  # 7
            bibtex_out,       # 8
            rw_out,           # 9
            kg_stats_out,     # 10
            kg_viz_out,       # 11
            session_state     # 12
        ]

        # ── Event bindings ────────────────────────────────────────
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                query_input, year_filter, exclude_surveys,
                max_papers, generate_rw_flag
            ],
            outputs=all_outputs,
            show_progress=True
        )

        query_input.submit(
            fn=run_pipeline,
            inputs=[
                query_input, year_filter, exclude_surveys,
                max_papers, generate_rw_flag
            ],
            outputs=all_outputs,
            show_progress=True
        )

        refine_btn.click(
            fn=apply_refinement,
            inputs=[refine_input, session_state],
            outputs=all_outputs,
            show_progress=True
        )

        clear_btn.click(
            fn=clear_search_cache,
            outputs=[clear_status]
        )

        dl_report_btn.click(
            fn=lambda c: save_file(c, "dl_report.json"),
            inputs=[report_json_out],
            outputs=[dl_report_btn]
        )
        dl_bib_btn.click(
            fn=lambda c: save_file(c, "dl_references.bib"),
            inputs=[bibtex_out],
            outputs=[dl_bib_btn]
        )
        dl_rw_btn.click(
            fn=lambda c: save_file(c, "dl_related_work.md"),
            inputs=[rw_out],
            outputs=[dl_rw_btn]
        )

        refresh_btn.click(
            fn=lambda: [
                [
                    s["session_id"][:8],
                    s["query"][:60],
                    s["created_at"][:19]
                ]
                for s in memory.list_sessions()[:20]
            ],
            outputs=[history_tbl]
        )

        refresh_status_btn.click(
            fn=get_system_status,
            outputs=[status_out_sys]
        )

        demo.load(
            fn=lambda: [
                [
                    s["session_id"][:8],
                    s["query"][:60],
                    s["created_at"][:19]
                ]
                for s in memory.list_sessions()[:20]
            ],
            outputs=[history_tbl]
        )

    return demo


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    Path("data/artifacts").mkdir(parents=True, exist_ok=True)
    logger.info("=" * 50)
    logger.info("  AI Research Assistant")
    logger.info("=" * 50)

    if not check_ollama_available():
        logger.warning(
            "⚠️  No LLM configured. Set GROQ_API_KEY in secrets."
        )
    else:
        try:
            from agents.llm_helper import get_active_provider
            logger.info(f"✅ LLM: {get_active_provider()}")
        except Exception:
            logger.info("✅ LLM configured.")

    # Hugging Face compatible launch
    build_ui().launch()