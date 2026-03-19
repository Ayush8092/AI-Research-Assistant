"""
CLI Runner — test the full pipeline without Gradio.
Usage: python run_cli.py --query "federated learning in healthcare"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workflows.langgraph_workflow import run_research_pipeline
from agents.llm_helper import check_ollama_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="AI Research Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cli.py --query "federated learning in healthcare"
  python run_cli.py --query "LLMs for EHR analysis" --year 2022 --max-papers 15
  python run_cli.py --query "medical imaging segmentation" --no-surveys --output results.json
        """
    )
    parser.add_argument("--query",       required=True,        help="Research query")
    parser.add_argument("--year",        type=int, default=0,  help="Filter papers after this year")
    parser.add_argument("--max-papers",  type=int, default=20, help="Max papers to retrieve")
    parser.add_argument("--no-surveys",  action="store_true",  help="Exclude survey/review papers")
    parser.add_argument("--output",      default="",           help="Save results to JSON file")
    parser.add_argument("--verbose",     action="store_true",  help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not check_ollama_available():
        print("\n⚠️  WARNING: Ollama is not running. LLM features will be degraded.")
        print("   Install:  curl https://ollama.ai/install.sh | sh")
        print("   Pull:     ollama pull phi3:mini")
        print("   Start:    ollama serve\n")

    filters = {"max_papers": args.max_papers}
    if args.year > 2000:
        filters["year_after"] = args.year
    if args.no_surveys:
        filters["exclude_surveys"] = True

    print(f"\n{'='*60}")
    print(f"  AI Research Assistant — CLI")
    print(f"{'='*60}")
    print(f"  Query:    {args.query}")
    print(f"  Filters:  {filters}")
    print(f"{'='*60}\n")

    state    = run_research_pipeline(query=args.query, filters=filters)
    ranked   = state.get("ranked_papers", [])
    metrics  = state.get("metrics", {})
    insights = state.get("insights", {})
    errors   = state.get("errors", [])

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Papers retrieved:  {metrics.get('papers_retrieved', 0)}")
    print(f"  Papers ranked:     {len(ranked)}")
    print(f"  Mean score:        {metrics.get('score_mean', 0):.3f}")
    print(f"  Std deviation:     {metrics.get('score_std', 0):.3f}")
    print(f"  Pipeline time:     {metrics.get('query_time_sec', 0):.1f}s")

    if errors:
        print(f"\n  ⚠️  Errors ({len(errors)}):")
        for e in errors:
            print(f"     - {e}")

    if ranked:
        print(f"\n  Top 5 Papers:")
        print(f"  {'─'*55}")
        for i, p in enumerate(ranked[:5], 1):
            print(f"  {i}. [{p['final_score']:.3f}] {p['title'][:65]}")
            print(f"      Year: {p.get('published','')[:4]}  |  "
                  f"Citations: {p.get('citation_count', 0)}  |  "
                  f"Venue: {p.get('venue', 'arXiv')[:25]}")
            sb = p.get("score_breakdown", {})
            print(f"      Scores → "
                  f"Citation:{sb.get('citation_score',0):.2f}  "
                  f"Recency:{sb.get('recency_score',0):.2f}  "
                  f"Venue:{sb.get('venue_score',0):.2f}  "
                  f"LLM:{sb.get('llm_quality_score',0):.2f}")

    subtopics = state.get("subtopics", [])
    if subtopics:
        print(f"\n  Subtopics identified:")
        for t in subtopics:
            print(f"    • {t}")

    if insights.get("research_gaps"):
        print(f"\n  Research Gaps:")
        for line in insights["research_gaps"].split("\n")[:5]:
            clean = line.strip().lstrip("•-*0123456789.)> ").strip()
            if clean:
                print(f"    • {clean}")

    if insights.get("future_directions"):
        print(f"\n  Future Directions:")
        for line in insights["future_directions"].split("\n")[:4]:
            clean = line.strip().lstrip("•-*0123456789.)> ").strip()
            if clean:
                print(f"    → {clean}")

    if args.output:
        output_data = {
            "query":    args.query,
            "filters":  filters,
            "metrics":  metrics,
            "subtopics": state.get("subtopics", []),
            "ranked_papers": [
                {k: v for k, v in p.items() if k != "abstract"}
                for p in ranked[:20]
            ],
            "insights": insights,
            "knowledge_graph": {
                "entities": state.get("knowledge_graph_entities", []),
                "edges":    state.get("knowledge_graph_edges", [])
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  ✅ Results saved to: {args.output}")

    artifacts = state.get("artifacts", {})
    print(f"\n  Generated Artifacts:")
    for atype in ["report", "bibtex", "related_work"]:
        if artifacts.get(atype):
            print(f"    ✓ {atype}")
    print(f"  Location: data/artifacts/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()