"""
Paper Clustering
Groups papers into themes using KMeans + LLM naming.

Fixes:
  1. Adds keyword extraction per cluster
  2. Better LLM theme naming prompt
  3. Strips ** from cluster names
  4. Adds top paper per cluster
  5. Min cluster size enforcement
"""

import logging
import re
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_CLUSTERS        = 4
MIN_PAPERS_FOR_CLUSTERING = 4


def cluster_papers(
    papers: list[dict],
    encoder,
    n_clusters: int = DEFAULT_N_CLUSTERS
) -> tuple[list[dict], list[dict]]:
    """Cluster papers into research themes."""
    if len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        logger.info(
            f"[Clustering] Too few papers ({len(papers)}). "
            f"Single cluster."
        )
        for paper in papers:
            paper["cluster_id"]    = 0
            paper["cluster_theme"] = "General Research"
            paper["cluster_keywords"] = []
        return papers, [{
            "cluster_id":   0,
            "theme":        "General Research",
            "paper_count":  len(papers),
            "paper_titles": [p.get("title","")[:60] for p in papers],
            "keywords":     [],
            "top_paper":    papers[0].get("title","")[:60] if papers else ""
        }]

    actual_clusters = min(n_clusters, len(papers))

    texts = [
        f"{p.get('title','')} {p.get('abstract','')}".strip()
        for p in papers
    ]

    logger.info(f"[Clustering] Encoding {len(texts)} papers...")
    embeddings = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    try:
        from sklearn.cluster import KMeans
        logger.info(
            f"[Clustering] KMeans with {actual_clusters} clusters..."
        )
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(embeddings)
    except ImportError:
        labels = [i % actual_clusters for i in range(len(papers))]
    except Exception as e:
        logger.error(f"[Clustering] KMeans failed: {e}")
        labels = [i % actual_clusters for i in range(len(papers))]

    for paper, label in zip(papers, labels):
        paper["cluster_id"]       = int(label)
        paper["cluster_theme"]    = ""
        paper["cluster_keywords"] = []

    cluster_info = []
    for cluster_id in range(actual_clusters):
        cluster_papers_list = [
            p for p in papers
            if p["cluster_id"] == cluster_id
        ]
        if not cluster_papers_list:
            continue

        theme    = _generate_theme(cluster_id, cluster_papers_list)
        keywords = _extract_cluster_keywords(cluster_papers_list)
        top_paper = sorted(
            cluster_papers_list,
            key=lambda x: x.get("final_score", x.get("citation_count", 0)),
            reverse=True
        )[0].get("title","")[:60]

        for paper in cluster_papers_list:
            paper["cluster_theme"]    = theme
            paper["cluster_keywords"] = keywords

        cluster_info.append({
            "cluster_id":   cluster_id,
            "theme":        theme,
            "paper_count":  len(cluster_papers_list),
            "paper_titles": [
                p.get("title","")[:60]
                for p in cluster_papers_list
            ],
            "keywords":     keywords,
            "top_paper":    top_paper
        })

        logger.info(
            f"[Clustering] Cluster {cluster_id}: '{theme}' "
            f"({len(cluster_papers_list)} papers) "
            f"keywords={keywords[:3]}"
        )

    logger.info(f"[Clustering] Done. {actual_clusters} clusters.")
    return papers, cluster_info


def _generate_theme(cluster_id: int, papers: list[dict]) -> str:
    """Generate specific LLM theme label."""
    from agents.llm_helper import llm_generate

    snippets = []
    for p in papers[:4]:
        title    = p.get("title","")[:70]
        abstract = p.get("abstract","")[:120]
        snippets.append(f"- {title}: {abstract}")

    papers_text = "\n".join(snippets)

    prompt = f"""These research papers belong to one cluster:

{papers_text}

Give a SPECIFIC 3-5 word theme label for this cluster.
The label must describe what these papers have in common.

Good examples:
- "LLM-based Drug Discovery"
- "Federated Learning Privacy"
- "Vision Transformers Medical Imaging"

Bad examples (too generic):
- "AI Research"
- "Machine Learning Methods"
- "Data Analysis"

Rules:
- 3-5 words maximum
- Specific to these papers
- No ** markdown
- No punctuation at end
- Return ONLY the label"""

    response = llm_generate(prompt, temperature=0.1, max_tokens=15)
    theme    = _clean_theme(response, cluster_id)
    return theme


def _clean_theme(response: str, cluster_id: int) -> str:
    """Clean theme label thoroughly."""
    theme = response.strip()
    # Remove ** markers
    theme = re.sub(r'\*\*', '', theme).strip()
    theme = re.sub(r'\*', '', theme).strip()
    # Remove quotes
    theme = theme.strip('"').strip("'").strip("`")
    # Remove trailing punctuation
    theme = theme.rstrip(".,;:!?")

    # Remove common prefixes
    for prefix in [
        "theme label:", "theme:", "label:", "title:",
        "cluster:", "topic:", "category:"
    ]:
        if theme.lower().startswith(prefix):
            theme = theme[len(prefix):].strip()
            break

    # Validate
    if not theme or len(theme) > 60 or len(theme.split()) > 7:
        theme = _fallback_theme(cluster_id)

    return theme


def _extract_cluster_keywords(papers: list[dict]) -> list[str]:
    """
    Extract top keywords from a cluster of papers.
    Uses simple frequency analysis on titles + abstracts.
    """
    stop_words = {
        "the","a","an","and","or","of","in","for","to","with",
        "using","based","on","is","are","we","this","that","by",
        "from","as","at","be","have","has","which","their","our",
        "paper","propose","present","show","study","approach",
        "method","model","system","result","performance","data"
    }

    word_freq: dict[str, int] = {}
    for paper in papers:
        text = (
            paper.get("title","") + " " +
            paper.get("abstract","")
        ).lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

    # Return top 5 keywords
    sorted_words = sorted(
        word_freq.items(), key=lambda x: x[1], reverse=True
    )
    return [w for w, c in sorted_words[:5] if c >= 2]


def _fallback_theme(cluster_id: int) -> str:
    fallbacks = [
        "Methods and Algorithms",
        "Clinical Applications",
        "Data and Evaluation",
        "Safety and Ethics",
        "System Architecture",
        "Performance Analysis"
    ]
    return fallbacks[cluster_id % len(fallbacks)]