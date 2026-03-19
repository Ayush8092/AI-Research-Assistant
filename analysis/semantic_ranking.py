"""
Semantic Ranking — Hybrid ArXiv + FAISS retrieval.

Fix: FAISS results filtered by must-have anchor terms
     before merging with ArXiv results.
     This prevents old papers from contaminating new searches.
"""

import logging
import re

logger = logging.getLogger(__name__)

COMMON_ML_WORDS = {
    "deep", "learning", "neural", "network", "model", "data",
    "based", "using", "method", "approach", "paper", "proposed",
    "results", "performance", "training", "test", "evaluation",
    "accuracy", "dataset", "feature", "classification", "detection",
    "prediction", "analysis", "system", "framework", "algorithm"
}


def hybrid_rank_papers(
    arxiv_papers: list[dict],
    faiss_store,
    query: str,
    top_k: int = 10,
    anchor_terms: set = None
) -> list[dict]:
    """
    Merge ArXiv + FAISS results with topic-aware filtering.
    FAISS results MUST contain anchor terms to be included.
    """
    # Determine must-have terms for FAISS filtering
    must_have = _get_must_have(query, anchor_terms)
    logger.info(f"[SemanticRanking] Must-have for FAISS: {must_have}")

    faiss_results = []
    try:
        if faiss_store.index.ntotal > 0:
            raw_faiss = faiss_store.search(query, top_k=top_k * 2)

            # Filter FAISS results — must contain at least 1 must-have term
            if must_have:
                faiss_results = [
                    r for r in raw_faiss
                    if _contains_must_have(r, must_have)
                ]
                logger.info(
                    f"[SemanticRanking] FAISS: "
                    f"{len(raw_faiss)} raw → "
                    f"{len(faiss_results)} after anchor filter"
                )
            else:
                faiss_results = raw_faiss
    except Exception as e:
        logger.warning(f"[SemanticRanking] FAISS search failed: {e}")

    # Build unified pool
    paper_pool: dict[str, dict] = {}

    # ArXiv papers first
    for paper in arxiv_papers:
        pid = paper["paper_id"]
        paper_pool[pid] = {
            **paper,
            "source":         "arxiv",
            "semantic_score": 0.0,
            "keyword_score":  paper.get("domain_relevance", 0.3)
        }

    # Filtered FAISS results
    for r in faiss_results:
        pid = r["paper_id"]
        sim = r.get("similarity_score", 0.0)
        if pid in paper_pool:
            paper_pool[pid]["semantic_score"] = sim
            paper_pool[pid]["source"]         = "arxiv+faiss"
        else:
            paper_pool[pid] = {
                "paper_id":         pid,
                "title":            r.get("title", ""),
                "abstract":         r.get("abstract", ""),
                "authors":          r.get("authors", []),
                "published":        r.get("published", ""),
                "arxiv_url":        r.get("arxiv_url", ""),
                "categories":       [],
                "domain_relevance": 0.0,
                "source":           "faiss",
                "semantic_score":   sim,
                "keyword_score":    0.0
            }

    # Compute hybrid score
    query_keywords = _extract_keywords(query)
    for paper in paper_pool.values():
        paper["hybrid_score"] = _hybrid_score(
            paper        = paper,
            query_kw     = query_keywords,
            anchor_terms = anchor_terms or set()
        )

    ranked = sorted(
        paper_pool.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )[:top_k]

    logger.info(
        f"[SemanticRanking] Pool: {len(paper_pool)} → "
        f"returning top {len(ranked)}"
    )
    return ranked


def _get_must_have(query: str, anchor_terms: set) -> set:
    """Get the most specific terms that FAISS results must contain."""
    if anchor_terms:
        # Use anchor terms that are NOT common ML words
        specific = {t for t in anchor_terms
                    if t not in COMMON_ML_WORDS}
        if specific:
            return specific
    # Fallback: extract from query directly
    words = re.findall(r'\b\w+\b', query.lower())
    stop  = {"in","for","of","the","a","an","and","or","on",
             "with","using","based","via","from","to","by","at"}
    return {w for w in words
            if w not in stop
            and w not in COMMON_ML_WORDS
            and len(w) > 4}


def _contains_must_have(paper: dict, must_have: set) -> bool:
    """Check if paper contains at least 1 must-have term."""
    text = (paper.get("title","") + " " +
            paper.get("abstract","")).lower()
    return any(t in text for t in must_have)


def _hybrid_score(
    paper: dict,
    query_kw: set,
    anchor_terms: set
) -> float:
    """
    Compute weighted hybrid relevance score.

    Components:
      - semantic_score : FAISS cosine similarity (50%)
      - keyword_score  : domain keyword overlap (25%)
      - anchor_score   : anchor term coverage (25%)
    """
    semantic = paper.get("semantic_score", 0.0)
    keyword  = paper.get("keyword_score",  0.0)
    anchor   = paper.get("anchor_score",   0.0)

    # If anchor_score not set, compute it
    if anchor == 0.0 and anchor_terms:
        text    = (paper.get("title","") + " " +
                   paper.get("abstract","")).lower()
        matches = sum(1 for t in anchor_terms if t in text)
        anchor  = min(matches / max(len(anchor_terms), 1), 1.0)

    score = (
        0.50 * semantic +
        0.25 * keyword  +
        0.25 * anchor
    )

    # Bonus: found in both ArXiv and FAISS
    if paper.get("source") == "arxiv+faiss":
        score = min(1.0, score + 0.05)

    return round(score, 4)


def _extract_keywords(query: str) -> set:
    stop = {"in","for","of","the","a","an","and","or","on",
            "with","using","based","via","from","to","by","at",
            "is","are","was","be","it","its"}
    words = re.findall(r'\b\w+\b', query.lower())
    return {w for w in words if w not in stop and len(w) > 2}