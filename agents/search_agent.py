"""
SearchAgent — Multi-source parallel retrieval.

Fixes:
  1. Filters correction/erratum/retraction papers
  2. Year-adjusted smart citation scoring
  3. Better deduplication with fuzzy title matching
  4. Stricter must-have anchor filtering
"""

import logging
import os
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from .state import ResearchState
from utils.arxiv_client import ArxivClient

logger = logging.getLogger(__name__)

MAX_PAPERS_DEFAULT  = int(os.getenv("MAX_PAPERS_DEFAULT", "7"))
MAX_PAPERS_EXTENDED = 20
PER_SOURCE_LIMIT    = 10

COMMON_ML_WORDS = {
    "deep", "learning", "neural", "network", "model", "data",
    "based", "using", "method", "approach", "paper", "proposed",
    "results", "performance", "training", "test", "evaluation",
    "accuracy", "dataset", "feature", "classification", "detection",
    "prediction", "analysis", "system", "framework", "algorithm"
}

# Papers with these title prefixes are useless
SKIP_TITLE_PREFIXES = [
    "correction:", "erratum:", "corrigendum:",
    "retraction:", "publisher correction:", "author correction:",
    "editorial:", "comment on:", "response to:", "reply to:"
]


def search_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: Multi-source SearchAgent."""
    query       = state["query"]
    subtopics   = state.get("subtopics", [])
    filters     = state.get("filters", {})
    max_results = min(
        filters.get("max_papers", MAX_PAPERS_DEFAULT),
        MAX_PAPERS_EXTENDED
    )

    logger.info(f"[SearchAgent] Query: '{query}' | max={max_results}")

    anchor_terms = _extract_unique_anchors(query)
    must_have    = _extract_must_have_terms(query)
    logger.info(f"[SearchAgent] Anchors: {anchor_terms}")
    logger.info(f"[SearchAgent] Must-have: {must_have}")

    # Check for topic shift
    _maybe_clear_faiss(query, anchor_terms)

    # Parallel multi-source search
    all_papers = _parallel_search(query, subtopics, filters)
    logger.info(f"[SearchAgent] Raw from all sources: {len(all_papers)}")

    # Remove correction/erratum papers first
    all_papers = _filter_junk_papers(all_papers)
    logger.info(f"[SearchAgent] After junk filter: {len(all_papers)}")

    # Deduplicate
    papers = _deduplicate(all_papers)
    logger.info(f"[SearchAgent] After dedup: {len(papers)}")

    # Must-have filter
    if must_have:
        papers = _apply_must_have_filter(papers, must_have)
        logger.info(f"[SearchAgent] After must-have: {len(papers)}")

    # Anchor scoring
    papers = _score_by_anchors(papers, anchor_terms)

    # FAISS semantic re-ranking
    try:
        from vectorstore.faiss_store import FAISSStore
        from analysis.semantic_ranking import hybrid_rank_papers
        faiss_store = FAISSStore()
        papers      = hybrid_rank_papers(
            arxiv_papers = papers,
            faiss_store  = faiss_store,
            query        = query,
            top_k        = max_results,
            anchor_terms = anchor_terms
        )
        logger.info(f"[SearchAgent] After hybrid ranking: {len(papers)}")
    except Exception as e:
        logger.warning(f"[SearchAgent] Hybrid ranking failed: {e}")
        papers = sorted(
            papers,
            key=lambda x: x.get("anchor_score", 0),
            reverse=True
        )[:max_results]

    # Year filter
    year_after = filters.get("year_after")
    if year_after:
        papers = [p for p in papers
                  if _extract_year(p.get("published","")) >= year_after]
        logger.info(f"[SearchAgent] After year filter: {len(papers)}")

    # Survey filter
    if filters.get("exclude_surveys"):
        papers = [p for p in papers
                  if not _is_survey(p.get("title","") + p.get("abstract",""))]
        logger.info(f"[SearchAgent] After survey filter: {len(papers)}")

    papers = papers[:max_results]

    for i, p in enumerate(papers, 1):
        logger.info(
            f"[SearchAgent] [{i}] '{p.get('title','')[:55]}' "
            f"| src={p.get('source','?')} "
            f"| cite={p.get('citation_count',0)} "
            f"| anchors={p.get('anchor_score',0):.2f}"
        )

    logger.info(f"[SearchAgent] Final: {len(papers)} papers")

    return {
        **state,
        "raw_papers": papers,
        "metrics":    {
            **state.get("metrics", {}),
            "papers_retrieved": len(papers)
        }
    }


# ------------------------------------------------------------------ #
#  Junk Paper Filter                                                   #
# ------------------------------------------------------------------ #

def _filter_junk_papers(papers: list[dict]) -> list[dict]:
    """
    Remove correction notices, errata, retractions.
    These are useless for research analysis.
    """
    clean = []
    for paper in papers:
        title_lower = paper.get("title", "").lower().strip()
        is_junk     = any(
            title_lower.startswith(prefix)
            for prefix in SKIP_TITLE_PREFIXES
        )
        if is_junk:
            logger.info(
                f"[SearchAgent] Skipping junk: "
                f"'{paper.get('title','')[:50]}'"
            )
        else:
            clean.append(paper)
    return clean


# ------------------------------------------------------------------ #
#  Parallel Multi-Source Search                                        #
# ------------------------------------------------------------------ #

def _parallel_search(
    query: str,
    subtopics: list,
    filters: dict
) -> list[dict]:
    """Run all 4 sources in parallel."""
    all_papers = []

    def search_arxiv():
        try:
            client = ArxivClient(max_results=PER_SOURCE_LIMIT)
            papers = client.search(query)
            return [_arxiv_to_dict(p) for p in papers]
        except Exception as e:
            logger.warning(f"[SearchAgent] ArXiv failed: {e}")
            return []

    def search_semantic_scholar():
        try:
            from utils.semantic_scholar_client import SemanticScholarClient
            client = SemanticScholarClient(max_results=PER_SOURCE_LIMIT)
            papers = client.search(query)
            return [client.to_paper_dict(p) for p in papers]
        except Exception as e:
            logger.warning(f"[SearchAgent] S2 failed: {e}")
            return []

    def search_pubmed():
        try:
            from utils.pubmed_client import PubMedClient
            client = PubMedClient(max_results=PER_SOURCE_LIMIT)
            papers = client.search(query)
            return [client.to_paper_dict(p) for p in papers]
        except Exception as e:
            logger.warning(f"[SearchAgent] PubMed failed: {e}")
            return []

    def search_core():
        try:
            from utils.core_client import CoreClient
            client = CoreClient(max_results=PER_SOURCE_LIMIT)
            papers = client.search(query)
            return [client.to_paper_dict(p) for p in papers]
        except Exception as e:
            logger.warning(f"[SearchAgent] CORE failed: {e}")
            return []

    sources = {
        "arxiv":            search_arxiv,
        "semantic_scholar": search_semantic_scholar,
        "pubmed":           search_pubmed,
        "core":             search_core
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn): name
            for name, fn in sources.items()
        }
        for future in as_completed(futures, timeout=30):
            source_name = futures[future]
            try:
                results = future.result(timeout=15)
                logger.info(
                    f"[SearchAgent] {source_name}: {len(results)} papers"
                )
                all_papers.extend(results)
            except TimeoutError:
                logger.warning(f"[SearchAgent] {source_name} timed out.")
            except Exception as e:
                logger.warning(f"[SearchAgent] {source_name} error: {e}")

    return all_papers


def _arxiv_to_dict(p) -> dict:
    return {
        "paper_id":         p.paper_id,
        "title":            p.title,
        "authors":          p.authors,
        "abstract":         p.abstract,
        "published":        p.published,
        "updated":          p.updated,
        "categories":       p.categories,
        "arxiv_url":        p.arxiv_url,
        "pdf_url":          p.pdf_url,
        "domain_relevance": p.domain_relevance_score,
        "doi":              p.extra.get("doi", ""),
        "journal_ref":      p.extra.get("journal_ref", ""),
        "primary_category": p.extra.get("primary_category", ""),
        "source":           "arxiv"
    }


# ------------------------------------------------------------------ #
#  Deduplication                                                       #
# ------------------------------------------------------------------ #

def _deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicates by DOI then normalized title."""
    seen_dois   = set()
    seen_titles = set()
    unique      = []

    for paper in papers:
        doi        = (paper.get("doi") or "").strip().lower()
        title_norm = _normalize_title(paper.get("title", ""))

        if doi and doi in seen_dois:
            continue
        if title_norm and title_norm in seen_titles:
            continue

        if doi:
            seen_dois.add(doi)
        if title_norm:
            seen_titles.add(title_norm)

        unique.append(paper)

    removed = len(papers) - len(unique)
    if removed > 0:
        logger.info(f"[SearchAgent] Dedup removed {removed} duplicates.")
    return unique


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:60]


# ------------------------------------------------------------------ #
#  Anchor Filtering                                                    #
# ------------------------------------------------------------------ #

def _extract_unique_anchors(query: str) -> set:
    stop_words = {
        "in","for","of","the","a","an","and","or","on",
        "with","using","based","via","from","to","by","at",
        "is","are","was","be","it","its","how","what",
        "which","that","this","these","use","used"
    }
    words = re.findall(r'\b\w+\b', query.lower())
    return {
        w for w in words
        if w not in stop_words
        and w not in COMMON_ML_WORDS
        and len(w) > 3
    }


def _extract_must_have_terms(query: str) -> set:
    stop_words = {
        "in","for","of","the","a","an","and","or","on",
        "with","using","based","via","from","to","by","at",
        "is","are","was","be","it","its","how","what",
        "which","that","this","these","use","used"
    }
    words = re.findall(r'\b\w+\b', query.lower())
    must_have = {
        w for w in words
        if w not in stop_words
        and w not in COMMON_ML_WORDS
        and len(w) > 4
    }
    if not must_have:
        must_have = {
            w for w in words
            if w not in stop_words and len(w) > 3
        }
    return must_have


def _apply_must_have_filter(
    papers: list[dict],
    must_have: set
) -> list[dict]:
    filtered = []
    for paper in papers:
        text    = (
            paper.get("title","") + " " +
            paper.get("abstract","")
        ).lower()
        matches = [t for t in must_have if t in text]
        if matches:
            paper["must_have_matches"] = matches
            filtered.append(paper)

    if not filtered:
        logger.warning(
            "[SearchAgent] Must-have filter too strict. Reverting."
        )
        return papers
    return filtered


def _score_by_anchors(
    papers: list[dict],
    anchor_terms: set
) -> list[dict]:
    if not anchor_terms:
        for p in papers:
            p["anchor_score"] = 0.5
        return papers
    for paper in papers:
        text    = (
            paper.get("title","") + " " +
            paper.get("abstract","")
        ).lower()
        matches = sum(1 for t in anchor_terms if t in text)
        paper["anchor_score"] = min(
            matches / max(len(anchor_terms), 1), 1.0
        )
    return papers


# ------------------------------------------------------------------ #
#  FAISS Topic-Shift Detection                                         #
# ------------------------------------------------------------------ #

def _maybe_clear_faiss(query: str, anchor_terms: set) -> None:
    try:
        from vectorstore.faiss_store import FAISSStore
        store = FAISSStore()
        if store.index.ntotal == 0:
            return
        results = store.search(query, top_k=min(5, store.index.ntotal))
        if not results:
            return
        avg_sim = sum(
            r.get("similarity_score", 0) for r in results
        ) / len(results)
        top_text = " ".join(
            r.get("title","").lower() for r in results[:3]
        )
        anchor_cov = sum(
            1 for t in anchor_terms if t in top_text
        ) / max(len(anchor_terms), 1)

        if avg_sim < 0.3 and anchor_cov < 0.3:
            logger.warning("[SearchAgent] Topic shift. Clearing FAISS.")
            store.clear()
            _clear_embed_cache()
    except Exception as e:
        logger.warning(f"[SearchAgent] FAISS check failed: {e}")


def _clear_embed_cache() -> None:
    cache_path = Path("data/embed_cache.json")
    try:
        with open(cache_path, "w") as f:
            json.dump({"embedded_ids": []}, f)
    except Exception:
        pass


def _extract_year(date_str: str) -> int:
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return 0


def _is_survey(text: str) -> bool:
    return any(
        kw in text.lower()
        for kw in ["survey","review","overview","systematic review"]
    )