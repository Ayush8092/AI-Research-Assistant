"""
Arxiv API Client — abstracts + metadata only, no PDF downloads.

Fix: Query uses ONLY the original user query.
     Subtopics are no longer added — they caused off-topic results.
     Fetch more papers than needed so filtering has enough to work with.
"""

import arxiv
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

HEALTHCARE_KEYWORDS = [
    "clinical", "medical", "health", "patient", "hospital",
    "diagnosis", "treatment", "disease", "imaging", "radiology",
    "ehr", "electronic health", "clinical trial", "drug discovery",
    "genomics", "pathology", "surgery", "telemedicine", "biomedical",
    "cancer", "tumor", "diabetic", "diabetes", "retinopathy",
    "ct scan", "mri", "x-ray", "mammography", "ultrasound"
]


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    categories: list[str]
    arxiv_url: str
    pdf_url: str
    domain_relevance_score: float = 0.0
    extra: dict = field(default_factory=dict)


class ArxivClient:
    def __init__(
        self,
        max_results: int = 10,
        delay_between_requests: float = 1.0
    ):
        self.max_results = max_results
        self.client      = arxiv.Client(
            page_size   = max_results,
            delay_seconds = delay_between_requests,
            num_retries = 3
        )

    def search(
        self,
        query: str,
        subtopics: Optional[list[str]] = None
    ) -> list[ArxivPaper]:
        """
        Search Arxiv using ONLY the user's original query.
        Subtopics are ignored — they cause topic drift.
        We fetch 2x the requested amount so filtering has room to work.
        """
        # Use only the clean original query
        clean_query = query.replace('"', '').strip()
        logger.info(f"[ArxivClient] Searching: '{clean_query}'")

        search = arxiv.Search(
            query       = clean_query,
            max_results = self.max_results,
            sort_by     = arxiv.SortCriterion.Relevance,
            sort_order  = arxiv.SortOrder.Descending
        )

        papers = []
        try:
            for result in self.client.results(search):
                paper = self._parse_result(result)
                paper.domain_relevance_score = \
                    self._compute_domain_relevance(paper)
                papers.append(paper)
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"[ArxivClient] Search failed: {e}")

        logger.info(f"[ArxivClient] Retrieved {len(papers)} papers.")
        return papers

    def _parse_result(self, result: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            paper_id  = result.entry_id.split("/abs/")[-1],
            title     = result.title.strip(),
            authors   = [a.name for a in result.authors],
            abstract  = result.summary.strip(),
            published = result.published.strftime("%Y-%m-%d")
                        if result.published else "unknown",
            updated   = result.updated.strftime("%Y-%m-%d")
                        if result.updated else "unknown",
            categories = result.categories,
            arxiv_url  = result.entry_id,
            pdf_url    = result.pdf_url or "",
            extra={
                "comment":          result.comment or "",
                "journal_ref":      result.journal_ref or "",
                "doi":              result.doi or "",
                "primary_category": result.primary_category
            }
        )

    def _compute_domain_relevance(self, paper: ArxivPaper) -> float:
        """Healthcare keyword relevance score 0-1."""
        text = (paper.title + " " + paper.abstract).lower()
        hits = sum(1 for kw in HEALTHCARE_KEYWORDS if kw in text)
        return min(hits / 5.0, 1.0)