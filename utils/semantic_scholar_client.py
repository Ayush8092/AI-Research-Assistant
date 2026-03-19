"""
Semantic Scholar API Client
Free API — 200M papers, built-in citations, no payment required.
Get free key at: https://www.semanticscholar.org/product/api
"""

import logging
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

S2_BASE    = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

FIELDS = (
    "paperId,title,abstract,year,citationCount,"
    "authors,venue,externalIds,openAccessPdf"
)


@dataclass
class S2Paper:
    paper_id:       str
    title:          str
    authors:        list[str]
    abstract:       str
    published:      str
    citation_count: int
    venue:          str
    arxiv_url:      str
    doi:            str
    is_open_access: bool


class SemanticScholarClient:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.headers     = {}
        if S2_API_KEY:
            self.headers["x-api-key"] = S2_API_KEY
        # Polite delay: 1s without key, 0.2s with key
        self.delay = 0.2 if S2_API_KEY else 1.1

    def search(self, query: str) -> list[S2Paper]:
        """Search Semantic Scholar for papers matching query."""
        logger.info(f"[S2Client] Searching: '{query}'")
        try:
            params = {
                "query":  query,
                "limit":  self.max_results,
                "fields": FIELDS
            }
            with httpx.Client(timeout=15, headers=self.headers) as client:
                resp = client.get(f"{S2_BASE}/paper/search", params=params)
                time.sleep(self.delay)

                if resp.status_code == 200:
                    data   = resp.json()
                    papers = [
                        self._parse(p)
                        for p in data.get("data", [])
                        if p.get("abstract")
                    ]
                    logger.info(f"[S2Client] Retrieved {len(papers)} papers.")
                    return papers
                else:
                    logger.warning(
                        f"[S2Client] HTTP {resp.status_code}: "
                        f"{resp.text[:100]}"
                    )
        except Exception as e:
            logger.error(f"[S2Client] Search failed: {e}")
        return []

    def _parse(self, data: dict) -> S2Paper:
        authors = [
            a.get("name", "") for a in data.get("authors", [])[:6]
        ]
        year    = data.get("year", 0)
        ext     = data.get("externalIds", {}) or {}
        arxiv_id = ext.get("ArXiv", "")
        doi      = ext.get("DOI", "")

        arxiv_url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id else ""
        )

        oa_pdf = data.get("openAccessPdf") or {}
        is_oa  = bool(oa_pdf.get("url", ""))

        return S2Paper(
            paper_id       = data.get("paperId", ""),
            title          = (data.get("title") or "").strip(),
            authors        = authors,
            abstract       = (data.get("abstract") or "").strip(),
            published      = f"{year}-01-01" if year else "unknown",
            citation_count = data.get("citationCount", 0),
            venue          = (data.get("venue") or "").strip(),
            arxiv_url      = arxiv_url,
            doi            = doi,
            is_open_access = is_oa
        )

    def to_paper_dict(self, p: S2Paper) -> dict:
        """Convert S2Paper to standard paper dict for the pipeline."""
        return {
            "paper_id":         f"s2_{p.paper_id[:20]}",
            "title":            p.title,
            "authors":          p.authors,
            "abstract":         p.abstract,
            "published":        p.published,
            "updated":          p.published,
            "categories":       [],
            "arxiv_url":        p.arxiv_url or
                                f"https://www.semanticscholar.org/paper/{p.paper_id}",
            "pdf_url":          "",
            "domain_relevance": 0.0,
            "doi":              p.doi,
            "journal_ref":      p.venue,
            "primary_category": "",
            "citation_count":   p.citation_count,
            "source":           "semantic_scholar"
        }