"""
CORE API Client
Free API — 200M+ open access papers.
Get free key at: https://core.ac.uk/services/api
No key = works but slower (10 req/min limit).
"""

import logging
import time
import os
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

CORE_BASE    = "https://api.core.ac.uk/v3"
CORE_API_KEY = os.getenv("CORE_API_KEY", "")


@dataclass
class CorePaper:
    paper_id:  str
    title:     str
    authors:   list[str]
    abstract:  str
    published: str
    doi:       str
    url:       str
    publisher: str


class CoreClient:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        if not CORE_API_KEY:
            logger.warning(
                "[CoreClient] No CORE_API_KEY set. "
                "Set CORE_API_KEY for better rate limits. "
                "Get free key at: https://core.ac.uk/services/api"
            )
        self.headers = {
            "Authorization": f"Bearer {CORE_API_KEY}"
        } if CORE_API_KEY else {}
        # 10 req/min without key, 100 req/min with key
        self.delay = 0.7 if CORE_API_KEY else 7.0

    def search(self, query: str) -> list[CorePaper]:
        """Search CORE for open access papers."""
        if not CORE_API_KEY:
            logger.info(
                "[CoreClient] Skipping — no API key set. "
                "Add CORE_API_KEY to .env to enable."
            )
            return []

        logger.info(f"[CoreClient] Searching: '{query}'")
        try:
            params = {
                "q":      query,
                "limit":  self.max_results,
                "offset": 0
            }
            with httpx.Client(
                timeout=20, headers=self.headers
            ) as client:
                resp = client.get(
                    f"{CORE_BASE}/search/works",
                    params=params
                )
                time.sleep(self.delay)

                if resp.status_code == 200:
                    data   = resp.json()
                    papers = [
                        self._parse(p)
                        for p in data.get("results", [])
                        if self._has_abstract(p)
                    ]
                    logger.info(
                        f"[CoreClient] Retrieved {len(papers)} papers."
                    )
                    return papers
                elif resp.status_code == 401:
                    logger.warning(
                        "[CoreClient] Invalid API key."
                    )
                else:
                    logger.warning(
                        f"[CoreClient] HTTP {resp.status_code}"
                    )
        except Exception as e:
            logger.error(f"[CoreClient] Search failed: {e}")
        return []

    def _has_abstract(self, data: dict) -> bool:
        abstract = data.get("abstract", "") or ""
        return len(abstract.strip()) > 50

    def _parse(self, data: dict) -> CorePaper:
        authors = []
        for a in (data.get("authors") or [])[:6]:
            name = a.get("name", "")
            if name:
                authors.append(name)

        year = data.get("yearPublished", 0)
        published = f"{year}-01-01" if year else "unknown"

        doi = (data.get("doi") or "").strip()
        url = (
            data.get("downloadUrl")
            or data.get("sourceFulltextUrls", [None])[0]
            or f"https://core.ac.uk/works/{data.get('id','')}"
        )

        return CorePaper(
            paper_id  = f"core_{data.get('id', '')}",
            title     = (data.get("title") or "").strip(),
            authors   = authors,
            abstract  = (data.get("abstract") or "").strip(),
            published = published,
            doi       = doi,
            url       = url,
            publisher = (data.get("publisher") or "").strip()
        )

    def to_paper_dict(self, p: CorePaper) -> dict:
        """Convert CorePaper to standard pipeline dict."""
        return {
            "paper_id":         p.paper_id,
            "title":            p.title,
            "authors":          p.authors,
            "abstract":         p.abstract,
            "published":        p.published,
            "updated":          p.published,
            "categories":       [],
            "arxiv_url":        p.url,
            "pdf_url":          p.url,
            "domain_relevance": 0.0,
            "doi":              p.doi,
            "journal_ref":      p.publisher,
            "primary_category": "",
            "source":           "core"
        }