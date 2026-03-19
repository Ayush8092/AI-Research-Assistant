"""
OpenAlex API Client — free citation metadata, no API key required.
"""

import httpx
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"

HIGH_TIER_VENUES = {
    "nature","science","cell","lancet","nejm","jama","neurips","icml","iclr",
    "cvpr","acl","emnlp","nature medicine","nature communications","bmj",
    "ieee transactions","journal of medical internet research"
}
MEDIUM_TIER_VENUES = {
    "arxiv","plos","frontiers","mdpi","bmc","international journal",
    "applied","computational"
}


@dataclass
class OpenAlexMetadata:
    openalex_id:      str
    doi:              str
    title:            str
    citation_count:   int
    venue:            str
    venue_tier:       str
    publication_year: int
    is_open_access:   bool
    extra:            dict = field(default_factory=dict)


class OpenAlexClient:
    def __init__(self, email: str = "researcher@example.com"):
        self.headers = {"User-Agent": f"AIResearchAssistant/1.0 (mailto:{email})"}
        self.delay   = 0.15

    def fetch_metadata(self, doi: str = "", title: str = "") -> Optional[OpenAlexMetadata]:
        result = None
        if doi:
            result = self._fetch_by_doi(doi)
        if result is None and title:
            result = self._fetch_by_title(title)
        time.sleep(self.delay)
        return result

    def fetch_batch(self, papers: list[dict]) -> dict[str, OpenAlexMetadata]:
        results = {}
        for paper in papers:
            pid  = paper.get("paper_id", "")
            meta = self.fetch_metadata(doi=paper.get("doi",""), title=paper.get("title",""))
            results[pid] = meta if meta else self._empty_metadata(paper.get("title",""))
        return results

    def _fetch_by_doi(self, doi: str) -> Optional[OpenAlexMetadata]:
        try:
            with httpx.Client(timeout=10, headers=self.headers) as client:
                resp = client.get(f"{OPENALEX_BASE}/works/doi:{doi}")
                if resp.status_code == 200:
                    return self._parse_work(resp.json())
        except Exception as e:
            logger.debug(f"[OpenAlex] DOI lookup failed: {e}")
        return None

    def _fetch_by_title(self, title: str) -> Optional[OpenAlexMetadata]:
        try:
            params = {"filter": f"title.search:{title[:100]}", "per-page": 1}
            with httpx.Client(timeout=10, headers=self.headers) as client:
                resp = client.get(f"{OPENALEX_BASE}/works", params=params)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        return self._parse_work(results[0])
        except Exception as e:
            logger.debug(f"[OpenAlex] Title lookup failed: {e}")
        return None

    def _parse_work(self, work: dict) -> OpenAlexMetadata:
        venue    = ""
        location = work.get("primary_location", {})
        if location:
            source = location.get("source", {}) or {}
            venue  = source.get("display_name", "")
        return OpenAlexMetadata(
            openalex_id=work.get("id",""),
            doi=work.get("doi","") or "",
            title=work.get("display_name",""),
            citation_count=work.get("cited_by_count", 0),
            venue=venue,
            venue_tier=self._classify_venue(venue),
            publication_year=work.get("publication_year", 0),
            is_open_access=work.get("open_access",{}).get("is_oa", False),
            extra={"type": work.get("type",""), "referenced_works_count": work.get("referenced_works_count",0)}
        )

    def _classify_venue(self, venue: str) -> str:
        v = venue.lower()
        for kw in HIGH_TIER_VENUES:
            if kw in v: return "high"
        for kw in MEDIUM_TIER_VENUES:
            if kw in v: return "medium"
        return "low" if venue else "unknown"

    def _empty_metadata(self, title: str = "") -> OpenAlexMetadata:
        return OpenAlexMetadata("","",title,0,"","unknown",0,False)