"""
PubMed E-utilities API Client
100% Free — US government funded, no key required.
Optional key increases rate limit.
36 million peer-reviewed biomedical papers.
"""

import logging
import time
import os
from dataclasses import dataclass
from typing import Optional
import httpx
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

PUBMED_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")


@dataclass
class PubMedPaper:
    paper_id:  str
    title:     str
    authors:   list[str]
    abstract:  str
    published: str
    journal:   str
    doi:       str
    pubmed_url: str


class PubMedClient:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        # Rate limits: 3/s without key, 10/s with key
        self.delay = 0.15 if PUBMED_API_KEY else 0.4
        self.base_params = {}
        if PUBMED_API_KEY:
            self.base_params["api_key"] = PUBMED_API_KEY

    def search(self, query: str) -> list[PubMedPaper]:
        """Search PubMed and return papers with abstracts."""
        logger.info(f"[PubMedClient] Searching: '{query}'")

        # Step 1: Get paper IDs
        pmids = self._search_ids(query)
        if not pmids:
            logger.info("[PubMedClient] No results found.")
            return []

        # Step 2: Fetch full records
        papers = self._fetch_details(pmids)
        logger.info(f"[PubMedClient] Retrieved {len(papers)} papers.")
        return papers

    def _search_ids(self, query: str) -> list[str]:
        """Get PubMed IDs for query."""
        try:
            params = {
                **self.base_params,
                "db":      "pubmed",
                "term":    query,
                "retmax":  self.max_results,
                "retmode": "json",
                "sort":    "relevance"
            }
            with httpx.Client(timeout=15) as client:
                resp = client.get(f"{PUBMED_BASE}/esearch.fcgi",
                                  params=params)
                time.sleep(self.delay)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"[PubMedClient] ID search failed: {e}")
        return []

    def _fetch_details(self, pmids: list[str]) -> list[PubMedPaper]:
        """Fetch full paper details for a list of PMIDs."""
        if not pmids:
            return []
        try:
            params = {
                **self.base_params,
                "db":      "pubmed",
                "id":      ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            with httpx.Client(timeout=20) as client:
                resp = client.get(f"{PUBMED_BASE}/efetch.fcgi",
                                  params=params)
                time.sleep(self.delay)
                if resp.status_code == 200:
                    return self._parse_xml(resp.text)
        except Exception as e:
            logger.error(f"[PubMedClient] Fetch failed: {e}")
        return []

    def _parse_xml(self, xml_text: str) -> list[PubMedPaper]:
        """Parse PubMed XML response."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper and paper.abstract:
                    papers.append(paper)
        except ET.ParseError as e:
            logger.error(f"[PubMedClient] XML parse failed: {e}")
        return papers

    def _parse_article(self, article) -> Optional[PubMedPaper]:
        """Parse a single PubMed article XML element."""
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_el = medline.find("PMID")
            pmid    = pmid_el.text if pmid_el is not None else ""

            art = medline.find("Article")
            if art is None:
                return None

            # Title
            title_el = art.find("ArticleTitle")
            title    = "".join(title_el.itertext()).strip() \
                       if title_el is not None else ""

            # Abstract
            abstract_el = art.find("Abstract/AbstractText")
            abstract    = ""
            if abstract_el is not None:
                abstract = "".join(abstract_el.itertext()).strip()

            # Fall back to structured abstract
            if not abstract:
                parts = art.findall("Abstract/AbstractText")
                abstract = " ".join(
                    "".join(p.itertext()).strip()
                    for p in parts
                ).strip()

            if not abstract:
                return None

            # Authors
            authors = []
            for author in art.findall("AuthorList/Author")[:6]:
                last  = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(
                        f"{last}, {first}".strip(", ")
                    )

            # Publication date
            pub_date = art.find("Journal/JournalIssue/PubDate")
            year     = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                if not year:
                    med_date = pub_date.findtext("MedlineDate", "")
                    year     = med_date[:4] if med_date else ""

            published = f"{year}-01-01" if year else "unknown"

            # Journal
            journal = art.findtext(
                "Journal/Title", ""
            ) or art.findtext("Journal/ISOAbbreviation", "")

            # DOI
            doi = ""
            for id_el in art.findall("ELocationID"):
                if id_el.get("EIdType") == "doi":
                    doi = id_el.text or ""
                    break

            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            return PubMedPaper(
                paper_id  = f"pm_{pmid}",
                title     = title,
                authors   = authors,
                abstract  = abstract,
                published = published,
                journal   = journal,
                doi       = doi,
                pubmed_url = pubmed_url
            )
        except Exception as e:
            logger.debug(f"[PubMedClient] Article parse error: {e}")
            return None

    def to_paper_dict(self, p: PubMedPaper) -> dict:
        """Convert PubMedPaper to standard pipeline dict."""
        return {
            "paper_id":         p.paper_id,
            "title":            p.title,
            "authors":          p.authors,
            "abstract":         p.abstract,
            "published":        p.published,
            "updated":          p.published,
            "categories":       [],
            "arxiv_url":        p.pubmed_url,
            "pdf_url":          "",
            "domain_relevance": 0.5,  # PubMed = medical = relevant
            "doi":              p.doi,
            "journal_ref":      p.journal,
            "primary_category": "medicine",
            "source":           "pubmed"
        }