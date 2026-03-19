from .arxiv_client import ArxivClient, ArxivPaper
from .openalex_client import OpenAlexClient, OpenAlexMetadata
from .semantic_scholar_client import SemanticScholarClient, S2Paper
from .pubmed_client import PubMedClient, PubMedPaper
from .core_client import CoreClient, CorePaper

__all__ = [
    "ArxivClient", "ArxivPaper",
    "OpenAlexClient", "OpenAlexMetadata",
    "SemanticScholarClient", "S2Paper",
    "PubMedClient", "PubMedPaper",
    "CoreClient", "CorePaper"
]