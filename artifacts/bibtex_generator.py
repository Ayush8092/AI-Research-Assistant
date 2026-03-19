"""
BibTeX Generator — references.bib
"""

import logging
import re
from pathlib import Path
from agents.state import ResearchState

logger     = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/artifacts")


def generate_bibtex(state: ResearchState) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked_papers = state.get("ranked_papers",[])
    session_id    = state.get("session_id","")

    if not ranked_papers:
        return "% No papers found.\n"

    bib_entries = [_make_bibtex_entry(p) for p in ranked_papers[:20]]
    bib_content = "\n\n".join(bib_entries)

    file_path = OUTPUT_DIR / f"references_{session_id[:8]}.bib"
    with open(file_path,"w",encoding="utf-8") as f:
        f.write(bib_content)

    logger.info(f"[BibTeX] Saved: {file_path} ({len(bib_entries)} entries)")
    return bib_content


def _make_bibtex_entry(paper: dict) -> str:
    cite_key   = _make_cite_key(paper)
    authors    = paper.get("authors",[])
    author_str = " and ".join(authors[:6])
    if len(authors) > 6:
        author_str += " and others"
    title      = paper.get("title","").replace("{","").replace("}","")
    year       = paper.get("published","2024")[:4]
    arxiv_id   = paper.get("paper_id","")
    url        = paper.get("arxiv_url", f"https://arxiv.org/abs/{arxiv_id}")
    venue      = paper.get("venue","")
    doi        = paper.get("doi","")

    if venue and "arxiv" not in venue.lower() and venue:
        entry_type    = "article"
        journal_field = f"  journal   = {{{venue}}},"
    else:
        entry_type    = "misc"
        journal_field = f"  howpublished = {{arXiv preprint arXiv:{arxiv_id}}},"

    doi_field = f"\n  doi       = {{{doi}}}," if doi else ""

    return f"""@{entry_type}{{{cite_key},
  title     = {{{{{title}}}}},
  author    = {{{author_str}}},
  year      = {{{year}}},
{journal_field}
  url       = {{{url}}},{doi_field}
  note      = {{Accessed: 2024}}
}}"""


def _make_cite_key(paper: dict) -> str:
    authors     = paper.get("authors",[])
    year        = paper.get("published","2024")[:4]
    title_words = paper.get("title","paper").split()

    last_name = ""
    if authors:
        parts     = authors[0].split()
        last_name = parts[-1] if parts else "Unknown"

    skip_words = {"a","an","the","of","in","on","for","with","and","or"}
    title_word = ""
    for word in title_words:
        clean = re.sub(r"[^a-zA-Z]","",word).lower()
        if clean and clean not in skip_words and len(clean) > 2:
            title_word = clean.capitalize()
            break

    return re.sub(r"[^a-zA-Z]","",last_name) + year + title_word