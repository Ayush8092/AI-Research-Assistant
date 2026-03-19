"""
FAISS Vector Store
Stores embeddings for paper abstracts and extracted insights.
Uses all-MiniLM-L6-v2 for lightweight 384-dim embeddings.

Optimizations applied:
  1. Batch Embeddings   — add_papers_batch() encodes all texts in one call
  5. Embedding Cache    — skips papers already in metadata index
  Fix: Singleton encoder — model loaded only ONCE per process (not 4x)
"""

import json
import logging
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MAX_DOCUMENTS     = 500
EMBEDDING_DIM     = 384
INDEX_PATH        = "data/faiss_index.bin"
META_PATH         = "data/faiss_meta.json"
ENCODE_BATCH_SIZE = 16

# ------------------------------------------------------------------ #
#  Singleton encoder — loaded ONCE per process, shared by all         #
#  FAISSStore instances and paper_clustering.py                       #
# ------------------------------------------------------------------ #

_encoder_singleton: SentenceTransformer = None


def _get_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Return a shared SentenceTransformer instance.
    The model is downloaded and loaded only on the first call.
    All subsequent calls return the same object — no re-loading.
    """
    global _encoder_singleton
    if _encoder_singleton is None:
        logger.info(f"[FAISSStore] Loading embedding model (once): {model_name}")
        _encoder_singleton = SentenceTransformer(model_name)
        logger.info(f"[FAISSStore] Embedding model ready.")
    return _encoder_singleton


# ------------------------------------------------------------------ #
#  FAISSStore class                                                    #
# ------------------------------------------------------------------ #

class FAISSStore:
    """
    Lightweight FAISS-based vector store.

    Key methods:
      add_papers()        — delegates to add_papers_batch()
      add_papers_batch()  — batch encode + add (Optimization 1)
      search()            — semantic similarity search
      is_embedded()       — O(1) cache check (Optimization 5)
      clear()             — reset index
      stats()             — index statistics
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        meta_path:  str = META_PATH,
        max_docs:   int = MAX_DOCUMENTS,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = Path(index_path)
        self.meta_path  = Path(meta_path)
        self.max_docs   = max_docs
        self.model_name = model_name

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # ✅ FIX: Use singleton — does NOT reload the model if already loaded
        self.encoder = _get_encoder(model_name)

        self.index    = self._load_or_create_index()
        self.metadata = self._load_metadata()

        # Optimization 5: in-memory ID set for O(1) duplicate check
        self._known_ids: set = {m["paper_id"] for m in self.metadata}

        logger.info(f"[FAISSStore] Ready. Documents: {len(self.metadata)}")

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_papers_batch(self, papers: list[dict]) -> int:
        """
        Optimization 1: Batch-encode all new papers in a single encoder call.

        - Filters already-known papers using in-memory ID set (Opt 5)
        - Encodes all new abstracts in one batch call (Opt 1)
        - Adds resulting vectors to FAISS index in one operation
        """
        import time

        # Filter duplicates using O(1) set lookup (Optimization 5)
        new_papers = [p for p in papers if p["paper_id"] not in self._known_ids]

        if not new_papers:
            logger.info("[FAISSStore] No new papers to embed (all cached).")
            return 0

        # Enforce document limit
        available  = self.max_docs - len(self.metadata)
        new_papers = new_papers[:available]

        if not new_papers:
            logger.warning("[FAISSStore] Index full. Cannot add more papers.")
            return 0

        # Build text list: title + abstract for richer embeddings
        texts = [
            f"{p.get('title', '')} {p.get('abstract', '')}".strip()
            for p in new_papers
        ]

        # Single batch encode call (Optimization 1)
        logger.info(f"[FAISSStore] Batch encoding {len(texts)} papers "
                    f"(batch_size={ENCODE_BATCH_SIZE})...")
        t0 = time.time()

        embeddings = self.encoder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        elapsed = time.time() - t0
        logger.info(f"[FAISSStore] Encoding complete in {elapsed:.2f}s.")

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Update metadata and known-ID cache
        base_offset = len(self.metadata)
        new_meta = [
            {
                "paper_id":     p["paper_id"],
                "title":        p.get("title", ""),
                "abstract":     p.get("abstract", ""),
                "published":    p.get("published", ""),
                "authors":      p.get("authors", []),
                "index_offset": base_offset + i
            }
            for i, p in enumerate(new_papers)
        ]

        self.metadata.extend(new_meta)
        self._known_ids.update(p["paper_id"] for p in new_papers)

        self._save()
        logger.info(f"[FAISSStore] Added {len(new_papers)} papers. "
                    f"Total: {len(self.metadata)}")
        return len(new_papers)

    def add_papers(self, papers: list[dict]) -> int:
        """
        Original add method — delegates to add_papers_batch().
        Kept for backward compatibility with any code calling add_papers().
        """
        return self.add_papers_batch(papers)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search over stored paper abstracts."""
        if self.index.ntotal == 0:
            logger.warning("[FAISSStore] Index is empty.")
            return []

        q = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        k       = min(top_k, self.index.ntotal)
        D, I    = self.index.search(q, k)
        results = []

        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["similarity_score"] = float(dist)
                results.append(meta)

        return results

    def is_embedded(self, paper_id: str) -> bool:
        """
        Optimization 5: O(1) check whether a paper is already embedded.
        Used by ReaderAgent before deciding whether to re-embed.
        """
        return paper_id in self._known_ids

    def clear(self) -> None:
        """Reset the index, metadata, and known-ID cache."""
        self.index      = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.metadata   = []
        self._known_ids = set()
        self._save()
        logger.info("[FAISSStore] Index cleared.")

    def stats(self) -> dict:
        return {
            "total_documents": len(self.metadata),
            "index_vectors":   self.index.ntotal,
            "max_documents":   self.max_docs,
            "embedding_dim":   EMBEDDING_DIM,
            "model":           self.model_name,
            "known_ids_cache": len(self._known_ids)
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                logger.info(f"[FAISSStore] Loaded index: {idx.ntotal} vectors.")
                return idx
            except Exception as e:
                logger.warning(f"[FAISSStore] Load failed: {e}. Creating new index.")
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    def _load_metadata(self) -> list[dict]:
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[FAISSStore] Metadata load failed: {e}")
        return []

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)