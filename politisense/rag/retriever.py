"""
rag/retriever.py
────────────────
Retrieval layer — embeds a query and fetches the most similar
whitehouse.gov document chunks from ChromaDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import TOP_K_WHITEHOUSE
from rag.embedder import get_embedder
from rag.chroma_store import ChromaStore


# ── result type ────────────────────────────────────────────────────────────────

@dataclass
class WhitehouseHit:
    """A single retrieved whitehouse document chunk."""
    chunk_id:   str
    post_id:    str
    title:      str
    url:        str
    categories: str
    date_iso:   str
    text:       str
    similarity: float     # 1 − cosine_distance  (higher = more relevant)


# ── Retriever ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Embeds a query and retrieves the top-k most relevant
    whitehouse.gov chunks from ChromaDB.

    Usage:
        retriever = Retriever()
        hits = retriever.retrieve_whitehouse_only("Trump steel tariff")
    """

    def __init__(self) -> None:
        self._embedder = get_embedder()
        self._store    = ChromaStore()

    def retrieve_whitehouse_only(
        self,
        query: str,
        top_k: int = TOP_K_WHITEHOUSE,
    ) -> List[WhitehouseHit]:
        q_vec = self._embedder.embed_one(query)
        raw   = self._store.query_whitehouse(q_vec, top_k=top_k)
        hits  = []
        for r in raw:
            m = r["metadata"]
            hits.append(
                WhitehouseHit(
                    chunk_id   = r["id"],
                    post_id    = m.get("post_id", ""),
                    title      = m.get("title", ""),
                    url        = m.get("url", ""),
                    categories = m.get("categories", ""),
                    date_iso   = m.get("date_iso", ""),
                    text       = r["document"],
                    similarity = round(1 - r["distance"], 4),
                )
            )
        return hits
