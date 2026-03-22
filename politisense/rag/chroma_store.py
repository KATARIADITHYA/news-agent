"""
rag/chroma_store.py
────────────────────
All ChromaDB interactions — single collection: whitehouse_actions.

Design notes
────────────
• PersistentClient — data survives between runs.
• Pre-computed embeddings passed in — we control the model (no ChromaDB built-in).
• IDs are chunk_ids from the chunker.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_WHITEHOUSE,
    TOP_K_WHITEHOUSE,
)


class ChromaStore:
    """Facade over the whitehouse_actions ChromaDB collection."""

    def __init__(self, db_path: str = CHROMA_DB_PATH) -> None:
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._wh_col = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_WHITEHOUSE,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Whitehouse collection ──────────────────────────────────────────────────

    def upsert_whitehouse_chunks(
        self,
        chunk_ids:  List[str],
        embeddings: List[List[float]],
        documents:  List[str],
        metadatas:  List[Dict[str, Any]],
    ) -> None:
        """Insert or update a batch of whitehouse document chunks."""
        self._wh_col.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_whitehouse(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K_WHITEHOUSE,
    ) -> List[Dict[str, Any]]:
        """Return top-k most relevant whitehouse chunks for a query vector."""
        results = self._wh_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return _flatten(results)

    def whitehouse_count(self) -> int:
        return self._wh_col.count()

    def reset_whitehouse(self) -> None:
        """Drop and recreate the whitehouse collection (for full re-index)."""
        self._client.delete_collection(CHROMA_COLLECTION_WHITEHOUSE)
        self._wh_col = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_WHITEHOUSE,
            metadata={"hnsw:space": "cosine"},
        )


# ── helpers ────────────────────────────────────────────────────────────────────

def _flatten(raw: Dict) -> List[Dict[str, Any]]:
    """ChromaDB returns parallel lists — flatten into list of dicts."""
    ids       = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]
    return [
        {
            "id":       ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]
