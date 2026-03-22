"""
rag/embedder.py
───────────────
Singleton wrapper around sentence-transformers.
Model loads once per process — never reloaded.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: "Embedder | None" = None


def get_embedder() -> "Embedder":
    """Return the shared Embedder instance (load once, reuse everywhere)."""
    global _instance
    if _instance is None:
        _instance = Embedder()
    return _instance


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print(f"[Embedder] Loading model: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        print("[Embedder] Model loaded.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
