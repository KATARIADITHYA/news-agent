"""
rag/indexer.py
──────────────
Stage 0 — Build the ChromaDB indices.

Run this once (or whenever the CSV is updated):
    python -m rag.indexer

What it does:
  1. Reads whitehouse_presidential_actions_full.csv
  2. Chunks every document's content column
  3. Embeds each chunk with sentence-transformers
  4. Upserts all chunks into the 'whitehouse_actions' ChromaDB collection

The academic-papers collection is populated lazily by entity_extractor.py
when it fetches papers for a specific query.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# allow running as a module from inside the project
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WHITEHOUSE_CSV_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from rag.chunker import chunk_document
from rag.embedder import get_embedder
from rag.chroma_store import ChromaStore


# ── constants ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64          # how many chunks to embed + upsert at once


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _build_metadata(chunk) -> Dict[str, Any]:
    """Convert a Chunk into a ChromaDB-safe metadata dict (strings only)."""
    return {
        "post_id":      chunk.post_id,
        "title":        chunk.title[:256],      # cap long titles
        "url":          chunk.url,
        "categories":   chunk.categories,
        "date_iso":     chunk.date_iso,
        "chunk_index":  str(chunk.chunk_index),
        "total_chunks": str(chunk.total_chunks),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def build_whitehouse_index(force_rebuild: bool = False) -> None:
    store = ChromaStore()
    existing = store.whitehouse_count()

    if existing > 0 and not force_rebuild:
        print(
            f"[Indexer] Whitehouse collection already has {existing} chunks. "
            "Pass force_rebuild=True to re-index."
        )
        return

    if force_rebuild and existing > 0:
        print("[Indexer] Dropping existing whitehouse collection …")
        store.reset_whitehouse()

    print(f"[Indexer] Loading CSV: {WHITEHOUSE_CSV_PATH}")
    rows = _load_csv(WHITEHOUSE_CSV_PATH)
    print(f"[Indexer] Loaded {len(rows)} presidential actions.")

    embedder = get_embedder()

    # ── collect all chunks ────────────────────────────────────────────────────
    all_chunks = []
    for row in rows:
        content = row.get("content", "").strip()
        if not content:
            continue
        chunks = chunk_document(
            post_id=row["post_id"],
            title=row.get("title", ""),
            url=row.get("url", ""),
            categories=row.get("categories", ""),
            date_iso=row.get("date_iso", ""),
            content=content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        all_chunks.extend(chunks)

    print(f"[Indexer] Total chunks to index: {len(all_chunks)}")

    # ── embed and upsert in batches ────────────────────────────────────────────
    for batch_start in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Indexing"):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]

        texts      = [c.text for c in batch]
        embeddings = embedder.embed(texts)
        ids        = [c.chunk_id for c in batch]
        metadatas  = [_build_metadata(c) for c in batch]

        store.upsert_whitehouse_chunks(
            chunk_ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    final_count = store.whitehouse_count()
    print(f"[Indexer] Done. Collection now has {final_count} chunks.")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB whitehouse index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing collection and re-index from scratch",
    )
    args = parser.parse_args()
    build_whitehouse_index(force_rebuild=args.rebuild)
