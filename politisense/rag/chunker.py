"""
rag/chunker.py
──────────────
Splits long whitehouse.gov document content into overlapping
fixed-size character chunks so each chunk fits inside the
embedding model's context window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: str           # "{post_id}__{chunk_index}"
    post_id: str
    title: str
    url: str
    categories: str
    date_iso: str
    text: str               # the actual chunk text
    chunk_index: int
    total_chunks: int


def chunk_document(
    post_id: str,
    title: str,
    url: str,
    categories: str,
    date_iso: str,
    content: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Splits `content` into overlapping character-level chunks.
    Always produces at least one chunk even for very short content.
    """
    content = content.strip()
    if not content:
        return []

    chunks: List[Chunk] = []
    start = 0
    index = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))
        text = content[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{post_id}__{index}",
                post_id=str(post_id),
                title=title,
                url=url,
                categories=categories,
                date_iso=date_iso,
                text=text,
                chunk_index=index,
                total_chunks=0,   # filled in below
            )
        )
        if end == len(content):
            break
        start += chunk_size - overlap
        index += 1

    # back-fill total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    return chunks
