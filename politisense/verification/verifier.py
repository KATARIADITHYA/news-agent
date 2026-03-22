"""
verification/verifier.py
─────────────────────────
Stage 2 — Fact-check a news claim against the whitehouse.gov corpus.

Confidence Calculation (same approach as deep-learning-rag-agent):
───────────────────────────────────────────────────────────────────
  1. Embed the news claim → query vector
  2. ChromaDB returns top-5 chunks with cosine distances
  3. Convert each distance → similarity score:
         score = 1.0 - (distance / 2.0)     range: 0.0 → 1.0
  4. Drop chunks below SIMILARITY_THRESHOLD (default 0.3)
  5. Confidence = average similarity across all chunks above threshold
         confidence = sum(scores) / len(scores)   range: 0.0 → 1.0
  6. Convert to 0–100 integer for display:
         confidence_pct = round(confidence * 100)

Status labels (based on confidence_pct):
  ≥ 75  → "verified"
  55–74 → "likely"
  35–54 → "disputed"
  < 35  → "unverified"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    CONFIDENCE_VERIFIED,
    CONFIDENCE_LIKELY,
    CONFIDENCE_DISPUTED,
    SIMILARITY_THRESHOLD,
)
from rag.retriever import Retriever, WhitehouseHit


# ── result types ───────────────────────────────────────────────────────────────

@dataclass
class ClaimVerification:
    claim:            str
    status:           str           # verified / likely / disputed / unverified
    confidence:       int           # 0–100  (avg similarity × 100)
    confidence_raw:   float         # 0.0–1.0 raw float (for debugging)
    explanation:      str
    top_hit:          Optional[WhitehouseHit] = None
    all_hits:         List[WhitehouseHit]     = field(default_factory=list)
    chunks_used:      int           = 0   # how many chunks passed threshold
    chunks_retrieved: int           = 0   # how many chunks ChromaDB returned


# ── Verifier ───────────────────────────────────────────────────────────────────

class Verifier:
    """
    Verifies a news claim against the whitehouse.gov corpus.

    Confidence = average cosine similarity of retrieved chunks (above threshold).
    Identical approach to deep-learning-rag-agent/store.py query().
    """

    def __init__(self) -> None:
        self._retriever = Retriever()

    def verify(self, claim: str) -> ClaimVerification:
        # Step 1: retrieve top-5 chunks from ChromaDB
        hits = self._retriever.retrieve_whitehouse_only(claim, top_k=5)

        chunks_retrieved = len(hits)

        if not hits:
            return ClaimVerification(
                claim=claim,
                status="unverified",
                confidence=0,
                confidence_raw=0.0,
                explanation="No matching documents found in the whitehouse.gov corpus.",
                chunks_used=0,
                chunks_retrieved=0,
            )

        # Step 2: filter chunks below similarity threshold
        # (same as deep-learning-rag-agent: drop if score < SIMILARITY_THRESHOLD)
        above_threshold = [h for h in hits if h.similarity >= SIMILARITY_THRESHOLD]

        if not above_threshold:
            # All chunks below threshold → unverified
            best = max(hits, key=lambda h: h.similarity)
            return ClaimVerification(
                claim=claim,
                status="unverified",
                confidence=0,
                confidence_raw=0.0,
                explanation=(
                    f"No chunks met the similarity threshold ({SIMILARITY_THRESHOLD}). "
                    f"Best score was {best.similarity:.3f}. "
                    f"Closest source: \"{best.title}\" — {best.url}"
                ),
                top_hit=best,
                all_hits=hits,
                chunks_used=0,
                chunks_retrieved=chunks_retrieved,
            )

        # Step 3: confidence = average similarity across chunks above threshold
        # Identical to: avg_confidence = sum(c.score for c in chunks) / len(chunks)
        avg_similarity   = sum(h.similarity for h in above_threshold) / len(above_threshold)
        confidence_pct   = round(avg_similarity * 100)

        # Step 4: pick best hit (highest similarity) for citation
        top = max(above_threshold, key=lambda h: h.similarity)

        # Step 5: classify into status label
        status, explanation = self._classify(confidence_pct, top)

        print(
            f"[Verifier] chunks_retrieved={chunks_retrieved}  "
            f"chunks_above_threshold={len(above_threshold)}  "
            f"avg_similarity={avg_similarity:.4f}  "
            f"confidence={confidence_pct}/100  "
            f"status={status}"
        )

        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence_pct,
            confidence_raw=avg_similarity,
            explanation=explanation,
            top_hit=top,
            all_hits=hits,
            chunks_used=len(above_threshold),
            chunks_retrieved=chunks_retrieved,
        )

    # ── classification ────────────────────────────────────────────────────────

    def _classify(self, confidence_pct: int, top: WhitehouseHit):
        citation = f'"{top.title}" ({top.date_iso[:10]}) — {top.url}'

        if confidence_pct >= CONFIDENCE_VERIFIED:
            status = "verified"
            explanation = (
                f"Strong match in official whitehouse.gov corpus. "
                f"Source: {citation}."
            )
        elif confidence_pct >= CONFIDENCE_LIKELY:
            status = "likely"
            explanation = (
                f"Partial match — content consistent with official records. "
                f"Closest source: {citation}."
            )
        elif confidence_pct >= CONFIDENCE_DISPUTED:
            status = "disputed"
            explanation = (
                f"Weak match — claim may refer to a different policy. "
                f"Closest source: {citation}."
            )
        else:
            status = "unverified"
            explanation = (
                "No sufficiently similar document found in the official "
                f"whitehouse.gov corpus. Closest source: {citation}."
            )

        return status, explanation
