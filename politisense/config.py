"""
config.py
──────────
PolitiSense — Centralised settings for the pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
WHITEHOUSE_CSV_PATH = os.getenv(
    "WHITEHOUSE_CSV_PATH",
    str(BASE_DIR.parent / "whitehouse_presidential_actions_full.csv"),
)

# ── API Keys ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Embedding model (local, no API key required) ───────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB collection name ───────────────────────────────────────────────────
CHROMA_COLLECTION_WHITEHOUSE = os.getenv(
    "CHROMA_COLLECTION_WHITEHOUSE", "whitehouse_actions"
)

# ── RAG settings ───────────────────────────────────────────────────────────────
TOP_K_WHITEHOUSE = 5          # chunks to retrieve per query
CHUNK_SIZE       = 512        # characters per chunk
CHUNK_OVERLAP    = 64         # overlap between chunks

# ── Verification thresholds ────────────────────────────────────────────────────

# Similarity threshold — chunks below this score are dropped before averaging
# Same as deep-learning-rag-agent SIMILARITY_THRESHOLD (default 0.3)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# Confidence % bands (avg_similarity × 100)
CONFIDENCE_VERIFIED  = 75     # ≥75  → "verified"
CONFIDENCE_LIKELY    = 55     # 55–74 → "likely"
CONFIDENCE_DISPUTED  = 35     # 35–54 → "disputed"
                              # <35   → "unverified"
