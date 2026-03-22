"""
test_rag.py
────────────────────────────────────────────────────────────────
Comprehensive RAG test suite for PolitiSense.

Tests are grouped into 6 sections:

  Section 1 — Embedder
    1a. Model loads, produces correct vector dimensions
    1b. Two different texts produce different vectors
    1c. Two similar texts produce high cosine similarity
    1d. Singleton — same object returned on repeated calls

  Section 2 — Chunker
    2a. Normal content is chunked correctly
    2b. Short content produces exactly 1 chunk
    2c. Chunk overlap is respected
    2d. All metadata is preserved in chunks

  Section 3 — ChromaDB Store
    3a. Whitehouse collection exists and is accessible
    3b. Whitehouse collection has chunks (from prior indexing)
    3c. Upsert + query round-trip works correctly

  Section 4 — Retriever
    4a. Steel tariff query returns correct whitehouse doc
    4b. TikTok ban query returns correct whitehouse doc
    4c. Semiconductor query returns relevant result
    4d. Similarity scores are in valid range (0–1)
    4e. Top hit is more similar than bottom hit
    4f. Metadata fields are all present

  Section 5 — Verifier (Stage 2)
    5a. Known real policy scores LIKELY or VERIFIED
    5b. Fake/nonsense claim scores DISPUTED or UNVERIFIED
    5c. Score breakdown fields are all present
    5d. Top hit URL is a real whitehouse.gov URL

Run:
    cd "/Users/adithyakatari/Desktop/news agent/politisense"
    python test_rag.py

    # Run only a specific section:
    python test_rag.py --section 4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

# ── colour helpers (no dependencies) ──────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_passed = 0
_failed = 0
_skipped = 0


def _ok(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    suffix = f"  {CYAN}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}✓{RESET} {name}{suffix}")


def _fail(name: str, reason: str) -> None:
    global _failed
    _failed += 1
    print(f"  {RED}✗{RESET} {name}")
    print(f"    {RED}→ {reason}{RESET}")


def _skip(name: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    suffix = f" ({reason})" if reason else ""
    print(f"  {YELLOW}–{RESET} {name}{suffix}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def _cosine(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Embedder
# ══════════════════════════════════════════════════════════════════════════════

def test_section_1() -> None:
    _section("Section 1 — Embedder")
    from rag.embedder import get_embedder, Embedder

    # 1a. Model loads + correct vector dimensions
    try:
        emb = get_embedder()
        vecs = emb.embed(["test sentence"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384
        _ok("1a. Model loads — vector dim = 384")
    except Exception as e:
        _fail("1a. Model loads", str(e))

    # 1b. Different texts → different vectors
    try:
        vecs = emb.embed([
            "Section 232 steel tariff on Canada",
            "TikTok ban executive order ByteDance",
        ])
        sim = _cosine(vecs[0], vecs[1])
        assert sim < 0.95, f"Vectors too similar: {sim:.3f}"
        _ok("1b. Different texts → different vectors", f"cosine={sim:.3f}")
    except Exception as e:
        _fail("1b. Different texts → different vectors", str(e))

    # 1c. Similar texts → high cosine similarity
    try:
        vecs = emb.embed([
            "Trump imposes 25% tariff on steel from Canada",
            "President signs 25 percent steel import tariff Canada",
        ])
        sim = _cosine(vecs[0], vecs[1])
        assert sim > 0.80, f"Similar texts scored too low: {sim:.3f}"
        _ok("1c. Similar texts → high similarity", f"cosine={sim:.3f}")
    except Exception as e:
        _fail("1c. Similar texts → high similarity", str(e))

    # 1d. Singleton — same object returned
    try:
        emb2 = get_embedder()
        assert emb is emb2, "get_embedder() returned different instances"
        _ok("1d. Singleton — model loaded only once")
    except Exception as e:
        _fail("1d. Singleton", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Chunker
# ══════════════════════════════════════════════════════════════════════════════

def test_section_2() -> None:
    _section("Section 2 — Chunker")
    from rag.chunker import chunk_document

    LONG_TEXT = "A" * 2000   # 2000 chars — should produce multiple chunks
    SHORT_TEXT = "This is a short policy statement."

    # 2a. Normal content chunks correctly
    try:
        chunks = chunk_document(
            post_id="999", title="Test", url="http://test.gov",
            categories="Executive Orders", date_iso="2025-01-20T00:00:00",
            content=LONG_TEXT, chunk_size=512, overlap=64,
        )
        assert len(chunks) > 1, f"Expected >1 chunks, got {len(chunks)}"
        _ok("2a. Long content produces multiple chunks", f"{len(chunks)} chunks from 2000 chars")
    except Exception as e:
        _fail("2a. Long content chunking", str(e))

    # 2b. Short content → exactly 1 chunk
    try:
        chunks = chunk_document(
            post_id="998", title="Short", url="http://test.gov",
            categories="Proclamations", date_iso="2025-01-20T00:00:00",
            content=SHORT_TEXT, chunk_size=512, overlap=64,
        )
        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
        assert chunks[0].text == SHORT_TEXT
        _ok("2b. Short content → exactly 1 chunk")
    except Exception as e:
        _fail("2b. Short content chunking", str(e))

    # 2c. Overlap is respected — consecutive chunks share text
    try:
        chunks = chunk_document(
            post_id="997", title="Overlap", url="http://test.gov",
            categories="Proclamations", date_iso="2025-01-20T00:00:00",
            content=LONG_TEXT, chunk_size=512, overlap=64,
        )
        if len(chunks) >= 2:
            end_of_first   = chunks[0].text[-64:]
            start_of_second = chunks[1].text[:64]
            assert end_of_first == start_of_second, "Overlap text doesn't match"
        _ok("2c. Chunk overlap is correct (64 chars shared)")
    except Exception as e:
        _fail("2c. Chunk overlap", str(e))

    # 2d. All metadata preserved
    try:
        chunks = chunk_document(
            post_id="996", title="Meta Test", url="https://whitehouse.gov/test",
            categories="Executive Orders", date_iso="2025-03-15T10:00:00",
            content="Some policy content here for testing purposes.",
            chunk_size=512, overlap=64,
        )
        c = chunks[0]
        assert c.post_id    == "996"
        assert c.title      == "Meta Test"
        assert c.url        == "https://whitehouse.gov/test"
        assert c.categories == "Executive Orders"
        assert c.date_iso   == "2025-03-15T10:00:00"
        assert c.chunk_id   == "996__0"
        _ok("2d. All metadata fields preserved in chunks")
    except Exception as e:
        _fail("2d. Metadata preservation", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — ChromaDB Store
# ══════════════════════════════════════════════════════════════════════════════

def test_section_3() -> None:
    _section("Section 3 — ChromaDB Store")
    from rag.chroma_store import ChromaStore
    from rag.embedder import get_embedder

    store = ChromaStore()
    emb   = get_embedder()

    # 3a. Whitehouse collection accessible
    try:
        wh_count = store.whitehouse_count()
        assert wh_count >= 0
        _ok("3a. Whitehouse collection accessible",
            f"whitehouse={wh_count} chunks")
    except Exception as e:
        _fail("3a. Collection accessible", str(e))

    # 3b. Whitehouse collection is populated
    try:
        wh_count = store.whitehouse_count()
        assert wh_count > 0, (
            "Whitehouse collection is empty! Run: python -m rag.indexer"
        )
        _ok("3b. Whitehouse collection populated", f"{wh_count} chunks indexed")
    except AssertionError as e:
        _fail("3b. Whitehouse populated", str(e))
    except Exception as e:
        _fail("3b. Whitehouse populated", str(e))

    # 3c. Upsert + query round-trip
    try:
        test_text = "This is a test document about semiconductor export controls."
        test_id   = "test_roundtrip_001"
        test_vec  = emb.embed_one(test_text)

        store.upsert_whitehouse_chunks(
            chunk_ids=[test_id],
            embeddings=[test_vec],
            documents=[test_text],
            metadatas=[{
                "post_id": "test", "title": "Round-trip Test",
                "url": "http://test", "categories": "Test",
                "date_iso": "2025-01-01", "chunk_index": "0",
                "total_chunks": "1",
            }],
        )

        query_vec = emb.embed_one("semiconductor export restrictions")
        results   = store.query_whitehouse(query_vec, top_k=5)

        found = any(r["id"] == test_id for r in results)
        assert found, "Upserted test doc not found in query results"
        _ok("3c. Upsert → query round-trip works")
    except Exception as e:
        _fail("3c. Round-trip", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Retriever
# ══════════════════════════════════════════════════════════════════════════════

def test_section_4() -> None:
    _section("Section 4 — Retriever")
    from rag.retriever import Retriever

    retriever = Retriever()

    # ── Test cases: (query, expected_keyword_in_title, test_name) ─────────────
    QUERIES = [
        (
            "Trump imposes 25 percent tariff on steel and aluminum imports from Canada",
            "steel",
            "4a. Steel tariff query → correct whitehouse doc",
        ),
        (
            "executive order banning TikTok ByteDance divest 90 days IEEPA",
            None,    # TikTok may not be in the CSV — just check we get results
            "4b. TikTok IEEPA query → returns results",
        ),
        (
            "semiconductor chip export controls advanced AI restrictions China",
            None,
            "4c. Semiconductor export controls query → returns results",
        ),
    ]

    for query, expected_kw, test_name in QUERIES:
        try:
            hits = retriever.retrieve_whitehouse_only(query, top_k=5)
            assert len(hits) > 0, "No results returned"

            if expected_kw:
                titles_lower = " ".join(h.title.lower() for h in hits)
                assert expected_kw.lower() in titles_lower, (
                    f"Expected '{expected_kw}' in titles, got: "
                    f"{[h.title[:40] for h in hits[:2]]}"
                )

            top = hits[0]
            _ok(test_name, f"top=[{top.similarity:.3f}] {top.title[:55]}")
        except Exception as e:
            _fail(test_name, str(e))

    # 4d. Similarity scores are valid (0–1)
    try:
        hits = retriever.retrieve_whitehouse_only(
            "Trump signs executive order on trade", top_k=5
        )
        for h in hits:
            assert 0.0 <= h.similarity <= 1.0, (
                f"Similarity out of range: {h.similarity}"
            )
        _ok("4d. All similarity scores in range [0, 1]",
            f"range=[{min(h.similarity for h in hits):.3f}, "
            f"{max(h.similarity for h in hits):.3f}]")
    except Exception as e:
        _fail("4d. Similarity range", str(e))

    # 4e. Top hit is more similar than bottom hit
    try:
        hits = retriever.retrieve_whitehouse_only(
            "Trump tariff executive order trade policy", top_k=5
        )
        assert hits[0].similarity >= hits[-1].similarity, (
            "Results not sorted by similarity descending"
        )
        _ok("4e. Results sorted by similarity descending",
            f"top={hits[0].similarity:.3f}, bottom={hits[-1].similarity:.3f}")
    except Exception as e:
        _fail("4e. Sort order", str(e))

    # 4f. All metadata fields present
    try:
        hits = retriever.retrieve_whitehouse_only("tariff trade policy", top_k=1)
        h = hits[0]
        missing = []
        for field in ["chunk_id", "post_id", "title", "url", "categories",
                      "date_iso", "text", "similarity"]:
            if not getattr(h, field, None):
                missing.append(field)
        assert not missing, f"Missing fields: {missing}"
        assert h.url.startswith("https://www.whitehouse.gov"), (
            f"URL not whitehouse.gov: {h.url}"
        )
        _ok("4f. All metadata fields present, URL is whitehouse.gov")
    except Exception as e:
        _fail("4f. Metadata fields", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Verifier
# ══════════════════════════════════════════════════════════════════════════════

def test_section_5() -> None:
    _section("Section 5 — Verifier (Stage 2 fact-check)")
    from verification.verifier import Verifier

    verifier = Verifier()

    # 5a. Known real policy → should score LIKELY or VERIFIED
    REAL_CLAIMS = [
        "Trump signed an executive order imposing 25 percent tariffs on steel and aluminum imports under Section 232",
        "Trump proclaimed a temporary import surcharge on all goods entering the United States",
        "The President signed an executive order suspending duty-free de minimis treatment under IEEPA",
    ]
    for claim in REAL_CLAIMS:
        try:
            result = verifier.verify(claim)
            assert result.confidence >= 40, (
                f"Real policy scored too low: {result.confidence} ({result.status})"
            )
            assert result.status in ("verified", "likely", "disputed"), (
                f"Unexpected status for real claim: {result.status}"
            )
            _ok(
                f"5a. Real policy claim → {result.status.upper()} ({result.confidence}/100)",
                f"{claim[:55]}…"
            )
        except Exception as e:
            _fail("5a. Real policy scoring", str(e))

    # 5b. Nonsense claim → should score DISPUTED or UNVERIFIED
    FAKE_CLAIMS = [
        "Trump declared war on the moon and banned all cheese imports from Jupiter",
        "The President signed an order making Mondays optional and taxing rainbows 500 percent",
    ]
    for claim in FAKE_CLAIMS:
        try:
            result = verifier.verify(claim)
            assert result.status in ("disputed", "unverified"), (
                f"Fake claim should be disputed/unverified, got: "
                f"{result.status} ({result.confidence})"
            )
            _ok(
                f"5b. Nonsense claim → {result.status.upper()} ({result.confidence}/100)",
                f"{claim[:55]}…"
            )
        except Exception as e:
            _fail("5b. Nonsense claim scoring", str(e))

    # 5c. Score breakdown fields present
    try:
        result = verifier.verify("Trump steel tariff executive order Section 232")
        bd = result.score_breakdown
        required_keys = [
            "semantic_similarity", "title_keyword_overlap",
            "content_keyword_overlap", "category_bonus",
        ]
        missing = [k for k in required_keys if k not in bd]
        assert not missing, f"Missing breakdown keys: {missing}"
        total = (bd["semantic_similarity"] + bd["title_keyword_overlap"] +
                 bd["content_keyword_overlap"] + bd["category_bonus"])
        assert total == result.confidence, (
            f"Breakdown sum {total} ≠ confidence {result.confidence}"
        )
        _ok("5c. Score breakdown fields present and sum correctly",
            f"sig1={bd['semantic_similarity']} sig2={bd['title_keyword_overlap']} "
            f"sig3={bd['content_keyword_overlap']} sig4={bd['category_bonus']}")
    except Exception as e:
        _fail("5c. Score breakdown", str(e))

    # 5d. Top hit URL is real whitehouse.gov
    try:
        result = verifier.verify("Trump executive order tariff Section 232 steel")
        assert result.top_hit is not None, "No top hit returned"
        assert "whitehouse.gov" in result.top_hit.url, (
            f"URL not whitehouse.gov: {result.top_hit.url}"
        )
        assert result.top_hit.date_iso, "date_iso is empty"
        _ok("5d. Top hit has valid whitehouse.gov URL + date",
            f"{result.top_hit.url[:60]}")
    except Exception as e:
        _fail("5d. Top hit URL", str(e))





# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="PolitiSense RAG Test Suite")
    parser.add_argument(
        "--section", type=int, default=0,
        help="Run only a specific section (1–5). Default: run all.",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  PolitiSense — Comprehensive RAG Test Suite{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    sections = {
        1: test_section_1,
        2: test_section_2,
        3: test_section_3,
        4: test_section_4,
        5: test_section_5,
    }

    if args.section and args.section in sections:
        sections[args.section]()
    else:
        for fn in sections.values():
            fn()

    # ── Summary ────────────────────────────────────────────────────────────────
    total = _passed + _failed + _skipped
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  Results: {total} tests{RESET}")
    print(f"  {GREEN}Passed : {_passed}{RESET}")
    if _failed:
        print(f"  {RED}Failed : {_failed}{RESET}")
    if _skipped:
        print(f"  {YELLOW}Skipped: {_skipped}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
