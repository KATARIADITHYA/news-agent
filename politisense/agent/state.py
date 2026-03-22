"""
agent/state.py
──────────────
PolitiSense Agent — State definition for LangGraph.

The AgentState flows through every node in the graph.
Each node reads from it and writes back a partial update.
"""

from __future__ import annotations

from typing import Any, Dict, List
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    news_text:       str

    # ── Stage 1: Fingerprint ───────────────────────────────────────────────
    fingerprint:     Dict[str, Any]    # legal_authority, policy_type, scope, target

    # ── Stage 2: Retrieval ─────────────────────────────────────────────────
    rewritten_query: str
    verification:    Dict[str, Any]    # status, confidence, top_hit, all_hits
    low_confidence:  bool
    retry_count:     int

    # ── Stage 3: Report ────────────────────────────────────────────────────
    risk_score:      int
    risk_label:      str
    risk_breakdown:  Dict[str, Any]
    narrative:       str
    final_report:    Dict[str, Any]

    # ── Stage 4: Market Analysis ───────────────────────────────────────────
    tickers:         Dict[str, str]    # sector name → ETF ticker
    event_date:      str               # EO announcement date (from verification)
    price_data:      Dict[str, Any]    # raw yfinance prices per ticker
    market_metrics:  Dict[str, Any]    # computed volatility + returns per ticker

    # ── Memory ─────────────────────────────────────────────────────────────
    messages:        List[Dict[str, str]]
