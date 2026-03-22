"""
agent/graph.py
──────────────
PolitiSense Agent — Full LangGraph graph.

Graph structure:

    [START]
       │
       ▼
  fingerprint_node      ← Stage 1: Claude NLP extracts policy fingerprint
       │
       ▼
  query_rewrite_node    ← rewrites query for better ChromaDB search
       │
       ▼
  retrieval_node        ← Stage 2: ChromaDB search + confidence score
       │
       ▼
  guard_node            ← AGENT DECISION: retry or proceed?
       │
       ├── "retry" ─────► query_rewrite_node  (loops back, max 2 retries)
       │
       └── "report" ────► report_node         ← Stage 3: risk score + narrative
                               │
                               ▼
                          sector_map_node      ← maps sectors → ETF tickers
                               │
                               ▼
                          market_data_node     ← yfinance: price data around EO date
                               │
                               ▼
                          compute_node         ← volatility + returns calculation
                               │
                               ▼
                             [END]
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState
from agent.nodes import (
    fingerprint_node,
    query_rewrite_node,
    retrieval_node,
    guard_node,
    report_node,
    route_after_guard,
)
from agent.market_nodes import (
    sector_map_node,
    market_data_node,
    compute_node,
)


class PolitiSenseAgentBuilder:
    """
    Builds and compiles the full PolitiSense LangGraph agent.

    Stages:
      1. Fingerprint  — Claude NLP extracts policy type, scope, targets
      2. Retrieval    — ChromaDB semantic search with confidence scoring
      3. Report       — Risk score + Claude analyst narrative
      4. Market       — Sector → ETF → yfinance → volatility + returns
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        graph = StateGraph(AgentState)

        # ── Register all nodes ─────────────────────────────────────────────
        graph.add_node("fingerprint",   fingerprint_node)
        graph.add_node("query_rewrite", query_rewrite_node)
        graph.add_node("retrieval",     retrieval_node)
        graph.add_node("guard",         guard_node)
        graph.add_node("report",        report_node)
        graph.add_node("sector_map",    sector_map_node)
        graph.add_node("market_data",   market_data_node)
        graph.add_node("compute",       compute_node)

        # ── Fixed edges ────────────────────────────────────────────────────
        graph.add_edge(START,           "fingerprint")
        graph.add_edge("fingerprint",   "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")
        graph.add_edge("retrieval",     "guard")

        # ── Conditional edge — agent decides to retry or proceed ───────────
        graph.add_conditional_edges(
            "guard",
            route_after_guard,
            {
                "retry":  "query_rewrite",
                "report": "report",
            },
        )

        # ── Market analysis chain (runs after report) ──────────────────────
        graph.add_edge("report",      "sector_map")
        graph.add_edge("sector_map",  "market_data")
        graph.add_edge("market_data", "compute")
        graph.add_edge("compute",     END)

        return graph.compile(checkpointer=self._checkpointer)


@lru_cache(maxsize=1)
def get_agent():
    """Singleton compiled agent — built once per process."""
    return PolitiSenseAgentBuilder().build()
