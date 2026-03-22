"""
run_with_charts.py
───────────────────
Runs the full PolitiSense pipeline then automatically calls
etf_graph.plot_etf_graph() for each mapped ETF ticker — using
the RAG-matched whitehouse.gov date as the base date.

Usage:
    cd "/Users/adithyakatari/Desktop/news agent/politisense"
    python run_with_charts.py
    python run_with_charts.py --news "Trump imposes 25% tariff on steel imports"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# ── Add both directories to path ───────────────────────────────────────────────
POLITISENSE_DIR = Path(__file__).parent
NEWS_AGENT_DIR  = POLITISENSE_DIR.parent

sys.path.insert(0, str(POLITISENSE_DIR))   # pipeline, config, etc.
sys.path.insert(0, str(NEWS_AGENT_DIR))    # etf_graph.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pipeline import run_agent, report_to_text
from etf_graph import plot_etf_graph      # your working etf_graph.py — untouched

SUSPENDED_TICKERS = {"ERUS", "RSX", "RSXJ"}


def run_with_charts(news_text: str):
    # ── Step 1: Run the full pipeline ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Running PolitiSense agent...")
    print(f"{'='*65}")

    report = run_agent(news_text=news_text, thread_id="charts")

    # ── Step 2: Print the text report ──────────────────────────────────────
    print("\n" + report_to_text(report))

    # ── Step 3: Extract RAG date ────────────────────────────────────────────
    raw_date = report.get("top_source_date") or report.get("event_date") or ""
    try:
        rag_date = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
        base_date_str = str(rag_date)
    except (ValueError, TypeError):
        print("\nNo RAG date found — cannot plot charts.")
        return

    print(f"\n{'='*65}")
    print(f"  RAG date: {rag_date.strftime('%B %d, %Y')}")
    print(f"  Source  : {report.get('top_source_title', 'N/A')}")
    print(f"{'='*65}")

    # ── Step 4: Get unique tickers (skip suspended) ─────────────────────────
    tickers = report.get("tickers", {})
    seen    = set()
    unique  = {}
    for name, ticker in tickers.items():
        if ticker not in seen and ticker not in SUSPENDED_TICKERS:
            seen.add(ticker)
            unique[name] = ticker

    if not unique:
        print("No valid ETF tickers mapped — skipping charts.")
        return

    print(f"\nPlotting {len(unique)} ETF chart(s): {list(unique.values())}")
    print("(Close each chart window to see the next one)\n")

    # ── Step 5: Call plot_etf_graph for each ticker ─────────────────────────
    for name, ticker in unique.items():
        print(f"\n── {ticker} ({name}) ──")
        plot_etf_graph(ticker, base_date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--news", type=str,
        default="Trump announces 25% tariff on India and unspecified penalties for buying Russian oil",
        help="News headline to analyse",
    )
    args = parser.parse_args()
    run_with_charts(args.news)
