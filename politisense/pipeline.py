"""
pipeline.py
────────────
PolitiSense — LangGraph Agent Entry Point

Replaces the old linear pipeline with a true agent that:
  - Reasons about confidence and retries if needed
  - Uses memory across multi-turn conversations
  - Decides autonomously what to do next

Usage (CLI):
    python pipeline.py
    python pipeline.py --news "Your news text here"
    python pipeline.py --news "..." --output report.txt
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).parent))

from agent.graph import get_agent


# ── Run Agent ──────────────────────────────────────────────────────────────────

def run_agent(
    news_text:  str,
    thread_id:  str = "default",     # thread_id enables memory per conversation
) -> dict:
    """
    Run the PolitiSense agent on a news text.

    Parameters
    ----------
    news_text  : Raw news article or headline
    thread_id  : Conversation ID — same thread_id = agent remembers past queries

    Returns
    -------
    dict : final_report with all pipeline outputs
    """
    agent  = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    _hdr("PolitiSense Agent")
    print(f"  News    : {news_text[:100]}…")
    print(f"  Thread  : {thread_id}")

    # Initial state — only news_text is set, agent figures out the rest
    initial_state = {
        "news_text":    news_text,
        "retry_count":  0,
        "low_confidence": False,
        "messages":     [],
    }

    # Run the agent — LangGraph executes the graph
    final_state = agent.invoke(initial_state, config=config)

    report = final_state.get("final_report", {})
    _hdr("Agent Complete")
    print(f"  Status     : {report.get('verification_status', '').upper()}")
    print(f"  Confidence : {report.get('verification_confidence', 0)}/100")
    print(f"  Risk Score : {report.get('risk_score', 0)}/90  [{report.get('risk_label', '')}]")
    print(f"  Retries    : {report.get('retries_used', 0)}")

    # Add market metrics to report
    report["market_metrics"] = final_state.get("market_metrics", {})
    report["tickers"]        = final_state.get("tickers", {})
    report["event_date"]     = final_state.get("event_date", "")

    return report


# ── Plain text report ──────────────────────────────────────────────────────────

def report_to_text(report: dict) -> str:
    sep  = "=" * 70
    sep2 = "-" * 70
    fp   = report.get("policy_fingerprint", {})
    bd   = report.get("risk_breakdown", {})

    lines = [
        sep,
        "POLITISENSE — TRUMP POLICY VOLATILITY REPORT",
        sep,
        f"Generated  : {datetime.utcnow().isoformat()}Z",
        f"Risk Score : {report.get('risk_score', 0)}/90  [{report.get('risk_label', '')}]",
        f"Retries    : {report.get('retries_used', 0)} (agent retried retrieval this many times)",
        "",
        "POLICY NEWS",
        sep2,
        report.get("news_text", ""),
        "",
        f"Policy type    : {fp.get('policy_type', 'N/A')}",
        f"Legal authority: {fp.get('legal_authority', 'N/A') or 'N/A'}",
        f"Scope          : {fp.get('scope', 'N/A') or 'N/A'}",
        f"Target         : {fp.get('target', 'N/A') or 'N/A'}",
        "",
        "STAGE 2 — FACT-CHECK",
        sep2,
        f"Status     : {report.get('verification_status', '').upper()}",
        f"Confidence : {report.get('verification_confidence', 0)}/100  (avg cosine similarity × 100)",
        f"Explanation: {report.get('verification_explanation', '')}",
        f"Source     : {report.get('top_source_title', '')}",
        f"URL        : {report.get('top_source_url', '')}",
        f"Date       : {report.get('top_source_date', '')}",
        "",
        "STAGE 3 — RISK SCORE",
        sep2,
        f"  Verification signal : {bd.get('verification_signal', 0):>3}/30",
        f"  Policy type         : {bd.get('policy_type', 0):>3}/20",
        f"  Legal authority     : {bd.get('legal_authority', 0):>3}/15",
        f"  Scope breadth       : {bd.get('scope_breadth', 0):>3}/25",
        f"  {'─'*25}",
        f"  TOTAL               : {report.get('risk_score', 0):>3}/90  [{report.get('risk_label', '')}]",
        "",
        "ANALYST NARRATIVE",
        sep2,
        report.get("analyst_narrative", ""),
    ]

    # Market metrics section
    market = report.get("market_metrics", {})
    if market:
        lines += [
            "",
            "STAGE 4 — MARKET ANALYSIS",
            sep2,
            f"Event date : {report.get('event_date', 'N/A')}",
            f"Tickers    : {', '.join(report.get('tickers', {}).values())}",
            "",
            f"{'Ticker':<8} {'Name':<25} {'Pre Ret':>8} {'Post Ret':>9} {'Pre Vol':>8} {'Post Vol':>9} {'ΔVol%':>7}",
            "-" * 80,
        ]
        for ticker, m in market.items():
            if ticker == "SPY":
                continue
            lines.append(
                f"  {ticker:<8} {m.get('name',''):<25} "
                f"{m.get('pre_return_pct', 0):>+7.1f}%  "
                f"{m.get('post_return_pct', 0):>+8.1f}%  "
                f"{m.get('pre_vol_annualised', 0):>7.1f}%  "
                f"{m.get('post_vol_annualised', 0):>8.1f}%  "
                f"{m.get('vol_change_pct', 0):>+6.1f}%"
            )
        # SPY benchmark
        if "SPY" in market:
            spy = market["SPY"]
            lines += [
                "",
                f"  Benchmark (SPY): pre={spy.get('pre_return_pct',0):+.1f}%  "
                f"post={spy.get('post_return_pct',0):+.1f}%",
            ]

    lines += ["", sep]
    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PolitiSense Agent")
    parser.add_argument(
        "--news", type=str,
        default=(
            "President Trump signed an executive order imposing 25% tariffs on all "
            "steel and aluminum imports from Canada and Mexico, effective immediately. "
            "The White House cited national security concerns under Section 232 of "
            "the Trade Expansion Act."
        ),
        help="News text to analyse",
    )
    parser.add_argument(
        "--thread", type=str, default="default",
        help="Thread ID for conversation memory (default: 'default')",
    )
    parser.add_argument("--output",      type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    report = run_agent(news_text=args.news, thread_id=args.thread)

    text = report_to_text(report)
    print("\n" + text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nText report saved to: {args.output}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"JSON report saved to: {args.output_json}")
