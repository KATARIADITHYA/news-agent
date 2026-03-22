"""
agent/market_nodes.py
──────────────────────
New nodes added to the PolitiSense agent:

  Node 6: sector_map_node    → maps sectors from fingerprint → ETF tickers
  Node 7: market_data_node   → yfinance fetches price data around EO date
  Node 8: compute_node       → calculates volatility + returns

These plug directly into the existing LangGraph graph after report_node.

ETF mappings are loaded from config/etf_maps.json via ETFLoader —
no hardcoded dicts here.
"""

from __future__ import annotations

import time
import warnings
from datetime import date, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from agent.state import AgentState
from agent.etf_loader import get_loader as _get_etf_loader  # ← replaces hardcoded maps


# ── Node 6: Sector Map ────────────────────────────────────────────────────────

def sector_map_node(state: AgentState) -> Dict[str, Any]:
    """
    Uses Claude to predict which sectors are affected by the EO,
    then maps those sectors to ETF tickers via ETFLoader.

    Flow:
      1. Claude reads the news + policy text → predicts affected sectors
      2. ETFLoader maps those sector/country names → ETF tickers
         (loaded from config/etf_maps.json — no hardcoded dicts)
    """
    import anthropic as _anthropic
    from config import ANTHROPIC_API_KEY as _KEY

    news_text    = state.get("news_text", "")
    fp           = state.get("fingerprint", {})
    verification = state.get("verification", {})
    top_hit      = (verification.get("top_hit") or {})
    policy_text  = top_hit.get("text", "")

    print(f"\n[Node: sector_map] Asking Claude which sectors are affected...")

    loader = _get_etf_loader()
    affected_sectors = _llm_predict_sectors(
        news_text=news_text,
        policy_fingerprint=fp,
        policy_text=policy_text,
        api_key=_KEY,
        loader=loader,
    )
    print(f"[Node: sector_map] Claude predicted sectors: {affected_sectors}")

    tickers = loader.lookup_list(affected_sectors)
    print(f"[Node: sector_map] Final tickers: {tickers}")
    return {"tickers": tickers}


def _llm_predict_sectors(
    news_text: str,
    policy_fingerprint: dict,
    policy_text: str,
    api_key: str,
    loader,
) -> List[str]:
    import json as _json
    import re as _re
    import anthropic as _anthropic

    if not api_key:
        items = []
        for field in ["scope", "target"]:
            val = policy_fingerprint.get(field, "")
            items.extend([x.strip() for x in val.split(",") if x.strip()])
        return items

    available_str = loader.available_for_prompt()

    prompt = f"""You are a financial analyst. Identify which sectors and countries 
are economically affected by this Trump policy.

NEWS:
{news_text}

POLICY FINGERPRINT:
- Type     : {policy_fingerprint.get('policy_type', 'N/A')}
- Authority: {policy_fingerprint.get('legal_authority', 'N/A')}
- Scope    : {policy_fingerprint.get('scope', 'N/A')}
- Target   : {policy_fingerprint.get('target', 'N/A')}

OFFICIAL POLICY TEXT (from whitehouse.gov):
{policy_text[:600] if policy_text else 'Not available'}

AVAILABLE SECTORS/COUNTRIES TO CHOOSE FROM:
{available_str}

TASK: From the list above, pick ONLY the sectors and countries that are 
DIRECTLY or SIGNIFICANTLY affected by this specific policy.
Think step by step:
  - What goods/commodities are targeted?
  - Which countries are named or implied?
  - Which industries depend on those goods or trade with those countries?
  - What is the second-order effect? (e.g. steel tariff → automotive affected too)

Return ONLY a JSON array of strings — names exactly as they appear in the available list.
Max 6 items. No explanation.

Example: ["india", "energy", "oil & gas", "trade & logistics"]

Your answer:"""

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        msg    = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = msg.content[0].text.strip()
        raw  = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = _json.loads(raw)
        return [str(x).strip() for x in data if x]
    except Exception as e:
        print(f"[sector_map] LLM sector prediction failed ({e}) — using fingerprint fallback")
        items = []
        for field in ["scope", "target"]:
            val = policy_fingerprint.get(field, "")
            items.extend([x.strip() for x in val.split(",") if x.strip()])
        return items


# ── Node 7: Market Data ───────────────────────────────────────────────────────

def market_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches price data from yfinance for each mapped ticker.

    Improvements:
      - Deduplicates tickers before fetching (avoids double requests for e.g. SPY, XLE)
      - Adds a 3s delay between requests to avoid Yahoo rate limiting
      - Retries up to 3x with exponential backoff on rate limit errors
    """
    tickers      = state.get("tickers", {})
    verification = state.get("verification", {})

    top_hit  = verification.get("top_hit") or {}
    date_iso = top_hit.get("date_iso", "")

    try:
        event_date = date.fromisoformat(date_iso[:10])
    except Exception:
        event_date = date.today() - timedelta(days=30)
        print(f"[Node: market_data] No date from EO — using {event_date}")

    fetch_start = str(event_date - timedelta(days=60))
    fetch_end   = str(event_date + timedelta(days=30))

    print(f"\n[Node: market_data] Event date: {event_date}  Window: {fetch_start} → {fetch_end}")

    # ── Deduplicate: build ticker → [name, ...] mapping ───────────────────
    # Multiple sectors can map to the same ETF (e.g. "energy" and "oil & gas" → XLE)
    # Fetch each unique ticker only once, store under the first name seen
    ticker_to_name: Dict[str, str] = {}
    for name, ticker in tickers.items():
        if ticker not in ticker_to_name:
            ticker_to_name[ticker] = name

    unique_count = len(ticker_to_name)
    print(f"[Node: market_data] Fetching {unique_count} unique tickers (deduplicated from {len(tickers)})")

    price_data = {}

    for i, (ticker, name) in enumerate(ticker_to_name.items()):
        # Polite delay between requests — avoids Yahoo rate limiting
        if i > 0:
            time.sleep(3)

        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=fetch_start,
                    end=fetch_end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if df.empty:
                    print(f"[Node: market_data] {ticker} — no data returned")
                    break

                if isinstance(df.columns, pd.MultiIndex):
                    close = df["Close"][ticker]
                else:
                    close = df["Close"]

                # Store with string keys so LangGraph checkpointer can serialise
                price_data[ticker] = {
                    "name":       name,
                    "prices":     {str(k): float(v) for k, v in close.dropna().items()},
                    "event_date": str(event_date),
                }
                print(f"[Node: market_data] {ticker} ✓  {len(close)} days")
                break

            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "too many" in err:
                    wait = 10 * (2 ** attempt)   # 10s → 20s → 40s
                    print(f"[Node: market_data] {ticker} rate limited — waiting {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"[Node: market_data] {ticker} error: {e}")
                    break

    print(f"[Node: market_data] Fetched {len(price_data)}/{unique_count} tickers successfully")
    return {
        "price_data": price_data,
        "event_date": str(event_date),
    }


# ── Node 8: Compute ───────────────────────────────────────────────────────────

def compute_node(state: AgentState) -> Dict[str, Any]:
    """
    Computes volatility and returns from the fetched price data.

    For each ticker:
      - Pre-event return  : 60-day return BEFORE the EO date
      - Post-event return : 30-day return AFTER the EO date
      - Pre-event vol     : std(daily returns) × sqrt(252) before EO
      - Post-event vol    : std(daily returns) × sqrt(252) after EO
      - Vol change        : (post_vol - pre_vol) / pre_vol × 100
    """
    price_data = state.get("price_data", {})
    event_date = state.get("event_date", "")

    try:
        event_dt = pd.Timestamp(event_date)
    except Exception:
        event_dt = pd.Timestamp.today()

    print(f"\n[Node: compute] Computing metrics for {len(price_data)} tickers")

    market_metrics = {}

    for ticker, data in price_data.items():
        try:
            prices = pd.Series(data["prices"])
            prices.index = pd.to_datetime(prices.index)
            prices = prices.sort_index()

            pre  = prices[prices.index <  event_dt]
            post = prices[prices.index >= event_dt]

            if len(pre) < 2 or len(post) < 2:
                continue

            pre_returns  = pre.pct_change().dropna()
            post_returns = post.pct_change().dropna()

            pre_total_return  = ((pre.iloc[-1]  - pre.iloc[0])  / pre.iloc[0])  * 100
            post_total_return = ((post.iloc[-1] - post.iloc[0]) / post.iloc[0]) * 100

            pre_vol    = float(pre_returns.std()  * np.sqrt(252) * 100)
            post_vol   = float(post_returns.std() * np.sqrt(252) * 100)
            vol_change = ((post_vol - pre_vol) / pre_vol * 100) if pre_vol > 0 else 0.0

            market_metrics[ticker] = {
                "name":                data["name"],
                "pre_return_pct":      round(float(pre_total_return),  2),
                "post_return_pct":     round(float(post_total_return), 2),
                "pre_vol_annualised":  round(pre_vol,  2),
                "post_vol_annualised": round(post_vol, 2),
                "vol_change_pct":      round(vol_change, 2),
                "pre_price_start":     round(float(pre.iloc[0]),   2),
                "pre_price_end":       round(float(pre.iloc[-1]),  2),
                "post_price_start":    round(float(post.iloc[0]),  2),
                "post_price_end":      round(float(post.iloc[-1]), 2),
                "prices_pre":          {str(k.date()): round(float(v), 2) for k, v in pre.items()},
                "prices_post":         {str(k.date()): round(float(v), 2) for k, v in post.items()},
            }

            print(
                f"[Node: compute] {ticker:6s}  "
                f"pre_ret={pre_total_return:+.1f}%  "
                f"post_ret={post_total_return:+.1f}%  "
                f"vol_change={vol_change:+.1f}%"
            )

        except Exception as e:
            print(f"[Node: compute] {ticker} compute error: {e}")

    return {"market_metrics": market_metrics}
