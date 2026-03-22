"""
test_chart.py
─────────────
Standalone test — fetch a stock/ETF and show before/after chart.
Uses Finnhub API for real-time + historical data (no rate limiting).

Usage:
    python test_chart.py
    python test_chart.py --ticker INDA --date 2026-02-06
    python test_chart.py --ticker SPY  --date 2025-01-20
    python test_chart.py --ticker XLE  --date 2025-03-15 --no-chart
"""

import argparse
import time
import os
import requests
from datetime import date, timedelta, datetime
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import numpy as np

load_dotenv(Path(__file__).parent / ".env")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


# ── Fetch via Finnhub ──────────────────────────────────────────────────────────

def fetch_finnhub(ticker: str, start: date, end: date) -> pd.Series | None:
    """
    Fetch daily OHLCV candles from Finnhub.
    Free tier: 60 requests/minute, full historical data, no scraping.
    """
    if not FINNHUB_API_KEY:
        print("  No FINNHUB_API_KEY found in .env")
        return None

    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end,   datetime.min.time()).timestamp())

    print(f"  Calling Finnhub API for {ticker}...")
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol":     ticker,
                "resolution": "D",
                "from":       period1,
                "to":         period2,
                "token":      FINNHUB_API_KEY,
            },
            timeout=10,
        )
        data = r.json()

        if data.get("s") != "ok":
            print(f"  Finnhub returned status: {data.get('s')} — ticker may not be supported")
            return None

        closes = data["c"]   # close prices
        times  = data["t"]   # unix timestamps

        series = pd.Series(
            closes,
            index=pd.to_datetime(times, unit="s").normalize(),
        )
        series = series.sort_index()
        print(f"  Finnhub OK — {len(series)} trading days  (${series.iloc[0]:.2f} → ${series.iloc[-1]:.2f})")
        return series

    except Exception as e:
        print(f"  Finnhub error: {e}")
        return None


# ── Main fetch ─────────────────────────────────────────────────────────────────

def fetch(ticker: str, event_date: date, days_before=60, days_after=30) -> pd.Series | None:
    start = event_date - timedelta(days=days_before + 10)
    end   = min(event_date + timedelta(days=days_after + 10), date.today())

    print(f"\nFetching {ticker}  [{start} → {end}]")
    return fetch_finnhub(ticker, start, end)


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(prices: pd.Series, event_date: date, ticker: str):
    ev  = pd.Timestamp(event_date)
    pre = prices[prices.index < ev]
    pos = prices[prices.index >= ev]

    if len(pre) < 2 or len(pos) < 2:
        print(f"\nNot enough data around {event_date}.")
        print(f"  Pre-event days  : {len(pre)}")
        print(f"  Post-event days : {len(pos)}")
        print(f"  Data range      : {prices.index[0].date()} → {prices.index[-1].date()}")
        return

    pre_ret = ((pre.iloc[-1] - pre.iloc[0]) / pre.iloc[0]) * 100
    pos_ret = ((pos.iloc[-1] - pos.iloc[0]) / pos.iloc[0]) * 100
    pre_vol = float(pre.pct_change().dropna().std() * np.sqrt(252) * 100)
    pos_vol = float(pos.pct_change().dropna().std() * np.sqrt(252) * 100)
    vol_chg = ((pos_vol - pre_vol) / pre_vol * 100) if pre_vol > 0 else 0.0

    sep = "─" * 54
    print(f"\n{sep}")
    print(f"  {ticker}  ·  Event: {event_date.strftime('%B %d, %Y')}")
    print(sep)
    print(f"  Pre-event  ({len(pre):>2} days)  ${pre.iloc[0]:.2f} → ${pre.iloc[-1]:.2f}   {pre_ret:+.1f}%")
    print(f"  Post-event ({len(pos):>2} days)  ${pos.iloc[0]:.2f} → ${pos.iloc[-1]:.2f}   {pos_ret:+.1f}%")
    print(f"  Volatility  pre={pre_vol:.1f}%  post={pos_vol:.1f}%  change={vol_chg:+.1f}%")
    print(sep)


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_chart(prices: pd.Series, event_date: date, ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ev       = pd.Timestamp(event_date)
        pre_data = prices[prices.index < ev]
        pos_data = prices[prices.index >= ev]

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.fill_between(pre_data.index, pre_data.values, alpha=0.12, color="#4a90e2")
        ax.plot(pre_data.index, pre_data.values, color="#4a90e2", linewidth=2, label="Pre-event")

        ax.fill_between(pos_data.index, pos_data.values, alpha=0.12, color="#e24b4a")
        ax.plot(pos_data.index, pos_data.values, color="#e24b4a", linewidth=2, label="Post-event")

        ax.axvline(x=ev, color="#fd7e14", linewidth=2, linestyle="--",
                   label=f"Event: {event_date.strftime('%b %d, %Y')}")

        # 30-day linear projection
        if len(pos_data) >= 3:
            x_num = np.arange(len(pos_data))
            slope, intercept = np.polyfit(x_num, pos_data.values.astype(float), 1)
            proj_dates = pd.date_range(pos_data.index[-1], periods=31, freq="D")[1:]
            proj_y     = [intercept + slope * (len(pos_data) + i) for i in range(30)]
            ax.plot(proj_dates, proj_y, color="#888", linewidth=1.5,
                    linestyle=":", label="30-day projection")

        ax.set_title(f"{ticker}  —  Before & After  {event_date.strftime('%B %d, %Y')}",
                     color="white", fontsize=14, pad=12)
        ax.set_xlabel("Date", color="#aaa", fontsize=11)
        ax.set_ylabel("Price (USD)", color="#aaa", fontsize=11)
        ax.tick_params(colors="#aaa")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#333")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.legend(facecolor="#252540", edgecolor="#333", labelcolor="white", fontsize=10)
        ax.grid(True, color="#333", linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        plt.show()
        print("Chart shown — close the window to exit.")

    except ImportError:
        print("\nInstall matplotlib:  pip install matplotlib")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="INDA",       help="Ticker (default: INDA)")
    parser.add_argument("--date",     default="2026-02-06", help="Event date YYYY-MM-DD")
    parser.add_argument("--before",   default=60, type=int, help="Days before event")
    parser.add_argument("--after",    default=30, type=int, help="Days after event")
    parser.add_argument("--no-chart", action="store_true",  help="Summary only, no chart")
    args = parser.parse_args()

    try:
        event_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Bad date: {args.date} — use YYYY-MM-DD")
        exit(1)

    prices = fetch(args.ticker.upper(), event_date, args.before, args.after)

    if prices is None or len(prices) < 5:
        print(f"\nNo data for {args.ticker}. Check your FINNHUB_API_KEY in .env")
        exit(1)

    print_summary(prices, event_date, args.ticker.upper())

    if not args.no_chart:
        plot_chart(prices, event_date, args.ticker.upper())
