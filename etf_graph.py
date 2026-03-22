# pip install yfinance pandas matplotlib numpy

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_with_retry(ticker_symbol, max_retries=5):
    last_error = None

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker_symbol,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if not df.empty:
                return df

        except Exception as e:
            last_error = e

        wait_time = 2 ** attempt
        print(f"Retrying in {wait_time} seconds...")
        time.sleep(wait_time)

    print("Download failed after retries.")
    if last_error:
        print("Last error:", last_error)
    return pd.DataFrame()


def plot_etf_graph(ticker_symbol, base_date_str):
    base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()

    df = download_with_retry(ticker_symbol)

    if df.empty:
        print(f"No data found for ticker: {ticker_symbol}")
        print("Yahoo may still be rate-limiting you.")
        return

    if "Close" in df.columns:
        close_prices = df["Close"].dropna()
    else:
        close_prices = df.iloc[:, -1].dropna()

    if len(close_prices) < 2:
        print("Not enough data to plot.")
        return

    # simple projection
    x_hist = np.arange(len(close_prices))
    y_hist = close_prices.values

    slope, intercept = np.polyfit(x_hist, y_hist, 1)

    future_days = 30
    x_future = np.arange(len(close_prices), len(close_prices) + future_days)
    y_future = slope * x_future + intercept

    future_dates = pd.date_range(
        start=pd.Timestamp(base_date) + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(close_prices.index, close_prices.values, label="Last 1 Month Actual Price")
    plt.plot(future_dates, y_future, linestyle="--", label="Next 1 Month Projection")
    plt.axvline(pd.Timestamp(base_date), linestyle=":", label=f"Base Date: {base_date}")

    plt.title(f"{ticker_symbol.upper()} ETF Graph")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ticker = input("Enter ETF ticker: ").strip().upper()
date_input = input("Enter base date (YYYY-MM-DD): ").strip()

plot_etf_graph(ticker, date_input)