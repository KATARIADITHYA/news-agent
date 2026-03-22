import os
import requests
from datetime import date, timedelta, datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv(Path(__file__).parent / ".env")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


def fetch_prices(ticker: str, event_date: date, days_before=60, days_after=30) -> pd.Series | None:
    start   = event_date - timedelta(days=days_before + 10)
    end     = min(event_date + timedelta(days=days_after + 10), date.today())
    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end,   datetime.min.time()).timestamp())

    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": ticker, "resolution": "D",
                "from": period1, "to": period2,
                "token": FINNHUB_API_KEY,
            },
            timeout=10,
        )
        data = r.json()
        if data.get("s") != "ok":
            return None
        series = pd.Series(data["c"], index=pd.to_datetime(data["t"], unit="s").normalize())
        return series.sort_index()
    except Exception:
        return None
