"""
market_analysis/sector_etf_map.py
───────────────────────────────────
Maps sectors/industries → ETF tickers for yfinance price data.
Used after LLM predicts which sectors are affected by an EO.
"""

SECTOR_ETF_MAP = {
    # Trade & Tariffs
    "steel":                  "SLX",
    "steel & metals":         "SLX",
    "aluminum":               "AA",
    "metals & mining":        "XME",
    "automotive":             "CARZ",
    "auto parts":             "CARZ",

    # Energy
    "oil & gas":              "XLE",
    "energy":                 "XLE",
    "oil":                    "XLE",
    "natural gas":            "UNG",
    "renewables":             "ICLN",
    "solar":                  "TAN",
    "nuclear":                "URA",

    # Technology
    "technology":             "XLK",
    "semiconductors":         "SOXX",
    "ai":                     "BOTZ",
    "ai / data centers":      "BOTZ",
    "cloud computing":        "SKYY",
    "software":               "IGV",
    "social media":           "SOCL",
    "cybersecurity":          "CIBR",

    # Finance
    "financials":             "XLF",
    "banking":                "KBE",
    "insurance":              "KIE",

    # Healthcare
    "healthcare":             "XLV",
    "pharmaceuticals":        "XPH",
    "biotech":                "XBI",
    "medical devices":        "IHI",

    # Defense
    "defense":                "ITA",
    "aerospace":              "ITA",
    "aerospace & defense":    "ITA",

    # Consumer
    "consumer goods":         "XLP",
    "retail":                 "XRT",
    "consumer discretionary": "XLY",

    # Agriculture & Materials
    "agriculture":            "MOO",
    "materials":              "XLB",
    "lumber":                 "CUT",
    "construction":           "ITB",
    "real estate":            "VNQ",

    # Trade & Logistics
    "trade & logistics":      "IYT",
    "industrials":            "XLI",
    "manufacturing":          "XLI",
    "shipping":               "BDRY",

    # Other
    "utilities":              "XLU",
    "telecom":                "IYZ",
    "media":                  "PBS",
}

# Company → ticker (for when LLM names specific companies)
COMPANY_TICKER_MAP = {
    "apple":              "AAPL",
    "microsoft":          "MSFT",
    "nvidia":             "NVDA",
    "google":             "GOOGL",
    "alphabet":           "GOOGL",
    "amazon":             "AMZN",
    "meta":               "META",
    "tesla":              "TSLA",
    "ford":               "F",
    "general motors":     "GM",
    "boeing":             "BA",
    "lockheed martin":    "LMT",
    "raytheon":           "RTX",
    "exxon":              "XOM",
    "chevron":            "CVX",
    "nucor":              "NUE",
    "us steel":           "X",
    "alcoa":              "AA",
    "intel":              "INTC",
    "tsmc":               "TSM",
    "samsung":            "SSNLF",
    "walmart":            "WMT",
    "jp morgan":          "JPM",
    "goldman sachs":      "GS",
    "pfizer":             "PFE",
    "johnson & johnson":  "JNJ",
}
