"""
agent/etf_loader.py
────────────────────
Loads ETF/ticker mappings from config/etf_maps.json.

Replaces the hardcoded COUNTRY_ETF_MAP, SECTOR_ETF_MAP, and
COMPANY_TICKER_MAP dicts that were in market_nodes.py.

Usage:
    from agent.etf_loader import ETFLoader
    loader = ETFLoader()
    ticker = loader.lookup("india")           # → "INDA"
    ticker = loader.lookup("semiconductors")  # → "SOXX"
    all_countries = loader.countries          # full dict
"""

from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional

_DEFAULT_MAP_PATH = Path(__file__).parent.parent / "config" / "etf_maps.json"


class ETFLoader:
    """
    Loads and queries the ETF maps from etf_maps.json.

    Priority order for lookup():
      1. countries
      2. sectors
      3. commodities
      4. companies
    """

    def __init__(self, map_path: Path = _DEFAULT_MAP_PATH):
        with open(map_path) as f:
            data = json.load(f)

        self.countries:   dict[str, str] = data.get("countries",   {})
        self.sectors:     dict[str, str] = data.get("sectors",     {})
        self.commodities: dict[str, str] = data.get("commodities", {})
        self.companies:   dict[str, str] = data.get("companies",   {})

        # Flat lookup index (lower-cased key → ticker)
        self._index: dict[str, str] = {}
        for mapping in [self.countries, self.sectors, self.commodities, self.companies]:
            for key, ticker in mapping.items():
                self._index[key.lower()] = ticker

    def lookup(self, term: str) -> Optional[str]:
        """
        Case-insensitive lookup. Tries exact match first,
        then substring containment in both directions.
        Returns None if no match found.
        """
        term_lower = term.lower().strip()

        # 1. Exact match
        if term_lower in self._index:
            return self._index[term_lower]

        # 2. Substring: does the index key contain the term?
        for key, ticker in self._index.items():
            if term_lower in key or key in term_lower:
                return ticker

        return None

    def lookup_list(self, terms: list[str]) -> dict[str, str]:
        """
        Lookup multiple terms. Returns {term: ticker} for matched terms only.
        Always adds SPY as market benchmark.
        """
        result = {}
        for term in terms:
            ticker = self.lookup(term)
            if ticker:
                result[term] = ticker
        result["market (SPY)"] = "SPY"
        return result

    @property
    def all_keys(self) -> list[str]:
        """All available lookup keys (for use in Claude prompts)."""
        return sorted(self._index.keys())

    def available_for_prompt(self) -> str:
        """
        Returns a formatted string of available sector/country keys
        to inject into Claude prompts (replaces the hardcoded list).
        """
        country_keys = ", ".join(sorted(self.countries.keys()))
        sector_keys  = ", ".join(sorted(self.sectors.keys()))
        return f"COUNTRIES: {country_keys}\nSECTORS: {sector_keys}"


# Singleton for import convenience
@lru_cache(maxsize=1)
def get_loader() -> ETFLoader:
    return ETFLoader()
