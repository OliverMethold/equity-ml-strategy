"""
Universe management.

ROADMAP: get_universe() now reads from the parquet cache rather than a
hard-coded list, enabling use of any stocks loaded via load_from_csv.py.
Falls back to LIQUID_UNIVERSE_30 if cache is empty.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import LIQUID_UNIVERSE_30, RAW_DIR, BENCHMARK_TICKER

logger = logging.getLogger(__name__)


def get_universe_from_cache() -> list[str]:
    """
    Return all tickers available in the parquet cache, excluding the benchmark.
    Applies a minimal date-range filter: must have data from before 2014.
    """
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet cache found. Falling back to LIQUID_UNIVERSE_30.")
        return list(LIQUID_UNIVERSE_30)

    tickers = []
    for p in parquet_files:
        ticker = p.stem
        if ticker == BENCHMARK_TICKER:
            continue
        try:
            df = pd.read_parquet(p, columns=["date"])
            df["date"] = pd.to_datetime(df["date"])
            if df["date"].min() <= pd.Timestamp("2014-01-01") and len(df) >= 252:
                tickers.append(ticker)
        except Exception:
            continue

    if not tickers:
        logger.warning("Cache exists but no tickers passed filters. Falling back to LIQUID_UNIVERSE_30.")
        return list(LIQUID_UNIVERSE_30)

    logger.info("Universe from cache: %d tickers.", len(tickers))
    return tickers


def get_universe(use_full_cache: bool = True) -> list[str]:
    """
    Return the trading universe.

    Parameters
    ----------
    use_full_cache : bool
        If True (default), use all tickers in the parquet cache.
        If False, use the hard-coded LIQUID_UNIVERSE_30 from config.
    """
    if use_full_cache:
        return get_universe_from_cache()
    logger.info("Using hard-coded 30-ticker universe from config.py.")
    return list(LIQUID_UNIVERSE_30)
