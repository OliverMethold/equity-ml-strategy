"""
Data loader with parquet caching.

Loads from local parquet cache in data/raw/.
Run scripts/load_from_csv.py first to populate the cache from Kaggle data.
Falls back to yfinance if cache miss (requires network access).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return RAW_DIR / f"{safe}.parquet"


def _load_from_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        logger.debug("Cache hit for %s (%d rows)", ticker, len(df))
        return df.copy()
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", ticker, exc)
    return None


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data for a list of tickers from local parquet cache.
    Run scripts/load_from_csv.py first to populate the cache.
    Falls back to yfinance for any tickers not in cache.
    """
    if end is None:
        end = pd.Timestamp("today").strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        if not force_refresh:
            cached = _load_from_cache(ticker)
            if cached is not None and not cached.empty:
                cached["date"] = pd.to_datetime(cached["date"])
                mask = cached["date"] >= pd.Timestamp(start)
                if end:
                    mask &= cached["date"] <= pd.Timestamp(end)
                filtered = cached[mask]
                if not filtered.empty:
                    frames.append(filtered)
                    continue

        # Fallback to yfinance
        logger.info("No cache for %s, attempting yfinance download...", ticker)
        try:
            import yfinance as yf
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.reset_index()
                raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
                raw = raw.rename(columns={"datetime": "date", "adj_close": "close"})
                raw["ticker"] = ticker
                raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
                raw = raw.dropna(subset=["close"])
                required = {"date", "open", "high", "low", "close", "volume"}
                if required.issubset(set(raw.columns)):
                    df = raw[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
                    df.to_parquet(_cache_path(ticker), index=False)
                    frames.append(df)
                    continue
        except Exception as exc:
            logger.debug("yfinance fallback failed for %s: %s", ticker, exc)

        logger.warning(
            "No data available for %s. Run scripts/load_from_csv.py first.", ticker
        )

    if not frames:
        logger.error("No data loaded. Run scripts/load_from_csv.py to populate the cache.")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("Loaded %d rows for %d tickers from cache.", len(result), result["ticker"].nunique())
    return result
