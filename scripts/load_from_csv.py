"""
Import stock data from the Huge Stock Market Dataset (Boris Marjanovic / Kaggle)
into the data cache so the pipeline can run without any network access.

File format: aapl.us.txt, spy.us.txt etc. (CSV content with .txt extension)

ROADMAP: Auto-discovery mode — loads ALL available tickers from csv_data/,
applies quality filters, and caches up to MAX_UNIVERSE_SIZE tickers.
The pipeline then uses whatever is cached.

Usage:
1. Download: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
2. Copy all files from the Stocks/ folder into csv_data/ in the project root.
3. Copy spy.us.txt from the ETFs/ folder into csv_data/ as well.
4. Run: python scripts/load_from_csv.py
5. Then run: python scripts/run_full_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("csv_loader")

from src.config import (
    RAW_DIR, LIQUID_UNIVERSE_30, BENCHMARK_TICKER,
    MIN_PRICE, MIN_HISTORY_DAYS, MIN_AVG_VOLUME, MAX_UNIVERSE_SIZE,
)

CSV_DIR = Path(__file__).resolve().parent.parent / "csv_data"
PRIORITY_TICKERS = set(LIQUID_UNIVERSE_30 + [BENCHMARK_TICKER])


def find_file(ticker: str) -> Path | None:
    if not CSV_DIR.exists():
        return None
    ticker_lower = ticker.lower().replace("-", ".")
    for ext in ["*.txt", "*.csv"]:
        for f in CSV_DIR.glob(ext):
            stem = f.stem.lower()
            base = stem.split(".")[0]
            if base == ticker_lower.split(".")[0]:
                return f
    return None


def parse_file(ticker: str, path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col is None:
            return None
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        # Prefer adj_close; this dataset uses Close (already adjusted)
        if "adj_close" in df.columns:
            df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        elif "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        else:
            return None

        for col in ["open", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = float("nan")

        df["ticker"] = ticker
        df = df.dropna(subset=["date", "close"])
        df = df[df["close"] > 0]
        df = df[df["date"] >= pd.Timestamp("2010-01-01")]
        df = df.sort_values("date").reset_index(drop=True)

        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None

        return df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()

    except Exception as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None


def passes_quality_filter(df: pd.DataFrame, ticker: str) -> bool:
    """
    ROADMAP quality filters:
    - Minimum price (exclude penny stocks)
    - Minimum history (252 trading days)
    - Minimum average volume (liquidity)
    - No long runs of identical prices (stale data)
    """
    if len(df) < MIN_HISTORY_DAYS:
        return False

    avg_price = df["close"].mean()
    if avg_price < MIN_PRICE:
        return False

    avg_volume = df["volume"].mean()
    if avg_volume < MIN_AVG_VOLUME:
        return False

    # Check for stale data: runs of >5 identical close prices
    price_runs = (df["close"] == df["close"].shift(1)).rolling(5).sum()
    if (price_runs >= 5).any():
        return False

    return True


def auto_discover_tickers() -> list[str]:
    """
    ROADMAP: Discover all tickers available in csv_data/,
    apply quality filters, return up to MAX_UNIVERSE_SIZE tickers.
    Priority tickers (LIQUID_UNIVERSE_30 + SPY) are always included if available.
    """
    if not CSV_DIR.exists():
        return []

    all_files = list(CSV_DIR.glob("*.txt")) + list(CSV_DIR.glob("*.csv"))
    logger.info("Found %d files in csv_data/ for auto-discovery.", len(all_files))

    tickers_from_files = []
    for f in all_files:
        stem = f.stem.lower()
        base = stem.split(".")[0].upper()
        # Skip obviously non-stock files
        if len(base) > 6 or len(base) < 1:
            continue
        tickers_from_files.append((base, f))

    return [t for t, _ in tickers_from_files]


def main(auto_discover: bool = True):
    if not CSV_DIR.exists():
        logger.error(
            "csv_data/ folder not found at %s\n"
            "Create this folder and place your Kaggle stock files in it.",
            CSV_DIR,
        )
        sys.exit(1)

    all_files = list(CSV_DIR.glob("*.txt")) + list(CSV_DIR.glob("*.csv"))
    logger.info("Found %d files in csv_data/", len(all_files))

    if len(all_files) == 0:
        logger.error("No files found in csv_data/. Copy the Stocks/ folder contents there.")
        sys.exit(1)

    # Phase 1: priority tickers (LIQUID_UNIVERSE_30 + SPY)
    priority_success = 0
    priority_missing = []

    for ticker in PRIORITY_TICKERS:
        path = find_file(ticker)
        if path is None:
            logger.warning("No file found for priority ticker %s", ticker)
            priority_missing.append(ticker)
            continue

        df = parse_file(ticker, path)
        if df is None or df.empty:
            priority_missing.append(ticker)
            continue

        safe = ticker.replace("/", "_")
        df.to_parquet(RAW_DIR / f"{safe}.parquet", index=False)
        logger.info("Cached %s: %d rows (%s to %s)",
                    ticker, len(df), df["date"].min().date(), df["date"].max().date())
        priority_success += 1

    # Phase 2: auto-discover additional tickers to expand universe
    if auto_discover:
        already_cached = {p.stem for p in RAW_DIR.glob("*.parquet")}
        discovered_count = 0
        skipped_quality = 0

        for f in all_files:
            stem = f.stem.lower()
            base = stem.split(".")[0].upper()

            if len(base) > 6 or len(base) < 1:
                continue
            if base in already_cached:
                continue
            if discovered_count + priority_success >= MAX_UNIVERSE_SIZE:
                break

            df = parse_file(base, f)
            if df is None or df.empty:
                continue

            if not passes_quality_filter(df, base):
                skipped_quality += 1
                continue

            safe = base.replace("/", "_")
            df.to_parquet(RAW_DIR / f"{safe}.parquet", index=False)
            discovered_count += 1

        logger.info(
            "Auto-discovery: added %d tickers (skipped %d for quality).",
            discovered_count, skipped_quality,
        )

    total_cached = len(list(RAW_DIR.glob("*.parquet")))
    print(f"\nResult: {priority_success}/{len(PRIORITY_TICKERS)} priority tickers cached.")
    print(f"Total universe size in cache: {total_cached} tickers.")
    if priority_missing:
        print(f"Missing priority tickers: {priority_missing}")
    if total_cached >= 15:
        print("\nReady to run the pipeline:")
        print("  python scripts/run_full_pipeline.py")


if __name__ == "__main__":
    main()
