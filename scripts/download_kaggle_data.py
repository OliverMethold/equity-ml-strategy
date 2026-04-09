"""
Downloads S&P 500 historical price data from Kaggle and loads it into
the data cache so the pipeline can run without any Yahoo Finance access.

Prerequisites:
1. pip install kaggle
2. Place your kaggle.json API key at C:\\Users\\YOUR_USERNAME\\.kaggle\\kaggle.json
   (Get it from kaggle.com -> profile -> Settings -> API -> Create New Token)

Usage:
    python scripts/download_kaggle_data.py
    python scripts/run_full_pipeline.py
"""

import sys
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("kaggle_loader")

from src.config import RAW_DIR, LIQUID_UNIVERSE_30, BENCHMARK_TICKER

ALL_TICKERS = LIQUID_UNIVERSE_30 + [BENCHMARK_TICKER]

# Where to store the downloaded zip and extracted files
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "kaggle_raw"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    """Download the Kaggle S&P 500 dataset."""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    logger.info("Downloading S&P 500 dataset from Kaggle...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "camnugent/sandp500",
            path=str(DOWNLOAD_DIR),
            unzip=True,
            quiet=False,
        )
        logger.info("Download complete.")
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        logger.info("Trying alternative dataset...")
        try:
            kaggle.api.dataset_download_files(
                "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
                path=str(DOWNLOAD_DIR),
                unzip=True,
                quiet=False,
            )
            logger.info("Alternative download complete.")
        except Exception as exc2:
            logger.error("Both downloads failed: %s", exc2)
            sys.exit(1)


def find_csv_for_ticker(ticker: str) -> Path | None:
    """Search the download directory for a CSV matching the ticker."""
    search_name = ticker.lower().replace("-", "_")
    search_name2 = ticker.lower().replace("-", ".")

    for csv_file in DOWNLOAD_DIR.rglob("*.csv"):
        stem = csv_file.stem.lower()
        if stem == ticker.lower() or stem == search_name or stem == search_name2:
            return csv_file
        # Handle BRK-B -> brk-b or brk_b
        if ticker.upper() in ["BRK-B", "BRK.B"]:
            if "brk" in stem and ("b" in stem):
                return csv_file

    # Also check csv_data folder if user placed files there manually
    csv_data_dir = Path(__file__).resolve().parent.parent / "csv_data"
    if csv_data_dir.exists():
        for csv_file in csv_data_dir.glob("*.csv"):
            stem = csv_file.stem.lower()
            if stem == ticker.lower() or stem == search_name:
                return csv_file

    return None


def parse_csv(ticker: str, path: Path) -> pd.DataFrame | None:
    """Parse a stock CSV into the standard format."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Normalise date column
        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col is None:
            logger.error("No date column in %s", path)
            return None
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        # Normalise close column — prefer adj close
        if "adj_close" in df.columns:
            df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        elif "adj._close" in df.columns:
            df["close"] = pd.to_numeric(df["adj._close"], errors="coerce")
        elif "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        else:
            logger.error("No close column in %s", path)
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
        df = df.sort_values("date")

        return df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()

    except Exception as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None


def main():
    # Download from Kaggle
    download_dataset()

    # List what was downloaded
    all_csvs = list(DOWNLOAD_DIR.rglob("*.csv"))
    logger.info("Found %d CSV files after download.", len(all_csvs))

    success = 0
    missing = []

    for ticker in ALL_TICKERS:
        path = find_csv_for_ticker(ticker)
        if path is None:
            logger.warning("No CSV found for %s", ticker)
            missing.append(ticker)
            continue

        df = parse_csv(ticker, path)
        if df is None or df.empty:
            logger.error("Could not parse data for %s", ticker)
            missing.append(ticker)
            continue

        cache_path = RAW_DIR / f"{ticker.replace('/', '_').replace('-', '-')}.parquet"
        df.to_parquet(cache_path, index=False)
        logger.info(
            "Cached %s: %d rows (%s to %s)",
            ticker, len(df),
            df["date"].min().date(),
            df["date"].max().date(),
        )
        success += 1

    print(f"\nResult: {success}/{len(ALL_TICKERS)} tickers cached.")
    if missing:
        print(f"Missing: {missing}")
        print("These tickers weren't in the Kaggle dataset.")
        print("You can manually download their CSVs from Yahoo Finance and")
        print("place them in the csv_data/ folder, then run:")
        print("  python scripts/load_from_csv.py")
    if success >= 20:
        print("\nEnough data to run the pipeline. Now run:")
        print("  python scripts/run_full_pipeline.py")


if __name__ == "__main__":
    main()
