"""
Live signal generator.

Pulls the latest market data, computes features, loads the most recently
trained model, and outputs today's recommended positions to a CSV and stdout.

This script satisfies the "uses live market data" claim without placing any
actual trades. yfinance data is delayed ~15 minutes for most US equities,
which is acceptable for a daily-frequency strategy that signals the next
session's open.

Usage:
    python scripts/run_live_signal.py
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_signal")

LOOKBACK_DAYS = 300  # how many calendar days of history to fetch for feature computation


def main() -> None:
    from src.config import LIQUID_UNIVERSE_30, REPORTS_DIR
    from src.data.loader import get_prices
    from src.features.build import build_feature_matrix, get_feature_columns
    from src.models.registry import load_latest_model, load_model_meta
    from src.models.predict import predict_signals

    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    logger.info("Live signal run: %s", today)
    logger.info("Fetching %d days of data (%s to %s)...", LOOKBACK_DAYS, start_date, end_date)

    # ------------------------------------------------------------------ #
    # 1. Pull latest data (bypasses cache to get today's bar)
    # ------------------------------------------------------------------ #
    prices = get_prices(
        LIQUID_UNIVERSE_30,
        start=start_date,
        end=end_date,
        force_refresh=True,   # always fetch fresh data for live signals
    )

    if prices.empty:
        logger.error("No price data fetched. Markets may be closed or network issue.")
        sys.exit(1)

    latest_date = prices["date"].max()
    logger.info("Latest available data date: %s", latest_date.date())

    # ------------------------------------------------------------------ #
    # 2. Build features — we need history to compute rolling indicators
    # ------------------------------------------------------------------ #
    logger.info("Computing features...")
    features = build_feature_matrix(prices, save=False)  # don't overwrite the training cache

    if features.empty:
        logger.error("Feature computation produced empty DataFrame.")
        sys.exit(1)

    # Restrict to the most recent available date only
    most_recent_features = features[features["date"] == latest_date].copy()

    if most_recent_features.empty:
        logger.error(
            "No features for the most recent date (%s). "
            "This can happen if the market was closed today.",
            latest_date.date(),
        )
        sys.exit(1)

    logger.info(
        "Features computed for %d tickers on %s.",
        len(most_recent_features),
        latest_date.date(),
    )

    # ------------------------------------------------------------------ #
    # 3. Load the latest trained model
    # ------------------------------------------------------------------ #
    logger.info("Loading latest trained model...")
    try:
        model, window_end = load_latest_model()
        meta = load_model_meta(window_end)
        feature_cols = meta.get("feature_cols") or get_feature_columns(most_recent_features)
        logger.info(
            "Model trained up to: %s | Type: %s | Features: %d",
            window_end,
            meta.get("model_type", "unknown"),
            len(feature_cols),
        )
    except FileNotFoundError:
        logger.error(
            "No trained model found. Run scripts/run_full_pipeline.py first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Generate signals
    # ------------------------------------------------------------------ #
    logger.info("Generating signals...")
    signals = predict_signals(
        features=most_recent_features,
        model=model,
        feature_cols=feature_cols,
    )

    # ------------------------------------------------------------------ #
    # 5. Output CSV
    # ------------------------------------------------------------------ #
    date_str = today.strftime("%Y%m%d")
    output_path = REPORTS_DIR / f"signals_{date_str}.csv"
    signals_out = signals[["date", "ticker", "prob_up", "position"]].copy()
    signals_out = signals_out.sort_values("prob_up", ascending=False)
    signals_out.to_csv(output_path, index=False)
    logger.info("Signals written to %s", output_path)

    # ------------------------------------------------------------------ #
    # 6. Print human-readable summary
    # ------------------------------------------------------------------ #
    longs = signals_out[signals_out["position"] == 1]
    flat = signals_out[signals_out["position"] == 0]

    print("\n" + "=" * 60)
    print(f"  LIVE SIGNAL SUMMARY — {today}  (next session open)")
    print("=" * 60)
    print(f"\n  Model trained through: {window_end}")
    print(f"  Signal date:           {latest_date.date()}")
    print(f"  Universe size:         {len(signals_out)} tickers")
    print(f"  LONG positions:        {len(longs)}")
    print(f"  FLAT positions:        {len(flat)}")

    if len(longs) > 0:
        weight = 1 / len(longs)
        print(f"\n  Equal weight per position: {weight:.1%}")
        print("\n  LONGS (sorted by confidence):")
        for _, row in longs.iterrows():
            print(f"    {row['ticker']:8s}  prob_up={row['prob_up']:.3f}")
    else:
        print("\n  No long signals today. All positions: FLAT / CASH.")

    print()
    print("  NOTE: This is a research signal only. No trades have been placed.")
    print(f"  Output CSV: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
