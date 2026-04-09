"""
Full pipeline: data -> features -> train -> backtest -> HTML report.

ROADMAP CHANGES implemented in this run:
  - Universe: reads all tickers from parquet cache (not just hard-coded 30)
  - Features: ~50 features including momentum quality, mean reversion, regime
  - Target: 5-day forward return (PREDICTION_HORIZON=5)
  - Model: rolling window training with embargo gap
  - Signals: entry/exit hysteresis + MAX_POSITIONS cap
  - Walk-forward: embargo between train and test windows

Usage:
    python scripts/run_full_pipeline.py

Environment variables:
    UNIVERSE=small   Use hard-coded 30-ticker list (faster for testing)
    UNIVERSE=full    Use all tickers in cache (default)
    FORCE_REFRESH=1  Rebuild feature matrix from scratch
"""

from __future__ import annotations

import logging
import os
import sys
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    t0 = time.time()

    use_full_cache = os.getenv("UNIVERSE", "full").lower() != "small"
    model_type = os.getenv("MODEL_TYPE", "xgboost")
    force_refresh = os.getenv("FORCE_REFRESH", "0") == "1"

    from src.config import BENCHMARK_TICKER, TRAIN_START, PROCESSED_DIR, INITIAL_CAPITAL, \
                       LONG_THRESHOLD, EXIT_THRESHOLD, MAX_POSITIONS
    from src.data.loader import get_prices
    from src.data.universe import get_universe
    from src.features.build import build_feature_matrix, load_feature_matrix
    from src.models.train import assemble_oos_predictions, walk_forward_train
    from src.models.predict import signals_to_position_matrix
    from src.backtest.engine import run_backtest, get_benchmark_equity, get_benchmark_returns
    from src.backtest.metrics import compute_all_metrics
    from src.reporting.plots import generate_html_report, save_html_report
    

    # ------------------------------------------------------------------ #
    # Step 1: Universe
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STEP 1/6  Universe selection")
    tickers = get_universe(use_full_cache=use_full_cache)
    all_tickers = tickers + [BENCHMARK_TICKER]
    logger.info("Universe: %d tickers (+ benchmark %s)", len(tickers), BENCHMARK_TICKER)

    # ------------------------------------------------------------------ #
    # Step 2: Load data
    # ------------------------------------------------------------------ #
    logger.info("STEP 2/6  Loading price data (%s onwards)", TRAIN_START)
    prices = get_prices(all_tickers, start=TRAIN_START)
    if prices.empty:
        logger.error("No price data loaded. Run scripts/load_from_csv.py first.")
        sys.exit(1)

    n_tickers = prices["ticker"].nunique()
    date_range = f"{prices['date'].min().date()} to {prices['date'].max().date()}"
    logger.info("Loaded %d rows for %d tickers (%s)", len(prices), n_tickers, date_range)

    universe_prices = prices[prices["ticker"].isin(tickers)].copy()
    benchmark_prices = prices[prices["ticker"] == BENCHMARK_TICKER].copy()

    # ------------------------------------------------------------------ #
    # Step 3: Feature engineering
    # ------------------------------------------------------------------ #
    logger.info("STEP 3/6  Building feature matrix")
    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists() and not force_refresh:
        logger.info("Loading cached feature matrix from %s", features_path)
        features = load_feature_matrix()
    else:
        logger.info("Building feature matrix (this takes a few minutes for large universes)...")
        features = build_feature_matrix(universe_prices, save=True)

    logger.info(
        "Feature matrix: %d rows, %d columns, %d tickers",
        len(features), features.shape[1], features["ticker"].nunique(),
    )

    # ------------------------------------------------------------------ #
    # Step 4: Walk-forward training
    # ------------------------------------------------------------------ #
    logger.info("STEP 4/6  Walk-forward training (%s)", model_type)
    fold_results = walk_forward_train(features, model_type=model_type, save_folds=True)
    logger.info("Completed %d walk-forward folds.", len(fold_results))

    oos_predictions = assemble_oos_predictions(fold_results)
    logger.info(
        "OOS predictions: %d rows from %s to %s",
        len(oos_predictions),
        oos_predictions["date"].min().date(),
        oos_predictions["date"].max().date(),
    )

    print("\n--- Walk-Forward Fold Metrics ---")
    for r in fold_results:
        m = r.metrics
        print(
            f"  Train end: {r.train_end}  |  Test: {r.test_start} to {r.test_end}  "
            f"|  Acc: {m.get('accuracy', 0):.3f}  "
            f"|  AUC: {m.get('roc_auc', 0):.3f}  "
            f"|  LogLoss: {m.get('log_loss', 0):.4f}"
        )
    print()

    # ------------------------------------------------------------------ #
    # Step 5: Signal construction + backtest
    # ------------------------------------------------------------------ #
    logger.info("STEP 5/6  Building signals and running backtest")

    # ── REPLACE the position-generation block with this ──────────────────

    oos_predictions = oos_predictions.copy().sort_values("date")
    oos_predictions["position"] = 0

    prev_longs: set = set()

    for date, group in oos_predictions.groupby("date"):
        idx = group.index
        probs = group.set_index("ticker")["prob_up"]

        # Hold existing longs if prob still above EXIT threshold
        held = {t for t in prev_longs if t in probs.index and probs[t] >= EXIT_THRESHOLD}
        # Add new longs above ENTRY threshold
        new_entries = set(probs[probs >= LONG_THRESHOLD].index)
        candidates = held | new_entries

        # Cap to MAX_POSITIONS — prefer highest prob
        if len(candidates) > MAX_POSITIONS:
            candidates = set(
                probs[probs.index.isin(candidates)]
                .nlargest(MAX_POSITIONS).index
            )

        # Write back
        oos_predictions.loc[
            idx[group["ticker"].isin(candidates).values], "position"
        ] = 1

        prev_longs = candidates
    # ─────────────────────────────────────────────────────────────────────

    # Only rebalance on Mondays — reduces unnecessary weight churn
    weights = signals_to_position_matrix(oos_predictions, universe_prices)
    weights = weights[weights.index.dayofweek == 0]  # Monday only
    weights = weights.reindex(
        pd.bdate_range(weights.index.min(), weights.index.max()),
        method="ffill"
    )

    logger.info(
        "Position matrix: %d dates, %d tickers, avg daily longs: %.1f",
        len(weights), weights.shape[1],
        (weights > 0).sum(axis=1).mean(),
    )

    oos_start = oos_predictions["date"].min()
    oos_end = oos_predictions["date"].max()
    oos_prices = universe_prices[
        (universe_prices["date"] >= oos_start) &
        (universe_prices["date"] <= oos_end)
    ].copy()

    result = run_backtest(weights, oos_prices)

    bm_prices = benchmark_prices[
        (benchmark_prices["date"] >= oos_start) &
        (benchmark_prices["date"] <= oos_end)
    ].copy()
    benchmark_ret = get_benchmark_returns(bm_prices, BENCHMARK_TICKER) if not bm_prices.empty else None
    benchmark_eq = get_benchmark_equity(bm_prices, BENCHMARK_TICKER, INITIAL_CAPITAL) if not bm_prices.empty else None

    # ------------------------------------------------------------------ #
    # Step 6: Metrics and report
    # ------------------------------------------------------------------ #
    logger.info("STEP 6/6  Computing metrics and generating report")

    strategy_curve = result.equity_curve
    daily_returns = result.daily_returns

    # Align benchmark if available
    benchmark_eq_aligned = None
    benchmark_ret_aligned = None
    if benchmark_eq is not None and len(benchmark_eq) > 0:
        common_dates = strategy_curve.index.intersection(benchmark_eq.index)
        if len(common_dates) > 0:
            benchmark_eq_aligned = benchmark_eq.loc[common_dates]
            benchmark_ret_aligned = benchmark_ret.loc[common_dates]

    metrics = compute_all_metrics(
        equity_curve=strategy_curve,
        daily_returns=daily_returns,
        weights=weights,
        benchmark_returns=benchmark_ret_aligned,
    )

    print("--- Backtest Performance Summary ---")
    print(f"  Total Return:     {metrics['total_return']:.1%}")
    print(f"  CAGR:             {metrics['cagr']:.1%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:    {metrics['sortino']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:.1%}")
    print(f"  Max DD Duration:  {metrics['max_drawdown_duration_days']:.0f} days")
    print(f"  Hit Rate:         {metrics['hit_rate']:.1%}")
    print(f"  Annual Turnover:  {metrics['annualised_turnover']:.0%}")
    if "benchmark_cagr" in metrics:
        print(f"  Benchmark CAGR:   {metrics['benchmark_cagr']:.1%}")
        print(f"  Benchmark Sharpe: {metrics['benchmark_sharpe']:.2f}")
    print()

    if metrics["sharpe"] > 2.0:
        logger.warning(
            "Sharpe of %.2f is suspiciously high — review for lookahead bias.",
            metrics["sharpe"],
        )

    last_fold = fold_results[-1]
    fold_metrics_list = [
        {**r.metrics, "fold": i + 1, "train_end": r.train_end,
         "test_start": r.test_start, "test_end": r.test_end}
        for i, r in enumerate(fold_results)
    ]

    html = generate_html_report(
        strategy_curve=strategy_curve,
        daily_returns=daily_returns,
        benchmark_curve=benchmark_eq_aligned,
        benchmark_returns=benchmark_ret_aligned,
        weights=weights,
        model=last_fold.model,
        feature_cols=last_fold.feature_cols,
        fold_metrics=fold_metrics_list,
    )

    report_path = save_html_report(html)
    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds. Report: %s", elapsed, report_path)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
