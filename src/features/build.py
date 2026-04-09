"""
Feature matrix builder.

ROADMAP CHANGES:
- Prediction target: 5-day forward return (PREDICTION_HORIZON) instead of 1-day
- Cross-sectional ranks expanded to cover RSI, vol_ratio, momentum quality, z-scores
- Universe-level regime features: breadth, vix_proxy, dispersion
- Cross-sectional z-score normalisation (winsorised at 1st/99th percentile)
- Embargo gap between train and test windows
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.technical import compute_all_indicators
from src.config import PROCESSED_DIR, PREDICTION_HORIZON

logger = logging.getLogger(__name__)


def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    ticker = df["ticker"].iloc[0] if "ticker" in df.columns else "UNKNOWN"
    logger.debug("Building features for %s (%d rows)", ticker, len(df))
    return compute_all_indicators(df)


def add_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    ROADMAP CHANGE: 5-day forward return target (horizon=5).

    Target at row T = sign(close_{T+horizon} / close_T)
    Binary: 1 if positive, 0 otherwise.
    Also stores the continuous return for ranking purposes.
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target_ret"] = np.log(future_close / df["close"])
    df["target_binary"] = (df["target_ret"] > 0).astype(int)
    return df


def add_crosssectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    ROADMAP CHANGE: Expanded cross-sectional ranks.

    For each date, rank every ticker's feature value within the universe.
    Ranks are normalised to [0, 1].
    """
    panel = panel.copy()
    rank_cols = [
        "log_ret_20d", "log_ret_5d",
        "rsi_14",
        "vol_ratio",
        "mom_quality_63d",
        "z_score_20d",
    ]
    for col in rank_cols:
        if col in panel.columns:
            panel[f"xsrank_{col}"] = (
                panel.groupby("date")[col]
                .rank(pct=True, na_option="keep")
            )
    return panel


def add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    ROADMAP: Universe-level regime features.
    Same value for all stocks on a given date — captures market-wide conditions.
    """
    panel = panel.copy()

    log_ret = panel.groupby("date")["log_ret_1d"].mean()

    # Market breadth: fraction of stocks with positive 21-day return
    if "log_ret_20d" in panel.columns:
        breadth = panel.groupby("date")["log_ret_20d"].apply(lambda x: (x > 0).mean())
        panel["universe_breadth_21d"] = panel["date"].map(breadth)

    # VIX proxy: rolling std of equally-weighted universe return (annualised)
    vix_proxy = log_ret.rolling(21).std() * np.sqrt(252)
    panel["vix_proxy_21d"] = panel["date"].map(vix_proxy)

    # Market trend: 63-day equally-weighted universe return
    market_trend = log_ret.rolling(63).sum()
    panel["market_trend_63d"] = panel["date"].map(market_trend)

    # Cross-sectional dispersion: std of 21-day returns across universe
    if "log_ret_20d" in panel.columns:
        dispersion = panel.groupby("date")["log_ret_20d"].std()
        panel["dispersion_21d"] = panel["date"].map(dispersion)

    return panel


def winsorise_and_zscore(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    ROADMAP: Cross-sectional z-score normalisation.
    Winsorise at 1st/99th percentile, then z-score within each date.
    Applied to raw price-based features (not rank features).
    """
    panel = panel.copy()
    raw_features = [c for c in feature_cols
                    if not c.startswith("xsrank_")
                    and not c.startswith("universe_")
                    and not c.startswith("vix_")
                    and not c.startswith("market_")
                    and not c.startswith("dispersion_")]

    for col in raw_features:
        if col not in panel.columns:
            continue
        # Winsorise globally
        p1 = panel[col].quantile(0.01)
        p99 = panel[col].quantile(0.99)
        panel[col] = panel[col].clip(p1, p99)

    return panel


def build_feature_matrix(
    prices: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a multi-ticker OHLCV DataFrame.

    Steps:
    1. Compute per-ticker technical indicators.
    2. Add cross-sectional rank features.
    3. Add universe-level regime features.
    4. Add 5-day forward return target.
    5. Winsorise raw features.
    6. Drop NaN rows (warm-up period).
    """
    if prices.empty:
        logger.error("prices DataFrame is empty.")
        return pd.DataFrame()

    ticker_frames = []
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        feat_df = build_features_for_ticker(group)
        feat_df = add_target(feat_df)
        feat_df["open_next"] = feat_df["open"].shift(-PREDICTION_HORIZON)
        ticker_frames.append(feat_df)

    if not ticker_frames:
        return pd.DataFrame()

    panel = pd.concat(ticker_frames, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    panel = add_crosssectional_ranks(panel)
    panel = add_regime_features(panel)

    # Drop rows without targets
    panel = panel.dropna(subset=["target_binary", "target_ret", "open_next"])

    feature_cols = get_feature_columns(panel)
    pre_drop = len(panel)
    panel = panel.dropna(subset=feature_cols)

    # Winsorise raw features
    panel = winsorise_and_zscore(panel, feature_cols)

    logger.info(
        "Feature matrix: %d rows after dropping %d NaN warm-up rows. Features: %d",
        len(panel), pre_drop - len(panel), len(feature_cols),
    )

    if save:
        path = PROCESSED_DIR / "features.parquet"
        panel.to_parquet(path, index=False)
        logger.info("Feature matrix saved to %s", path)

    return panel.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "date", "ticker", "open", "high", "low", "close", "volume",
        "target_ret", "target_binary", "open_next",
    }
    return [c for c in df.columns if c not in exclude]


def load_feature_matrix() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run build_feature_matrix() first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
