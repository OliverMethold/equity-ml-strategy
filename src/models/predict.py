"""
Inference module.

ROADMAP CHANGES:
- Entry/exit hysteresis: only enter at LONG_THRESHOLD, only exit below EXIT_THRESHOLD
- Position capped at MAX_POSITIONS — take top N by prob_up when more qualify
- Position persistence: existing longs are retained unless prob drops below EXIT_THRESHOLD
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import LONG_THRESHOLD, EXIT_THRESHOLD, MAX_POSITIONS
from src.features.build import get_feature_columns
from src.models.registry import load_latest_model, load_model_meta

logger = logging.getLogger(__name__)


def predict_signals(
    features: pd.DataFrame,
    model=None,
    scaler: StandardScaler | None = None,
    feature_cols: list[str] | None = None,
    previous_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate trading signals with hysteresis and position capping.

    ROADMAP: Positions use a band threshold:
    - New long: prob_up >= LONG_THRESHOLD (0.55)
    - Hold existing long: prob_up >= EXIT_THRESHOLD (0.45)
    - Maximum MAX_POSITIONS longs per day; if more qualify, take top N by prob_up.
    """
    if model is None:
        model, window_end = load_latest_model()
        meta = load_model_meta(window_end)
        feature_cols = meta.get("feature_cols") or get_feature_columns(features)
        scaler = None

    if feature_cols is None:
        feature_cols = get_feature_columns(features)

    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing columns: {missing}")

    X = features[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
        X = ss.fit_transform(X)
        logger.warning("No scaler provided — fitting on current data (use training scaler for production).")

    prob_up = model.predict_proba(X)[:, 1]

    out = features[["date", "ticker"]].copy().reset_index(drop=True)
    out["prob_up"] = prob_up
    out["position"] = 0

    # Build previous position set for hysteresis
    prev_long_tickers: set[str] = set()
    if previous_positions is not None and not previous_positions.empty:
        prev_long_tickers = set(
            previous_positions.loc[previous_positions["position"] == 1, "ticker"]
        )

    # Apply hysteresis: new entry >= LONG_THRESHOLD, hold >= EXIT_THRESHOLD
    for i, row in out.iterrows():
        ticker = row["ticker"]
        p = row["prob_up"]
        if ticker in prev_long_tickers:
            # Already long — hold unless prob drops below exit threshold
            out.at[i, "position"] = 1 if p >= EXIT_THRESHOLD else 0
        else:
            # New entry only if above entry threshold
            out.at[i, "position"] = 1 if p >= LONG_THRESHOLD else 0

    # Cap positions: if more than MAX_POSITIONS qualify, keep top N by prob_up
    longs = out[out["position"] == 1]
    if len(longs) > MAX_POSITIONS:
        top_tickers = longs.nlargest(MAX_POSITIONS, "prob_up")["ticker"].values
        out.loc[out["position"] == 1, "position"] = 0
        out.loc[out["ticker"].isin(top_tickers), "position"] = 1

    logger.info(
        "Signal generation: %d longs (capped at %d), %d flat.",
        (out["position"] == 1).sum(), MAX_POSITIONS, (out["position"] == 0).sum(),
    )
    return out


def signals_to_position_matrix(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])

    # ── DEFENSIVE CHECK ──────────────────────────────────────────────
    if "position" not in signals.columns:
        raise ValueError(
            f"signals DataFrame is missing 'position' column. "
            f"Got columns: {signals.columns.tolist()}. "
            f"Make sure you call predict_signals() before signals_to_position_matrix()."
        )
    # ─────────────────────────────────────────────────────────────────

    wide = signals.pivot_table(index="date", columns="ticker", values="position", fill_value=0)
    row_sums = wide.sum(axis=1).replace(0, 1)
    weights = wide.div(row_sums, axis=0)
    return weights
