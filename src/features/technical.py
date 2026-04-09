"""
Technical indicator wrappers — original features plus all roadmap additions.

ROADMAP ADDITIONS:
  Momentum quality:  mom_quality_63d, mom_quality_21d, trend_consistency_21d,
                     ret_skewness_63d, max_drawdown_21d, autocorr_5d
  Mean reversion:    z_score_20d, z_score_60d, price_vs_52w_high, price_vs_52w_low,
                     rsi_divergence
  Volume:            up_volume_ratio, volume_trend_10d, intraday_range_norm
  Regime:            (cross-sectional regime features added in build.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats

from src.config import (
    ATR_PERIOD, BB_STD, BB_WINDOW, MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    RETURN_WINDOWS, RSI_PERIOD, SMA_LONG, SMA_SHORT, VOL_MA_WINDOW,
    VOL_WINDOW, MOM_WINDOWS,
)


# ── Original features ───────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_1d"] = log_ret
    for w in RETURN_WINDOWS:
        df[f"log_ret_{w}d"] = np.log(df["close"] / df["close"].shift(w))
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Medium-term momentum, lagged 1 day to avoid lookahead."""
    df = df.copy()
    for w in MOM_WINDOWS:
        df[f"mom_{w}d"] = np.log(df["close"].shift(1) / df["close"].shift(w + 1))
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df[f"vol_{VOL_WINDOW}d"] = log_ret.rolling(VOL_WINDOW).std()
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[f"rsi_{RSI_PERIOD}"] = ta.rsi(df["close"], length=RSI_PERIOD)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    macd_df = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_df is not None and not macd_df.empty:
        df["macd"] = macd_df.iloc[:, 0]
        df["macd_signal"] = macd_df.iloc[:, 2]
        df["macd_hist"] = macd_df.iloc[:, 1]
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bb = ta.bbands(df["close"], length=BB_WINDOW, std=BB_STD)
    if bb is not None and not bb.empty:
        for col_key, out_name in [("BBL", "bb_lower"), ("BBU", "bb_upper"),
                                   ("BBB", "bb_width"), ("BBP", "bb_pct")]:
            matched = [c for c in bb.columns if col_key in c]
            df[out_name] = bb[matched[0]].values if matched else np.nan
    else:
        df["bb_lower"] = df["bb_upper"] = df["bb_width"] = df["bb_pct"] = np.nan
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df[f"atr_{ATR_PERIOD}"] = atr if atr is not None else np.nan
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vol_ma = df["volume"].rolling(VOL_MA_WINDOW).mean()
    df["vol_ratio"] = df["volume"] / vol_ma
    return df


def add_sma_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sma_short = df["close"].rolling(SMA_SHORT).mean()
    sma_long = df["close"].rolling(SMA_LONG).mean()
    df[f"close_to_sma{SMA_SHORT}"] = df["close"] / sma_short - 1
    df[f"close_to_sma{SMA_LONG}"] = df["close"] / sma_long - 1
    return df


# ── ROADMAP: Momentum quality features ─────────────────────────────────────

def add_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum quality = return / volatility over the window.
    High values = strong, low-noise trend (more predictive than raw return).
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for w in [21, 63]:
        rolling_ret = np.log(df["close"].shift(1) / df["close"].shift(w + 1))
        rolling_vol = log_ret.rolling(w).std()
        df[f"mom_quality_{w}d"] = rolling_ret / (rolling_vol + 1e-8)

    # Trend consistency: fraction of up days over prior 21 days
    df["trend_consistency_21d"] = (log_ret > 0).rolling(21).mean()

    # Max drawdown over prior 21 days
    def rolling_max_dd(series, window=21):
        result = []
        for i in range(len(series)):
            if i < window:
                result.append(np.nan)
                continue
            window_prices = series.iloc[i - window:i]
            roll_max = window_prices.cummax()
            dd = ((window_prices - roll_max) / roll_max).min()
            result.append(dd)
        return pd.Series(result, index=series.index)

    df["max_drawdown_21d"] = rolling_max_dd(df["close"])

    # Return autocorrelation (5-day)
    df["autocorr_5d"] = log_ret.rolling(20).apply(
        lambda x: x.autocorr(lag=5) if len(x) >= 10 else np.nan,
        raw=False
    )

    # Return skewness (63-day)
    df["ret_skewness_63d"] = log_ret.rolling(63).apply(
        lambda x: stats.skew(x) if len(x) >= 20 else np.nan, raw=True
    )

    return df


# ── ROADMAP: Mean reversion features ───────────────────────────────────────

def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-scores and price range features identifying stretched/compressed stocks.
    """
    df = df.copy()

    # Z-score: distance from rolling mean in std devs
    for w in [20, 60]:
        roll_mean = df["close"].rolling(w).mean()
        roll_std = df["close"].rolling(w).std()
        df[f"z_score_{w}d"] = (df["close"] - roll_mean) / (roll_std + 1e-8)

    # Position relative to 52-week range
    df["price_vs_52w_high"] = df["close"] / df["close"].rolling(252).max()
    df["price_vs_52w_low"] = df["close"] / df["close"].rolling(252).min()

    # RSI divergence: RSI vs its own moving average
    rsi = ta.rsi(df["close"], length=RSI_PERIOD)
    if rsi is not None:
        df["rsi_divergence"] = rsi - rsi.rolling(14).mean()
    else:
        df["rsi_divergence"] = np.nan

    return df


# ── ROADMAP: Enhanced volume features ──────────────────────────────────────

def add_volume_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume direction and trend features capturing buying/selling pressure.
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # Up volume ratio: volume on up days / total volume (21-day)
    up_vol = (df["volume"] * (log_ret > 0)).rolling(21).sum()
    total_vol = df["volume"].rolling(21).sum()
    df["up_volume_ratio"] = up_vol / (total_vol + 1e-8)

    # Volume trend: slope of log(volume) over 10 days
    def vol_trend_slope(series, window=10):
        result = []
        x = np.arange(window)
        for i in range(len(series)):
            if i < window:
                result.append(np.nan)
                continue
            y = np.log(series.iloc[i - window:i].values + 1)
            if np.std(y) < 1e-8:
                result.append(0.0)
                continue
            slope = np.polyfit(x, y, 1)[0]
            result.append(slope / (np.mean(y) + 1e-8))  # normalise
        return pd.Series(result, index=series.index)

    df["volume_trend_10d"] = vol_trend_slope(df["volume"])

    # Intraday range normalised to its 20-day average
    intraday_range = (df["high"] - df["low"]) / df["close"]
    df["intraday_range_norm"] = intraday_range / (intraday_range.rolling(20).mean() + 1e-8)

    # Relative volume (today vs 63-day average)
    df["relative_volume"] = df["volume"] / (df["volume"].rolling(63).mean() + 1e-8)

    return df


# ── Master function ─────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all indicators to a single-ticker price DataFrame.
    Input must have [date, open, high, low, close, volume], sorted ascending.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df = add_return_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_sma_features(df)
    # ROADMAP additions
    df = add_momentum_quality(df)
    df = add_mean_reversion_features(df)
    df = add_volume_direction_features(df)
    return df
