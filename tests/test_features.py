"""
Tests for feature engineering.

The most important property tested here is the absence of lookahead bias:
no feature computed at row T should use information from row T+1 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def price_df() -> pd.DataFrame:
    """200 rows of synthetic OHLCV data for a single ticker."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    open_ = close * (1 + np.random.randn(n) * 0.002)
    high = np.maximum(close, open_) * (1 + np.abs(np.random.randn(n) * 0.003))
    low = np.minimum(close, open_) * (1 - np.abs(np.random.randn(n) * 0.003))
    volume = np.random.randint(500_000, 2_000_000, n)
    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture()
def panel_df(price_df) -> pd.DataFrame:
    """Multi-ticker panel with 3 tickers."""
    frames = []
    for ticker in ["AAA", "BBB", "CCC"]:
        t = price_df.copy()
        t["ticker"] = ticker
        t["close"] = t["close"] * (1 + np.random.randn(len(t)) * 0.001)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Technical indicator tests
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:

    def test_log_ret_1d_matches_manual(self, price_df):
        from src.features.technical import add_return_features
        df = add_return_features(price_df)
        expected = np.log(price_df["close"] / price_df["close"].shift(1))
        pd.testing.assert_series_equal(
            df["log_ret_1d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_return_features_present(self, price_df):
        from src.features.technical import add_return_features
        df = add_return_features(price_df)
        for w in [1, 5, 10, 20]:
            assert f"log_ret_{w}d" in df.columns, f"Missing log_ret_{w}d"

    def test_rsi_bounded(self, price_df):
        from src.features.technical import add_rsi
        df = add_rsi(price_df)
        rsi_col = [c for c in df.columns if "rsi" in c][0]
        rsi = df[rsi_col].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI outside [0, 100]"

    def test_bollinger_band_upper_above_lower(self, price_df):
        from src.features.technical import add_bollinger_bands
        df = add_bollinger_bands(price_df)
        valid = df.dropna(subset=["bb_lower", "bb_upper"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_atr_non_negative(self, price_df):
        from src.features.technical import add_atr
        df = add_atr(price_df)
        atr_col = [c for c in df.columns if "atr" in c][0]
        assert (df[atr_col].dropna() >= 0).all()

    def test_vol_ratio_positive(self, price_df):
        from src.features.technical import add_volume_features
        df = add_volume_features(price_df)
        assert (df["vol_ratio"].dropna() > 0).all()

    def test_compute_all_indicators_columns(self, price_df):
        from src.features.technical import compute_all_indicators
        df = compute_all_indicators(price_df)
        expected_cols = [
            "log_ret_1d", "log_ret_5d", "log_ret_10d", "log_ret_20d",
            "rsi_14", "macd", "macd_signal",
            "bb_width", "vol_ratio",
            "close_to_sma50", "close_to_sma200",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Expected column '{col}' not found"


# ---------------------------------------------------------------------------
# Lookahead bias tests — the most important correctness tests
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_features_dont_use_shift_minus_1(self, price_df):
        """
        Verify that features at row T do not depend on row T+1 close.

        Method: set the last row's close to an extreme outlier and verify
        that no feature in the second-to-last row reflects that outlier.
        """
        from src.features.technical import compute_all_indicators

        df_modified = price_df.copy()
        df_original = price_df.copy()

        # Inject a 10x spike in the last row's close
        df_modified.iloc[-1, df_modified.columns.get_loc("close")] *= 10

        feat_mod = compute_all_indicators(df_modified)
        feat_orig = compute_all_indicators(df_original)

        # The second-to-last row's features should be identical in both
        # (the future spike should not affect them)
        feature_cols = [
            c for c in feat_orig.columns
            if c not in {"date", "ticker", "open", "high", "low", "close", "volume"}
        ]
        row_idx = len(feat_orig) - 2  # second-to-last row

        for col in feature_cols:
            v_orig = feat_orig.iloc[row_idx][col]
            v_mod = feat_mod.iloc[row_idx][col]
            if pd.isna(v_orig) and pd.isna(v_mod):
                continue
            assert v_orig == pytest.approx(v_mod, rel=1e-6), (
                f"Feature '{col}' at T-1 changed when we modified T's close. "
                f"Lookahead bias detected! orig={v_orig}, modified={v_mod}"
            )

    def test_target_uses_next_close(self, price_df):
        """Target at row T should equal log(close[T+1] / close[T])."""
        from src.features.build import add_target
        df = add_target(price_df)
        for i in range(len(df) - 1):
            expected = np.log(df["close"].iloc[i + 1] / df["close"].iloc[i])
            actual = df["target_ret"].iloc[i]
            assert actual == pytest.approx(expected, rel=1e-9), (
                f"Target at row {i} does not match log(close[{i+1}]/close[{i}])"
            )

    def test_momentum_features_lagged(self, price_df):
        """Momentum features must be lagged by 1 day to avoid using today's close."""
        from src.features.technical import add_momentum_features
        df = add_momentum_features(price_df)
        # mom_63d at row T = log(close[T-1] / close[T-64])
        # i.e. close.shift(1) / close.shift(64)
        if "mom_63d" in df.columns:
            for i in range(65, min(len(df), 80)):
                expected = np.log(df["close"].iloc[i - 1] / df["close"].iloc[i - 64])
                actual = df["mom_63d"].iloc[i]
                assert actual == pytest.approx(expected, rel=1e-9), (
                    f"mom_63d at row {i} is not correctly lagged"
                )


# ---------------------------------------------------------------------------
# Feature matrix builder tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_output_has_target_columns(self, panel_df):
        from src.features.build import build_feature_matrix
        result = build_feature_matrix(panel_df, save=False)
        assert "target_binary" in result.columns
        assert "target_ret" in result.columns

    def test_target_binary_is_0_or_1(self, panel_df):
        from src.features.build import build_feature_matrix
        result = build_feature_matrix(panel_df, save=False)
        values = result["target_binary"].unique()
        assert set(values).issubset({0, 1}), f"Unexpected target values: {values}"

    def test_no_nan_in_feature_cols(self, panel_df):
        from src.features.build import build_feature_matrix, get_feature_columns
        result = build_feature_matrix(panel_df, save=False)
        feat_cols = get_feature_columns(result)
        nan_counts = result[feat_cols].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert cols_with_nan.empty, (
            f"NaN values remain in feature columns after build: {cols_with_nan.to_dict()}"
        )

    def test_crosssectional_rank_bounded(self, panel_df):
        from src.features.build import build_feature_matrix
        result = build_feature_matrix(panel_df, save=False)
        rank_cols = [c for c in result.columns if c.startswith("xsrank_")]
        for col in rank_cols:
            vals = result[col].dropna()
            assert ((vals >= 0) & (vals <= 1)).all(), (
                f"Cross-sectional rank '{col}' outside [0, 1]"
            )
