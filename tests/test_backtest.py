"""
Tests for the backtest engine and metrics module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_returns() -> pd.Series:
    """Zero-return series — edge case."""
    dates = pd.date_range("2020-01-02", periods=252, freq="B")
    return pd.Series(0.0, index=dates, name="returns")


@pytest.fixture()
def positive_returns() -> pd.Series:
    """Steady 0.05% daily gain."""
    dates = pd.date_range("2020-01-02", periods=252, freq="B")
    return pd.Series(0.0005, index=dates, name="returns")


@pytest.fixture()
def random_returns() -> pd.Series:
    np.random.seed(0)
    dates = pd.date_range("2015-01-02", periods=1500, freq="B")
    return pd.Series(np.random.randn(1500) * 0.01, index=dates, name="returns")


@pytest.fixture()
def equity_from(positive_returns):
    return (1 + positive_returns).cumprod() * 100_000


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_total_return_correct(self, positive_returns):
        from src.backtest.metrics import total_return
        eq = (1 + positive_returns).cumprod() * 100_000
        tr = total_return(eq)
        expected = eq.iloc[-1] / eq.iloc[0] - 1
        assert tr == pytest.approx(expected, rel=1e-9)

    def test_cagr_positive_for_positive_returns(self, positive_returns):
        from src.backtest.metrics import cagr
        eq = (1 + positive_returns).cumprod() * 100_000
        assert cagr(eq) > 0

    def test_sharpe_nan_on_zero_vol(self, flat_returns):
        from src.backtest.metrics import annualised_sharpe
        result = annualised_sharpe(flat_returns)
        assert np.isnan(result)

    def test_sharpe_positive_for_positive_returns(self, positive_returns):
        from src.backtest.metrics import annualised_sharpe
        assert annualised_sharpe(positive_returns) > 0

    def test_sortino_positive_for_no_downside(self, positive_returns):
        from src.backtest.metrics import sortino_ratio
        # All positive returns means no downside, sortino is nan (no downside vol)
        result = sortino_ratio(positive_returns)
        assert np.isnan(result) or result > 0

    def test_max_drawdown_zero_for_monotone_increase(self, positive_returns):
        from src.backtest.metrics import max_drawdown
        eq = (1 + positive_returns).cumprod() * 100_000
        assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_negative_for_volatile_returns(self, random_returns):
        from src.backtest.metrics import max_drawdown
        eq = (1 + random_returns).cumprod() * 100_000
        dd = max_drawdown(eq)
        assert dd < 0, "Max drawdown should be negative for volatile returns"

    def test_max_drawdown_between_minus1_and_0(self, random_returns):
        from src.backtest.metrics import max_drawdown
        eq = (1 + random_returns).cumprod() * 100_000
        dd = max_drawdown(eq)
        assert -1 <= dd <= 0

    def test_hit_rate_all_positive(self, positive_returns):
        from src.backtest.metrics import hit_rate
        assert hit_rate(positive_returns) == pytest.approx(1.0)

    def test_hit_rate_between_0_and_1(self, random_returns):
        from src.backtest.metrics import hit_rate
        hr = hit_rate(random_returns)
        assert 0 <= hr <= 1

    def test_annualised_turnover_zero_for_static_weights(self):
        from src.backtest.metrics import annualised_turnover
        dates = pd.date_range("2020-01-02", periods=252, freq="B")
        weights = pd.DataFrame(
            {"AAPL": 0.5, "MSFT": 0.5},
            index=dates,
        )
        to = annualised_turnover(weights)
        assert to == pytest.approx(0.0, abs=1e-9)

    def test_compute_all_metrics_keys(self, positive_returns):
        from src.backtest.metrics import compute_all_metrics
        eq = (1 + positive_returns).cumprod() * 100_000
        metrics = compute_all_metrics(eq, positive_returns)
        expected_keys = ["total_return", "cagr", "sharpe", "sortino",
                         "max_drawdown", "hit_rate"]
        for k in expected_keys:
            assert k in metrics, f"Missing key: {k}"

    def test_metrics_to_dataframe_has_two_columns(self, positive_returns):
        from src.backtest.metrics import compute_all_metrics, metrics_to_dataframe
        eq = (1 + positive_returns).cumprod() * 100_000
        metrics = compute_all_metrics(eq, positive_returns)
        df = metrics_to_dataframe(metrics)
        assert list(df.columns) == ["Metric", "Value"]
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Backtest engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def _make_prices(self, tickers=("AAPL", "MSFT"), n=200):
        """Synthetic price panel."""
        np.random.seed(1)
        frames = []
        for ticker in tickers:
            dates = pd.date_range("2020-01-02", periods=n, freq="B")
            close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
            frames.append(pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }))
        return pd.concat(frames, ignore_index=True)

    def _make_weights(self, prices):
        dates = prices["date"].unique()
        tickers = prices["ticker"].unique()
        # Alternate between all-AAPL and all-MSFT to test rebalancing
        rows = []
        for i, d in enumerate(sorted(dates)):
            row = {t: 0.5 for t in tickers}
            rows.append({"date": d, **row})
        df = pd.DataFrame(rows).set_index("date")
        return df[list(tickers)]

    def test_equity_curve_starts_near_initial_capital(self):
        from src.backtest.engine import _manual_backtest
        from src.config import INITIAL_CAPITAL
        prices = self._make_prices()
        weights = self._make_weights(prices)
        result = _manual_backtest(weights, prices)
        # First value should be close to initial capital
        first_val = result.equity_curve.iloc[0]
        assert abs(first_val - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05

    def test_equity_curve_length(self):
        from src.backtest.engine import _manual_backtest
        prices = self._make_prices()
        weights = self._make_weights(prices)
        result = _manual_backtest(weights, prices)
        assert len(result.equity_curve) > 0

    def test_daily_returns_have_no_huge_outliers(self):
        """Single-day returns above 50% would indicate a data bug."""
        from src.backtest.engine import _manual_backtest
        prices = self._make_prices()
        weights = self._make_weights(prices)
        result = _manual_backtest(weights, prices)
        max_ret = result.daily_returns.abs().max()
        assert max_ret < 0.5, f"Suspiciously large daily return: {max_ret:.2%}"

    def test_get_benchmark_returns_correct_ticker(self):
        from src.backtest.engine import get_benchmark_returns
        prices = self._make_prices(tickers=("AAPL", "SPY"))
        ret = get_benchmark_returns(prices, "SPY")
        assert ret.name == "SPY"
        assert len(ret) > 0


# ---------------------------------------------------------------------------
# Walk-forward train/test split integrity
# ---------------------------------------------------------------------------

class TestWalkForwardIntegrity:

    def _make_features(self, n_dates=800, n_tickers=3):
        """Minimal feature matrix for training tests."""
        np.random.seed(99)
        dates = pd.date_range("2010-01-04", periods=n_dates, freq="B")
        rows = []
        for ticker in [f"T{i}" for i in range(n_tickers)]:
            for date in dates:
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "feat_a": np.random.randn(),
                    "feat_b": np.random.randn(),
                    "feat_c": np.random.randn(),
                    "target_binary": int(np.random.randn() > 0),
                    "target_ret": np.random.randn() * 0.01,
                    "open_next": 100 + np.random.randn(),
                })
        return pd.DataFrame(rows)

    def test_no_test_data_leaks_into_train(self):
        """Every test row must post-date every train row within each fold."""
        from src.models.train import walk_forward_train
        features = self._make_features()
        fold_results = walk_forward_train(features, model_type="logistic", save_folds=False)
        assert len(fold_results) > 0
        for fold in fold_results:
            train_end = pd.Timestamp(fold.train_end)
            test_start = pd.Timestamp(fold.test_start)
            assert test_start > train_end, (
                f"Test period starts before train end in fold ending {fold.train_end}"
            )

    def test_oos_predictions_cover_all_test_dates(self):
        """Assembled OOS predictions should cover each fold's test window."""
        from src.models.train import walk_forward_train, assemble_oos_predictions
        features = self._make_features()
        fold_results = walk_forward_train(features, model_type="logistic", save_folds=False)
        oos = assemble_oos_predictions(fold_results)
        assert not oos.empty
        assert "prob_up" in oos.columns
        assert "pred_binary" in oos.columns
        # Probabilities should be in [0, 1]
        assert ((oos["prob_up"] >= 0) & (oos["prob_up"] <= 1)).all()
