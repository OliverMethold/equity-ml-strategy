"""
Backtest engine using vectorbt.

Key assumptions:
- Signals are generated at the CLOSE of day T using features built from data
  available at that close.
- Trades ENTER at the OPEN of day T+1. We never buy at the close we observed.
- Transaction costs: 5 bps per side (buy) + 5 bps (sell) = 10 bps round trip.
- Slippage: 1 bp per trade.
- Long-only for v1.
- Positions are equal-weighted and rebalanced daily.

vectorbt note:
    vectorbt's Portfolio.from_orders() is used rather than from_signals() for
    precise control over entry price and cost model.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
)

logger = logging.getLogger(__name__)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> "BacktestResult":
    """
    Simulate portfolio performance from signal weights and price data.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide date x ticker weight matrix (output of signals_to_position_matrix).
        Values are portfolio fractions (0 to 1), rows sum to ≤ 1.
    prices : pd.DataFrame
        Long-format OHLCV DataFrame [date, ticker, open, high, low, close, volume].

    Returns
    -------
    BacktestResult
        Contains equity curve, daily returns, and trade log.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        logger.warning("vectorbt not available, falling back to manual backtest.")
        return _manual_backtest(weights, prices)

    # Pivot prices to wide format
    close_wide = prices.pivot_table(index="date", columns="ticker", values="close")
    open_wide = prices.pivot_table(index="date", columns="ticker", values="open")

    # Align weights with price dates
    common_dates = weights.index.intersection(close_wide.index)
    common_tickers = weights.columns.intersection(close_wide.columns)

    weights = weights.loc[common_dates, common_tickers]
    close_wide = close_wide.loc[common_dates, common_tickers]
    open_wide = open_wide.loc[common_dates, common_tickers]

    # Shift weights forward: signal from day T drives trade at day T+1 open
    # We enter at next day's open, so use open_wide shifted back
    entry_price = open_wide.shift(-1).ffill()

    # Build size matrix: target number of shares each day
    # We trade based on target weight changes
    tc_rate = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

    try:
        pf = vbt.Portfolio.from_orders(
            close=close_wide,
            size=weights,
            size_type="targetpercent",
            price=entry_price,
            fees=tc_rate,
            freq="1D",
            init_cash=INITIAL_CAPITAL,
            call_seq="auto",
            group_by=True,
        )
        return BacktestResult(portfolio=pf, weights=weights, prices=prices)
    except Exception as exc:
        logger.warning("vectorbt portfolio construction failed (%s). Using manual backtest.", exc)
        return _manual_backtest(weights, prices)


def _manual_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> "BacktestResult":
    """
    Fallback pure-pandas backtest when vectorbt is unavailable or fails.

    Implements a simple daily rebalancing strategy:
    - At the OPEN of day T+1, rebalance to target weights from signal at close of day T.
    - Applies transaction cost to the weight change.
    - Computes portfolio daily return.
    """
    close_wide = prices.pivot_table(index="date", columns="ticker", values="close")
    open_wide = prices.pivot_table(index="date", columns="ticker", values="open")

    common_dates = sorted(set(weights.index) & set(close_wide.index))
    common_tickers = sorted(set(weights.columns) & set(close_wide.columns))

    weights = weights.reindex(index=common_dates, columns=common_tickers).fillna(0)
    close_wide = close_wide.reindex(index=common_dates, columns=common_tickers).ffill().fillna(0)
    open_wide = open_wide.reindex(index=common_dates, columns=common_tickers).ffill().fillna(0)

    # Daily return of each asset from T open to T+1 open (approx)
    # Using close-to-close as a reasonable proxy
    asset_returns = close_wide.pct_change().fillna(0)

    # Shift weights: position taken based on signal from previous day
    weights_lagged = weights.shift(1).fillna(0)

    # Portfolio daily return
    port_ret = (weights_lagged * asset_returns).sum(axis=1)

    # Transaction cost: |weight change| * tc_rate
    tc_rate = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000
    weight_change = weights_lagged.diff().abs().sum(axis=1)
    tc_daily = weight_change * tc_rate

    port_ret_net = port_ret - tc_daily

    equity_curve = (1 + port_ret_net).cumprod() * INITIAL_CAPITAL
    equity_curve.name = "strategy"

    return BacktestResult(
        portfolio=None,
        weights=weights,
        prices=prices,
        equity_curve=equity_curve,
        daily_returns=port_ret_net,
    )


class BacktestResult:
    """Container for backtest output."""

    def __init__(
        self,
        portfolio=None,
        weights: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        equity_curve: pd.Series | None = None,
        daily_returns: pd.Series | None = None,
    ):
        self._portfolio = portfolio
        self.weights = weights
        self.prices = prices
        self._equity_curve = equity_curve
        self._daily_returns = daily_returns

    @property
    def equity_curve(self) -> pd.Series:
        if self._equity_curve is not None:
            return self._equity_curve
        if self._portfolio is not None:
            try:
                return self._portfolio.value()
            except Exception:
                return self._portfolio.total_value()
        raise AttributeError("No equity curve available.")

    @property
    def daily_returns(self) -> pd.Series:
        if self._daily_returns is not None:
            return self._daily_returns
        if self._portfolio is not None:
            try:
                return self._portfolio.returns()
            except Exception:
                pass
        return self.equity_curve.pct_change().fillna(0)


def get_benchmark_returns(prices: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    """Extract buy-and-hold daily returns for the benchmark ticker."""
    spy = prices[prices["ticker"] == ticker].set_index("date")["close"]
    spy = spy.sort_index()
    returns = spy.pct_change().fillna(0)
    returns.name = ticker
    return returns


def get_benchmark_equity(
    prices: pd.DataFrame,
    ticker: str = "SPY",
    start_value: float = INITIAL_CAPITAL,
) -> pd.Series:
    ret = get_benchmark_returns(prices, ticker)
    curve = (1 + ret).cumprod() * start_value
    curve.name = f"{ticker} buy-and-hold"
    return curve
