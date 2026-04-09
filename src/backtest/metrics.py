"""
Performance metrics for the backtest.

Risk-free rate assumed to be 0%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series) -> float:
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    n_years = (end - start).days / 365.25
    if n_years <= 0:
        return float("nan")
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1)


def annualised_sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    return float(dd.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    roll_max = equity_curve.cummax()
    in_dd = (equity_curve < roll_max)
    if not in_dd.any():
        return 0
    dd_start = None
    max_dur = 0
    for date, is_dd in in_dd.items():
        if is_dd and dd_start is None:
            dd_start = date
        elif not is_dd and dd_start is not None:
            dur = (date - dd_start).days
            max_dur = max(max_dur, dur)
            dd_start = None
    if dd_start is not None:
        dur = (equity_curve.index[-1] - dd_start).days
        max_dur = max(max_dur, dur)
    return max_dur


def hit_rate(daily_returns: pd.Series) -> float:
    nonzero = daily_returns[daily_returns != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def avg_win_loss(daily_returns: pd.Series) -> tuple[float, float]:
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else float("nan")
    avg_loss = float(losses.mean()) if len(losses) > 0 else float("nan")
    return avg_win, avg_loss


def annualised_turnover(weights: pd.DataFrame) -> float:
    """Annualised one-way turnover based on position entries and exits."""
    positions = (weights > 0).astype(float)
    # Only count days where something actually changes
    changes = positions.diff().abs().sum(axis=1)
    avg_positions = positions.sum(axis=1).mean()
    if avg_positions == 0:
        return 0.0
    return (changes.mean() * 252) / max(avg_positions, 1)


def compute_all_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    weights: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    avg_win, avg_loss = avg_win_loss(daily_returns)
    metrics = {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve),
        "sharpe": annualised_sharpe(daily_returns),
        "sortino": sortino_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "max_drawdown_duration_days": max_drawdown_duration(equity_curve),
        "hit_rate": hit_rate(daily_returns),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "annualised_turnover": annualised_turnover(weights) if weights is not None else float("nan"),
    }
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm_eq = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
        if len(bm_eq) > 0:
            metrics["benchmark_total_return"] = total_return(bm_eq)
            metrics["benchmark_cagr"] = cagr(bm_eq)
            metrics["benchmark_sharpe"] = annualised_sharpe(benchmark_returns)
            metrics["benchmark_max_drawdown"] = max_drawdown(bm_eq)
    return metrics


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    rows = []
    fmt_map = {
        "total_return": ("{:.1%}", "Total Return"),
        "cagr": ("{:.1%}", "CAGR"),
        "sharpe": ("{:.2f}", "Sharpe Ratio"),
        "sortino": ("{:.2f}", "Sortino Ratio"),
        "max_drawdown": ("{:.1%}", "Max Drawdown"),
        "max_drawdown_duration_days": ("{:.0f} days", "Max DD Duration"),
        "hit_rate": ("{:.1%}", "Hit Rate (% up days)"),
        "avg_win": ("{:.3%}", "Avg Winning Day"),
        "avg_loss": ("{:.3%}", "Avg Losing Day"),
        "annualised_turnover": ("{:.0%}", "Annualised Turnover"),
        "benchmark_total_return": ("{:.1%}", "Benchmark Total Return"),
        "benchmark_cagr": ("{:.1%}", "Benchmark CAGR"),
        "benchmark_sharpe": ("{:.2f}", "Benchmark Sharpe"),
        "benchmark_max_drawdown": ("{:.1%}", "Benchmark Max Drawdown"),
    }
    for key, (fmt, label) in fmt_map.items():
        if key in metrics:
            val = metrics[key]
            try:
                formatted = fmt.format(val)
            except (ValueError, TypeError):
                formatted = str(val)
            rows.append({"Metric": label, "Value": formatted})
    return pd.DataFrame(rows)
