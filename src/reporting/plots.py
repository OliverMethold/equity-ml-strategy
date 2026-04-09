"""
Reporting and visualisation. Generates interactive Plotly charts and HTML report.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_all_metrics,
    metrics_to_dataframe,
)

logger = logging.getLogger(__name__)

STRATEGY_COLOUR = "#2563EB"
BENCHMARK_COLOUR = "#6B7280"
DRAWDOWN_COLOUR = "#DC2626"


def plot_equity_curve(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series | None = None,
) -> go.Figure:
    fig = go.Figure()
    norm_strat = strategy_curve / strategy_curve.iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=norm_strat.index, y=norm_strat.values,
        name="Strategy", line=dict(color=STRATEGY_COLOUR, width=2),
    ))

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        norm_bm = benchmark_curve / benchmark_curve.iloc[0] * 100
        common = norm_strat.index.intersection(norm_bm.index)
        if len(common) > 0:
            norm_bm = norm_bm.loc[common]
            fig.add_trace(go.Scatter(
                x=norm_bm.index, y=norm_bm.values,
                name="SPY (buy-and-hold)",
                line=dict(color=BENCHMARK_COLOUR, width=2, dash="dash"),
            ))

    fig.update_layout(
        title="Portfolio Equity Curve (rebased to 100)",
        xaxis_title="Date", yaxis_title="Portfolio Value (rebased)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
    )
    return fig


def plot_drawdown(equity_curve: pd.Series) -> go.Figure:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="Drawdown", fill="tozeroy",
        line=dict(color=DRAWDOWN_COLOUR, width=1),
        fillcolor="rgba(220, 38, 38, 0.15)",
    ))
    fig.update_layout(
        title="Strategy Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", ticksuffix="%"),
    )
    return fig


def plot_rolling_sharpe(daily_returns: pd.Series, window: int = 60) -> go.Figure:
    roll_mean = daily_returns.rolling(window).mean()
    roll_std = daily_returns.rolling(window).std()
    rolling_sharpe = roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1, dash="dash"))
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        name=f"Rolling {window}d Sharpe",
        line=dict(color=STRATEGY_COLOUR, width=1.5),
    ))
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title="Date", yaxis_title="Sharpe Ratio (annualised)",
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
    )
    return fig


def plot_feature_importance(model, feature_cols: list[str], top_n: int = 20) -> go.Figure:
    if not hasattr(model, "feature_importances_"):
        fig = go.Figure()
        fig.update_layout(title="Feature Importance (not available for this model type)")
        return fig

    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        feature_cols = feature_cols[:len(importances)]

    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    imp_series = imp_series.tail(top_n)

    fig = go.Figure(go.Bar(
        x=imp_series.values, y=imp_series.index,
        orientation="h", marker_color=STRATEGY_COLOUR,
    ))
    fig.update_layout(
        title=f"Feature Importance (top {top_n})",
        xaxis_title="Importance Score", yaxis_title="Feature",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        height=max(440, top_n * 22),
    )
    return fig


def plot_monthly_returns_heatmap(daily_returns: pd.Series) -> go.Figure:
    monthly = (
        daily_returns
        .resample("ME")
        .apply(lambda r: (1 + r).prod() - 1)
    )
    df = monthly.reset_index()
    df.columns = ["date", "ret"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")

    pivot = df.pivot_table(index="year", columns="month", values="ret")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    z = pivot.values * 100
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        text=text, texttemplate="%{text}",
        colorscale=[[0.0, "#DC2626"], [0.5, "#F9FAFB"], [1.0, "#16A34A"]],
        zmid=0, colorbar=dict(title="Return %"),
    ))
    fig.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month", yaxis_title="Year",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def generate_html_report(
    strategy_curve: pd.Series,
    daily_returns: pd.Series,
    benchmark_curve: pd.Series | None,
    benchmark_returns: pd.Series | None,
    weights: pd.DataFrame,
    model=None,
    feature_cols: list[str] | None = None,
    fold_metrics: list[dict] | None = None,
) -> str:
    metrics = compute_all_metrics(
        equity_curve=strategy_curve,
        daily_returns=daily_returns,
        weights=weights,
        benchmark_returns=benchmark_returns,
    )
    metrics_df = metrics_to_dataframe(metrics)

    figs = {
        "equity": plot_equity_curve(strategy_curve, benchmark_curve),
        "drawdown": plot_drawdown(strategy_curve),
        "rolling_sharpe": plot_rolling_sharpe(daily_returns),
        "monthly": plot_monthly_returns_heatmap(daily_returns),
    }
    if model is not None and feature_cols is not None:
        figs["feature_importance"] = plot_feature_importance(model, feature_cols)

    from plotly.io import to_html
    chart_htmls = {k: to_html(fig, include_plotlyjs=False, full_html=False) for k, fig in figs.items()}

    metrics_table_rows = "".join(
        f"<tr><td>{row['Metric']}</td><td>{row['Value']}</td></tr>"
        for _, row in metrics_df.iterrows()
    )

    fold_table_html = ""
    if fold_metrics:
        fold_rows = "".join(
            f"<tr>"
            f"<td>{m.get('fold', i+1)}</td>"
            f"<td>{m.get('train_end', '')}</td>"
            f"<td>{m.get('test_start', '')} to {m.get('test_end', '')}</td>"
            f"<td>{m.get('accuracy', float('nan')):.3f}</td>"
            f"<td>{m.get('roc_auc', float('nan')):.3f}</td>"
            f"<td>{m.get('log_loss', float('nan')):.4f}</td>"
            f"</tr>"
            for i, m in enumerate(fold_metrics)
        )
        fold_table_html = f"""
        <h2>Walk-Forward Fold Performance</h2>
        <table>
          <thead>
            <tr><th>Fold</th><th>Train End</th><th>Test Window</th>
                <th>Accuracy</th><th>ROC AUC</th><th>Log Loss</th></tr>
          </thead>
          <tbody>{fold_rows}</tbody>
        </table>
        """

    feature_imp_html = chart_htmls.get("feature_importance", "")
    run_date = datetime.date.today().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Equity ML Strategy — Backtest Report ({run_date})</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #F9FAFB; color: #111827; padding: 24px; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 4px; color: #111827; }}
    h2 {{ font-size: 1.2rem; font-weight: 600; margin: 32px 0 12px; color: #1F2937; }}
    .subtitle {{ color: #6B7280; margin-bottom: 32px; font-size: 0.9rem; }}
    .disclaimer {{ background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px;
      padding: 16px; margin-bottom: 32px; font-size: 0.875rem; color: #92400E; }}
    .disclaimer strong {{ display: block; margin-bottom: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
      border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th, td {{ padding: 10px 16px; text-align: left; font-size: 0.875rem; }}
    th {{ background: #F3F4F6; font-weight: 600; color: #374151; }}
    tr:nth-child(even) {{ background: #F9FAFB; }}
    td:last-child {{ font-weight: 500; }}
    .chart-container {{ background: white; border-radius: 8px; padding: 16px;
                        margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Equity ML Strategy — Backtest Report</h1>
  <p class="subtitle">Generated: {run_date} &nbsp;|&nbsp;
     Strategy: XGBoost classification, long-only, 5-day horizon, hysteresis signals</p>

  <div class="disclaimer">
    <strong>Research purposes only. Not financial advice.</strong>
    This backtest uses the <em>current</em> S&amp;P 500 constituent list applied retroactively,
    which introduces survivorship bias and overstates historical returns. The strategy has
    not been traded with real capital. Transaction costs of 5 bps per trade and 1 bp slippage
    are modelled. Past backtest performance does not predict future returns.
  </div>

  <h2>Performance Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{metrics_table_rows}</tbody>
  </table>

  <h2>Equity Curve</h2>
  <div class="chart-container">{chart_htmls['equity']}</div>

  <div class="chart-grid">
    <div class="chart-container">{chart_htmls['drawdown']}</div>
    <div class="chart-container">{chart_htmls['rolling_sharpe']}</div>
  </div>

  <h2>Monthly Returns</h2>
  <div class="chart-container">{chart_htmls['monthly']}</div>

  {"<h2>Feature Importance</h2><div class=\"chart-container\">" + feature_imp_html + "</div>" if feature_imp_html else ""}

  {fold_table_html}

</body>
</html>"""

    return html


def save_html_report(html: str) -> str:
    from src.config import REPORTS_DIR
    run_date = datetime.date.today().strftime("%Y%m%d")
    path = REPORTS_DIR / f"backtest_{run_date}.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Report saved to %s", path)
    return str(path)
