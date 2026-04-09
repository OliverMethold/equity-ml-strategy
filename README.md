# Equity ML Strategy

A machine learning pipeline that ingests historical US equity data, trains a tree-based classification model to predict next-day price direction, backtests the resulting strategy, and produces a signal for the next trading session.

Built as a research project to demonstrate sound methodology. It has not been traded with real capital.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/equity-ml-strategy.git
cd equity-ml-strategy

uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

python scripts/run_full_pipeline.py
```

The pipeline runs end-to-end in roughly 5-10 minutes on the default 30-ticker universe and writes an interactive HTML report to `reports/backtest_YYYYMMDD.html`.

To generate today's signal after the model is trained:

```bash
python scripts/run_live_signal.py
```

---

## What this does

The pipeline has six stages:

1. **Universe selection.** Defaults to 30 liquid S&P 500 constituents defined in `src/config.py`. Full S&P 500 is available via `UNIVERSE=full python scripts/run_full_pipeline.py`.

2. **Data download.** Uses `yfinance` with `auto_adjust=True` to pull split and dividend-adjusted OHLCV data from 2010 to the present. Data is cached as parquet files in `data/raw/` so subsequent runs are fast.

3. **Feature engineering.** Computes ~20 technical features per ticker per day from OHLCV only. All features are computed strictly from data available at the close of day T — no lookahead. A cross-sectional rank feature (each ticker's 20-day return ranked within the universe) gives the model a relative signal.

4. **Walk-forward training.** Trains an XGBoost classifier using an expanding window: train on years 1-3, predict year 4; retrain on years 1-4, predict year 5; and so on. A StandardScaler is fit on the training window only and applied to the test window. This prevents data leakage.

5. **Backtest.** Simulates a long-only daily-rebalanced portfolio. Signals enter at the next day's open, not the close observed when the signal was generated. Transaction costs of 5 basis points per trade and 1 basis point slippage are applied. Results are compared against SPY buy-and-hold.

6. **Reporting.** Generates an interactive HTML report with equity curve, drawdown chart, rolling Sharpe, monthly returns heatmap, feature importance, and a full metrics table.

---

## What to realistically expect

A well-implemented version of this pipeline typically produces a Sharpe ratio between 0.3 and 0.8 before costs, and often underperforms SPY on a risk-adjusted basis once transaction costs are included. That is the expected result and is the more interesting finding. If the backtest shows a Sharpe above 2, treat it as a bug to investigate, not a win to celebrate.

The project's value is demonstrating sound methodology, not printing money.

---

## Known limitations (important)

### Survivorship bias

This is the most significant limitation. The universe is built from the *current* S&P 500 constituent list, applied retroactively to 2010. This means the model only trains and tests on companies that survived and remained index constituents through to today. Businesses that went bankrupt, were acquired at a loss, or were removed from the index are excluded.

This effect typically inflates backtest returns by 1-3% per year. For a production system, a point-in-time constituent list (e.g. Norgate Data) would be required.

### No live trading

The `run_live_signal.py` script pulls fresh market data (delayed ~15 minutes via yfinance) and produces a signal. It does not connect to a broker or place trades. The "live" element is data freshness only.

### Risk-free rate assumed to be zero

All Sharpe and Sortino calculations use a risk-free rate of 0%. This overstates risk-adjusted performance in high-rate environments (e.g. 2023-2024 when cash earned 5%+).

### Transaction cost model is simplified

The 5 bps per-trade cost is a reasonable retail assumption for liquid large-cap equities but does not account for market impact, short-selling costs, or borrowing fees.

### Only OHLCV features

Version 1 uses no fundamental data. A strategy with access to earnings, balance sheet ratios, or analyst revisions would have more signal and better robustness.

### Single model type

XGBoost is the main model. No attempt has been made to combine models, apply regime detection, or use position sizing proportional to confidence.

---

## Project structure

```
equity-ml-strategy/
├── src/
│   ├── config.py               central parameters: tickers, dates, model settings
│   ├── data/
│   │   ├── loader.py           yfinance wrapper with parquet caching
│   │   └── universe.py         S&P 500 scraper and hard-coded 30-ticker list
│   ├── features/
│   │   ├── technical.py        pandas_ta indicator wrappers
│   │   └── build.py            full feature matrix builder with target construction
│   ├── models/
│   │   ├── train.py            walk-forward training loop
│   │   ├── predict.py          signal generation from trained model
│   │   └── registry.py         save/load models as joblib files
│   ├── backtest/
│   │   ├── engine.py           vectorbt-based backtest with pandas fallback
│   │   └── metrics.py          Sharpe, Sortino, max drawdown, CAGR, hit rate, etc.
│   └── reporting/
│       └── plots.py            Plotly charts and HTML report generation
├── scripts/
│   ├── run_full_pipeline.py    end-to-end orchestrator
│   └── run_live_signal.py      fetch latest data, load model, output signal CSV
├── notebooks/
│   ├── 01_eda.ipynb            exploratory data analysis
│   └── 02_feature_analysis.ipynb feature distributions and target correlations
└── tests/
    ├── test_loader.py           data loader with mocked yfinance
    ├── test_features.py         indicator correctness and lookahead bias checks
    └── test_backtest.py         metrics and backtest engine
```

---

## Features used

| Category | Features |
|---|---|
| Returns | Log returns at 1, 5, 10, 20 days |
| Momentum | 3-month and 6-month returns, lagged 1 day |
| Volatility | Rolling 20-day standard deviation of returns |
| Momentum oscillator | RSI (14) |
| Trend | MACD, MACD signal line, MACD histogram |
| Mean reversion | Bollinger band width and %B (20-day, 2 std) |
| Range | ATR (14) |
| Volume | Volume ratio vs 20-day average |
| Trend following | Distance from 50-day and 200-day SMA |
| Cross-sectional | Rank by 5-day and 20-day return within universe |

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite does not require network access. yfinance is mocked in `test_loader.py`. The lookahead bias test in `test_features.py` injects a synthetic price spike into the last row and verifies that no feature in the previous row changes value.

---

## Configuration

Edit `src/config.py` to change:
- Ticker universe (`LIQUID_UNIVERSE_30`)
- Date range (`TRAIN_START`)
- Signal thresholds (`LONG_THRESHOLD`, `FLAT_LOWER`)
- Transaction cost assumptions (`TRANSACTION_COST_BPS`, `SLIPPAGE_BPS`)
- Walk-forward window sizes (`WALK_FORWARD_TRAIN_YEARS`, `WALK_FORWARD_STEP_YEARS`)
- XGBoost hyperparameters (`XGBOOST_PARAMS`)

---

## Potential next steps

- Point-in-time S&P 500 constituent list (Norgate Data) to eliminate survivorship bias
- Fundamental features from SimFin (P/E, revenue growth, debt ratios)
- Long-short market-neutral construction
- Regime detection (high vs low volatility) with separate models per regime
- Position sizing by model confidence (Kelly criterion capped at 25% full Kelly)
- LSTM or transformer comparison (usually does not outperform tree models on daily equity data)
- Paper trading via Alpaca API

---

## Disclaimer

This is a research project. It has not been traded with real capital. Nothing in this repository constitutes financial or investment advice. Past backtest performance does not predict future returns.
