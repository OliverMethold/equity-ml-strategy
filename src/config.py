"""
Central configuration for the equity-ml-strategy pipeline.
Edit this file to change universes, date ranges, or model parameters.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_DIR = DATA_DIR / "universe"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

for _dir in [RAW_DIR, UNIVERSE_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
TRAIN_START = "2010-01-01"
TRAIN_END = None  # None = use all available data

# ---------------------------------------------------------------------------
# Universe
# NOTE: Survivorship bias applies — uses current S&P 500 constituents
# retroactively. For research purposes only.
# ---------------------------------------------------------------------------
LIQUID_UNIVERSE_30 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    "META", "TSLA", "BRK-B", "UNH", "JPM",
    "V", "XOM", "JNJ", "WMT", "PG",
    "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "COST", "BAC", "LLY",
    "AVGO", "TMO", "MCD", "CSCO", "ACN",
]

BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Universe expansion — set MAX_UNIVERSE_SIZE to cap auto-discovery
# ---------------------------------------------------------------------------
MAX_UNIVERSE_SIZE = 200   # cap when using load_from_csv auto-discovery
MIN_PRICE = 5.0           # exclude penny stocks
MIN_HISTORY_DAYS = 252    # require at least 1 year of history
MIN_AVG_VOLUME = 500_000  # minimum average daily volume

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
RETURN_WINDOWS = [1, 5, 10, 20]
MOM_WINDOWS = [63, 126]
VOL_WINDOW = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
ATR_PERIOD = 14
VOL_MA_WINDOW = 20
SMA_SHORT = 50
SMA_LONG = 200

# ---------------------------------------------------------------------------
# Prediction target
# ROADMAP CHANGE: 5-day forward return instead of 1-day
# Reduces noise by ~sqrt(5), naturally reduces turnover
# ---------------------------------------------------------------------------
PREDICTION_HORIZON = 5    # days forward to predict
EMBARGO_DAYS = 5          # gap between train end and test start (= horizon)

# ---------------------------------------------------------------------------
# Walk-forward validation
# ROADMAP CHANGE: rolling 2-year window instead of expanding from 2010
# ---------------------------------------------------------------------------
WALK_FORWARD_TRAIN_YEARS = 3
WALK_FORWARD_STEP_YEARS = 1
ROLLING_WINDOW = True     # True = rolling 3yr window, False = expanding

# ---------------------------------------------------------------------------
# Signal thresholds
# ROADMAP CHANGE: hysteresis band to cut turnover
# Entry at 0.55, exit at 0.45 — prevents churning on borderline signals
# ---------------------------------------------------------------------------
LONG_THRESHOLD = 0.55     # enter long when prob_up >= this
EXIT_THRESHOLD = 0.45     # exit long when prob_up < this (hysteresis)
MAX_POSITIONS = 25        # cap simultaneous longs; rank by prob_up

# ---------------------------------------------------------------------------
# Backtest assumptions
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 1
INITIAL_CAPITAL = 100_000

# ---------------------------------------------------------------------------
# Model defaults
# ROADMAP CHANGE: shallower trees, lower learning rate, stronger regularisation
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,           # shallower = less overfit
    "learning_rate": 0.03,    # slower learning with more trees
    "subsample": 0.75,        # row sampling per tree
    "colsample_bytree": 0.70, # feature sampling per tree
    "min_child_weight": 8,    # stronger leaf regularisation
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}
