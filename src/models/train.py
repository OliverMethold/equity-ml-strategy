"""
Walk-forward model training.

ROADMAP CHANGES:
- Rolling training window (ROLLING_WINDOW=True) instead of always expanding from 2010
- Embargo gap (EMBARGO_DAYS) between train end and test start to prevent label overlap
  with 5-day prediction horizon
- Inner validation split for early stopping (last 15% of training rows)
- Scale class imbalance with scale_pos_weight
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    WALK_FORWARD_STEP_YEARS,
    WALK_FORWARD_TRAIN_YEARS,
    XGBOOST_PARAMS,
    ROLLING_WINDOW,
    EMBARGO_DAYS,
)
from src.features.build import get_feature_columns
from src.models.registry import save_model

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    train_end: str
    test_start: str
    test_end: str
    model: object
    scaler: StandardScaler
    feature_cols: list[str]
    predictions: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def _make_model(model_type: str = "xgboost", pos_weight: float = 1.0) -> object:
    if model_type == "logistic":
        return LogisticRegression(max_iter=500, random_state=42, C=0.1)
    elif model_type == "xgboost":
        params = {k: v for k, v in XGBOOST_PARAMS.items() if k != "use_label_encoder"}
        params["scale_pos_weight"] = pos_weight
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model_type: {model_type}")


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_prob)
    except ValueError:
        ll = float("nan")
    return {"accuracy": accuracy, "roc_auc": auc, "log_loss": ll}


def walk_forward_train(
    features: pd.DataFrame,
    model_type: str = "xgboost",
    save_folds: bool = True,
) -> list[FoldResult]:
    """
    Walk-forward training with rolling window and embargo gap.
    """
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])

    feature_cols = get_feature_columns(features)
    logger.info("Training with %d features.", len(feature_cols))

    min_date = features["date"].min()
    max_date = features["date"].max()

    first_test_start = min_date + pd.DateOffset(years=WALK_FORWARD_TRAIN_YEARS)
    fold_starts = pd.date_range(
        start=first_test_start,
        end=max_date,
        freq=f"{WALK_FORWARD_STEP_YEARS}YS",
    )

    if len(fold_starts) == 0:
        fold_starts = [max_date - pd.DateOffset(years=1)]

    results: list[FoldResult] = []

    for fold_idx, test_start in enumerate(fold_starts):
        test_end = min(test_start + pd.DateOffset(years=WALK_FORWARD_STEP_YEARS), max_date)
        train_end = test_start - pd.Timedelta(days=1)

        # ROADMAP: embargo gap to prevent label overlap with 5-day horizon
        embargo_end = test_start + pd.Timedelta(days=EMBARGO_DAYS)
        actual_test_start = embargo_end

        # ROADMAP: rolling window — drop data older than WALK_FORWARD_TRAIN_YEARS
        if ROLLING_WINDOW:
            window_start = train_end - pd.DateOffset(years=WALK_FORWARD_TRAIN_YEARS)
            train_mask = (features["date"] >= window_start) & (features["date"] <= train_end)
        else:
            train_mask = features["date"] <= train_end

        test_mask = (features["date"] >= actual_test_start) & (features["date"] <= test_end)

        X_train_all = features.loc[train_mask, feature_cols].values
        y_train_all = features.loc[train_mask, "target_binary"].values
        X_test = features.loc[test_mask, feature_cols].values
        y_test = features.loc[test_mask, "target_binary"].values

        if len(X_train_all) < 200 or len(X_test) < 20:
            logger.warning(
                "Fold %d skipped: insufficient data (train=%d, test=%d).",
                fold_idx + 1, len(X_train_all), len(X_test),
            )
            continue

        # ROADMAP: inner validation split (last 15% of training rows)
        split_idx = int(len(X_train_all) * 0.85)
        X_train = X_train_all[:split_idx]
        y_train = y_train_all[:split_idx]
        X_val = X_train_all[split_idx:]
        y_val = y_train_all[split_idx:]

        # Class balance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model = _make_model(model_type, pos_weight=pos_weight)

        if model_type == "xgboost":
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics = _evaluate(y_test, y_prob)

        logger.info(
            "Fold %d | Train: %s to %s | Test: %s to %s | "
            "Acc: %.3f | AUC: %.3f | LogLoss: %.4f | N_train: %d | N_test: %d",
            fold_idx + 1,
            train_mask.index[train_mask][0] if hasattr(train_mask.index[train_mask], '__len__') else "?",
            train_end.date(),
            actual_test_start.date(),
            test_end.date(),
            metrics["accuracy"],
            metrics["roc_auc"],
            metrics["log_loss"],
            len(X_train),
            len(X_test),
        )

        if metrics["accuracy"] > 0.65:
            logger.warning(
                "Fold %d accuracy of %.1f%% is high — check for lookahead bias.",
                fold_idx + 1, metrics["accuracy"] * 100,
            )

        test_rows = features.loc[test_mask, ["date", "ticker"]].copy().reset_index(drop=True)
        test_rows["prob_up"] = y_prob
        test_rows["pred_binary"] = (y_prob >= 0.5).astype(int)

        train_end_str = train_end.strftime("%Y-%m-%d")
        if save_folds:
            save_model(
                model,
                window_end=train_end_str,
                meta={
                    "model_type": model_type,
                    "feature_cols": feature_cols,
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "metrics": metrics,
                    "train_end": train_end_str,
                    "test_start": actual_test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "rolling_window": ROLLING_WINDOW,
                    "embargo_days": EMBARGO_DAYS,
                },
            )

        results.append(
            FoldResult(
                train_end=train_end_str,
                test_start=actual_test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                predictions=test_rows,
                metrics=metrics,
            )
        )

    if not results:
        raise RuntimeError(
            "Walk-forward training produced no folds. "
            "Check that the feature matrix covers enough years."
        )

    return results


def assemble_oos_predictions(fold_results: list[FoldResult]) -> pd.DataFrame:
    oos = pd.concat([r.predictions for r in fold_results], ignore_index=True)
    oos = oos.sort_values(["date", "ticker"]).reset_index(drop=True)
    oos["date"] = pd.to_datetime(oos["date"])
    return oos
