"""
Model registry: save and load trained models with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


def save_model(model, window_end: str, meta: dict | None = None) -> Path:
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : sklearn-compatible estimator
    window_end : str
        The last date of the training window (YYYY-MM-DD). Used as part of filename.
    meta : dict, optional
        Arbitrary metadata to store alongside the model (e.g. feature list, metrics).

    Returns
    -------
    Path
        Path to the saved .joblib file.
    """
    stem = f"model_{window_end.replace('-', '')}"
    model_path = MODELS_DIR / f"{stem}.joblib"
    meta_path = MODELS_DIR / f"{stem}.json"

    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    if meta is not None:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.debug("Saved model metadata to %s", meta_path)

    return model_path


def load_model(window_end: str):
    """Load a model by its training window end date."""
    stem = f"model_{window_end.replace('-', '')}"
    model_path = MODELS_DIR / f"{stem}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)


def load_latest_model():
    """Load the most recently trained model (by filename sort)."""
    paths = sorted(MODELS_DIR.glob("model_*.joblib"))
    if not paths:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_DIR}. "
            "Run the full pipeline first."
        )
    path = paths[-1]
    logger.info("Loading latest model: %s", path)
    return joblib.load(path), path.stem.replace("model_", "")


def load_model_meta(window_end: str) -> dict:
    """Load metadata for a model."""
    stem = f"model_{window_end.replace('-', '')}"
    meta_path = MODELS_DIR / f"{stem}.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def list_models() -> list[str]:
    """Return a list of available training window end dates, sorted."""
    paths = sorted(MODELS_DIR.glob("model_*.joblib"))
    return [p.stem.replace("model_", "") for p in paths]
