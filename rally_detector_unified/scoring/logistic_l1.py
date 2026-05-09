"""
Logistic Regression L1 (Lasso) models — one per target.
Targets: 9 combinations of (threshold_pct, horizon_hours).
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from .feature_pipeline import build_sklearn_pipeline, build_feature_matrix
from ..config import TARGETS

logger = logging.getLogger(__name__)


def _target_name(threshold: int, horizon: int) -> str:
    return f"rally_{threshold}_{horizon}h"


@dataclass
class TrainedModel:
    target: str
    pipeline: Pipeline
    threshold: int
    horizon_hours: int
    n_train: int
    pos_rate: float
    feature_names: list[str] = field(default_factory=list)


def build_logistic_pipeline() -> Pipeline:
    base = build_sklearn_pipeline()
    base.steps.append((
        "model",
        LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="saga",
            max_iter=2000,
            scoring="average_precision",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    ))
    return base


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    threshold: int,
    horizon_hours: int,
) -> Optional[TrainedModel]:
    """Train a single model for one target. Returns None if not enough positives."""
    valid = y.notna()
    X_fit = X[valid]
    y_fit = y[valid]

    pos_rate = y_fit.mean()
    if len(y_fit) < 100 or pos_rate == 0 or pos_rate == 1:
        logger.warning(
            "Skipping %s: n=%d, pos_rate=%.4f", target_name, len(y_fit), pos_rate
        )
        return None

    pipeline = build_logistic_pipeline()
    pipeline.fit(X_fit.values, y_fit.values)

    logger.info(
        "Trained %s: n=%d, pos_rate=%.3f%%",
        target_name, len(y_fit), pos_rate * 100
    )
    return TrainedModel(
        target=target_name,
        pipeline=pipeline,
        threshold=threshold,
        horizon_hours=horizon_hours,
        n_train=len(y_fit),
        pos_rate=pos_rate,
    )


def train_all_models(
    df: pd.DataFrame,
) -> dict[str, TrainedModel]:
    """
    Train all 9 models on the full feature matrix.
    Expects df to have target columns pre-computed.
    """
    X = build_feature_matrix(df)
    models: dict[str, TrainedModel] = {}

    for threshold, horizon in TARGETS:
        col = _target_name(threshold, horizon)
        if col not in df.columns:
            logger.warning("Target column %s not in DataFrame. Skipping.", col)
            continue
        y = df[col]
        model = train_model(X, y, col, threshold, horizon)
        if model is not None:
            models[col] = model

    return models


def predict_proba_all(
    df: pd.DataFrame,
    models: dict[str, TrainedModel],
) -> pd.DataFrame:
    """
    Apply all trained models to df.
    Returns DataFrame with one probability column per target.
    """
    X = build_feature_matrix(df)
    proba_cols: dict[str, np.ndarray] = {}

    for target, model in models.items():
        try:
            proba = model.pipeline.predict_proba(X.values)[:, 1]
        except Exception as exc:
            logger.error("predict_proba failed for %s: %s", target, exc)
            proba = np.full(len(X), np.nan)
        proba_cols[f"proba_{target}"] = proba

    return pd.DataFrame(proba_cols, index=df.index)
