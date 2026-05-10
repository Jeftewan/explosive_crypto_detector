"""
Logistic Regression L1 (Lasso) models — one per target.

Optimizations vs the original implementation:
  * solver="liblinear" instead of saga (10-50× faster on medium data, 8 GB friendly)
  * Cs/cv reduced (4/3 instead of 10/5) — sub-fits per target drop from 50 to 12
  * max_iter=500, tol=1e-3 — saga over-iterated by default
  * Imputer + scaler fit ONCE per dataset and shared across all targets
    (instead of refitting the same preprocessing 9 times)
  * Optional frozen-C dict to skip C search across walk-forward folds
  * Stratified negative subsampling for very large train sets
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline

from .feature_pipeline import build_sklearn_pipeline, build_feature_matrix
from ..config import (
    TARGETS,
    LR_SOLVER,
    LR_CS_GRID,
    LR_CV_FOLDS,
    LR_MAX_ITER,
    LR_TOL,
    LR_N_JOBS,
    TRAIN_SUBSAMPLE_NEG_RATIO,
    TRAIN_SUBSAMPLE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _target_name(threshold: int, horizon: int) -> str:
    return f"rally_{threshold}_{horizon}h"


@dataclass
class TrainedModel:
    target: str
    pipeline: Pipeline           # full pipeline (preprocess + classifier) for predict
    threshold: int
    horizon_hours: int
    n_train: int
    pos_rate: float
    C: float = 1.0
    feature_names: list[str] = field(default_factory=list)


def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, neg_ratio: int, threshold: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positives, sample `neg_ratio` negatives per positive when len > threshold."""
    if len(y) <= threshold:
        return X, y
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_keep_neg = min(len(neg_idx), len(pos_idx) * neg_ratio)
    if n_keep_neg == 0 or len(pos_idx) == 0:
        return X, y
    sampled_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    keep.sort()
    return X[keep], y[keep]


def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[float],
) -> tuple[LogisticRegression, float]:
    """
    Fit a single L1 logistic regression. If C is None, run LogisticRegressionCV
    to pick C; otherwise fit a plain LogisticRegression with the given C.
    Returns (fitted_classifier, C_used).
    """
    if C is None:
        cv_clf = LogisticRegressionCV(
            Cs=LR_CS_GRID,
            cv=LR_CV_FOLDS,
            penalty="l1",
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            tol=LR_TOL,
            scoring="average_precision",
            class_weight="balanced",
            n_jobs=LR_N_JOBS,
            random_state=42,
        )
        cv_clf.fit(X, y)
        chosen_C = float(cv_clf.C_[0])
        # Guard against degenerate values returned by some sklearn versions
        # (e.g. when the search hits a numerically unstable grid edge).
        if not np.isfinite(chosen_C) or chosen_C <= 0:
            chosen_C = 1.0
        chosen_C = float(np.clip(chosen_C, 1e-4, 1e4))
        # Refit a regular LR with the chosen C so the returned object is lightweight.
        clf = LogisticRegression(
            C=chosen_C,
            penalty="l1",
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            tol=LR_TOL,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X, y)
        return clf, chosen_C
    clf = LogisticRegression(
        C=C,
        penalty="l1",
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        tol=LR_TOL,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    return clf, C


def train_all_models(
    df: pd.DataFrame,
    frozen_Cs: Optional[dict[str, float]] = None,
) -> tuple[dict[str, TrainedModel], dict[str, float]]:
    """
    Train all targets on `df`. Imputer+scaler are fit ONCE on the full feature
    matrix and shared by every target.

    If `frozen_Cs` is provided, skips C search and uses the given C for each
    target. Returns (models, chosen_Cs).
    """
    X_df = build_feature_matrix(df)
    feature_names = list(X_df.columns)
    X_raw = X_df.values

    # Fit shared preprocessing once
    preprocess = build_sklearn_pipeline()
    X_pre = preprocess.fit_transform(X_raw)

    rng = np.random.default_rng(42)
    models: dict[str, TrainedModel] = {}
    chosen_Cs: dict[str, float] = {}

    for threshold, horizon in TARGETS:
        col = _target_name(threshold, horizon)
        if col not in df.columns:
            logger.warning("Target column %s not in DataFrame. Skipping.", col)
            continue
        y_full = df[col].values
        valid = ~np.isnan(y_full)
        if valid.sum() < 100:
            logger.warning("Skipping %s: only %d valid rows", col, valid.sum())
            continue
        X_fit_full = X_pre[valid]
        y_fit_full = y_full[valid].astype(np.int8)
        pos_rate = float(y_fit_full.mean())
        if pos_rate == 0 or pos_rate == 1:
            logger.warning("Skipping %s: pos_rate=%.4f (degenerate)", col, pos_rate)
            continue

        # Stratified subsample for very large training sets
        X_fit, y_fit = _stratified_subsample(
            X_fit_full,
            y_fit_full,
            neg_ratio=TRAIN_SUBSAMPLE_NEG_RATIO,
            threshold=TRAIN_SUBSAMPLE_THRESHOLD,
            rng=rng,
        )

        C = frozen_Cs.get(col) if frozen_Cs else None
        clf, used_C = _fit_classifier(X_fit, y_fit, C=C)
        chosen_Cs[col] = used_C

        # Wrap preprocess + classifier into a single pipeline for prediction.
        # We clone the already-fitted preprocess by reusing its steps; sklearn
        # Pipelines accept pre-fitted estimators when constructed directly.
        full_pipeline = Pipeline(
            steps=[*preprocess.steps, ("model", clf)]
        )

        models[col] = TrainedModel(
            target=col,
            pipeline=full_pipeline,
            threshold=threshold,
            horizon_hours=horizon,
            n_train=int(len(y_fit_full)),
            pos_rate=pos_rate,
            C=used_C,
            feature_names=feature_names,
        )
        logger.info(
            "Trained %s: n=%d (fit on %d), pos_rate=%.3f%%, C=%.4g%s",
            col, len(y_fit_full), len(y_fit), pos_rate * 100, used_C,
            " [frozen]" if C is not None else "",
        )

    return models, chosen_Cs


def predict_proba_all(
    df: pd.DataFrame,
    models: dict[str, TrainedModel],
) -> pd.DataFrame:
    """
    Apply all trained models to df.
    Returns DataFrame with one probability column per target.
    """
    X = build_feature_matrix(df).values
    proba_cols: dict[str, np.ndarray] = {}

    for target, model in models.items():
        try:
            proba = model.pipeline.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.error("predict_proba failed for %s: %s", target, exc)
            proba = np.full(len(X), np.nan)
        proba_cols[f"proba_{target}"] = proba

    return pd.DataFrame(proba_cols, index=df.index)
