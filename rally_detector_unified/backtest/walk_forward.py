"""
Purged walk-forward cross-validation with embargo (López de Prado, 2018).

Schema for 365 days, 5 folds:
  Fold 1:  Train [0-150d]  Embargo 21d  Test [171-200d]
  Fold 2:  Train [0-200d]  Embargo 21d  Test [221-250d]
  Fold 3:  Train [0-250d]  Embargo 21d  Test [271-300d]
  Fold 4:  Train [0-300d]  Embargo 21d  Test [321-350d]
  Fold 5:  Train [0-350d]  Embargo 21d  Test [350-365d]
"""
import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from ..scoring.feature_pipeline import build_feature_matrix
from ..scoring.logistic_l1 import build_logistic_pipeline, predict_proba_all, TrainedModel
from ..config import WF_EMBARGO_DAYS, WF_FOLDS, WF_MIN_TRAIN_DAYS, TARGETS

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    models: dict[str, TrainedModel] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    actuals: pd.DataFrame = field(default_factory=pd.DataFrame)


def _make_folds(
    index: pd.DatetimeIndex,
    n_folds: int = WF_FOLDS,
    embargo_days: int = WF_EMBARGO_DAYS,
    min_train_days: int = WF_MIN_TRAIN_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate (train_start, train_end, test_start, test_end) tuples.
    All times are UTC DatetimeIndex boundaries.
    """
    start = index.min()
    end = index.max()
    total_days = (end - start).days

    # Split remaining days (after min_train) into n_folds equal test windows
    available = total_days - min_train_days - embargo_days
    if available <= 0:
        raise ValueError(f"Not enough data for {n_folds} folds. Need > {min_train_days + embargo_days} days.")

    test_window = available // n_folds
    folds = []

    for i in range(n_folds):
        test_start_day = min_train_days + embargo_days + i * test_window
        test_end_day = test_start_day + test_window if i < n_folds - 1 else total_days

        train_end_dt = start + timedelta(days=test_start_day - embargo_days)
        test_start_dt = start + timedelta(days=test_start_day)
        test_end_dt = start + timedelta(days=test_end_day)

        folds.append((start, train_end_dt, test_start_dt, test_end_dt))

    return folds


def _purge(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    max_horizon_hours: int,
) -> pd.DataFrame:
    """
    Remove rows from train whose forward-return window overlaps with test period.
    A train row at time t has a forward window [t, t + max_horizon_hours].
    It overlaps test if t + max_horizon_hours >= test_start.
    """
    cutoff = test_start - timedelta(hours=max_horizon_hours)
    return df[df.index <= cutoff]


def run_walk_forward(
    feature_df: pd.DataFrame,
    n_folds: int = WF_FOLDS,
    embargo_days: int = WF_EMBARGO_DAYS,
) -> list[FoldResult]:
    """
    Run purged walk-forward CV. Returns one FoldResult per fold.
    feature_df: long-format DataFrame indexed by UTC timestamp,
                with feature columns and target columns (rally_*).
    """
    target_cols = [f"rally_{t}_{h}h" for t, h in TARGETS]
    max_horizon = max(h for _, h in TARGETS)

    # Use a single-symbol or cross-symbol index
    time_index = feature_df.index.unique().sort_values()
    fold_defs = _make_folds(time_index, n_folds=n_folds, embargo_days=embargo_days)

    results: list[FoldResult] = []

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(fold_defs, start=1):
        logger.info(
            "Fold %d — Train: %s → %s | Test: %s → %s",
            fold_i, tr_start.date(), tr_end.date(), te_start.date(), te_end.date()
        )

        # Train slice with purging
        train_raw = feature_df[(feature_df.index >= tr_start) & (feature_df.index <= tr_end)]
        train_df = _purge(train_raw, te_start, te_end, max_horizon)

        # Test slice
        test_df = feature_df[(feature_df.index >= te_start) & (feature_df.index < te_end)]

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test split. Skipping.", fold_i)
            continue

        X_train = build_feature_matrix(train_df)
        X_test = build_feature_matrix(test_df)

        # Train one model per target
        fold_models: dict[str, TrainedModel] = {}
        for threshold, horizon in TARGETS:
            col = f"rally_{threshold}_{horizon}h"
            if col not in train_df.columns:
                continue
            y = train_df[col].dropna()
            if y.empty or y.nunique() < 2:
                continue

            from ..scoring.logistic_l1 import train_model
            model = train_model(
                X_train.loc[y.index],
                y,
                target_name=col,
                threshold=threshold,
                horizon_hours=horizon,
            )
            if model is not None:
                fold_models[col] = model

        # Predict on test
        predictions = predict_proba_all(test_df, fold_models)
        actuals = test_df[[c for c in target_cols if c in test_df.columns]]

        results.append(FoldResult(
            fold=fold_i,
            train_start=tr_start,
            train_end=tr_end,
            test_start=te_start,
            test_end=te_end,
            models=fold_models,
            predictions=predictions,
            actuals=actuals,
        ))
        logger.info("Fold %d done: %d test rows, %d models trained", fold_i, len(test_df), len(fold_models))

    return results
