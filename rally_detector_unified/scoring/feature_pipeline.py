"""
Feature pipeline: assembles all indicators into a feature matrix,
handles NaN via median imputation + missing indicator flags, and standardizes.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features that are always NaN in folds before the 30-day OI/LS/Taker window
_OI_LS_TAKER_FEATURES = [
    "oi_zscore", "oi_regime", "oi_change_pct",
    "ls_account_ratio", "ls_zscore", "ls_extreme_long", "ls_extreme_short",
    "top_trader_ls_ratio", "ls_divergence",
    "taker_bs_ratio", "taker_bs_momentum", "taker_bs_extreme",
]

FEATURE_COLUMNS = [
    # FR indicators
    "fr_pct_positive",
    "fr_consecutive_streak",
    "fr_zscore",
    "fr_momentum",
    "fr_percentile",
    # Technical
    "rsi",
    "bb_squeeze",
    "bb_squeeze_bars",
    "bb_pct_b",
    "bb_width",
    "volume_zscore",
    "volume_spike",
    "obv_trend",
    "obv_accumulation",
    "price_roc_7d",
    "price_roc_14d",
    "price_roc_21d",
    "volatility_compression",
    "volatility_ratio",
    # Derivatives (NaN before day -30)
    "oi_zscore",
    "oi_regime",
    "oi_change_pct",
    "ls_zscore",
    "ls_extreme_long",
    "ls_extreme_short",
    "top_trader_ls_ratio",
    "ls_divergence",
    "taker_bs_ratio",
    "taker_bs_momentum",
    "taker_bs_extreme",
    # Regime
    "btc_regime_bull",
    # Cross-exchange
    "fr_cross_exchange_std_zscore",
    "fr_cross_exchange_range_zscore",
    "fr_cross_exchange_elevated",
    # Volume rank
    "volume_rank_pct",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the feature columns that exist in df.
    Missing columns are filled with NaN (will be imputed downstream).
    """
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.debug("Missing features (will be NaN): %s", missing)

    X = df[available].copy()
    for c in missing:
        X[c] = np.nan

    return X[FEATURE_COLUMNS]


def build_sklearn_pipeline() -> Pipeline:
    """
    Scikit-learn pipeline:
      1. Median imputation with missing-value indicator columns
      2. Standard scaling (mean=0, std=1)
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])
