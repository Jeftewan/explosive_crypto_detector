"""
Cross-exchange funding rate dispersion indicator.
Data comes from Postgres via postgres_loader; this module only
computes derived features from the raw std/range columns.
"""
import numpy as np
import pandas as pd

from ..config import FR_ZSCORE_WINDOW


def add_cross_exchange_features(df: pd.DataFrame, window_days: int = FR_ZSCORE_WINDOW) -> pd.DataFrame:
    """
    Compute rolling z-scores of the cross-exchange FR dispersion.
    Expects columns: fr_cross_exchange_std, fr_cross_exchange_range
    """
    for col in ["fr_cross_exchange_std", "fr_cross_exchange_range"]:
        if col not in df.columns:
            df[f"{col}_zscore"] = np.nan
            continue
        w = window_days * 6
        rolling = df[col].rolling(window=w, min_periods=w // 2)
        df[f"{col}_zscore"] = (df[col] - rolling.mean()) / (rolling.std() + 1e-10)

    # High dispersion flag: FR std > 1 std dev above its rolling mean
    if "fr_cross_exchange_std_zscore" in df.columns:
        df["fr_cross_exchange_elevated"] = (
            df["fr_cross_exchange_std_zscore"] > 1.0
        ).astype(float)
    else:
        df["fr_cross_exchange_elevated"] = np.nan

    return df
