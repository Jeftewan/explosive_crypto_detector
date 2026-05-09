"""
Funding Rate indicators.
All functions operate on a DataFrame with a 'funding_rate' column
and return a DataFrame with new indicator columns added.
"""
import numpy as np
import pandas as pd

from ..config import FR_WINDOW_DAYS, FR_ZSCORE_WINDOW, FR_MOMENTUM_PERIODS


def add_fr_pct_positive(df: pd.DataFrame, window_days: int = FR_WINDOW_DAYS) -> pd.DataFrame:
    """Fraction of funding rate periods that are positive within rolling window."""
    periods = window_days * 3  # ~3 FR payments per day (8h interval)
    df["fr_pct_positive"] = (
        (df["funding_rate"] > 0)
        .rolling(window=periods, min_periods=periods // 2)
        .mean()
    )
    return df


def add_fr_consecutive_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signed consecutive streak of positive/negative FR.
    Positive streak = how many consecutive positive FR payments.
    Negative streak (stored as negative number) = consecutive negative.
    """
    sign = np.sign(df["funding_rate"].fillna(0))
    # Use a group-by cumcount trick
    streak = []
    current = 0
    prev_sign = 0
    for s in sign:
        if s == 0:
            streak.append(0)
            current = 0
            prev_sign = 0
        elif s == prev_sign:
            current += 1
            streak.append(current * s)
        else:
            current = 1
            streak.append(s)
            prev_sign = s
    df["fr_consecutive_streak"] = streak
    return df


def add_fr_zscore(df: pd.DataFrame, window_days: int = FR_ZSCORE_WINDOW) -> pd.DataFrame:
    """Z-score of funding rate vs rolling mean/std."""
    w = window_days * 3
    rolling = df["funding_rate"].rolling(window=w, min_periods=w // 2)
    df["fr_zscore"] = (df["funding_rate"] - rolling.mean()) / (rolling.std() + 1e-10)
    return df


def add_fr_momentum(df: pd.DataFrame, periods: int = FR_MOMENTUM_PERIODS) -> pd.DataFrame:
    """Rate of change of funding rate over N periods."""
    df["fr_momentum"] = df["funding_rate"].pct_change(periods=periods)
    return df


def add_fr_percentile(df: pd.DataFrame, window_days: int = FR_ZSCORE_WINDOW) -> pd.DataFrame:
    """Rolling percentile rank of funding rate (0–1)."""
    w = window_days * 3
    df["fr_percentile"] = (
        df["funding_rate"]
        .rolling(window=w, min_periods=w // 2)
        .rank(pct=True)
    )
    return df


def add_all_fr_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_fr_pct_positive(df)
    df = add_fr_consecutive_streak(df)
    df = add_fr_zscore(df)
    df = add_fr_momentum(df)
    df = add_fr_percentile(df)
    return df
