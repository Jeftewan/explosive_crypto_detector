"""
Analysis 3: Pre-explosion footprint.
What did the indicator values look like N hours before tokens that exploded?
"""
import numpy as np
import pandas as pd

from ..scoring.feature_pipeline import FEATURE_COLUMNS


def pre_explosion_analysis(
    feature_df: pd.DataFrame,
    target_col: str = "rally_100_72h",
    lookback_hours: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compare mean feature values for:
      - Rows that eventually hit the target (positive class)
      - All other rows (negative class)

    Returns DataFrame with columns:
        [feature, mean_positive, mean_negative, ratio, p_value]
    """
    from scipy import stats as scipy_stats

    if lookback_hours is None:
        lookback_hours = [0]

    if target_col not in feature_df.columns:
        return pd.DataFrame()

    available_features = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    y = feature_df[target_col]
    pos_mask = y == 1
    neg_mask = y == 0

    records = []
    for feat in available_features:
        vals = feature_df[feat]
        pos_vals = vals[pos_mask].dropna()
        neg_vals = vals[neg_mask].dropna()

        if len(pos_vals) < 5 or len(neg_vals) < 5:
            continue

        try:
            _, p_val = scipy_stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        except Exception:
            p_val = float("nan")

        records.append({
            "feature": feat,
            "mean_positive": round(float(pos_vals.mean()), 4),
            "mean_negative": round(float(neg_vals.mean()), 4),
            "std_positive": round(float(pos_vals.std()), 4),
            "ratio": round(float(pos_vals.mean()) / (abs(float(neg_vals.mean())) + 1e-10), 3),
            "p_value": round(p_val, 4),
            "significant": p_val < 0.05,
        })

    df = pd.DataFrame(records)
    return df.sort_values("p_value") if not df.empty else df
