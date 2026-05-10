"""
Analysis 6: Feature correlations — Pearson and Spearman.
"""
import pandas as pd
from scipy import stats as scipy_stats

from ..scoring.feature_pipeline import FEATURE_COLUMNS


def correlation_analysis(
    feature_df: pd.DataFrame,
    target_col: str = "rally_100_72h",
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlation of each feature with the target.
    Returns sorted DataFrame by |spearman_r|.
    """
    if target_col not in feature_df.columns:
        return pd.DataFrame()

    available = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    records = []

    for feat in available:
        # Drop rows where either column is NaN. Positional alignment is
        # preserved because both come from the same row of feature_df, so
        # no reindex is needed (and reindex would crash on the duplicated
        # multi-symbol DatetimeIndex).
        pair = feature_df[[feat, target_col]].dropna()
        if len(pair) < 20:
            continue
        x = pair[feat].values
        y = pair[target_col].values

        try:
            pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
        except Exception:
            pearson_r, pearson_p = float("nan"), float("nan")

        try:
            spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
        except Exception:
            spearman_r, spearman_p = float("nan"), float("nan")

        records.append({
            "feature": feat,
            "n": int(len(pair)),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "abs_spearman": round(abs(spearman_r), 4),
        })

    df = pd.DataFrame(records)
    return df.sort_values("abs_spearman", ascending=False) if not df.empty else df
