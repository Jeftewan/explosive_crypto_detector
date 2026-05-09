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

    y = feature_df[target_col].dropna()
    available = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    records = []

    for feat in available:
        x = feature_df[feat].reindex(y.index).dropna()
        common = y.reindex(x.index).dropna()
        x = x.reindex(common.index)

        if len(common) < 20:
            continue

        try:
            pearson_r, pearson_p = scipy_stats.pearsonr(x, common)
        except Exception:
            pearson_r, pearson_p = float("nan"), float("nan")

        try:
            spearman_r, spearman_p = scipy_stats.spearmanr(x, common)
        except Exception:
            spearman_r, spearman_p = float("nan"), float("nan")

        records.append({
            "feature": feat,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "abs_spearman": round(abs(spearman_r), 4),
        })

    df = pd.DataFrame(records)
    return df.sort_values("abs_spearman", ascending=False) if not df.empty else df
