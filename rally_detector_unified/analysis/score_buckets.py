"""
Analysis 1: Probability buckets vs actual hit rate (calibration check).
"""
import numpy as np
import pandas as pd


def score_buckets_analysis(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    n_buckets: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    For each target model, bin predictions into n_buckets probability buckets
    and compute actual hit rate per bucket.

    Returns dict: target_name → DataFrame with columns
        [bucket_low, bucket_high, n, actual_hit_rate, expected_hit_rate, lift]
    """
    results = {}
    proba_cols = [c for c in predictions.columns if c.startswith("proba_")]

    for pcol in proba_cols:
        target = pcol.replace("proba_", "")
        if target not in actuals.columns:
            continue

        df = pd.DataFrame({
            "proba": predictions[pcol],
            "actual": actuals[target],
        }).dropna()

        if df.empty:
            continue

        df["bucket"] = pd.cut(df["proba"], bins=n_buckets)
        base = df["actual"].mean()

        summary = (
            df.groupby("bucket", observed=True)["actual"]
            .agg(n="count", actual_hit_rate="mean")
            .reset_index()
        )
        summary["bucket_mid"] = summary["bucket"].apply(lambda x: (x.left + x.right) / 2)
        summary["expected_hit_rate"] = summary["bucket_mid"]
        summary["lift"] = summary["actual_hit_rate"] / (base + 1e-10)
        results[target] = summary

    return results
