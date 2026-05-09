"""
Analysis 9: Optimal holding period per signal profile.
For each profile, computes expected return at 24h/72h/168h/504h horizons.
"""
import numpy as np
import pandas as pd

from ..config import TARGETS
from ..analysis.multi_profile import PROFILES, _apply_profile


def optimal_holding_analysis(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each profile × horizon combination, compute:
        mean_return, median_return, hit_rate, expected_value

    Returns DataFrame with columns:
        [profile, horizon_h, mean_return, median_return, hit_rate, expected_value]
    """
    records = []

    for profile_name, conditions in PROFILES.items():
        mask = _apply_profile(feature_df, conditions)
        n = mask.sum()
        if n < 10:
            continue

        for threshold, horizon in TARGETS:
            target_col = f"rally_{threshold}_{horizon}h"
            if target_col not in feature_df.columns or "close" not in feature_df.columns:
                continue

            fwd_return = feature_df["close"].pct_change(periods=horizon).shift(-horizon)
            signal_returns = fwd_return[mask].dropna()

            if len(signal_returns) < 5:
                continue

            hit = feature_df[target_col][mask].mean()

            records.append({
                "profile": profile_name,
                "threshold_pct": threshold,
                "horizon_h": horizon,
                "n": int(n),
                "mean_return": round(float(signal_returns.mean()), 4),
                "median_return": round(float(signal_returns.median()), 4),
                "hit_rate": round(float(hit), 4),
                "expected_value": round(float(signal_returns.mean()), 4),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(["profile", "horizon_h"])
