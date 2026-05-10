"""
Analysis 11: Rally type breakdown — Type A, B, C classification.

Type A (silent accumulation): BB Squeeze + OBV positive + FR neutral
Type B (short squeeze):       OI drops + FR negative + L/S extreme
Type C (leveraged):           FR↑ + OI↑ + Taker Buy↑
"""
import numpy as np
import pandas as pd

from ._helpers import forward_returns_by_symbol, positional_select


RALLY_TYPES = {
    "A_silent_accumulation": {
        "bb_squeeze": (1, 1),
        "obv_accumulation": (1, 1),
        "fr_zscore": (-0.5, 0.5),
    },
    "B_short_squeeze": {
        "oi_change_pct": (None, -0.02),       # OI dropping
        "funding_rate": (None, 0),             # FR negative
        "ls_extreme_short": (1, 1),            # Extreme short positioning
    },
    "C_leveraged_long": {
        "fr_momentum": (0.01, None),           # FR rising
        "oi_change_pct": (0.02, None),         # OI rising
        "taker_bs_ratio": (1.2, None),         # Taker buy dominant
    },
}


def _apply_type(df: pd.DataFrame, conditions: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in conditions.items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi
    return mask


def rally_type_breakdown(
    feature_df: pd.DataFrame,
    target_col: str = "rally_100_72h",
) -> pd.DataFrame:
    """
    For each rally type, compute: n_signals, hit_rate, mean_return.
    Returns DataFrame with columns:
        [rally_type, n, hit_rate, base_rate, lift, mean_fwd_return]
    """
    if target_col not in feature_df.columns:
        return pd.DataFrame()

    try:
        horizon = int(target_col.split("_")[-1].replace("h", ""))
    except Exception:
        horizon = 72

    fwd_return = (
        forward_returns_by_symbol(feature_df, horizon=horizon)
        if "close" in feature_df.columns else None
    )
    y = feature_df[target_col]
    base = y.mean()
    records = []

    for rtype, conditions in RALLY_TYPES.items():
        mask_arr = _apply_type(feature_df, conditions).values
        n = int(mask_arr.sum())
        if n < 5:
            continue

        hit = float(np.nanmean(positional_select(y.values, mask_arr)))
        mean_ret = (
            float(np.nanmean(positional_select(fwd_return, mask_arr)))
            if fwd_return is not None else float("nan")
        )
        records.append({
            "rally_type": rtype,
            "n": int(n),
            "hit_rate": round(float(hit), 4),
            "base_rate": round(float(base), 4),
            "lift": round(float(hit) / (float(base) + 1e-10), 2),
            "mean_fwd_return": round(float(mean_ret), 4),
        })

    return pd.DataFrame(records)
