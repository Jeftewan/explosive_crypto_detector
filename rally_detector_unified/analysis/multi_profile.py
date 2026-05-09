"""
Analysis 2: Multi-indicator profile combinations and their hit rates.
Profiles defined as conjunctions of indicator conditions.
"""
import pandas as pd


PROFILES = {
    "bb_squeeze_vol_spike": {
        "bb_squeeze": (1, 1),
        "volume_spike": (1, 1),
    },
    "bb_squeeze_rsi_low": {
        "bb_squeeze": (1, 1),
        "rsi": (None, 50),
    },
    "squeeze_vol_rsi": {
        "bb_squeeze": (1, 1),
        "volume_spike": (1, 1),
        "rsi": (None, 50),
    },
    "high_fr_zscore": {
        "fr_zscore": (1.5, None),
    },
    "accumulation_squeeze": {
        "obv_accumulation": (1, 1),
        "bb_squeeze": (1, 1),
    },
    "smart_divergence": {
        "ls_divergence": (0.1, None),   # retail long, smart short
        "bb_squeeze": (1, 1),
    },
    "oi_expansion_taker_buy": {
        "oi_change_pct": (0.05, None),  # OI growing
        "taker_bs_ratio": (1.2, None),  # aggressive buyers
    },
}


def _apply_profile(df: pd.DataFrame, conditions: dict) -> pd.Series:
    """Return boolean mask for rows satisfying all conditions in the profile."""
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in conditions.items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi
    return mask


def multi_profile_analysis(
    feature_df: pd.DataFrame,
    actuals: pd.DataFrame,
    profiles: dict | None = None,
) -> pd.DataFrame:
    """
    For each profile, compute: n_signals, hit_rate per target, lift vs base.
    Returns long-format DataFrame with columns:
        [profile, target, n, hit_rate, base_rate, lift]
    """
    if profiles is None:
        profiles = PROFILES

    target_cols = [c for c in actuals.columns if c.startswith("rally_")]
    records = []

    for profile_name, conditions in profiles.items():
        mask = _apply_profile(feature_df, conditions)
        n = mask.sum()
        if n < 10:
            continue

        for target in target_cols:
            if target not in actuals.columns:
                continue
            y = actuals[target]
            base = y.mean()
            hit = y[mask].mean()
            records.append({
                "profile": profile_name,
                "target": target,
                "n": int(n),
                "hit_rate": round(hit, 4),
                "base_rate": round(base, 4),
                "lift": round(hit / (base + 1e-10), 2),
            })

    return pd.DataFrame(records)
