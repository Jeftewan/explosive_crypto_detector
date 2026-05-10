"""
Analysis 10: Market regime breakdown — hit rate and Sharpe in BTC bull vs bear.
"""
import numpy as np
import pandas as pd

from ..backtest.metrics import compute_fold_metrics
from ..config import TARGETS


def market_regime_breakdown(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split feature_df by BTC regime (bull=1 / bear=0) and compute metrics per target.

    Returns DataFrame with columns:
        [regime, target, n, base_rate, hit_rate_50, precision_top10pct, lift_top10pct]
    """
    if "btc_regime_bull" not in feature_df.columns:
        return pd.DataFrame()

    records = []
    for regime_val, regime_label in [(1, "bull"), (0, "bear")]:
        # Positional alignment: predictions and feature_df share row order
        # (predict_proba_all preserves input index), but their DatetimeIndex
        # is duplicated across symbols, so use .iloc with a numpy mask.
        mask = (feature_df["btc_regime_bull"] == regime_val).values
        sub_feat = feature_df.iloc[mask]
        sub_pred = predictions.iloc[mask]

        for threshold, horizon in TARGETS:
            target_col = f"rally_{threshold}_{horizon}h"
            proba_col = f"proba_{target_col}"

            if target_col not in sub_feat.columns or proba_col not in sub_pred.columns:
                continue

            y = sub_feat[target_col].values
            yp = sub_pred[proba_col].values
            m = compute_fold_metrics(y, yp)
            m["regime"] = regime_label
            m["target"] = target_col
            records.append(m)

    return pd.DataFrame(records)
