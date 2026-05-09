"""
Analysis 5: Top N explosions — tokens that moved the most and their indicators.
"""
import pandas as pd


def top_explosions_analysis(
    feature_df: pd.DataFrame,
    target_col: str = "rally_100_72h",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find the top_n highest forward-return events among positives,
    and return their indicator snapshot at signal time.
    """
    if target_col not in feature_df.columns or "close" not in feature_df.columns:
        return pd.DataFrame()

    try:
        horizon = int(target_col.split("_")[-1].replace("h", ""))
    except Exception:
        horizon = 72

    fwd_return = feature_df["close"].pct_change(periods=horizon).shift(-horizon)
    y = feature_df[target_col]

    positives = feature_df[y == 1].copy()
    positives["fwd_return"] = fwd_return[y == 1]
    positives = positives.dropna(subset=["fwd_return"])

    top = positives.nlargest(top_n, "fwd_return")

    indicator_cols = [
        "symbol", "fwd_return", "rsi", "bb_squeeze", "volume_zscore",
        "fr_zscore", "fr_pct_positive", "obv_trend",
        "oi_zscore", "ls_zscore", "taker_bs_ratio", "btc_regime_bull",
    ]
    available = [c for c in indicator_cols if c in top.columns]
    return top[available].round(4)
