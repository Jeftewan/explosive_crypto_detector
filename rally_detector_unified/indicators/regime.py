"""
Market regime indicators based on BTC price.
Requires a DataFrame that includes BTC data (passed separately).
"""
import numpy as np
import pandas as pd


_MA200_PERIODS = 200 * 6  # 200 days × 6 4h-candles per day


def compute_btc_regime(btc_close: pd.Series) -> pd.Series:
    """
    Binary BTC market regime: 1 = bull (price > MA200), 0 = bear.
    Input: BTC close prices indexed by UTC timestamp.
    Returns: Series with same index, values 0 or 1.
    """
    ma200 = btc_close.rolling(window=_MA200_PERIODS, min_periods=_MA200_PERIODS // 2).mean()
    regime = (btc_close > ma200).astype(int)
    regime.name = "btc_regime_bull"
    return regime


def add_btc_regime(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join BTC market regime onto any symbol DataFrame.
    btc_df must have a 'close' column indexed by UTC timestamp.
    """
    if btc_df.empty or "close" not in btc_df.columns:
        df["btc_regime_bull"] = np.nan
        return df

    regime = compute_btc_regime(btc_df["close"])
    # Reindex to match df's hourly grid
    regime_h = regime.reindex(df.index, method="ffill")
    df["btc_regime_bull"] = regime_h.values
    return df
