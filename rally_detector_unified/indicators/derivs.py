"""
Derivatives market indicators: Open Interest, Long/Short ratios, Top Trader
positioning, Taker buy/sell ratio.  All are NaN-aware — many rows will be NaN
for folds before the 30-day OI/LS/Taker window.
"""
import numpy as np
import pandas as pd

from ..config import FR_ZSCORE_WINDOW, FR_MOMENTUM_PERIODS


# ─── Open Interest ────────────────────────────────────────────────────────────

def add_oi_indicators(df: pd.DataFrame, window: int = FR_ZSCORE_WINDOW) -> pd.DataFrame:
    """OI z-score and regime based on Binance OI data."""
    if "open_interest" not in df.columns:
        df["oi_zscore"] = np.nan
        df["oi_regime"] = np.nan
        return df

    oi = df["open_interest"]
    rolling = oi.rolling(window=window * 6, min_periods=window * 3)  # 6 4h-candles per day
    df["oi_zscore"] = (oi - rolling.mean()) / (rolling.std() + 1e-10)
    df["oi_regime"] = pd.cut(
        df["oi_zscore"],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=[-1, 0, 1, 2],
    ).astype(float)
    df["oi_change_pct"] = oi.pct_change(periods=6)  # 1-day change
    return df


# ─── Long/Short Account Ratio ─────────────────────────────────────────────────

def add_ls_indicators(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """L/S account ratio z-score and extreme flags."""
    if "ls_account_ratio" not in df.columns:
        df["ls_zscore"] = np.nan
        df["ls_extreme_long"] = np.nan
        df["ls_extreme_short"] = np.nan
        return df

    ls = df["ls_account_ratio"]
    w = window_days * 6  # 4h candles per day
    rolling = ls.rolling(window=w, min_periods=w // 2)
    df["ls_zscore"] = (ls - rolling.mean()) / (rolling.std() + 1e-10)
    df["ls_extreme_long"] = (df["ls_zscore"] > 1.5).astype(float)
    df["ls_extreme_short"] = (df["ls_zscore"] < -1.5).astype(float)
    return df


# ─── Top Trader L/S Position Ratio ───────────────────────────────────────────

def add_top_trader_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Top trader ratio and divergence vs retail (L/S divergence signal)."""
    has_top = "top_trader_ls_ratio" in df.columns
    has_ls = "ls_account_ratio" in df.columns

    if not has_top:
        df["top_trader_ls_ratio"] = np.nan

    if has_top and has_ls:
        # Divergence: retail long bias while top traders are short (or vice-versa)
        retail_net = df["ls_account_ratio"] - 1.0   # > 0 = net long retail
        smart_net = df["top_trader_ls_ratio"] - 1.0  # > 0 = net long smart
        df["ls_divergence"] = retail_net - smart_net  # positive = retail long, smart short
    else:
        df["ls_divergence"] = np.nan

    return df


# ─── Taker Buy/Sell Ratio ─────────────────────────────────────────────────────

def add_taker_indicators(df: pd.DataFrame, momentum_periods: int = FR_MOMENTUM_PERIODS * 2) -> pd.DataFrame:
    """Taker B/S ratio and its momentum."""
    if "taker_bs_ratio" not in df.columns:
        df["taker_bs_momentum"] = np.nan
        df["taker_bs_extreme"] = np.nan
        return df

    df["taker_bs_momentum"] = df["taker_bs_ratio"].pct_change(periods=momentum_periods)
    df["taker_bs_extreme"] = (df["taker_bs_ratio"] > 1.5).astype(float)
    return df


# ─── Volume rank percentile across universe ───────────────────────────────────

def add_volume_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative volume rank across universe at each timestamp.
    Requires the full multi-symbol DataFrame (long format with 'symbol' column).
    """
    if "symbol" not in df.columns or "quote_volume" not in df.columns:
        df["volume_rank_pct"] = np.nan
        return df

    df["volume_rank_pct"] = df.groupby(df.index)["quote_volume"].rank(pct=True)
    return df


# ─── Composite ───────────────────────────────────────────────────────────────

def add_all_deriv_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_oi_indicators(df)
    df = add_ls_indicators(df)
    df = add_top_trader_indicators(df)
    df = add_taker_indicators(df)
    return df
