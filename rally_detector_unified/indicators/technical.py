"""
Technical indicators: RSI, Bollinger Bands, Volume z-score/spike, OBV, Price ROC,
Volatility compression.  All expect a DataFrame with OHLCV columns.
"""
import numpy as np
import pandas as pd

from ..config import RSI_PERIOD, BB_WINDOW, BB_STD, VOLUME_ZSCORE_WINDOW, OBV_SMOOTHING


# ─── RSI ──────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = BB_WINDOW,
    num_std: float = BB_STD,
) -> pd.DataFrame:
    rolling = df["close"].rolling(window=window, min_periods=window // 2)
    mid = rolling.mean()
    std = rolling.std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = upper - lower

    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = band_width / (mid + 1e-10)
    df["bb_pct_b"] = (df["close"] - lower) / (band_width + 1e-10)

    # Squeeze: band width < 80th percentile of its own rolling history
    bw_pct80 = df["bb_width"].rolling(window=window * 5, min_periods=window).quantile(0.8)
    df["bb_squeeze"] = (df["bb_width"] < bw_pct80).astype(int)

    # Consecutive bars in squeeze
    squeeze_bars = []
    count = 0
    for v in df["bb_squeeze"]:
        count = count + 1 if v == 1 else 0
        squeeze_bars.append(count)
    df["bb_squeeze_bars"] = squeeze_bars

    return df


# ─── Volume ───────────────────────────────────────────────────────────────────

def add_volume_indicators(
    df: pd.DataFrame,
    window: int = VOLUME_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Volume z-score and spike flag."""
    rolling = df["volume"].rolling(window=window, min_periods=window // 2)
    df["volume_zscore"] = (df["volume"] - rolling.mean()) / (rolling.std() + 1e-10)
    df["volume_spike"] = (df["volume_zscore"] > 2.0).astype(int)
    return df


# ─── OBV ─────────────────────────────────────────────────────────────────────

def add_obv(df: pd.DataFrame, smoothing: int = OBV_SMOOTHING) -> pd.DataFrame:
    """On-Balance Volume with trend and accumulation signal."""
    direction = np.sign(df["close"].diff().fillna(0))
    obv = (direction * df["volume"]).cumsum()
    df["obv"] = obv
    df["obv_ma"] = obv.ewm(span=smoothing, min_periods=smoothing // 2).mean()
    df["obv_trend"] = np.sign(df["obv"] - df["obv_ma"])
    df["obv_accumulation"] = (df["obv_trend"] == 1).astype(int)
    return df


# ─── Price ROC ───────────────────────────────────────────────────────────────

def add_price_roc(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of change over 7, 14, 21 day windows (assuming 4h candles = 6 per day)."""
    candles_per_day = 6  # 4h candles
    for days in [7, 14, 21]:
        p = days * candles_per_day
        df[f"price_roc_{days}d"] = df["close"].pct_change(periods=p)
    return df


# ─── Volatility compression ───────────────────────────────────────────────────

def add_volatility_compression(df: pd.DataFrame, short: int = 10, long: int = 50) -> pd.DataFrame:
    """
    Volatility compression: short-window ATR < long-window ATR.
    Signals that volatility has contracted (coiled spring).
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_short = tr.rolling(window=short, min_periods=short // 2).mean()
    atr_long = tr.rolling(window=long, min_periods=long // 2).mean()

    df["atr"] = atr_short
    df["volatility_compression"] = (atr_short < atr_long).astype(int)
    df["volatility_ratio"] = atr_short / (atr_long + 1e-10)
    return df


# ─── Composite ───────────────────────────────────────────────────────────────

def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_volume_indicators(df)
    df = add_obv(df)
    df = add_price_roc(df)
    df = add_volatility_compression(df)
    return df
