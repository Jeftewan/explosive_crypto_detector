"""
Builds the full feature matrix with all indicators and return targets
from the unified grid DataFrame.
"""
import logging

import numpy as np
import pandas as pd

from ..indicators.fr import add_all_fr_indicators
from ..indicators.technical import add_all_technical_indicators
from ..indicators.derivs import add_all_deriv_indicators, add_volume_rank
from ..indicators.regime import add_btc_regime
from ..indicators.cross_exchange import add_cross_exchange_features
from ..config import TARGETS

logger = logging.getLogger(__name__)


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward return targets for each (threshold_pct, horizon_hours).
    Uses real-hour forward returns from the close price on the hourly grid.

    Target = 1 if max close price in [t+1, t+horizon] exceeds close[t] * (1 + threshold/100).
    """
    close = df["close"]
    for threshold, horizon in TARGETS:
        col = f"rally_{threshold}_{horizon}h"
        # Max future price within the horizon window
        target_price = close * (1 + threshold / 100)
        # Rolling max over next `horizon` bars (forward-looking)
        # We shift by -horizon so future max aligns with current bar
        future_max = close[::-1].rolling(window=horizon, min_periods=1).max()[::-1].shift(-horizon)
        df[col] = (future_max >= target_price).astype(float)
        # Last `horizon` rows have no future → set to NaN to avoid look-ahead
        df.loc[df.index[-horizon:], col] = np.nan

    return df


def build_features_for_symbol(
    symbol_df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply all indicator functions and compute targets for a single-symbol DataFrame.
    Input: hourly-grid DataFrame for one symbol (must have OHLCV + auxiliary columns).
    Output: same DataFrame with ~28 feature columns + 9 target columns.
    """
    df = symbol_df.copy()

    # Check minimum required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Missing OHLCV columns for %s: %s", df.get("symbol", ["?"])[0] if "symbol" in df.columns else "?", missing)
        return pd.DataFrame()

    # FR indicators — require 'funding_rate' column
    if "funding_rate" in df.columns:
        df = add_all_fr_indicators(df)
    else:
        for col in ["fr_pct_positive", "fr_consecutive_streak", "fr_zscore", "fr_momentum", "fr_percentile"]:
            df[col] = np.nan

    # Technical indicators
    df = add_all_technical_indicators(df)

    # Derivatives indicators
    df = add_all_deriv_indicators(df)

    # BTC market regime
    if btc_df is not None:
        df = add_btc_regime(df, btc_df)
    else:
        df["btc_regime_bull"] = np.nan

    # Cross-exchange FR dispersion
    df = add_cross_exchange_features(df)

    # Volume rank (needs full universe, added later in build_features_universe)
    if "volume_rank_pct" not in df.columns:
        df["volume_rank_pct"] = np.nan

    # Forward-return targets
    df = _compute_targets(df)

    return df


def build_features_universe(
    unified_df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply feature building to all symbols in the unified long-format DataFrame.
    Also adds volume rank percentile across universe.
    """
    frames = []
    for sym, grp in unified_df.groupby("symbol"):
        feat_df = build_features_for_symbol(grp, btc_df=btc_df)
        if not feat_df.empty:
            feat_df["symbol"] = sym
            frames.append(feat_df)

    if not frames:
        raise RuntimeError("No features built for any symbol.")

    result = pd.concat(frames)
    result.sort_index(inplace=True)

    # Compute volume rank across universe at each timestamp
    result = add_volume_rank(result)

    logger.info(
        "Feature matrix: %d rows × %d cols, %d symbols",
        len(result), result.shape[1], result["symbol"].nunique()
    )
    return result
