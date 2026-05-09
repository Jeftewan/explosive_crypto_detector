"""
Validate model predictions against the 13 real trades from user_history.
Serves as a sanity check (not statistical validation — n=13 is too small).
"""
import logging

import pandas as pd

from ..data.postgres_loader import load_user_history

logger = logging.getLogger(__name__)


def validate_against_user_history(
    predictions_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each trade in user_history, find the model's predicted probability
    at entry_time for that symbol.

    Returns DataFrame with columns:
        symbol, entry_time, exit_time, pnl_pct,
        proba_rally_* (one column per model target)
    """
    trades = load_user_history()
    if trades.empty:
        logger.warning("No user history loaded — skipping ground truth validation.")
        return pd.DataFrame()

    target_cols = [c for c in predictions_df.columns if c.startswith("proba_")]
    records = []

    for _, trade in trades.iterrows():
        sym = trade["symbol"]
        entry = trade["entry_time"]

        # Nearest prediction within ±2 hours of entry_time
        sym_pred = predictions_df[feature_df["symbol"] == sym] if "symbol" in feature_df.columns else pd.DataFrame()
        if sym_pred.empty:
            logger.debug("No predictions for %s", sym)
            records.append({"symbol": sym, "entry_time": entry, "pnl_pct": trade.get("pnl_pct"), **{c: None for c in target_cols}})
            continue

        time_diff = (sym_pred.index - entry).total_seconds().abs()
        nearest_idx = time_diff.argmin()
        if time_diff.iloc[nearest_idx] > 7200:  # 2h tolerance
            logger.debug("No close prediction for %s at %s (nearest: %ds away)", sym, entry, time_diff.iloc[nearest_idx])
            records.append({"symbol": sym, "entry_time": entry, "pnl_pct": trade.get("pnl_pct"), **{c: None for c in target_cols}})
            continue

        row = sym_pred.iloc[nearest_idx]
        record = {
            "symbol": sym,
            "entry_time": entry,
            "exit_time": trade.get("exit_time"),
            "pnl_pct": trade.get("pnl_pct"),
        }
        for col in target_cols:
            record[col] = row.get(col)
        records.append(record)

    result = pd.DataFrame(records)
    logger.info(
        "Ground truth validation: %d trades matched out of %d total",
        result[target_cols[0]].notna().sum() if target_cols and not result.empty else 0,
        len(trades),
    )
    return result
