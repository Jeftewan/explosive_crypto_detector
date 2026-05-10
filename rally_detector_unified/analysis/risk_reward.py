"""
Analysis 4: Risk/reward simulation — equal capital per signal, fixed holding.
Also: the "1 of 10 that rises x20" analysis (SPECIAL SECTION from plan).
"""
import numpy as np
import pandas as pd

from ._helpers import forward_returns_by_symbol, positional_select


def simulate_equal_weight(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    target_col: str = "rally_100_72h",
    proba_col: str | None = None,
    min_proba: float = 0.5,
    n_simulations: int = 10_000,
    basket_size: int = 10,
    capital: float = 1000.0,
) -> dict:
    """
    Monte Carlo simulation: pick `basket_size` signals at random from the
    top-predicted subset, invest equal capital, hold for horizon.

    Returns dict with distribution statistics and probability questions answered.
    """
    if proba_col is None:
        proba_col = f"proba_{target_col}"

    if proba_col not in predictions.columns or target_col not in feature_df.columns:
        return {}

    # Build signal universe: rows above min_proba threshold
    mask = (predictions[proba_col] >= min_proba).values
    y = positional_select(feature_df[target_col].values, mask)
    # Forward 1-bar return computed per-symbol so it doesn't cross
    # cripto boundaries in the long-format multi-symbol DataFrame.
    returns = forward_returns_by_symbol(feature_df, horizon=1)
    signal_returns = pd.Series(positional_select(returns, mask))

    if len(y) < basket_size:
        return {"error": f"Not enough signals ({len(y)}) for basket_size={basket_size}"}

    # Monte Carlo
    rng = np.random.default_rng(42)
    basket_pnls = []
    valid_returns = signal_returns.dropna().values

    for _ in range(n_simulations):
        picks = rng.choice(valid_returns, size=basket_size, replace=False)
        # Equal capital per pick
        basket_pnl = (picks * (capital / basket_size)).sum()
        basket_pnls.append(basket_pnl)

    pnls = np.array(basket_pnls)

    return {
        "target": target_col,
        "n_signals": int(len(y)),
        "hit_rate": float(np.nanmean(y)),
        "return_distribution": {
            "p10": float(np.percentile(pnls, 10)),
            "p25": float(np.percentile(pnls, 25)),
            "p50": float(np.percentile(pnls, 50)),
            "p75": float(np.percentile(pnls, 75)),
            "p90": float(np.percentile(pnls, 90)),
            "p95": float(np.percentile(pnls, 95)),
            "p99": float(np.percentile(pnls, 99)),
        },
        "prob_pnl_positive": float((pnls > 0).mean()),
        "prob_pnl_50pct": float((pnls > capital * 0.5).mean()),
        "prob_pnl_200pct": float((pnls > capital * 2.0).mean()),
        "expected_pnl": float(pnls.mean()),
    }


def outlier_contribution_analysis(
    feature_df: pd.DataFrame,
    target_col: str = "rally_100_72h",
) -> dict:
    """
    Analyze the contribution of top outliers (x5, x10, x20 movers) to total PnL.
    Answers: "1 of 10 that rises x20 — is the overall portfolio positive?"
    """
    if target_col not in feature_df.columns or "close" not in feature_df.columns:
        return {}

    # Forward return at horizon derived from target name
    try:
        horizon = int(target_col.split("_")[-1].replace("h", ""))
    except Exception:
        horizon = 72

    fwd_return = forward_returns_by_symbol(feature_df, horizon=horizon)
    y = feature_df[target_col]

    pos_mask = (y == 1).values
    signal_returns = pd.Series(positional_select(fwd_return, pos_mask)).dropna()
    if len(signal_returns) == 0:
        return {}

    total_return = signal_returns.sum()
    thresholds = {
        ">=500%": 5.0,
        ">=1000%": 10.0,
        ">=2000%": 20.0,
    }

    result = {
        "n_signals": len(signal_returns),
        "median_return": float(signal_returns.median()),
        "mean_return": float(signal_returns.mean()),
    }

    for label, t in thresholds.items():
        mask = signal_returns >= t
        freq = mask.mean()
        contribution = signal_returns[mask].sum() / (total_return + 1e-10)
        result[f"freq_{label}"] = round(float(freq), 4)
        result[f"pnl_contribution_{label}"] = round(float(contribution), 4)

    return result
