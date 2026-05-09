"""
Analysis 7: Walk-forward stability — Sharpe, hit rate, and Kelly across folds.
Detects performance degradation over time.
"""
import pandas as pd

from ..backtest.walk_forward import FoldResult
from ..backtest.metrics import compute_fold_metrics, sharpe_ratio, pbo_score, deflated_sharpe_ratio
import numpy as np


def walk_forward_stability_analysis(
    fold_results: list[FoldResult],
    target_col: str = "rally_100_72h",
) -> dict:
    """
    For each fold, compute metrics and return stability summary.
    Also computes PBO and DSR across folds.
    """
    proba_col = f"proba_{target_col}"
    fold_summaries = []
    fold_sharpes = []

    for fr in fold_results:
        if proba_col not in fr.predictions.columns:
            continue
        if target_col not in fr.actuals.columns:
            continue

        y = fr.actuals[target_col].values
        yp = fr.predictions[proba_col].values

        m = compute_fold_metrics(y, yp)
        m["fold"] = fr.fold
        m["test_start"] = str(fr.test_start.date())
        m["test_end"] = str(fr.test_end.date())
        fold_summaries.append(m)

        if not np.isnan(m.get("sharpe", float("nan"))):
            fold_sharpes.append(m["sharpe"])

    stability_df = pd.DataFrame(fold_summaries)
    sharpe_list = [m.get("sharpe", float("nan")) for m in fold_summaries]
    sharpe_clean = [s for s in sharpe_list if not np.isnan(s)]

    n_obs = stability_df["n"].sum() if not stability_df.empty else 0
    obs_sharpe = float(np.nanmean(sharpe_clean)) if sharpe_clean else float("nan")

    return {
        "fold_metrics": stability_df,
        "pbo": pbo_score(sharpe_clean),
        "dsr": deflated_sharpe_ratio(
            sharpe_obs=obs_sharpe,
            n_trials=len(fold_results),
            n_obs=int(n_obs),
        ),
        "mean_sharpe": obs_sharpe,
        "sharpe_std": float(np.nanstd(sharpe_clean)) if sharpe_clean else float("nan"),
        "degrading": (
            len(sharpe_clean) >= 3
            and sharpe_clean[-1] < sharpe_clean[0]
        ),
    }
