"""
Backtest metrics: hit rate, Kelly, Sharpe, max drawdown, PBO, DSR.
All functions are stateless and operate on arrays/Series.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)


# ─── Basic metrics ────────────────────────────────────────────────────────────

def hit_rate(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    predicted_pos = y_pred_proba >= threshold
    if predicted_pos.sum() == 0:
        return float("nan")
    return float(y_true[predicted_pos].mean())


def precision_at_decile(y_true: np.ndarray, y_pred_proba: np.ndarray, decile: float = 0.9) -> float:
    """Hit rate in the top `decile` fraction of predictions."""
    cutoff = np.quantile(y_pred_proba, decile)
    mask = y_pred_proba >= cutoff
    if mask.sum() == 0:
        return float("nan")
    return float(y_true[mask].mean())


def base_rate(y_true: np.ndarray) -> float:
    return float(np.nanmean(y_true))


# ─── Return-based metrics ─────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """Annualized Sharpe (assuming hourly returns → 8760 periods/year)."""
    r = returns[~np.isnan(returns)]
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a fraction (0 to 1)."""
    if len(equity_curve) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / (peak + 1e-10)
    return float(dd.min())


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion: f = (W/L * p - (1 - p)) / (W/L)
    where p = win_rate, W = avg_win, L = avg_loss (positive value).
    Returns fraction capped at 0.25 (quarter-Kelly safety).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    f = (b * p - (1 - p)) / b
    return float(np.clip(f, 0, 0.25))


# ─── PBO — Probability of Backtest Overfitting ────────────────────────────────

def pbo_score(fold_sharpes: list[float]) -> float:
    """
    Simplified PBO: fraction of all pairings where the best in-sample fold
    is NOT the best out-of-sample fold.
    Full Bailey et al. 2016 CSCV requires more data; this is a quick proxy.
    """
    n = len(fold_sharpes)
    if n < 2:
        return float("nan")
    overfit_count = 0
    total = 0
    for i, j in combinations(range(n), 2):
        total += 1
        # "In-sample best" = fold i, "out-of-sample" = fold j
        if fold_sharpes[i] > fold_sharpes[j]:
            overfit_count += 1
    return overfit_count / total if total > 0 else float("nan")


# ─── DSR — Deflated Sharpe Ratio ─────────────────────────────────────────────

def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Bailey & López de Prado (2014) DSR.
    Deflates observed Sharpe by expected maximum Sharpe over n_trials strategies.
    """
    if n_obs <= 1 or n_trials <= 1:
        return float("nan")
    # Euler-Mascheroni constant (not exposed by scipy.stats.norm)
    gamma = 0.5772156649015329
    # Expected maximum Sharpe under n_trials iid trials
    sr_max = (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    # Variance of Sharpe ratio
    sr_var = (1 + 0.5 * sharpe_obs ** 2 - skew * sharpe_obs + (kurt - 3) / 4 * sharpe_obs ** 2) / n_obs
    z = (sharpe_obs - sr_max) / (np.sqrt(sr_var) + 1e-10)
    return float(stats.norm.cdf(z))


# ─── Fold summary ─────────────────────────────────────────────────────────────

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> dict:
    """Compute all metrics for a single fold's predictions."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    yt = y_true[mask]
    yp = y_pred_proba[mask]

    metrics = {
        "n": int(mask.sum()),
        "base_rate": base_rate(yt),
        "hit_rate_50": hit_rate(yt, yp, threshold=0.5),
        "hit_rate_70": hit_rate(yt, yp, threshold=0.7),
        "precision_top10pct": precision_at_decile(yt, yp, decile=0.9),
        "lift_top10pct": (
            precision_at_decile(yt, yp, 0.9) / (base_rate(yt) + 1e-10)
            if base_rate(yt) > 0 else float("nan")
        ),
    }

    if returns is not None and len(returns) > 0:
        r = returns[mask]
        metrics["sharpe"] = sharpe_ratio(r)
        metrics["max_drawdown"] = max_drawdown(np.cumprod(1 + np.nan_to_num(r)))

        pos_mask = yt == 1
        neg_mask = yt == 0
        avg_win = float(np.nanmean(r[pos_mask])) if pos_mask.sum() > 0 else 0.0
        avg_loss = float(abs(np.nanmean(r[neg_mask]))) if neg_mask.sum() > 0 else 1.0
        metrics["kelly"] = kelly_fraction(float(yt.mean()), avg_win, avg_loss)

    return metrics
