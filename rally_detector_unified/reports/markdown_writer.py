"""
Generates the full backtest report in Markdown format.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import REPORTS_DIR, TARGETS


def _section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n\n"


def _df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    return df.head(max_rows).to_markdown(index=False) + "\n"


def generate_report(
    fold_results,
    stability: dict,
    score_buckets: dict,
    multi_profile: pd.DataFrame,
    pre_explosion: pd.DataFrame,
    risk_reward: dict,
    top_explosions: pd.DataFrame,
    correlations: pd.DataFrame,
    feature_importance: pd.DataFrame,
    optimal_holding: pd.DataFrame,
    regime_breakdown: pd.DataFrame,
    rally_types: pd.DataFrame,
    ground_truth: pd.DataFrame,
    output_path: Path | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append(f"# Rally Detector Unified — Backtest Report\n")
    lines.append(f"**Generated:** {now}\n")
    lines.append(f"**Targets:** {len(TARGETS)} models × {len(fold_results)} folds\n")

    # ── Walk-forward stability ────────────────────────────────────────────────
    lines.append(_section("Walk-Forward Stability"))
    lines.append(f"- **Mean Sharpe:** {stability.get('mean_sharpe', 'N/A'):.3f}\n")
    lines.append(f"- **Sharpe Std:** {stability.get('sharpe_std', 'N/A'):.3f}\n")
    lines.append(f"- **PBO:** {stability.get('pbo', 'N/A'):.3f}  _(lower = less overfit)_\n")
    lines.append(f"- **DSR:** {stability.get('dsr', 'N/A'):.3f}  _(higher = more robust)_\n")
    lines.append(f"- **Degrading over time?** {'⚠️ Yes' if stability.get('degrading') else '✅ No'}\n\n")
    lines.append(_df_to_md(stability.get("fold_metrics", pd.DataFrame())))

    # ── Score buckets (calibration) ───────────────────────────────────────────
    lines.append(_section("Calibration: Score Buckets vs Hit Rate"))
    for target, df in (score_buckets or {}).items():
        lines.append(f"\n**{target}**\n\n")
        lines.append(_df_to_md(df))

    # ── Multi-profile ─────────────────────────────────────────────────────────
    lines.append(_section("Multi-Indicator Profiles"))
    lines.append(_df_to_md(multi_profile))

    # ── Pre-explosion footprint ───────────────────────────────────────────────
    lines.append(_section("Pre-Explosion Indicator Footprint"))
    lines.append(_df_to_md(pre_explosion.head(15) if not pre_explosion.empty else pre_explosion))

    # ── SPECIAL: 1 de 10 que sube x20 ────────────────────────────────────────
    lines.append(_section("🎰 Special Analysis: '1 of 10 That Rises ×20'"))
    if risk_reward:
        lines.append(f"- **Signals above threshold:** {risk_reward.get('n_signals')}\n")
        lines.append(f"- **Hit rate:** {risk_reward.get('hit_rate', 0):.1%}\n")
        lines.append(f"- **Expected PnL (median basket):** ${risk_reward.get('return_distribution', {}).get('p50', 0):,.0f}\n")
        lines.append(f"- **P(PnL > 0):** {risk_reward.get('prob_pnl_positive', 0):.1%}\n")
        lines.append(f"- **P(PnL > +50%):** {risk_reward.get('prob_pnl_50pct', 0):.1%}\n")
        lines.append(f"- **P(PnL > +200%):** {risk_reward.get('prob_pnl_200pct', 0):.1%}\n")
        dist = risk_reward.get("return_distribution", {})
        if dist:
            lines.append("\n**Return distribution (basket PnL):**\n\n")
            lines.append("| Percentile | PnL |\n|---|---|\n")
            for k, v in dist.items():
                lines.append(f"| {k} | ${v:,.0f} |\n")
    lines.append("\n")

    # ── Top explosions ────────────────────────────────────────────────────────
    lines.append(_section("Top Explosions"))
    lines.append(_df_to_md(top_explosions))

    # ── Correlations ──────────────────────────────────────────────────────────
    lines.append(_section("Feature Correlations with Target"))
    lines.append(_df_to_md(correlations.head(20) if not correlations.empty else correlations))

    # ── Feature importance ────────────────────────────────────────────────────
    lines.append(_section("Feature Importance (Lasso Coefficients)"))
    lines.append(_df_to_md(feature_importance.head(30) if not feature_importance.empty else feature_importance))

    # ── Optimal holding ───────────────────────────────────────────────────────
    lines.append(_section("Optimal Holding Period by Profile"))
    lines.append(_df_to_md(optimal_holding))

    # ── Market regime ─────────────────────────────────────────────────────────
    lines.append(_section("Performance by BTC Market Regime"))
    lines.append(_df_to_md(regime_breakdown))

    # ── Rally type breakdown ──────────────────────────────────────────────────
    lines.append(_section("Rally Type Breakdown (A/B/C)"))
    lines.append(_df_to_md(rally_types))

    # ── Ground truth validation ───────────────────────────────────────────────
    lines.append(_section("Ground Truth Validation (13 Real Trades)"))
    lines.append("> ⚠️ **Limitation:** n=13 is too small for statistical conclusions. Sanity check only.\n\n")
    lines.append(_df_to_md(ground_truth))

    # ── Limitations ───────────────────────────────────────────────────────────
    lines.append(_section("Known Limitations"))
    lines.append("""1. **No fees/slippage** — returns are gross.
2. **Survivorship bias** — only symbols listed today. Delisted tokens not included.
3. **OI/L-S/Taker only last 30 days** — signal quality for those indicators validated on 30d only.
4. **Single market cycle** — 365 days may capture only one bull/bear regime.
5. **user_history: 13 trades** — sanity check, not validation.
6. **Predictive, not causal** — learned correlations may break if market structure changes.
""")

    report = "".join(lines)

    if output_path is None:
        output_path = REPORTS_DIR / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.md"

    output_path.write_text(report, encoding="utf-8")
    return report
