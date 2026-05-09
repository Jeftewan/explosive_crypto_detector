"""
Optional Plotly HTML dashboard for interactive exploration.
Only imported if plotly is available.
"""
import logging
from pathlib import Path

import pandas as pd

from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)


def _require_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        return go, px, make_subplots
    except ImportError:
        raise ImportError("plotly is required for HTML dashboard. Run: pip install plotly")


def generate_html_dashboard(
    fold_metrics: pd.DataFrame,
    score_buckets: dict,
    feature_importance: pd.DataFrame,
    multi_profile: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Generate an interactive Plotly HTML dashboard.
    Returns the path to the generated file.
    """
    go, px, make_subplots = _require_plotly()

    if output_path is None:
        from datetime import datetime, timezone
        output_path = REPORTS_DIR / f"dashboard_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.html"

    figs = []

    # ── Fold Sharpe over time ─────────────────────────────────────────────────
    if not fold_metrics.empty and "sharpe" in fold_metrics.columns:
        fig = px.bar(
            fold_metrics,
            x="fold",
            y="sharpe",
            title="Walk-Forward: Sharpe Ratio per Fold",
            labels={"fold": "Fold", "sharpe": "Sharpe Ratio"},
            color="sharpe",
            color_continuous_scale="RdYlGn",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # ── Calibration plot (first target) ──────────────────────────────────────
    if score_buckets:
        first_target = list(score_buckets.keys())[0]
        cal_df = score_buckets[first_target]
        if not cal_df.empty and "bucket_mid" in cal_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cal_df["bucket_mid"].astype(str),
                y=cal_df["actual_hit_rate"],
                name="Actual Hit Rate",
            ))
            fig.add_trace(go.Scatter(
                x=cal_df["bucket_mid"].astype(str),
                y=cal_df["expected_hit_rate"],
                name="Perfect Calibration",
                mode="lines",
                line={"dash": "dash"},
            ))
            fig.update_layout(title=f"Calibration: {first_target}", xaxis_title="Predicted Proba", yaxis_title="Actual Hit Rate")
            figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ── Feature importance ────────────────────────────────────────────────────
    if not feature_importance.empty:
        first_target = feature_importance["target"].iloc[0]
        fi = feature_importance[feature_importance["target"] == first_target].head(15)
        fig = px.bar(
            fi,
            x="abs_coef",
            y="feature",
            orientation="h",
            title=f"Top Features — {first_target}",
            labels={"abs_coef": "|Coefficient|", "feature": "Feature"},
            color="coefficient",
            color_continuous_scale="RdBu",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # ── Profile lift ──────────────────────────────────────────────────────────
    if not multi_profile.empty and "lift" in multi_profile.columns:
        fig = px.bar(
            multi_profile.sort_values("lift", ascending=False).head(20),
            x="profile",
            y="lift",
            color="target",
            barmode="group",
            title="Profile Lift vs Base Rate",
        )
        fig.update_layout(xaxis_tickangle=-45)
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Rally Detector Dashboard</title></head>
<body>
<h1>Rally Detector Unified — Interactive Dashboard</h1>
{''.join(figs)}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)
    return output_path
