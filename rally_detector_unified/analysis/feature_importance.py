"""
Analysis 8: Feature importance from Lasso coefficients.
"""
import numpy as np
import pandas as pd

from ..scoring.logistic_l1 import TrainedModel
from ..scoring.feature_pipeline import FEATURE_COLUMNS


def feature_importance_analysis(
    models: dict[str, TrainedModel],
) -> pd.DataFrame:
    """
    For each trained model, extract L1 coefficients and return sorted by |coef|.
    Returns long-format DataFrame with columns:
        [target, feature, coefficient, abs_coef, rank]
    """
    records = []

    for target, model in models.items():
        pipeline = model.pipeline
        # Get the logistic model from the pipeline
        if "model" not in pipeline.named_steps:
            continue
        lr = pipeline.named_steps["model"]

        if not hasattr(lr, "coef_"):
            continue

        coefs = lr.coef_[0]

        # Feature names after imputer adds indicator columns
        imputer = pipeline.named_steps.get("imputer")
        if imputer is not None and hasattr(imputer, "get_feature_names_out"):
            feat_names = list(imputer.get_feature_names_out(FEATURE_COLUMNS))
        else:
            feat_names = FEATURE_COLUMNS[:len(coefs)]

        # Pad/trim to match coef length
        feat_names = feat_names[:len(coefs)]

        for feat, coef in zip(feat_names, coefs):
            if abs(coef) > 1e-6:  # only non-zero (Lasso sparsity)
                records.append({
                    "target": target,
                    "feature": feat,
                    "coefficient": round(float(coef), 6),
                    "abs_coef": round(abs(float(coef)), 6),
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["rank"] = df.groupby("target")["abs_coef"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["target", "rank"])
