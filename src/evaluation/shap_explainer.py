# SHAP explanations for the LightGBM PD model

# - Global importance: averaging |SHAP value| per feature across many
#  loans ranks which features actually moved predictions
# - Local accuracy: for any single loan, expected_value + sum(that
# loan's SHAP values) reconstructs the model's raw prediction
# - answers why did this specific loan get this particular score?

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    report = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    return report.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_instance(shap_values: np.ndarray, X: pd.DataFrame, index: int) -> pd.DataFrame:
    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "value": X.iloc[index].to_numpy(),
            "shap_contribution": shap_values[index],
        }
    )
    return (
        explanation.assign(_abs=explanation["shap_contribution"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )


def log_odds_to_probability(log_odds: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-log_odds))
