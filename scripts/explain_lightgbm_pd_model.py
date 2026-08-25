# SHAP interpretability for the LightGBM PD model

# Requires models/lightgbm_pd_model.joblib and data/processed/raw_val.csv

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import (  # noqa: E402
    SELECTED_CATEGORICAL_FEATURES,
    SELECTED_FEATURES,
)
from src.evaluation.shap_explainer import (  # noqa: E402
    compute_shap_values,
    explain_instance,
    global_importance,
    log_odds_to_probability,
)
from src.models.pd_lightgbm import prepare_features  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    model = joblib.load(MODELS_DIR / "lightgbm_pd_model.joblib")
    val = pd.read_csv(PROCESSED_DIR / "raw_val.csv")
    X = prepare_features(val, SELECTED_FEATURES, SELECTED_CATEGORICAL_FEATURES)

    shap_values, expected_value = compute_shap_values(model, X)

    print("Global feature importance (mean |SHAP value|):\n")
    print(global_importance(shap_values, SELECTED_FEATURES).to_string(index=False))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False)
    plot_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary plot to {plot_path}")

    # Example: explain the single riskiest-predicted loan in this split
    proba = model.predict_proba(X)[:, 1]
    riskiest_idx = int(proba.argmax())

    print(f"\n{'=' * 60}")
    print(f"Explaining loan at row {riskiest_idx} (highest predicted PD in this split)")
    print(f"{'=' * 60}")
    print(f"Base rate (expected value): {log_odds_to_probability(expected_value):.2%}")
    print(f"This loan's predicted PD:   {proba[riskiest_idx]:.2%}")
    print("\nPer feature contribution with most influential first:\n")
    print(explain_instance(shap_values, X, riskiest_idx).to_string(index=False))


if __name__ == "__main__":
    main()
