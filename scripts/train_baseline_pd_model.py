# Train the baseline PD model
#
# Run: python -m scripts.train_baseline_pd_model
# Requires data/processed/woe_{train,val}.csv

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import SELECTED_FEATURES  # noqa: E402
from src.models.pd_logistic_regression import (  # noqa: E402
    coefficient_report,
    fit_baseline_pd_model,
    quick_auc,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [f"{feat}_woe" for feat in SELECTED_FEATURES]


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "woe_train.csv")
    val = pd.read_csv(PROCESSED_DIR / "woe_val.csv")

    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_val, y_val = val[FEATURE_COLUMNS], val["target"]

    model = fit_baseline_pd_model(X_train, y_train)

    print("Coefficient report:\n")
    print(coefficient_report(model, FEATURE_COLUMNS).to_string(index=False))

    train_auc = quick_auc(model, X_train, y_train)
    val_auc = quick_auc(model, X_val, y_val)
    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Val AUC:   {val_auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    joblib.dump(model, out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
