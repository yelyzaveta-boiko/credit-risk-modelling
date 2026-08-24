# PD logistic regression evaluation on val and test: AUC, KS,
# Gini, and a calibration table

# requires data/processed/woe_{val,test}.csv.

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import SELECTED_FEATURES  # noqa: E402
from src.evaluation.pd_metrics import evaluate_pd_model  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [f"{feat}_woe" for feat in SELECTED_FEATURES]


def main() -> None:
    model = joblib.load(MODELS_DIR / "baseline_logistic_regression.joblib")

    for split_name in ["val", "test"]:
        split = pd.read_csv(PROCESSED_DIR / f"woe_{split_name}.csv")
        X, y = split[FEATURE_COLUMNS], split["target"]
        proba = model.predict_proba(X)[:, 1]

        result = evaluate_pd_model(y, proba)

        print(f"\n{'=' * 60}\n{split_name}\n{'=' * 60}")
        print(f"AUC:  {result['auc']:.4f}")
        print(f"KS:   {result['ks']:.4f}")
        print(f"Gini: {result['gini']:.4f}")
        print("\nCalibration (decile 0 = lowest predicted risk):")
        print(result["calibration"].to_string(index=False))


if __name__ == "__main__":
    main()
