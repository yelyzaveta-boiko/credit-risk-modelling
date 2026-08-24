# Train PD model using LightGBM gradient boosted trees
# Requires data/processed/raw_{train,val}.csv

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import (  # noqa: E402
    SELECTED_CATEGORICAL_FEATURES,
    SELECTED_FEATURES,
)
from src.models.pd_lightgbm import (  # noqa: E402
    feature_importance_report,
    fit_lightgbm_pd_model,
    quick_auc,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SELECTED_FEATURES].copy()
    # native LightGBM support for categorical columns requires the "category" dtype
    for col in SELECTED_CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "raw_train.csv")
    val = pd.read_csv(PROCESSED_DIR / "raw_val.csv")

    X_train, y_train = prepare_features(train), train["target"]
    X_val, y_val = prepare_features(val), val["target"]

    model = fit_lightgbm_pd_model(X_train, y_train)

    print("Feature importance report by split count:\n")
    print(feature_importance_report(model, SELECTED_FEATURES).to_string(index=False))

    train_auc = quick_auc(model, X_train, y_train)
    val_auc = quick_auc(model, X_val, y_val)
    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Val AUC:   {val_auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "lightgbm_pd_model.joblib"
    joblib.dump(model, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
