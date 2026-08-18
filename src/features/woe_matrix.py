# WOE feature matrix
# for numeric features via WOEBinner, categorical features via
# CategoricalWOEEncoder
# fit on the train split

from __future__ import annotations

import pandas as pd

from src.features.categorical_woe import (
    CategoricalWOEEncoder,
    fit_categorical_features,
    transform_with_categorical_encoders,
)
from src.features.woe_binning import WOEBinner, fit_baseline_features, transform_with_binners

WOEEncoder = WOEBinner | CategoricalWOEEncoder


def fit_woe_encoders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, WOEEncoder]:
    # Fit one encoder per baseline feature
    encoders: dict[str, WOEEncoder] = {}
    encoders.update(fit_baseline_features(X_train, y_train, numeric_features))
    encoders.update(fit_categorical_features(X_train, y_train, categorical_features))
    return encoders


def transform_woe_matrix(X: pd.DataFrame, encoders: dict[str, WOEEncoder]) -> pd.DataFrame:
    numeric_encoders = {f: e for f, e in encoders.items() if isinstance(e, WOEBinner)}
    categorical_encoders = {
        f: e for f, e in encoders.items() if isinstance(e, CategoricalWOEEncoder)
    }
    numeric_part = transform_with_binners(X, numeric_encoders)
    categorical_part = transform_with_categorical_encoders(X, categorical_encoders)
    return pd.concat([numeric_part, categorical_part], axis=1)


def iv_summary(encoders: dict[str, WOEEncoder]) -> pd.DataFrame:
    # Rank features by Information Value
    # IV score:
    # <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious
    rows = [
        {
            "feature": feat,
            "type": "numeric" if isinstance(enc, WOEBinner) else "categorical",
            "n_bins": len(enc.stats_),
            "iv": round(enc.iv(), 4),
        }
        for feat, enc in encoders.items()
    ]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
