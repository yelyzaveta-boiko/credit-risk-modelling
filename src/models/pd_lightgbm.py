# PD model using LightGBM gradient boosted trees on the raw baseline features

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def fit_lightgbm_pd_model(
    X_train: pd.DataFrame, y_train: pd.Series, **kwargs
) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "random_state": 42,
        "importance_type": "gain",
        "verbosity": -1,
        **kwargs,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def quick_auc(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)


def feature_importance_report(
    model: lgb.LGBMClassifier, feature_names: list[str]
) -> pd.DataFrame:
    report = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    return report.sort_values("importance", ascending=False).reset_index(drop=True)
