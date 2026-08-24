import numpy as np
import pandas as pd
import pytest

from src.models.pd_lightgbm import (
    feature_importance_report,
    fit_lightgbm_pd_model,
    prepare_features,
    quick_auc,
)


@pytest.fixture
def raw_frame():
    rng = np.random.default_rng(0)
    n = 2000

    # A strongly predictive numeric feature, a weak categorical one, and a
    # column with missing values
    numeric = rng.uniform(0, 100, n)
    category = pd.Series(rng.choice(["A", "B", "C"], size=n)).astype("category")
    with_missing = rng.uniform(0, 1, n)
    with_missing[rng.choice(n, size=100, replace=False)] = np.nan

    p_bad = np.clip(numeric / 150, 0.01, 0.99)
    y = pd.Series((rng.uniform(size=n) < p_bad).astype(int))

    X = pd.DataFrame({"numeric": numeric, "category": category, "with_missing": with_missing})
    return X, y


def test_prepare_features_selects_columns_and_casts_categoricals(raw_frame):
    X, _ = raw_frame
    raw = X.assign(category=X["category"].astype(str), extra_column="unused")

    prepared = prepare_features(raw, ["numeric", "category"], ["category"])

    assert list(prepared.columns) == ["numeric", "category"]
    assert prepared["category"].dtype.name == "category"
    assert prepared["numeric"].dtype == raw["numeric"].dtype
    # the original frame must be untouched
    assert raw["category"].dtype.name != "category"


def test_fit_lightgbm_pd_model_handles_categorical_and_missing_columns(raw_frame):
    X, y = raw_frame
    # Should not raise even if the category dtype column and NaNs
    model = fit_lightgbm_pd_model(X, y, n_estimators=20)

    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_feature_importance_report_ranks_predictive_feature_first(raw_frame):
    X, y = raw_frame
    model = fit_lightgbm_pd_model(X, y, n_estimators=50)

    report = feature_importance_report(model, list(X.columns))

    assert set(report["feature"]) == {"numeric", "category", "with_missing"}
    assert report.iloc[0]["feature"] == "numeric"
    assert report["importance"].is_monotonic_decreasing


def test_quick_auc_is_above_chance_on_a_separable_signal(raw_frame):
    X, y = raw_frame
    model = fit_lightgbm_pd_model(X, y, n_estimators=50)

    auc = quick_auc(model, X, y)
    assert 0.5 < auc <= 1.0
