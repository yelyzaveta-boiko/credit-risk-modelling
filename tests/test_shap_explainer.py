# Unit tests:
#   1. global_importance ranks the genuinely predictive feature first
#   2. explain_instance reconstructs the model's raw prediction for a single row
#   3. log_odds_to_probability matches a hand computed sigmoid

import numpy as np
import pandas as pd
import pytest

from src.evaluation.shap_explainer import (
    compute_shap_values,
    explain_instance,
    global_importance,
    log_odds_to_probability,
)
from src.models.pd_lightgbm import fit_lightgbm_pd_model


@pytest.fixture
def raw_frame():
    rng = np.random.default_rng(0)
    n = 1000

    # A strongly predictive numeric feature, a weak categorical one, and a
    # column with no relationship to the target
    numeric = rng.uniform(0, 100, n)
    category = pd.Series(rng.choice(["A", "B", "C"], size=n)).astype("category")
    noise = rng.uniform(0, 1, n)

    p_bad = np.clip(numeric / 150, 0.01, 0.99)
    y = pd.Series((rng.uniform(size=n) < p_bad).astype(int))

    X = pd.DataFrame({"numeric": numeric, "category": category, "noise": noise})
    return X, y


def test_global_importance_ranks_predictive_feature_first(raw_frame):
    X, y = raw_frame
    model = fit_lightgbm_pd_model(X, y, n_estimators=50)

    shap_values, _ = compute_shap_values(model, X)
    report = global_importance(shap_values, list(X.columns))

    assert set(report["feature"]) == {"numeric", "category", "noise"}
    assert report.iloc[0]["feature"] == "numeric"
    assert report["mean_abs_shap"].is_monotonic_decreasing


def test_explain_instance_reconstructs_the_raw_prediction(raw_frame):
    X, y = raw_frame
    model = fit_lightgbm_pd_model(X, y, n_estimators=50)

    shap_values, expected_value = compute_shap_values(model, X)
    row = 7

    explanation = explain_instance(shap_values, X, row)

    # SHAP's local accuracy property: base value + this row's contributions
    # must equal the model's actual raw log-odds prediction for that row
    raw_prediction = model.predict(X.iloc[[row]], raw_score=True)[0]
    reconstructed = expected_value + explanation["shap_contribution"].sum()
    assert reconstructed == pytest.approx(raw_prediction, abs=1e-6)

    # most influential feature first
    assert set(explanation["feature"]) == {"numeric", "category", "noise"}
    abs_contrib = explanation["shap_contribution"].abs()
    assert abs_contrib.is_monotonic_decreasing


def test_log_odds_to_probability_matches_hand_computed_sigmoid():
    assert log_odds_to_probability(0.0) == pytest.approx(0.5)
    assert log_odds_to_probability(np.array([0.0, 1.0])) == pytest.approx(
        [0.5, 1 / (1 + np.exp(-1.0))]
    )
