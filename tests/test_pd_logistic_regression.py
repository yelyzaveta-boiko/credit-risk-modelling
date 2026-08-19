import numpy as np
import pandas as pd
import pytest

from src.models.pd_logistic_regression import (
    coefficient_report,
    fit_baseline_pd_model,
    quick_auc,
)


@pytest.fixture
def woe_frame():
    rng = np.random.default_rng(0)
    n = 2000

    # Two WOE like features: higher WOE = safer, so the true relationship
    # a well specified model should learn is a negative coefficient on both
    woe_a = rng.uniform(-2, 2, n)
    woe_b = rng.uniform(-2, 2, n)
    log_odds = -1.2 * woe_a - 0.8 * woe_b
    p_bad = 1 / (1 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < p_bad).astype(int)

    X = pd.DataFrame({"feature_a_woe": woe_a, "feature_b_woe": woe_b})
    return X, pd.Series(y)


def test_fit_baseline_pd_model_predicts_higher_woe_as_safer(woe_frame):
    X, y = woe_frame
    model = fit_baseline_pd_model(X, y)

    # A high WOE row should get a lower predicted PD than a low WOE row
    safe_row = pd.DataFrame({"feature_a_woe": [2.0], "feature_b_woe": [2.0]})
    risky_row = pd.DataFrame({"feature_a_woe": [-2.0], "feature_b_woe": [-2.0]})

    p_safe = model.predict_proba(safe_row)[0, 1]
    p_risky = model.predict_proba(risky_row)[0, 1]
    assert p_safe < p_risky


def test_coefficient_report_flags_expected_sign(woe_frame):
    X, y = woe_frame
    model = fit_baseline_pd_model(X, y)

    report = coefficient_report(model, list(X.columns))

    assert set(report["feature"]) == {"feature_a_woe", "feature_b_woe"}
    # Both features were constructed with a genuinely negative relationship,
    # so both should come back with the expected negative sign
    assert report["sign_as_expected"].all()
    # sorted most negative first
    assert report["coefficient"].is_monotonic_increasing


def test_quick_auc_is_above_chance_on_a_separable_signal(woe_frame):
    X, y = woe_frame
    model = fit_baseline_pd_model(X, y)

    auc = quick_auc(model, X, y)
    assert 0.5 < auc <= 1.0
