import numpy as np
import pandas as pd
import pytest

from src.evaluation.pd_metrics import calibration_table, gini_coefficient, ks_statistic


def test_ks_and_gini_are_one_under_perfect_separation():
    y_true = pd.Series([0] * 500 + [1] * 500)
    y_pred_proba = np.array(y_true, dtype=float)

    assert ks_statistic(y_true, y_pred_proba) == pytest.approx(1.0)
    assert gini_coefficient(y_true, y_pred_proba) == pytest.approx(1.0)


def test_ks_and_gini_are_between_chance_and_perfect_under_partial_separation():
    rng = np.random.default_rng(0)
    n = 2000
    score = rng.uniform(0, 1, n)
    p_bad = np.clip(score + rng.normal(0, 0.25, n), 0.01, 0.99)
    y_true = pd.Series((rng.uniform(size=n) < p_bad).astype(int))

    ks = ks_statistic(y_true, score)
    gini = gini_coefficient(y_true, score)

    assert 0.0 < ks < 1.0
    assert 0.0 < gini < 1.0


def test_calibration_table_reports_predicted_vs_actual_by_decile():
    rng = np.random.default_rng(1)
    n = 5000

    score = rng.uniform(0, 1, n)
    p_bad = np.clip(score + rng.normal(0, 0.1, n), 0.01, 0.99)
    y_true = pd.Series((rng.uniform(size=n) < p_bad).astype(int))

    table = calibration_table(y_true, score, n_bins=10)

    assert list(table.columns) == ["decile", "count", "predicted_pd", "actual_default_rate", "gap"]
    assert table["count"].sum() == n
    assert table["predicted_pd"].is_monotonic_increasing
    assert table["actual_default_rate"].iloc[-1] > table["actual_default_rate"].iloc[0]
