# Unit tests:
#   1. No leakage from test set into bin edges
#   2. Correct WOE formula output on a synthetic small example
#   3. Rejection of non-monotonic bins.

import numpy as np
import pandas as pd
import pytest

from src.features.woe_binning import WOEBinner


# 1. No leakage: fitting on train must not be affected by what's in test,
#    and transform() on test/val must reuse train's edges and WOE unchanged
def test_transform_does_not_refit_or_leak():
    rng = np.random.default_rng(0)
    n = 2000

    x_train = pd.Series(rng.uniform(0, 100, n), name="dti")
    # np.clip(a, a_min, a_max) to keep probabilities in [a_min, a_max] to avoid extreme WOE values
    p_bad = np.clip(x_train / 150 + rng.normal(0, 0.05, n), 0.01, 0.99)
    # turn probabilities into binary target with some noise
    y_train = pd.Series((rng.uniform(size=n) < p_bad).astype(int))

    binner = WOEBinner(feature_name="dti", initial_bins=10)
    binner.fit(x_train, y_train)

    edges_after_fit = binner.bin_edges_.copy()
    woe_after_fit = dict(binner.woe_lookup_)

    # Test data: values outside train's range,
    # and a different relationship with target. If transform()
    # silently refit on this data, edges/woe would change
    x_test = pd.Series([-500.0, 1000.0, 50.0, 3.0])
    _ = binner.transform(x_test)

    np.testing.assert_array_equal(binner.bin_edges_, edges_after_fit)
    assert binner.woe_lookup_ == woe_after_fit

    # Out-of-range test values must still get the nearest bin's WOE, and no NaNs
    transformed = binner.transform(x_test)
    assert transformed.notna().all()
    assert len(transformed) == len(x_test)


def test_val_and_test_share_same_lookup_fit_only_on_train():
    rng = np.random.default_rng(1)
    n = 1500
    x_train = pd.Series(rng.uniform(0, 50, n))
    y_train = pd.Series((rng.uniform(size=n) < (x_train / 100)).astype(int))

    binner = WOEBinner(feature_name="f", initial_bins=8)
    binner.fit(x_train, y_train)

    # Two different splits - same feature value must always map
    # to the same WOE, because both use the train fitted lookup.
    val_point = pd.Series([25.0])
    test_point = pd.Series([25.0])
    assert binner.transform(val_point).iloc[0] == binner.transform(test_point).iloc[0]


# 2. Test WOE formula

def test_woe_formula_matches_hand_calculation():
    # Single bin containing everything 8 good, 2 bad out of totals
    # WOE = ln( (good+eps)/(total_good+eps) / ((bad+eps)/(total_bad+eps)) )
    good_count, bad_count = 8, 2
    total_good, total_bad = 8, 2
    eps = 0.5

    expected = np.log(
        ((good_count + eps) / (total_good + eps))
        / ((bad_count + eps) / (total_bad + eps))
    )
    assert expected == pytest.approx(0.0, abs=1e-9)

    x = pd.Series([1.0] * 10)  # all in one bin via qcut duplicates
    y = pd.Series([0] * 8 + [1] * 2)

    binner = WOEBinner(feature_name="const", initial_bins=5)
    binner.fit(x, y)

    # check if the binning produced a single bin as expected
    assert len(binner.stats_) == 1
    # check if the WOE of that bin is approximately 0 as expected
    assert binner.stats_[0].woe == pytest.approx(0.0, abs=1e-9)

    # asymmetric two population case hand computed vs through binner:
    # Population: 100 good, 50 bad total
    # Bin A: 40 good, 5 bad  -> should have positive WOE (safer than average)
    # Bin B: 60 good, 45 bad -> should have negative WOE (riskier than average)
    total_good, total_bad = 100, 50
    woe_a_expected = np.log(((40 + eps) / (total_good + eps)) / ((5 + eps) / (total_bad + eps)))
    woe_b_expected = np.log(((60 + eps) / (total_good + eps)) / ((45 + eps) / (total_bad + eps)))

    woe_a = WOEBinner._woe(40, 5, total_good, total_bad)
    woe_b = WOEBinner._woe(60, 45, total_good, total_bad)

    assert woe_a == pytest.approx(woe_a_expected, abs=1e-9)
    assert woe_b == pytest.approx(woe_b_expected, abs=1e-9)
    assert woe_a > 0 > woe_b


# 3. Non monotonic bins get merged

def test_non_monotonic_bins_are_merged_into_monotonic_sequence():
    rng = np.random.default_rng(42)

    # Feature with a zig zag bad rate pattern:
    # low x -> low risk, mid x -> artificial spike in risk, high x -> low risk again
    n_per_segment = 400
    x1 = rng.uniform(0, 10, n_per_segment)
    x2 = rng.uniform(10, 20, n_per_segment)
    x3 = rng.uniform(20, 30, n_per_segment)

    y1 = (rng.uniform(size=n_per_segment) < 0.05).astype(int)  # 5% bad rate
    y2 = (rng.uniform(size=n_per_segment) < 0.40).astype(int)  # spike 40% bad rate
    y3 = (rng.uniform(size=n_per_segment) < 0.06).astype(int)  # back down 6% bad rate

    x = pd.Series(np.concatenate([x1, x2, x3]))
    y = pd.Series(np.concatenate([y1, y2, y3]))

    binner = WOEBinner(feature_name="zigzag", initial_bins=15)
    binner.fit(x, y)

    bad_rates = [s.bad_rate for s in binner.stats_]
    diffs = np.diff(bad_rates)

    is_non_decreasing = np.all(diffs >= -1e-9)
    is_non_increasing = np.all(diffs <= 1e-9)
    assert is_non_decreasing or is_non_increasing, (
        f"Final bins are not monotonic: bad rates = {bad_rates}"
    )

    # The merge step produced fewer final bins than the initial candidate count
    assert len(binner.stats_) < binner.initial_bins
    assert len(binner.overrides_) > 0
    assert all("merged" in reason for reason in binner.overrides_)


def test_missing_values_get_their_own_woe_and_transform_cleanly():
    rng = np.random.default_rng(7)
    n = 1000
    x = pd.Series(rng.uniform(0, 100, n))
    y = pd.Series((rng.uniform(size=n) < (x / 200)).astype(int))

    # inject informative missingness missing rows are riskier
    missing_idx = rng.choice(n, size=100, replace=False)
    y.iloc[missing_idx] = (rng.uniform(size=100) < 0.5).astype(int)
    x.iloc[missing_idx] = np.nan

    binner = WOEBinner(feature_name="has_nans", initial_bins=10)
    binner.fit(x, y)

    # check that there is a WOE value for the missing bin
    assert binner.missing_woe_ is not None

    x_new = pd.Series([np.nan, 10.0, np.nan, 90.0])
    transformed = binner.transform(x_new)
    # check that transform() does not produce NaNs
    assert transformed.notna().all()
    # check that the missing values all get the same WOE as the fitted missing bin
    assert transformed.iloc[0] == transformed.iloc[2] == binner.missing_woe_
