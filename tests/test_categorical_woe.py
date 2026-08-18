# Unit tests:
#   1. No leakage from test set into the category lookup
#   2. Correct WOE formula output on a synthetic small example
#   3. Rare categories get pooled into "other"
#   4. Missing values get their own WOE
#   5. Unseen categories fall back to "other", or raise if there is none

import numpy as np
import pandas as pd
import pytest

from src.features.categorical_woe import CategoricalWOEEncoder


# 1 No leakage
def test_transform_does_not_refit_or_leak():
    rng = np.random.default_rng(0)

    # A, B, C are common (above 5% threshold);
    # D, E are rare and get pooled into "other" during fit
    x_train = pd.Series(
        ["A"] * 700 + ["B"] * 700 + ["C"] * 500 + ["D"] * 50 + ["E"] * 50, name="purpose"
    )
    bad_prob = x_train.map({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.5, "E": 0.05})
    y_train = pd.Series((rng.uniform(size=len(x_train)) < bad_prob).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="purpose")
    encoder.fit(x_train, y_train)

    lookup_after_fit = dict(encoder.category_lookup_)
    woe_after_fit = dict(encoder.woe_lookup_)

    # Test a category never seen during fit. If transform() silently
    # refit on this data, the lookup tables would change
    x_test = pd.Series(["A", "B", "never_seen_category"])
    _ = encoder.transform(x_test)

    assert encoder.category_lookup_ == lookup_after_fit
    assert encoder.woe_lookup_ == woe_after_fit

    # WOE of unseen cat from "other" bucket
    transformed = encoder.transform(x_test)
    assert transformed.notna().all()
    assert len(transformed) == len(x_test)


def test_val_and_test_share_same_lookup_fit_only_on_train():
    rng = np.random.default_rng(1)
    n = 1500
    x_train = pd.Series(rng.choice(["RENT", "OWN", "MORTGAGE"], size=n))
    bad_prob = x_train.map({"RENT": 0.3, "OWN": 0.1, "MORTGAGE": 0.15})
    y_train = pd.Series((rng.uniform(size=n) < bad_prob).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="home_ownership")
    encoder.fit(x_train, y_train)

    # Two different splits - same category = the same WOE
    val_point = pd.Series(["RENT"])
    test_point = pd.Series(["RENT"])
    assert encoder.transform(val_point).iloc[0] == encoder.transform(test_point).iloc[0]


# 2 WOE formula

def test_woe_formula_matches_hand_calculation():
    # Single category containing everything: 8 good, 2 bad
    good_count, bad_count = 8, 2
    total_good, total_bad = 8, 2
    eps = 0.5

    expected = np.log(
        ((good_count + eps) / (total_good + eps))
        / ((bad_count + eps) / (total_bad + eps))
    )
    assert expected == pytest.approx(0.0, abs=1e-9)

    x = pd.Series(["only"] * 10)
    y = pd.Series([0] * 8 + [1] * 2)

    encoder = CategoricalWOEEncoder(feature_name="const")
    encoder.fit(x, y)

    # single category produces a single bin ?
    assert len(encoder.stats_) == 1
    # WOE of that bin is approximately 0 ?
    assert encoder.stats_[0].woe == pytest.approx(0.0, abs=1e-9)

    # hand computed vs through encoder
    # Category A: 40 good, 5 bad  -> should have positive WOE (safer than average)
    # Category B: 60 good, 45 bad -> should have negative WOE (riskier than average)
    total_good, total_bad = 100, 50
    woe_a_expected = np.log(((40 + eps) / (total_good + eps)) / ((5 + eps) / (total_bad + eps)))
    woe_b_expected = np.log(((60 + eps) / (total_good + eps)) / ((45 + eps) / (total_bad + eps)))

    woe_a = CategoricalWOEEncoder._woe(40, 5, total_good, total_bad)
    woe_b = CategoricalWOEEncoder._woe(60, 45, total_good, total_bad)

    assert woe_a == pytest.approx(woe_a_expected, abs=1e-9)
    assert woe_b == pytest.approx(woe_b_expected, abs=1e-9)
    assert woe_a > 0 > woe_b


# Rare categories pooled into "other"

def test_rare_categories_are_pooled_into_other():
    rng = np.random.default_rng(42)

    # A, B, C are common; D, E are rare (< 5% of 2000 = 100 rows each)
    x = pd.Series(["A"] * 700 + ["B"] * 700 + ["C"] * 500 + ["D"] * 50 + ["E"] * 50)
    bad_prob = x.map({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.5, "E": 0.05})
    y = pd.Series((rng.uniform(size=len(x)) < bad_prob).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="purpose")
    encoder.fit(x, y)

    # 3 common categories + 1 "other" bucket
    assert len(encoder.stats_) == 4
    assert encoder.other_bin_id_ is not None
    assert len(encoder.overrides_) > 0
    assert all("pooled" in reason for reason in encoder.overrides_)

    other_stats = next(s for s in encoder.stats_ if s.bin_id == encoder.other_bin_id_)
    assert set(other_stats.categories) == {"D", "E"}
    assert other_stats.count == 100


def test_no_pooling_when_every_category_is_common():
    rng = np.random.default_rng(3)
    n = 900
    x = pd.Series(rng.choice(["A", "B", "C"], size=n))
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="term")
    encoder.fit(x, y)

    assert encoder.other_bin_id_ is None
    assert encoder.overrides_ == []
    assert len(encoder.stats_) == 3


# missing values with their own WOE

def test_missing_values_get_their_own_woe_and_transform_cleanly():
    rng = np.random.default_rng(7)
    n = 1000
    x = pd.Series(rng.choice(["A", "B", "C"], size=n))
    bad_prob = x.map({"A": 0.1, "B": 0.3, "C": 0.5})
    y = pd.Series((rng.uniform(size=n) < bad_prob).astype(int))

    # inject informative missingness: missing rows are riskier
    missing_idx = rng.choice(n, size=100, replace=False)
    y.iloc[missing_idx] = (rng.uniform(size=100) < 0.9).astype(int)
    x.iloc[missing_idx] = np.nan

    encoder = CategoricalWOEEncoder(feature_name="has_nans")
    encoder.fit(x, y)

    # WOE value for the missing bin
    assert encoder.missing_woe_ is not None

    x_new = pd.Series([np.nan, "A", np.nan, "B"])
    transformed = encoder.transform(x_new)
    # transform() does not produce NaNs
    assert transformed.notna().all()
    # missing values all get the same WOE as the fitted missing bin
    assert transformed.iloc[0] == transformed.iloc[2] == encoder.missing_woe_


# Unseen categories fall back to "other"

def test_unseen_category_falls_back_to_other_bucket():
    rng = np.random.default_rng(42)
    x = pd.Series(["A"] * 700 + ["B"] * 700 + ["C"] * 500 + ["D"] * 50 + ["E"] * 50)
    bad_prob = x.map({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.5, "E": 0.05})
    y = pd.Series((rng.uniform(size=len(x)) < bad_prob).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="purpose")
    encoder.fit(x, y)

    other_woe = encoder.woe_lookup_[encoder.other_bin_id_]
    transformed = encoder.transform(pd.Series(["brand_new_category"]))
    assert transformed.iloc[0] == pytest.approx(other_woe)


def test_unseen_category_without_other_bucket_raises():
    rng = np.random.default_rng(3)
    n = 900
    x = pd.Series(rng.choice(["A", "B", "C"], size=n))
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="term")
    encoder.fit(x, y)

    with pytest.raises(ValueError, match="unseen category"):
        encoder.transform(pd.Series(["never_seen"]))


# Refitting the same instance must not leak state from a previous fit

def test_refit_resets_other_bucket_and_overrides():
    rng = np.random.default_rng(42)

    # First fit: has rare categories, so other_bin_id_/overrides_ get populated
    x_with_rare = pd.Series(["A"] * 700 + ["B"] * 700 + ["C"] * 500 + ["D"] * 50 + ["E"] * 50)
    bad_prob = x_with_rare.map({"A": 0.1, "B": 0.3, "C": 0.2, "D": 0.5, "E": 0.05})
    y_with_rare = pd.Series((rng.uniform(size=len(x_with_rare)) < bad_prob).astype(int))

    encoder = CategoricalWOEEncoder(feature_name="purpose")
    encoder.fit(x_with_rare, y_with_rare)
    assert encoder.other_bin_id_ is not None
    assert encoder.overrides_ != []

    # Refit the same instance on data with no rare categories
    n = 900
    x_no_rare = pd.Series(rng.choice(["A", "B", "C"], size=n))
    y_no_rare = pd.Series((rng.uniform(size=n) < 0.2).astype(int))
    encoder.fit(x_no_rare, y_no_rare)

    # Stale state from the first fit must not leak into the second
    assert encoder.other_bin_id_ is None
    assert encoder.overrides_ == []
    with pytest.raises(ValueError, match="unseen category"):
        encoder.transform(pd.Series(["never_seen"]))
