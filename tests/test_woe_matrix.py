# Unit tests:
#   1. fit_woe_encoders fits the right encoder class per feature type
#   2. transform_woe_matrix produces one *_woe column per feature, no NaNs
#   3. iv_summary ranks features by IV

import numpy as np
import pandas as pd
import pytest

from src.features.categorical_woe import CategoricalWOEEncoder
from src.features.woe_binning import WOEBinner
from src.features.woe_matrix import fit_woe_encoders, iv_summary, transform_woe_matrix


@pytest.fixture
def train_frame():
    rng = np.random.default_rng(0)
    n = 2000

    # Strongly predictive numeric feature
    x_num = rng.uniform(0, 100, n)
    p_bad = np.clip(x_num / 150, 0.01, 0.99)

    # Weakly predictive categorical feature, independent noise added
    x_cat = rng.choice(["A", "B", "C"], size=n)
    cat_bump = pd.Series(x_cat).map({"A": -0.05, "B": 0.0, "C": 0.05}).to_numpy()
    p_bad = np.clip(p_bad + cat_bump, 0.01, 0.99)

    y = (rng.uniform(size=n) < p_bad).astype(int)

    return pd.DataFrame({"num_feature": x_num, "cat_feature": x_cat, "target": y})


def test_fit_woe_encoders_uses_correct_encoder_per_feature(train_frame):
    encoders = fit_woe_encoders(
        train_frame, train_frame["target"], ["num_feature"], ["cat_feature"]
    )

    assert set(encoders) == {"num_feature", "cat_feature"}
    assert isinstance(encoders["num_feature"], WOEBinner)
    assert isinstance(encoders["cat_feature"], CategoricalWOEEncoder)


def test_transform_woe_matrix_has_one_column_per_feature_and_no_nans(train_frame):
    encoders = fit_woe_encoders(
        train_frame, train_frame["target"], ["num_feature"], ["cat_feature"]
    )
    matrix = transform_woe_matrix(train_frame, encoders)

    assert set(matrix.columns) == {"num_feature_woe", "cat_feature_woe"}
    assert len(matrix) == len(train_frame)
    assert matrix.notna().all().all()


def test_iv_summary_ranks_the_more_predictive_feature_first(train_frame):
    encoders = fit_woe_encoders(
        train_frame, train_frame["target"], ["num_feature"], ["cat_feature"]
    )
    summary = iv_summary(encoders)

    assert list(summary.columns) == ["feature", "type", "n_bins", "iv"]
    ivs = list(summary["iv"])
    assert ivs == sorted(ivs, reverse=True)
    # num_feature drives bad rate directly, so it must outrank the noisier categorical
    assert summary.iloc[0]["feature"] == "num_feature"
    assert (summary["iv"] > 0).all()
