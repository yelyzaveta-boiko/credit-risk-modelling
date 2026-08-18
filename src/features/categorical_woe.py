# WOE for categorical features

#  1. Each raw category value starts as its own candidate bucket.
#  2. Categories that cover less than min_bin_fraction on its own of
#     training rows get pooled together into a single "other" bucket
#  3. Save the final category -> bucket -> WOE lookup table
#  A category never seen during fit falls back to the "other" bucket if one exists


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPSILON = 0.5

MISSING_BIN_LABEL = "missing"
OTHER_BIN_LABEL = "other"


@dataclass
class CategoryBinStats:
    bin_id: int
    categories: list[str]
    count: int
    bad_count: int
    good_count: int
    bad_rate: float
    woe: float

# Fits WOE buckets for a single categorical feature on training data,
# then applies that same lookup table to any other split.
@dataclass
class CategoricalWOEEncoder:

    # Usage
    # -----
    # encoder = CategoricalWOEEncoder(feature_name="purpose")
    # encoder.fit(X_train["purpose"], y_train)
    # woe_train = encoder.transform(X_train["purpose"])
    # woe_val = encoder.transform(X_val["purpose"])      # uses train's lookup
    # woe_test = encoder.transform(X_test["purpose"])    # uses train's lookup

    feature_name: str
    min_bin_fraction: float = 0.05
    category_lookup_: dict[str, int] | None = field(default=None, init=False)
    woe_lookup_: dict[int, float] | None = field(default=None, init=False)
    stats_: list[CategoryBinStats] | None = field(default=None, init=False)
    missing_woe_: float | None = field(default=None, init=False)
    other_bin_id_: int | None = field(default=None, init=False)
    overrides_: list[str] = field(default_factory=list, init=False)

    # fit: learn category -> WOE lookup from only training data

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> CategoricalWOEEncoder:
        # Reset learned state in case fit() is called more than once on the same
        # instance - other_bin_id_ and overrides_ are only ever appended/set
        # conditionally below, so a stale value could otherwise leak from a
        # previous fit into this one.
        self.category_lookup_ = None
        self.woe_lookup_ = None
        self.stats_ = None
        self.missing_woe_ = None
        self.other_bin_id_ = None
        self.overrides_ = []

        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        missing_mask = x_train.isna()
        x_obs = x_train[~missing_mask].astype(str)
        y_obs = y_train[~missing_mask]

        if x_obs.empty:
            raise ValueError(f"'{self.feature_name}' has no observed non missing values to bin.")

        total_good = int((y_obs == 0).sum())
        total_bad = int((y_obs == 1).sum())
        if total_good == 0 or total_bad == 0:
            raise ValueError(
                f"'{self.feature_name}' training slice has only one class present, "
                "cannot compute WOE"
            )

        stats = self._compute_category_stats(x_obs, y_obs, total_good, total_bad)

        min_count = int(self.min_bin_fraction * len(x_obs))
        stats = self._pool_rare_categories(stats, min_count, total_good, total_bad)

        self.category_lookup_ = {cat: s.bin_id for s in stats for cat in s.categories}
        self.woe_lookup_ = {s.bin_id: s.woe for s in stats}
        self.stats_ = stats

        # Handling missing values
        if missing_mask.any():
            n_missing_bad = int((y_train[missing_mask] == 1).sum())
            n_missing_good = int((y_train[missing_mask] == 0).sum())
            self.missing_woe_ = self._woe(n_missing_good, n_missing_bad, total_good, total_bad)
        else:
            self.missing_woe_ = None

        return self

    # transform: apply the fitted lookup table to any split - train/val/test

    def transform(self, x: pd.Series) -> pd.Series:
        if self.category_lookup_ is None or self.woe_lookup_ is None:
            raise RuntimeError(
                f"{type(self).__name__}.transform() called before fit(). Fit on train first."
            )

        x = x.copy()
        out = pd.Series(index=x.index, dtype=float)

        missing_mask = x.isna()
        if missing_mask.any():
            if self.missing_woe_ is None:
                raise ValueError(
                    f"'{self.feature_name}' has missing values here but none were seen "
                    "during fit(); no WOE was learned for missingness."
                )
            out[missing_mask] = self.missing_woe_

        observed = x[~missing_mask].astype(str)
        if not observed.empty:
            bin_ids = observed.map(self._lookup_bin_id)
            out[~missing_mask] = bin_ids.map(self.woe_lookup_)

        return out

    def fit_transform(self, x_train: pd.Series, y_train: pd.Series) -> pd.Series:
        return self.fit(x_train, y_train).transform(x_train)


    def _lookup_bin_id(self, category: str) -> int:
        if category in self.category_lookup_:
            return self.category_lookup_[category]
        if self.other_bin_id_ is not None:
            return self.other_bin_id_
        raise ValueError(
            f"'{self.feature_name}' saw unseen category '{category}' at transform time, "
            "and no 'other' bucket was learned during fit() to fall back to "
        )

    @staticmethod
    def _woe(good_count: int, bad_count: int, total_good: int, total_bad: int) -> float:
        pct_good = (good_count + EPSILON) / (total_good + EPSILON)
        pct_bad = (bad_count + EPSILON) / (total_bad + EPSILON)
        return float(np.log(pct_good / pct_bad))

    def _compute_category_stats(
        self, x_obs: pd.Series, y_obs: pd.Series, total_good: int, total_bad: int
    ) -> list[CategoryBinStats]:
        stats = []
        for bin_id, category in enumerate(sorted(x_obs.unique())):
            mask = x_obs == category
            count = int(mask.sum())
            bad_count = int((y_obs[mask] == 1).sum())
            good_count = count - bad_count
            woe = self._woe(good_count, bad_count, total_good, total_bad)
            stats.append(
                CategoryBinStats(
                    bin_id=bin_id,
                    categories=[category],
                    count=count,
                    bad_count=bad_count,
                    good_count=good_count,
                    bad_rate=bad_count / count if count else 0.0,
                    woe=woe,
                )
            )
        return stats

    # min_count category check - pulled into single "other" bucket
    def _pool_rare_categories(
        self,
        stats: list[CategoryBinStats],
        min_count: int,
        total_good: int,
        total_bad: int,
    ) -> list[CategoryBinStats]:
        common = [s for s in stats if s.count >= min_count]
        rare = [s for s in stats if s.count < min_count]

        if not rare:
            for new_id, s in enumerate(common):
                s.bin_id = new_id
            return common

        pooled_categories = [cat for s in rare for cat in s.categories]
        count = sum(s.count for s in rare)
        bad_count = sum(s.bad_count for s in rare)
        good_count = sum(s.good_count for s in rare)
        woe = self._woe(good_count, bad_count, total_good, total_bad)
        other = CategoryBinStats(
            bin_id=-1,  # placeholder
            categories=pooled_categories,
            count=count,
            bad_count=bad_count,
            good_count=good_count,
            bad_rate=bad_count / count if count else 0.0,
            woe=woe,
        )
        self.overrides_.append(
            f"[{self.feature_name}] pooled {len(rare)} rare categories "
            f"({', '.join(pooled_categories)}) into '{OTHER_BIN_LABEL}': "
            f"combined n={count} (< {min_count} min per category)"
        )

        merged = common + [other]
        for new_id, s in enumerate(merged):
            s.bin_id = new_id
        self.other_bin_id_ = other.bin_id
        return merged

    # Reporting
    def summary(self) -> pd.DataFrame:
        if self.stats_ is None:
            raise RuntimeError("Call fit() first.")
        rows = [
            {
                "bin": s.bin_id,
                "categories": ", ".join(s.categories),
                "count": s.count,
                "bad_count": s.bad_count,
                "bad_rate": round(s.bad_rate, 4),
                "woe": round(s.woe, 4),
            }
            for s in self.stats_
        ]
        if self.missing_woe_ is not None:
            rows.append(
                {
                    "bin": MISSING_BIN_LABEL,
                    "categories": "NaN",
                    "count": None,
                    "bad_count": None,
                    "bad_rate": None,
                    "woe": round(self.missing_woe_, 4),
                }
            )
        return pd.DataFrame(rows)


# Fit one CategoricalWOEEncoder per categorical feature in the baseline set
def fit_categorical_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    **kwargs,
) -> dict[str, CategoricalWOEEncoder]:
    encoders = {}
    for feat in categorical_features:
        encoder = CategoricalWOEEncoder(feature_name=feat, **kwargs)
        encoder.fit(X_train[feat], y_train)
        encoders[feat] = encoder
    return encoders


def transform_with_categorical_encoders(
    X: pd.DataFrame, encoders: dict[str, CategoricalWOEEncoder]
) -> pd.DataFrame:
    # Apply fitted encoders to any split
    out = pd.DataFrame(index=X.index)
    for feat, encoder in encoders.items():
        out[f"{feat}_woe"] = encoder.transform(X[feat])
    return out
