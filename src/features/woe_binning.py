"""
Weight of Evidence binning for numeric feature with monotonicity
enforcement

  1. Split feature values into 15 candidate buckets using quantiles,
     so each bucket starts with roughly the same number of loans in it
  2. For each bucket, figure out: what fraction of loans in that bucket
     actually defaulted = bad rate and a related WOE score that
     compares the share of good loans vs bad loans in that bucket
  3. Check that the bad rate moves in one consistent direction
     from bucket 1 to bucket 15. If it doesn't,
     merge neighboring buckets together until it does
  4. Save the final bucket edges and their WOE scores as a lookup table,
     so the exact same buckets can be applied to validation/test data
     without ever recomputing them from that data to prevent leakage

    WOE_bucket = ln( (share of GOOD loans in this bucket)
                      / (share of BAD loans in this bucket) )

A positive WOE means "this bucket has relatively more good loans than bad
loans compared to the whole portfolio" -> lower risk than average.
A negative WOE means the opposite -> higher risk than average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Small constant added to bin counts
# For "Laplace smoothing" to never take log(0),
# a bin with zero bads (common in the safest tail bin) would make WOE blow up to infinity.
EPSILON = 0.5

# A bin with missing value gets its own bucket, because in this
# dataset missingness might be sometimes informative rather
# than random - it deserves its own WOE score, not to be dropped or
# silently imputed away at this stage.
MISSING_BIN_LABEL = "missing"


@dataclass
class BinStats:
    #used for documenting
    bin_id: int
    left_edge: float
    right_edge: float
    count: int
    bad_count: int
    good_count: int
    bad_rate: float
    woe: float


@dataclass
class WOEBinner:
    """
    Fits monotonic WOE bins for a single numeric feature on training data,
    then applies that same lookup table to any other split.

    Usage
    -----
    binner = WOEBinner(feature_name="dti")
    binner.fit(X_train["dti"], y_train)
    woe_train = binner.transform(X_train["dti"])
    woe_val = binner.transform(X_val["dti"])      # uses train's edges/WOE
    woe_test = binner.transform(X_test["dti"])    # uses train's edges/WOE
    """

    feature_name: str
    initial_bins: int = 15
    min_bin_fraction: float = 0.05  # smallest allowed bin size, as a share of train

    # Populated by fit()
    bin_edges_: np.ndarray | None = field(default=None, init=False)
    woe_lookup_: dict[int, float] | None = field(default=None, init=False)
    stats_: list[BinStats] | None = field(default=None, init=False)
    missing_woe_: float | None = field(default=None, init=False)
    # "increasing" / "decreasing"
    monotonic_direction_: str | None = field(default=None, init=False)
    overrides_: list[str] = field(default_factory=list, init=False)  # human readable merge log

    # fit: learn bin edges + WOE values from only training data

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> WOEBinner:
        # Reset learned state in case fit() is called more than once on the same
        # instance - overrides_ is only ever appended to below, so a stale merge
        # log could otherwise leak from a previous fit into this one.
        self.bin_edges_ = None
        self.woe_lookup_ = None
        self.stats_ = None
        self.missing_woe_ = None
        self.monotonic_direction_ = None
        self.overrides_ = []

        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        missing_mask = x_train.isna() # bollean array of missing values
        x_obs = x_train[~missing_mask] # filter only the observed non missing values
        y_obs = y_train[~missing_mask]

        if x_obs.empty:
            raise ValueError(f"'{self.feature_name}' has no observed non missing values to bin.")

        # split the data into quantile-based bins, then compute WOE for each bin
        # if two boundaries are identical, get rid of the duplicate boundary and reduce the
        # number of bins. It doesnt merge two existing bins, but prevent from creating them
        # with the same boundary
        _, edges = pd.qcut(
            x_obs,
            q=self.initial_bins,
            duplicates="drop",
            retbins=True
        )
        edges = np.asarray(edges, dtype=float)

        if len(edges) < 2:
            # The feature has so few unique values that
            # quantile cuts becomes a single edge
            # - one bin covering everything
            edges = np.array([x_obs.min(), x_obs.max()])

        # we dont want the model to break if the validation/test set has values outside the
        # train range, so we add -inf +inf edges so transform () never sees an out of range value
        edges[0], edges[-1] = -np.inf, np.inf

        # assign each observed value to a bin and compute WOE for each bin
        # trim the first and last edges to keep only boundaries that actually separate two bins
        bin_ids = np.digitize(x_obs, edges[1:-1], right=True)

        total_good = int((y_obs == 0).sum())
        total_bad = int((y_obs == 1).sum())
        if total_good == 0 or total_bad == 0:
            raise ValueError(
                f"'{self.feature_name}' training slice has only one class present, "
                "cannot compute WOE"
            )

        stats = self._compute_bin_stats(x_obs, y_obs, bin_ids, edges, total_good, total_bad)

        # Decide which direction bad rate should move based on
        # the feature's overall relationship with default
        direction = self._infer_direction(stats)

        # for merging neighboring bins if monotonicity is violated
        stats = self._enforce_monotonicity(stats, direction, total_good, total_bad)

        #save final edges and WOE lookup table for transform() to use on any split
        self.bin_edges_ = np.array([stats[0].left_edge] + [s.right_edge for s in stats])
        self.woe_lookup_ = {i: s.woe for i, s in enumerate(stats)}
        self.stats_ = stats
        self.monotonic_direction_ = direction

        # Missing values get their own WOE
        if missing_mask.any():
            n_missing_bad = int((y_train[missing_mask] == 1).sum())
            n_missing_good = int((y_train[missing_mask] == 0).sum())
            self.missing_woe_ = self._woe(n_missing_good, n_missing_bad, total_good, total_bad)
        else:
            self.missing_woe_ = None

        return self

    # transform: apply the fitted lookup table to any split - train/val/test

    def transform(self, x: pd.Series) -> pd.Series:
        if self.bin_edges_ is None or self.woe_lookup_ is None:
            raise RuntimeError("WOEBinner.transform() called before fit(). Fit on train first.")

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

        observed = x[~missing_mask]
        if not observed.empty:
            # assign each observed value to a bin (0..k-1) based on train's edges
            bin_ids = np.digitize(observed, self.bin_edges_[1:-1], right=True)
            # apply saved WOE lookup to each bin id
            out[~missing_mask] = [self.woe_lookup_[b] for b in bin_ids]

        return out

    def fit_transform(self, x_train: pd.Series, y_train: pd.Series) -> pd.Series:
        # learn the rules and apply them to the same training data, returning the
        # WOE-transformed series
        return self.fit(x_train, y_train).transform(x_train)

    # Internal helpers
    @staticmethod
    def _woe(good_count: int, bad_count: int, total_good: int, total_bad: int) -> float:
        pct_good = (good_count + EPSILON) / (total_good + EPSILON)
        pct_bad = (bad_count + EPSILON) / (total_bad + EPSILON)
        return float(np.log(pct_good / pct_bad))

    def _compute_bin_stats(
        self,
        x_obs: pd.Series,
        y_obs: pd.Series,
        bin_ids: np.ndarray,
        edges: np.ndarray,
        total_good: int,
        total_bad: int,
    ) -> list[BinStats]:
        # store the stats for each bin, including WOE, bad rate, counts, etc.
        stats = []
        # removes duplicates and sorts the bin ids
        unique_bins = sorted(set(bin_ids))
        for new_id, old_id in enumerate(unique_bins):
            # filter the observed values that belong to the current bin
            mask = bin_ids == old_id
            count = int(mask.sum())
            bad_count = int((y_obs[mask] == 1).sum())
            good_count = count - bad_count
            woe = self._woe(good_count, bad_count, total_good, total_bad)
            stats.append(
                BinStats(
                    bin_id=new_id,
                    left_edge=edges[old_id],
                    right_edge=edges[old_id + 1],
                    count=count,
                    bad_count=bad_count,
                    good_count=good_count,
                    bad_rate=bad_count / count if count else 0.0,
                    woe=woe,
                )
            )
        return stats

    @staticmethod
    # Discover the relationship from the training data:
    # does bad rate increase or decrease as the feature increases?
    # Used to guide the monotonicity enforcement step
    def _infer_direction(stats: list[BinStats]) -> str:
        bad_rates = [s.bad_rate for s in stats]
        first_half = (
            np.mean(bad_rates[: len(bad_rates) // 2]) if len(bad_rates) > 1 else bad_rates[0]
        )
        second_half = np.mean(bad_rates[len(bad_rates) // 2 :])
        return "decreasing" if first_half > second_half else "increasing"

    # Merge neighboring bins until the bad rate moves in one consistent direction (monotonicity)
    # To avoid counter intuitive risk patterns
    def _enforce_monotonicity(
        self,
        stats: list[BinStats],
        direction: str,
        total_good: int,
        total_bad: int,
    ) -> list[BinStats]:
        stats = list(stats)

        def violates(a: BinStats, b: BinStats) -> bool:
            # two neighbouring bins that break the rule
            if direction == "increasing":
                return b.bad_rate < a.bad_rate
            return b.bad_rate > a.bad_rate

        # keep checking the bins for monotonicity violations and merge them until none remain
        merged_any = True
        while merged_any and len(stats) > 1:
            merged_any = False
            i = 0
            while i < len(stats) - 1:
                if violates(stats[i], stats[i + 1]):
                    a, b = stats[i], stats[i + 1]
                    # merge the two bins into one and recompute stats
                    merged = self._merge_bins(a, b, total_good, total_bad)
                    # log the merge for human readable audit
                    reason = (
                        f"[{self.feature_name}] merged bins "
                        f"({a.left_edge:.4g}, {a.right_edge:.4g}] "
                        f"and ({b.left_edge:.4g}, {b.right_edge:.4g}]: bad rate "
                        f"{a.bad_rate:.3%} -> {b.bad_rate:.3%} broke '{direction}' trend; "
                        f"combined bad rate {merged.bad_rate:.3%}, n={merged.count}"
                    )
                    self.overrides_.append(reason)
                    # Bin 0 | Bin 1 | Bin 2 | Bin 3 becomes
                    # Bin 0 | Merged Bin | Bin 3, so we replace the two bins with the merged one
                    stats = stats[:i] + [merged] + stats[i + 2 :]
                    merged_any = True
                    i = 0
                else:
                    i += 1

        # renumber bin_id after merges
        for new_id, s in enumerate(stats):
            s.bin_id = new_id

        # enforce a minimum bin size to protects against tiny bins
        # that are not statistically meaningful and can cause overfitting
        # now by def the bin should contain at least 5% of training observations
        min_count = int(self.min_bin_fraction * sum(s.count for s in stats))
        # merge any bins that are smaller than the minimum count into their neighbors
        stats = self._merge_small_bins(stats, min_count, total_good, total_bad)

        return stats

    def _merge_small_bins(
        self, stats: list[BinStats], min_count: int, total_good: int, total_bad: int
    ) -> list[BinStats]:
        changed = True
        # keep merging small bins until all bins meet the minimum count requirement
        while changed and len(stats) > 1:
            # for this round lets assume no merges will happen, and if we do merge,
            # we will set changed to True
            changed = False
            for i, s in enumerate(stats):
                if s.count < min_count:
                    # merge with whichever neighbor keeps bad rate closer
                    if i == 0:
                        neighbor_idx = 1
                    elif i == len(stats) - 1:
                        neighbor_idx = i - 1
                    else:
                        # how different the bad rate of the small bin is from its
                        # left and right neighbors
                        left_diff = abs(stats[i - 1].bad_rate - s.bad_rate)
                        right_diff = abs(stats[i + 1].bad_rate - s.bad_rate)
                        # we chose the one that is more similar to that tiny bin
                        neighbor_idx = i - 1 if left_diff <= right_diff else i + 1
                    # merge the small bin with its chosen neighbor and recompute stats
                    lo, hi = sorted((i, neighbor_idx))
                    merged = self._merge_bins(stats[lo], stats[hi], total_good, total_bad)
                    self.overrides_.append(
                        f"[{self.feature_name}] merged small bin n={s.count} "
                        f"(< {min_count} min) into neighbor; combined n={merged.count}"
                    )
                    # two bins disappear and merged one appears in their place
                    stats = stats[:lo] + [merged] + stats[hi + 1 :]
                    changed = True
                    # restart the loop since the list has changed
                    break
        for new_id, s in enumerate(stats):
            s.bin_id = new_id
        return stats

    def _merge_bins(
        self, a: BinStats, b: BinStats, total_good: int, total_bad: int
    ) -> BinStats:
        count = a.count + b.count
        bad_count = a.bad_count + b.bad_count
        good_count = a.good_count + b.good_count
        woe = self._woe(good_count, bad_count, total_good, total_bad)
        return BinStats(
            bin_id=a.bin_id,
            left_edge=a.left_edge,
            right_edge=b.right_edge,
            count=count,
            bad_count=bad_count,
            good_count=good_count,
            bad_rate=bad_count / count if count else 0.0,
            woe=woe,
        )

    # Reporting helpers
    def summary(self) -> pd.DataFrame:
        if self.stats_ is None:
            raise RuntimeError("Call fit() first.")
        rows = [
            {
                "bin": s.bin_id,
                "range": f"({s.left_edge:.4g}, {s.right_edge:.4g}]",
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
                    "range": "NaN",
                    "count": None,
                    "bad_count": None,
                    "bad_rate": None,
                    "woe": round(self.missing_woe_, 4),
                }
            )
        return pd.DataFrame(rows)


# Fit one WOEBinner per numeric feature in the baseline set
def fit_baseline_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    **kwargs,
) -> dict[str, WOEBinner]:
    binners = {}
    for feat in numeric_features:
        binner = WOEBinner(feature_name=feat, **kwargs)
        binner.fit(X_train[feat], y_train)
        binners[feat] = binner
    return binners


def transform_with_binners(
    X: pd.DataFrame, binners: dict[str, WOEBinner]
) -> pd.DataFrame:
    #Apply already fitted binners to any split (val/test/production)
    out = pd.DataFrame(index=X.index)
    for feat, binner in binners.items():
        out[f"{feat}_woe"] = binner.transform(X[feat])
    return out
