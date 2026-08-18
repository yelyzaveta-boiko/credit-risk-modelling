# Build vs buy benchmark for numeric WOE binning

# Compares src.features.woe_binning.WOEBinner against
# optbinning.OptimalBinning (library) on a handful of baseline features

import functools
import sys
from pathlib import Path

import pandas as pd
import sklearn.utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_labeled_loans  # noqa: E402
from src.data.split import time_based_split  # noqa: E402
from src.features.woe_binning import WOEBinner  # noqa: E402

_orig_check_array = sklearn.utils.check_array


@functools.wraps(_orig_check_array)
def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


sklearn.utils.check_array = _patched_check_array

from optbinning import OptimalBinning  # noqa: E402

FEATURES = ["dti", "annual_inc", "revol_util"]


def resolved_direction(bad_rates: list[float]) -> str:
    increases = sum(b > a for a, b in zip(bad_rates, bad_rates[1:]))
    decreases = sum(b < a for a, b in zip(bad_rates, bad_rates[1:]))
    return "increasing" if increases >= decreases else "decreasing"


def own_result(feature: str, x_train: pd.Series, y_train: pd.Series) -> dict:
    binner = WOEBinner(feature_name=feature)
    binner.fit(x_train, y_train)
    return {
        "n_bins": len(binner.stats_),
        "direction": binner.monotonic_direction_,
        "iv": binner.iv(),
        "table": binner.summary(),
        "n_overrides": len(binner.overrides_),
    }


def optbinning_result(feature: str, x_train: pd.Series, y_train: pd.Series) -> dict:
    optb = OptimalBinning(name=feature, dtype="numerical", solver="cp", min_prebin_size=0.05)
    optb.fit(x_train.to_numpy(dtype=float), y_train.to_numpy())
    table = optb.binning_table.build()
    # the totals row's "Bin" value is "" (its label lives in the index as "Totals"),
    # so it isn't caught by an isin(["Special", "Missing", "Totals"]) filter
    bin_rows = table[(table["Bin"] != "") & ~table["Bin"].isin(["Special", "Missing"])]
    return {
        "n_bins": len(bin_rows),
        "direction": resolved_direction(bin_rows["Event rate"].tolist()),
        "iv": float(optb.binning_table.iv),
        "table": bin_rows[["Bin", "Count", "Event rate", "WoE"]],
        "status": optb.status,
    }


def main() -> None:
    df = load_labeled_loans()
    train, _val, _test = time_based_split(df)
    y_train = train["target"]

    for feature in FEATURES:
        x_train = train[feature]
        own = own_result(feature, x_train, y_train)
        lib = optbinning_result(feature, x_train, y_train)

        print(f"\n{' ' * 70}\n{feature}\n{' ' * 70}")
        print(
            f"WOEBinner: {own['n_bins']} bins, direction={own['direction']}, "
            f"IV = {own['iv']:.4f}, {own['n_overrides']} monotonicity/size merges"
        )
        print(
            f"OptimalBinning: {lib['n_bins']} bins, direction={lib['direction']}, "
            f"IV={lib['iv']:.4f}, solver status={lib['status']}"
        )
        print("\n-- WOEBinner bins --")
        print(own["table"].to_string(index=False))
        print("\n-- OptimalBinning bins --")
        print(lib["table"].to_string(index=False))


if __name__ == "__main__":
    main()
