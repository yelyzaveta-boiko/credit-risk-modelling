# WOE transformed feature matrices across
# the 15 baseline features, and rank features by Information Value.
#
# Writes data/processed/woe_{train,val,test}.csv and prints the IV ranking.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.data.load_data import load_labeled_loans  # noqa: E402
from src.data.split import time_based_split  # noqa: E402
from src.features.woe_matrix import fit_woe_encoders, iv_summary, transform_woe_matrix  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    df = load_labeled_loans()
    train, val, test = time_based_split(df)

    encoders = fit_woe_encoders(train, train["target"], NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    print("Feature ranking by Information Value (train split):\n")
    print(iv_summary(encoders).to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        matrix = transform_woe_matrix(split, encoders)
        matrix["target"] = split["target"].to_numpy()
        out_path = PROCESSED_DIR / f"woe_{name}.csv"
        matrix.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(matrix):,} rows, {matrix.shape[1]} columns)")


if __name__ == "__main__":
    main()
