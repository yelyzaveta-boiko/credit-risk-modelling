# Build the raw non WOE feature matrices for the tree model
# LightGBM finds non linear relationships by itself

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline_features import SELECTED_FEATURES  # noqa: E402
from src.data.load_data import load_labeled_loans  # noqa: E402
from src.data.split import time_based_split  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMNS = [*SELECTED_FEATURES, "target"]


def main() -> None:
    df = load_labeled_loans()
    train, val, test = time_based_split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = PROCESSED_DIR / f"raw_{name}.csv"
        split[COLUMNS].to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(split):,} rows, {len(COLUMNS)} columns)")


if __name__ == "__main__":
    main()
