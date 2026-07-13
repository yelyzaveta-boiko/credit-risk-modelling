# Download the Lending Club dataset from Kaggle into data/raw/
# Requires a Kaggle API token at ~/.kaggle/kaggle.json — see data/README.md.

import subprocess
import sys
from pathlib import Path

DATASET = "wordsforthewise/lending-club"
RAW_DIR = Path("data/raw")
EXPECTED_FILE_PREFIX = "accepted_"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Download failed. Kaggle CLI output:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    matches = list(RAW_DIR.glob(f"{EXPECTED_FILE_PREFIX}*"))
    if not matches:
        print(
            f"Download command succeeded but no file starting with "
            f"'{EXPECTED_FILE_PREFIX}' was found in {RAW_DIR}. "
            "Check the dataset's current file naming on Kaggle.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloaded successfully: {matches[0].name}")


if __name__ == "__main__":
    download()
