# Reusable loaders for the Lending Club dataset

# Import this module rather than reading CSVs directly in notebooks

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
ACCEPTED_LOANS_FILES = (
    "accepted_2007_to_2018Q4.csv.gz",
    "accepted_2007_to_2018Q4.csv",
    "accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
)


LEAKAGE_COLUMNS = [
    "recoveries",
    "collection_recovery_fee",
    "total_rec_late_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "out_prncp",
    "out_prncp_inv",
    "debt_settlement_flag",
]

def _accepted_loans_path() -> Path:
    for filename in ACCEPTED_LOANS_FILES:
        filepath = RAW_DIR / filename
        if filepath.is_file():
            return filepath

    expected_paths = ", ".join(str(RAW_DIR / filename) for filename in ACCEPTED_LOANS_FILES)
    raise FileNotFoundError(
        "Accepted loans file not found. Run `python src/data/download_data.py` first, "
        f"or see data/README.md for manual download instructions. Expected one of: {expected_paths}"
    )


# Load the accepted-loans dataset, dropping known post-default leakage columns.
def load_accepted_loans(
    nrows: int | None = None,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    filepath = _accepted_loans_path()

    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    if drop_leakage:
        cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_present)

    return df
