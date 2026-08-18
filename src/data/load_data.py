import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_raw_dir_env = os.getenv("DATA_RAW_DIR")
RAW_DIR = (PROJECT_ROOT / _raw_dir_env) if _raw_dir_env else (PROJECT_ROOT / "data" / "raw")

ACCEPTED_LOANS_FILES = (
    "accepted_2007_to_2018Q4.csv",
    )


# Columns populated after a loan has already defaulted/resolved,
# or that contain information not knowable at origination
LEAKAGE_COLUMNS = [
    "recoveries",
    "collection_recovery_fee",
    "total_rec_late_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "out_prncp",
    "out_prncp_inv",
    "debt_settlement_flag",
    # Most recently pulled FICO score — updated during loan servicing
    # fico_range_high/low already capture the origination-time value
    "last_fico_range_high",
    "last_fico_range_low",
    # Hardship plan fields populated after origination when a borrower is already struggling.
    "hardship_length",
    "hardship_last_payment_amount",
    "hardship_payoff_balance_amount",
    "hardship_loan_status",
    "hardship_dpd",
    "hardship_amount",
    "hardship_end_date",
    "hardship_type",
    "hardship_reason",
    "hardship_start_date",
    "hardship_status",
    "payment_plan_start_date",
    "deferral_term",
    "orig_projected_additional_accrued_interest",
    # Settlement fields — post default
    "settlement_amount",
    "settlement_percentage",
    "settlement_date",
    "settlement_status",
    "debt_settlement_flag_date",
    "settlement_term",
    # Forward looking servicing field
    "next_pymnt_d",
]


# Columns to drop uninformative and structurally too sparse - no leakage, but useless for this scope
UNINFORMATIVE_COLUMNS = [
    "member_id",
    "desc",  # free-text 99.99% missing
    # Joint application fields missing almost everywhere since joint apps are rare
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
    "revol_bal_joint",
    "dti_joint",
    "annual_inc_joint",
    "verification_status_joint",
    # Credit bureau trade line fields available for later loans only 89% missing
    "il_util",
    "mths_since_rcnt_il",
    "inq_last_12m",
    "total_cu_tl",
    "all_util",
    "max_bal_bc",
    "open_rv_24m",
    "open_rv_12m",
    "total_bal_il",
    "open_il_24m",
    "open_il_12m",
    "open_act_il",
    "open_acc_6m",
    "inq_fi",
]


# NaN informative columns- "this event never happened"
# impute with a large sentinel value rather than dropping.
NEVER_HAPPENED_SENTINEL_COLUMNS = [
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq",
]

# Populate 'months since X' columns where NaN means the event never occurred
def handle_sentinel_missingness(df: pd.DataFrame, sentinel: int = 999) -> pd.DataFrame:
    df = df.copy()
    for col in NEVER_HAPPENED_SENTINEL_COLUMNS:
        if col in df.columns:
            df[f"{col}_never_occurred"] = df[col].isna()
            df[col] = df[col].fillna(sentinel)
    return df


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


def load_accepted_loans(
    nrows: int | None = None,
    drop_leakage: bool = True,
    drop_uninformative: bool = True,
) -> pd.DataFrame:

    filepath = _accepted_loans_path()
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    if drop_leakage:
        cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_present)

    if drop_uninformative:
        cols_present = [c for c in UNINFORMATIVE_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_present)

    return df

# only known final outcome values
TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
}


def load_labeled_loans(**kwargs) -> pd.DataFrame:
    df = load_accepted_loans(**kwargs)
    df = df[df["loan_status"].isin(TARGET_MAP)].copy()
    df["target"] = df["loan_status"].map(TARGET_MAP)
    return df
