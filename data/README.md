# Data

## Source

**Dataset:** Lending Club Loan Data

**Provider:** Kaggle — [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**Downloaded:** 13.07.2026

## Contents

- `accepted_2007_to_2018Q4.csv` — loan level data for funded loans, 2007–2018. This file is used for PD and LGD modeling
- `rejected_2007_to_2018Q4.csv` — rejected loan applications. **Not used** in this project

## How to reproduce

```bash
pip install kaggle
# Place Kaggle API token at ~/.kaggle/kaggle.json first

kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip
```

Or run:

```bash
python src/data/download_data.py
```

this wraps the same command and confirms the expected files are present afterward.

## Known data leakage columns

Below are the columns that are only populated **after** a loan has already defaulted or been resolved, therefore they must be excluded from PD and LGD feature sets to avoid leakage:

- `recoveries`, `collection_recovery_fee` — only exist post default (collections process)
- `total_rec_late_fee` — accumulates over the loan's life, correlated with eventual default
- `last_pymnt_d`, `last_pymnt_amnt` — encode the loan's eventual outcome
- `out_prncp`, `out_prncp_inv` — outstanding principal, evolves with repayment/default status
- `debt_settlement_flag` — only set after default
