# Tests for src/data/load_data.py using a small artificial CSV

import pandas as pd
import pytest

from src.data.load_data import LEAKAGE_COLUMNS, load_accepted_loans


# Create a fixture that sets up a temporary raw data directory with a small fake accepted loans CSV
@pytest.fixture
def fake_raw_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    fake_df = pd.DataFrame(
        {
            "loan_amnt": [1000, 2000, 3000],
            "int_rate": [5.0, 7.5, 10.0],
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "recoveries": [0.0, 150.0, 0.0],
            "last_pymnt_d": ["Jan-2019", "Feb-2019", "Mar-2019"],
        }
    )
    fake_df.to_csv(raw_dir / "accepted_2007_to_2018Q4.csv", index=False)

    monkeypatch.setattr("src.data.load_data.RAW_DIR", raw_dir)
    return raw_dir


def test_load_drops_leakage_columns_by_default(fake_raw_dir):
    df = load_accepted_loans()
    for col in LEAKAGE_COLUMNS:
        assert col not in df.columns


def test_load_can_keep_leakage_columns_when_requested(fake_raw_dir):
    df = load_accepted_loans(drop_leakage=False)
    assert "recoveries" in df.columns


def test_load_respects_nrows(fake_raw_dir):
    df = load_accepted_loans(nrows=2)
    assert len(df) == 2


def test_load_raises_clear_error_when_file_missing(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr("src.data.load_data.RAW_DIR", empty_dir)

    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_accepted_loans()
