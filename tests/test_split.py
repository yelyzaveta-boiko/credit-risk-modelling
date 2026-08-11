# Tests for src/data/split.py using a small artificial set of issue dates

import pandas as pd
import pytest

from src.data.split import parse_issue_date, time_based_split


@pytest.fixture
def fake_loans():
    return pd.DataFrame(
        {
            "loan_amnt": range(6),
            "issue_d": ["Jan-2015", "Jun-2016", "Dec-2016", "Jan-2017", "Jun-2017", "Feb-2018"],
        }
    )


def test_split_sizes(fake_loans):
    train, val, test = time_based_split(fake_loans, train_end="2017-01-01", val_end="2018-01-01")
    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 1


def test_split_has_no_date_overlap(fake_loans):
    train, val, test = time_based_split(fake_loans, train_end="2017-01-01", val_end="2018-01-01")

    train_dates = parse_issue_date(train["issue_d"])
    val_dates = parse_issue_date(val["issue_d"])
    test_dates = parse_issue_date(test["issue_d"])

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()


def test_split_covers_every_row_exactly_once(fake_loans):
    train, val, test = time_based_split(fake_loans, train_end="2017-01-01", val_end="2018-01-01")

    combined_index = train.index.append([val.index, test.index])
    assert sorted(combined_index) == sorted(fake_loans.index)


def test_split_uses_default_boundaries(fake_loans):
    train, val, test = time_based_split(fake_loans)
    assert len(train) + len(val) + len(test) == len(fake_loans)


def test_split_raises_on_missing_date_column(fake_loans):
    with pytest.raises(KeyError, match="issue_d"):
        time_based_split(fake_loans.drop(columns=["issue_d"]))


def test_split_raises_when_train_end_not_before_val_end(fake_loans):
    with pytest.raises(ValueError, match="train_end must be before val_end"):
        time_based_split(fake_loans, train_end="2018-01-01", val_end="2017-01-01")
