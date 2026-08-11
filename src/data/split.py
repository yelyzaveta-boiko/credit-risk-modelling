# Time based train/validation/test split, keyed on loan issue date
# no random shuffle => model is never trained on loans issued after
# the ones it's evaluated on

import pandas as pd

DATE_COLUMN = "issue_d"
DATE_FORMAT = "%b-%Y"

DEFAULT_TRAIN_END = "2017-01-01"
DEFAULT_VAL_END = "2018-01-01"


def parse_issue_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT)


# Half open splits on the boundary dates: 
# train < train_end <= val < val_end <= test.
def time_based_split(
    df: pd.DataFrame,
    date_col: str = DATE_COLUMN,
    train_end: str = DEFAULT_TRAIN_END,
    val_end: str = DEFAULT_VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' column not found in DataFrame")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if not train_end_ts < val_end_ts:
        raise ValueError("train_end must be before val_end")

    dates = parse_issue_date(df[date_col])

    train = df.loc[dates < train_end_ts]
    val = df.loc[(dates >= train_end_ts) & (dates < val_end_ts)]
    test = df.loc[dates >= val_end_ts]

    return train, val, test
