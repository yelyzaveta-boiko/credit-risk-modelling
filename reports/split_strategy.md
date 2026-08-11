# Train / validation / test split strategy

## Why not a random split

A random row level split would let the model train on loans issued for example in
December 2017 and get validated on loans issued in June 2016, leading to the future information leakage into training. Therefore the model should only
see loans that haven't been originated yet, so the split needs to respect
time: train on the past, validate and test on the future.

## Split key

`issue_d` — the loan's origination month (`Mon-YYYY` - example `Dec-2015`) — is
known at the point of loan scoring, therefore no leakage and is safe to
split on.

## Cutoff dates

The dataset consists of the loans issued `2007-06` through `2018-12`, but volume is
heavily skewed toward later years. Deriving cutoffs directly from row counts (as an example picking date that gives an exact 60/20/20 split) would land on an
arbitrary date with no meaning of its own, and that would shift on every data refresh. Thereofre cutoffs were chosen on calendar year boundaries for
stability and interpretability:

| split | range                                | rows      | share |
| ----- | ------------------------------------ | --------- | ----- |
| train | `issue_d` < 2017-01-01               | 1,321,847 | 58%   |
| val   | 2017-01-01 <= `issue_d` < 2018-01-01 | 443,579   | 20%   |
| test  | `issue_d` >= 2018-01-01              | 495,242   | 22%   |

Boundaries are half open (`train_end <= val < val_end <= test`), so every loan falls into exactly one split and no split shares a boundary date with
another.

## Usage

```python
from src.data.load_data import load_accepted_loans
from src.data.split import time_based_split

df = load_accepted_loans()
train, val, test = time_based_split(df)
```

`train_end` and `val_end` are keyword arguments on `time_based_split`
