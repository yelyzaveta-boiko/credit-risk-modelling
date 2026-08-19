# Feature selection: IV-based screening

## Why screen before modeling, not after

The plan was originally to fit the baseline logistic regression on all 15 baseline features, and prune afterward based on what the model showed. But due to the time constraints, I decided to screen first instead. IV is computed independently of any model - it only needs the fitted WOE bins - so a feature whose IV says it carries no signal isn't going to reveal something different once it's inside a logistic regression. If I was to fit full set first that would mean spending time letting the model "discover" what the IV table already shows, and risk the regression fitting a spurious non zero coefficient to what is noise on almost zero IV features

## The threshold

Standard rule of thumb from scorecard development

| IV range   | Predictive power                        |
| ---------- | --------------------------------------- |
| < 0.02     | Useless                                 |
| 0.02 - 0.1 | Weak                                    |
| 0.1 - 0.3  | Medium                                  |
| 0.3 - 0.5  | Strong                                  |
| > 0.5      | Suspiciously strong - check for leakage |

I used IV < 0.02 ("useless") as the cutoff for dropping a feature from the baseline model entirely.

## Full ranking and decision

From `scripts/build_woe_matrix.py` on the train split:

| feature             | type        | IV     | band    | decision |
| ------------------- | ----------- | ------ | ------- | -------- |
| grade               | categorical | 0.4674 | strong  | keep     |
| int_rate            | numeric     | 0.4615 | strong  | keep     |
| term                | categorical | 0.1972 | medium  | keep     |
| fico_range_low      | numeric     | 0.1252 | medium  | keep     |
| dti                 | numeric     | 0.0775 | weak    | keep     |
| verification_status | categorical | 0.0532 | weak    | keep     |
| loan_amnt           | numeric     | 0.0322 | weak    | keep     |
| annual_inc          | numeric     | 0.0304 | weak    | keep     |
| home_ownership      | categorical | 0.0265 | weak    | keep     |
| revol_util          | numeric     | 0.0203 | weak    | keep     |
| purpose             | categorical | 0.0138 | useless | **drop** |
| open_acc            | numeric     | 0.0075 | useless | **drop** |
| emp_length          | categorical | 0.0009 | useless | **drop** |
| total_acc           | numeric     | 0.0004 | useless | **drop** |
| pub_rec             | numeric     | 0.0000 | useless | **drop** |

**10 kept, 5 dropped.**

## Reasons for feature dropping feature

- **`pub_rec`** (IV = 0.0000): shrinked to a single bin during WOE fitting - most borrowers in this data have zero derogatory public records, so there's no split left that separates a meaningfully different bad rate. A single bin feature contributes a constant, not a signal, a logistic regression coefficient on it would be pure noise.
- **`total_acc`** (IV = 0.0004) and **`open_acc`** (IV = 0.0075): total and open credit line counts. Distinguish between "many accounts" and "few accounts" borrowers, but that split doesn't track with default risk in this data once `revol_util` and `dti` (which already capture utilization/leverage) are in the picture.
- **`emp_length`** (IV = 0.0009): employment length in years. No relationship with default rate in this data.
- **`purpose`** (IV = 0.0138): stated loan purpose. Closest of the five to the 0.02 cutoff, so this one is rather my judgment decision than an obvious drop, but it still belongs to the "useless" category in the taxonomy I ve mentioned earlier, and 13 of `purpose`'s raw categories are rare enough to get pooled into "other" bucket.

## What's kept, and why the weak ones stay

The 10 surviving features ranging from strong (`grade`, `int_rate`) down to weak but not useless down to `revol_util` at 0.0203, just above the cutoff. I'm keeping the whole weak band rather than raising the bar further, because:

- Weak individually doesn't mean weak in combination - a scorecard's value comes from combining several weak to medium signals, not from a handful of strong ones alone.
- `loan_amnt`, `annual_inc`, `revol_util`, and `home_ownership` are all standard scorecard features in published Lending Club PD models; dropping them on IV alone with no other justification would be a harder call to defend than keeping them.
- IV only measures univariate strength. Multicollinearity between the kept features like`int_rate` is set by `grade`, so the two may be highly correlated is a separate problem to check once the model is actually fit.

What has been modified is:
`configs/baseline_features.py` keeps `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` / `BASELINE_FEATURES` as the initial 15 feature candidate set. `scripts/build_woe_matrix.py` fits and ranks all 15, so we can obtain the reproducible table with the IV values all the time. However a new `SELECTED_FEATURES` was added alongside it, where we have just 10 that survived screening. And the baseline logistic regression will be trained on that reduced set.
