# Logistic regression vs. LightGBM: PD model comparison

Both models are trained on the same 10 IV-selected features (`reports/feature_selection.md`) and the same time based train/val/test split. The difference is only in data preprocessing step: the logistic regression trains on WOE encoded features (`data/processed/woe_*.csv`), LightGBM trains on the raw values directly (`data/processed/raw_*.csv`), since trees don't need WOE to find non linear relationships on their own.

## Discrimination: LightGBM wins with tiny advantage

|                            | AUC    | KS     | Gini   |
| -------------------------- | ------ | ------ | ------ |
| Logistic regression - val  | 0.6961 | 0.2840 | 0.3922 |
| LightGBM - val             | 0.7009 | 0.2921 | 0.4018 |
| Logistic regression - test | 0.6905 | 0.2776 | 0.3811 |
| LightGBM - test            | 0.7002 | 0.2938 | 0.4003 |

LightGBM comes out ahead on every metric, on both splits, but the gap is really small, about half a point to one point of AUC. This is a useful finding because it confirms the WOE encoded scorecard has already captured most of the non linear structure in these features on its own, so the trees don't have much extra signal left to exploit. A well built logistic regression scorecard being close to a tree ensemble is a normal outcome and isn't a sign that either model is broken.

One point worth mentioning is the LightGBM's val and test AUC are nearly identical 0.7009 vs 0.7002, while the logistic regression's drops more between the two 0.6961 vs 0.6905. Which signifies that LightGBM's _ranking_ of risky vs. safe loans held up a bit better across the 2017 to 2018+ time gap.

## Calibration: same drift direction

Both models show the same underlying pattern documented in `reports/baseline_pd_model.md`: risk is underestimated on the 2017 validation loans and overestimated on the 2018+ test loans, proving the assumption that lending conditions most likely shift over time rather than either model being miscalibrated.

Where they differ is the riskiest 10% of test loans:

|                                     | Predicted PD | Actual default rate | Gap     |
| ----------------------------------- | ------------ | ------------------- | ------- |
| Logistic regression - test decile 9 | 43.6%        | 32.9%               | -10.8pp |
| LightGBM - test decile 9            | 46.6%        | 34.3%               | -12.3pp |

LightGBM is _more_ overconfident than the logistic regression in exactly the bucket where being wrong matters most - the loans a lender would flag as highest risk. Better ranking doesn't automatically mean better calibrated probabilities, and this is example of that: LightGBM separates goods from bads slightly better overall, but its predicted probabilities in the tail should be trusted even less than the baseline's.

## Feature importance: a multicollinearity finding

The logistic regression's coefficient report flagged `revol_util_woe` with a positive coefficient - the wrong sign, because a higher WOE - safer bucket - should always push default risk down, not up. The likely explanation was multicollinearity between the WOE features

LightGBM's feature importance independently points at the same underlying issue, on a different pair of features. `grade` and `int_rate` have nearly identical IV 0.4674 vs 0.4615, but LightGBM's gain based importance ranks `grade` almost 4 times higher than `int_rate` 342,712 vs 83,026:

| feature  | IV univariate | LightGBM gain importance |
| -------- | ------------- | ------------------------ |
| grade    | 0.4674        | 342,712                  |
| int_rate | 0.4615        | 83,026                   |

IV measures a feature's predictive power in isolation; gain importance measures what it contributes _on top of_ everything else already in the model. Lending Club sets `int_rate` directly from `grade`, so the two features carry a lot of duplicate information - once the trees have split on `grade`, there's little new signal left for `int_rate` to add. Two different models, two different symptoms, same root cause: `grade` and `int_rate` are highly correlated.

## Interpretability tradeoff

A logistic regression's coefficients are the whole model - any loan's score can be reconstructed by hand from its WOE values and the fitted coefficients in `reports/baseline_pd_model.md`. That directly answers "why did the model score this loan as risky," which matters in lending, because that question tends to come from a regulator or applicant, not just a data scientist.

LightGBM doesn't offer that out of the box - a prediction comes from hundreds of trees voting together, so explaining any single score needs a separate tool. The interpretability here has to be built and cannot be read off the fitted model.

## Where this leaves the two models

Given a 0.5-1 point AUC improvement against a real loss of built-in interpretability, the logistic regression scorecard is the more defensible choice as the primary model in a lending context, and LightGBM's value here is more as a **benchmark**: it confirms the WOE scorecard isn't leaving a lot of performance on the table, and its feature importance ranking cross checks findings from the logistic regression (the `grade`/`int_rate` collinearity) that would have been easy to miss from either model alone.
