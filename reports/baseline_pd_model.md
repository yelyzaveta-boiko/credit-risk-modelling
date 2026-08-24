# Baseline PD model

A **PD model** predicts the **P**robability of **D**efault - given everything we know about a loan at the time it was issued, how likely is this borrower to stop paying it back?

The baseline model is a **logistic regression** a standard statistics technique for predicting a yes/no outcome as a probability between 0% and 100%. It's the industry standard starting point for credit scoring - simple, fast, and every input feeds into the prediction in a way a human can inspect and explain, which matters a lot in lending, where regulators have to be able to ask "why did the model score this loan as risky?" and get a real answer.

It was trained using 10 of the loan's characteristics (income, interest rate, credit history, explained in`reports/feature_selection.md`), converted beforehand into a WOE format (explained in `reports/feature_engineering.md`) that makes each feature's relationship with risk easy for a simple model to pick up on.

`src/models/pd_logistic_regression.py` is the reusable model fitting logic and `scripts/train_baseline_pd_model.py` is the script that runs it on the data and saves the result

## Model correctness

A model that just guessed "this loan is fine" for every single loan would be right about 80% of the time, because most loans in this data don't default - so accuracy is not a good metric to measure the validity of the model. Instead we use four checks, where each one of them answers a different question:

- **AUC** - if we picked one loan that defaulted and one that didn't at random, how often would the model correctly say the defaulted one was riskier? 50% a coin flip means the model knows nothing while 100% means it always gets it right. It is the standard first check to understand if the model separate risky from safe loans at all
- **KS** - finds the one risk cutoff where the model does the best job of separating defaulters from non defaulters, and reports how big that separation is. 0% would mean no separation and 100% would mean a perfect dividing line exists.
- **Gini** - similar to AUC, but rescaled where 0% means no better than guessing and 100% means perfect, which is the scale credit risk reports conventionally use.
- **Calibration** - AUC/KS/Gini check if the model can _rank_ loans from safest to riskiest correctly. Calibration checks the folllowing if the model says a loan has a 20% chance of default, does it actually default about 20% of the time in reality? A model can be excellent at ranking and still be badly calibrated.

IMplemented in `src/evaluation/pd_metrics.py` the four checks described above and `scripts/evaluate_baseline_pd_model.py` to run them on the validation and test loans.

## Results

The model was trained on loans issued before 2017, then checked against loans it never saw during training: a 2017 "validation" batch, and a 2018+ "test" batch.

|                       | AUC | KS  | Gini |
| --------------------- | --- | --- | ---- |
| Validation 2017 loans | 70% | 28% | 39%  |
| Test 2018+ loans      | 69% | 28% | 38%  |

So the model correctly ranks a riskier loan above a safer one 7 times out of 10. This is a reasonable baseline - better than guessing, but it has a room for a more sophisticated model like the gradient-boosted-trees model to improve on. What is important to notice is that the score remained the same between the 2017 and 2018+ batches, which means the model isn't overfit to the specific loans it was trained on - it generalizes to loans from later years reasonably well.

## However woth mentioning calibration drifted over time

Ranking held up well, but the _calibrated_ risk numbers didn't:

- On the **2017 validation loans**, the model consistently **underestimated** risk - actual default rates ran up to 5 percentage points higher than what the model predicted.
- On the **2018+ test loans**, the model consistently **overestimated** risk instead - and for the riskiest 10% of loans, it overestimated by a full **11 percentage points** (predicted 44% chance of default, actual was 33%).

The most likely explanation isn't a bug in the model - it's that lending conditions genuinely changed over time. Loans issued in different years were underwritten differently and defaulted at different rates for reasons that have nothing to do with any individual loan's characteristics, and a model trained only on pre-2017 loans has no way to know that.

This means that the baseline model's _ranking_ of "who's riskier" can be trusted reasonably well, but exact predicted percentages should not be read too literally, especially for the riskiest loans in later years. This is a normal expected finding at the baseline model stage, but it's worth keeping in mind and worth comparing with the resulats obtained from the gradient boosted trees model

## To be done next

- Build the gradient-boosted-trees PD model and compare it against this baseline on the same four checks.
- Use SHAP to explain individual predictions from whichever model performs better.
