# EDA Summary

## What's in the dataset

I'm working with Lending Club's accepted loans data (2007–2018). Full file has 119 columns. I'm using a 200k row sample for now when iterate so that it runs faster.

## Defining default

Lending Club doesn't give a yes/no default label, it gives a `loan_status` field with 7 different values. So the first decision was to assign labels to them, which of them count as "good" and which count as "bad" .

| loan_status | my label | why                                       |
| ----------- | -------- | ----------------------------------------- |
| Fully Paid  | 0 (good) | loan is done, paid back in full           |
| Charged Off | 1 (bad)  | LC's written off status                   |
| Default     | 1 (bad)  | one step before Charged Off, same outcome |

I dropped the rest `Current`, `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period` because they are still ongoing, and its outcome is unknown yet if they'll end up paid off or defaulted. Labeling a "Current" loan as good and then month later it defaults, would mean feeding a wrong answer into training data. So they just get excluded from the labeled set.

After that filtering, the labeled dataset is about 20% default (35,090 charged off vs 140,992 fully paid). That's not a critical imbalance, but it's enough to undersatnd that we cannot rely purely on accuracy later in the evaluation phase, because a model that predicts "never defaults" for every single loan would already be 80% "accurate" and therefore completely useless. We will keep that in mind for training and for evaluation metrics.

## Missing data — and why it's missing matters

I went column by column checking missingness and realized that "missing" values should be treated differently in different cases. Therefore I splitted them into three categories:

**1. Missing because it's actually leakage in disguise.** A lot of columns are 97–99% empty — all the hardship-plan columns (`hardship_*`, `payment_plan_start_date`, `deferral_term`, etc.) and settlement columns (`settlement_*`, `debt_settlement_flag_date`). They're empty for almost everyone because they only get filled in once a borrower is already in trouble — hardship plans and settlements happen _after_ someone's struggling to pay, not at the point we are trying to predict from. So these aren't just "missing data," they're leakage, and I added them to `LEAKAGE_COLUMNS`.

**2. Missing because the column just isn't useful.** Things like `member_id` (pure ID), `desc` (free text), the joint application only fields (`sec_app_*`, `dti_joint`, `annual_inc_joint`), and some bureau fields that were only collected for more recent loans (`il_util`, `all_util`, `inq_fi` — around 89% missing). These are missing because not everyone applied jointly or some data collection started later, not because of leakage. I put these in a separate `UNINFORMATIVE_COLUMNS` group because the reason for dropping them is different.

**3. Missing because the event just never happened.** Columns like `mths_since_last_record`, `mths_since_recent_bc_dlq`, `mths_since_last_major_derog`, `mths_since_recent_revol_delinq` are 64–82% empty — but that's not bad data, it's because most people have never had a derogatory mark on their credit, so there's no "months since" to report. If I dropped these or imputed some average number, it would erase a good signal namely never having a delinquency is a strong sign of low risk. So instead I built `handle_sentinel_missingness()` which fills these with a sentinel value and adds a `_never_occurred` flag column, so the model threats is as "this never happened" case instead of losing it.

## More leakage I found

On top of the original 8 columns I'd already excluded, digging through missingness and correlations turned up 20 more (the hardship/settlement ones above) plus two more:

- `last_fico_range_high` / `last_fico_range_low` — this looks like a normal credit score field, but it's the borrower's _most recent_ FICO score pulled during servicing, not their score when they took out the loan. If someone's credit already dropped on the way to default, this field would already show it, same problem as `last_pymnt_d`.

## Features that overlap too much with each other

Ranning a correlation check showed two kinds of overlap:

- A lot of the strongest correlations were just leakage columns correlating with other leakage columns (`recoveries` with `collection_recovery_fee`, `hardship_amount` with `orig_projected_additional_accrued_interest`). Not a problem because once leakage columns are excluded, these pairs are gone too.
- But there's also real overlap among features I actually want to keep: `fico_range_high` and `fico_range_low` are almost identical (LC reports FICO as a fixed band, so they move together), `funded_amnt` and `funded_amnt_inv` are the same loan size number twice, and `open_acc`/`num_sats` are two different ways of counting the same thing (open credit lines).

Instead of trying to manually prune all 119 columns pair by pair, I decided to just start with a curated 15 feature set for the baseline (`configs/baseline_features.py`), picked to avoid this kind of duplication from the start. This matters more for logistic regression than for tree models — correlated inputs mess with how read the coefficients in a logistic regression, since the model can't tell which of two almost identical features should get the weight. I'll revisit the full feature space later for XGBoost since tree models handle correlated features much better.

## What I'm carrying forward into modeling

- Target: the binary good/bad label above, only using loans with a known outcome (Fully Paid / Charged Off / Default) without loans that are still in progress
- Starting features: the curated 15 feature baseline set, not the full raw columns
- Leakage exclusion and the "uninformative" column exclusion
- Need to account for the class imbalance 20% default rate during training
