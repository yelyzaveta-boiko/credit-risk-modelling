# Logistic regression PD model
# WOE linearizes each feature's relationship with default
# Each fitted coefficient is directly interpretable as how much that feature's
# bucket shifts the log odds of default.


from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def fit_baseline_pd_model(
    X_train: pd.DataFrame, y_train: pd.Series, **kwargs
) -> LogisticRegression:
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def coefficient_report(model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    # higher WOE always = safer bucket = more good loans than bad
    # => a model should give every feature a negative coefficient
    # higher WOE pushing the log odds of default down
    # A positive sign is a red flag meaning multicollinearity between WOE features
    coefs = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coefs["sign_as_expected"] = coefs["coefficient"] < 0
    return coefs.sort_values("coefficient").reset_index(drop=True)


def quick_auc(model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> float:
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)
