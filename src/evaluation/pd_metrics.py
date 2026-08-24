# Evaluation metrics

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# how well the model separates bad loans from good loans
def ks_statistic(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    order = np.argsort(y_pred_proba)
    y_sorted = y_true[order]

    # [0, 0, 0, 1, 1] = > total_good = 3, total_bad = 2
    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        raise ValueError("Need at least one good and one bad to compute KS.")

    # [0, 0, 0, 1, 1, 1, 1] => cum_bad = [0/4, 0/4, 0/4, 1/4, 2/4, 3/4, 4/4]
    #                                  = [0, 0, 0, 0.25, 0.5, 0.75, 1]
    cum_bad = np.cumsum(y_sorted == 1) / total_bad
    # [0, 0, 0, 1, 1, 1, 1] => cum_good = [1/3, 2/3, 3/3, 3/3, 3/3, 3/3, 3/3]
    #                                   = [0.333, 0.667, 1, 1, 1, 1, 1]
    cum_good = np.cumsum(y_sorted == 0) / total_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def gini_coefficient(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def calibration_table(
    y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": np.asarray(y_true), "pred_proba": np.asarray(y_pred_proba)})
    df["decile"] = pd.qcut(df["pred_proba"], q=n_bins, labels=False, duplicates="drop")

    table = (
        df.groupby("decile")
        .agg(
            count=("y_true", "size"),
            predicted_pd=("pred_proba", "mean"),
            actual_default_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["actual_default_rate"] - table["predicted_pd"]
    return table


def evaluate_pd_model(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "ks": ks_statistic(y_true, y_pred_proba),
        "gini": gini_coefficient(y_true, y_pred_proba),
        "calibration": calibration_table(y_true, y_pred_proba, n_bins=n_bins),
    }
