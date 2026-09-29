"""Metrics and statistical checks (docs/task2_plan.md, sections 3.4-3.5).

With ~177 test days, a single accuracy number is not enough to claim news
helps. Every check here exists to answer "is this difference real, or could
it be noise at this sample size?" rather than to report one more number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from statsmodels.stats.contingency_tables import mcnemar

from src.modeling.train import attach_tfidf_svd, fit_predict, model_feature_columns, uses_tfidf


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test: does model B disagree-and-win against model A more than
    the reverse, on the same test days?"""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    table = np.array([[np.sum(correct_a & correct_b), a_only], [b_only, np.sum(~correct_a & ~correct_b)]])
    result = mcnemar(table, exact=(a_only + b_only) < 25, correction=True)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "a_only_correct": a_only, "b_only_correct": b_only}


def binomial_test_vs_rate(y_true, y_pred, reference_rate: float) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    correct = int(np.sum(y_true == y_pred))
    n = len(y_true)
    result = stats.binomtest(correct, n, reference_rate, alternative="greater")
    return {"n_correct": correct, "n": n, "reference_rate": reference_rate, "p_value": float(result.pvalue)}


def block_bootstrap_ci(y_true, y_pred, block_size: int, n_resamples: int, random_state: int, ci: float = 0.95) -> dict:
    """Moving-block bootstrap CI: resamples contiguous blocks of days so that
    day-to-day dependence in the series isn't broken, unlike an i.i.d. resample."""
    rng = np.random.default_rng(random_state)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_size))
    accuracies, macro_f1s = [], []
    for _ in range(n_resamples):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]
        accuracies.append(accuracy_score(y_true[idx], y_pred[idx]))
        macro_f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    alpha = (1 - ci) / 2
    return {
        "accuracy_ci": [float(np.quantile(accuracies, alpha)), float(np.quantile(accuracies, 1 - alpha))],
        "macro_f1_ci": [float(np.quantile(macro_f1s, alpha)), float(np.quantile(macro_f1s, 1 - alpha))],
        "accuracy_mean": float(np.mean(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
    }


def walk_forward_evaluate(
    table: pd.DataFrame,
    article_features: pd.DataFrame,
    config: dict,
    best_configs: dict,
    model_ids=("B2", "M4"),
    initial_months: int = 12,
) -> pd.DataFrame:
    """Expanding-window, monthly-refit accuracy for B2 vs M4 (section 3.5).

    Reported separately from the headline test-set result and NOT used for
    model selection: it exists only to show whether the single 177-day test
    window's outcome generalizes across many more out-of-sample months.
    Hyperparameters are fixed to each model's already-tuned `best_configs`
    entry; only the model coefficients (and, for M4, the TF-IDF/SVD fit) are
    refit at each step, on all data strictly before that month.
    """
    dates = table["date"]
    months = dates.dt.to_period("M")
    unique_months = sorted(months.unique())
    random_state = config["models"]["random_state"]

    records = []
    for month in unique_months[initial_months:]:
        train_mask = months < month
        predict_mask = months == month
        if train_mask.sum() < 30 or predict_mask.sum() == 0:
            continue
        train_idx = np.where(train_mask.to_numpy())[0]
        predict_idx = np.where(predict_mask.to_numpy())[0]

        for model_id in model_ids:
            best = best_configs[model_id]
            logreg_params = {k: best[k] for k in ("C", "penalty", "class_weight", "solver")}
            if uses_tfidf(model_id):
                fit_end_date = dates.iloc[train_idx].max()
                working_table, _, _ = attach_tfidf_svd(table, article_features, fit_end_date, config, best["svd_k"])
            else:
                working_table = table
            feature_columns = model_feature_columns(model_id, config, best.get("svd_k"))
            _, pred, _ = fit_predict(working_table, feature_columns, train_idx, predict_idx, logreg_params, random_state)
            y_true = working_table.iloc[predict_idx]["y"].to_numpy()
            records.append({
                "month": str(month),
                "model_id": model_id,
                "n_days": len(predict_idx),
                "accuracy": float(accuracy_score(y_true, pred)),
            })
    return pd.DataFrame.from_records(records)
