"""Grid search on validation, then a single refit-and-predict on test.

docs/task2_plan.md, sections 2.4 and 3.1-3.3. TF-IDF/SVD is fit fresh for
every `svd_k` tried, using only headlines up to the fit window's end date
(train dates while tuning; train+val dates for the final model) -- this is
the leakage boundary the whole design protects (section 1.3).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.tfidf_features import daily_mean_svd, fit_tfidf_svd, top_terms_per_component
from src.modeling.dataset import count_feature_columns, lm_feature_columns, price_feature_columns

ROOT = Path(__file__).resolve().parents[2]

NLP_MODEL_IDS = ("M1", "M2", "M3", "M4")
ALL_MODEL_IDS = ("B2", "M1", "M2", "M3", "M4")


def uses_tfidf(model_id: str) -> bool:
    return model_id in ("M3", "M4")


def model_feature_columns(model_id: str, config: dict, svd_k: int | None = None) -> list[str]:
    categories = config["categories"]
    columns = list(price_feature_columns(config))
    if model_id in ("M1", "M4"):
        columns += count_feature_columns(categories)
    if model_id in ("M2", "M4"):
        columns += lm_feature_columns(categories)
    if model_id in ("M3", "M4"):
        columns += [f"svd_{i}" for i in range(svd_k)]
    return columns


def logistic_grid(config: dict) -> list[dict]:
    grid_cfg = config["models"]["logistic_grid"]
    solver_map = config["models"]["solver_by_penalty"]
    combos = []
    for C, penalty, class_weight in itertools.product(
        grid_cfg["C"], grid_cfg["penalty"], grid_cfg["class_weight"]
    ):
        combos.append({"C": C, "penalty": penalty, "class_weight": class_weight, "solver": solver_map[penalty]})
    return combos


def fit_predict(
    table: pd.DataFrame,
    feature_columns: list[str],
    fit_idx: np.ndarray,
    predict_idx: np.ndarray,
    logreg_params: dict,
    random_state: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    X_fit = table.iloc[fit_idx][feature_columns].to_numpy()
    y_fit = table.iloc[fit_idx]["y"].to_numpy()
    X_pred = table.iloc[predict_idx][feature_columns].to_numpy()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, random_state=random_state, **logreg_params)),
    ])
    pipeline.fit(X_fit, y_fit)
    predictions = pipeline.predict(X_pred)
    probabilities = pipeline.predict_proba(X_pred)[:, 1]
    return pipeline, predictions, probabilities


def attach_tfidf_svd(
    table: pd.DataFrame,
    article_features: pd.DataFrame,
    fit_end_date,
    config: dict,
    svd_k: int,
):
    """Return a copy of `table` with svd_0..svd_{k-1} columns attached.

    Fit on headlines with effective_trade_date <= fit_end_date only; every
    row of `table` (train/val/test alike) is transformed with that fixed fit
    and its daily mean SVD, aligned by position (not merged), since both
    `table["date"]` and the aggregated daily frame are already unique,
    identically ordered date sequences.
    """
    texts = article_features["normalized_title"]
    dates = article_features["effective_trade_date"]
    fit_mask = dates <= fit_end_date

    tfidf_cfg = config["tfidf"]
    vectorizer, svd, svd_matrix = fit_tfidf_svd(
        texts, fit_mask,
        ngram_range=tfidf_cfg["ngram_range"], min_df=tfidf_cfg["min_df"], max_df=tfidf_cfg["max_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"], max_features=tfidf_cfg["max_features"],
        svd_k=svd_k, random_state=tfidf_cfg["random_state"],
    )
    daily = daily_mean_svd(svd_matrix, dates)
    daily = daily.reindex(table["date"]).fillna(0.0)
    assert len(daily) == len(table), "daily SVD reindex must match table row count exactly"

    working = table.copy().reset_index(drop=True)
    for column in daily.columns:
        working[column] = daily[column].to_numpy()
    return working, vectorizer, svd


def tune_model(model_id: str, table: pd.DataFrame, article_features: pd.DataFrame, splits: dict, config: dict):
    """Grid search `model_id` on validation. Returns (best_config, all_tried_records)."""
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]
    random_state = config["models"]["random_state"]
    svd_k_grid = config["tfidf"]["svd_k_grid"] if uses_tfidf(model_id) else [None]
    logreg_combos = logistic_grid(config)

    records, best = [], None
    for svd_k in svd_k_grid:
        if uses_tfidf(model_id):
            fit_end_date = table["date"].iloc[train_idx].max()
            working_table, _, _ = attach_tfidf_svd(table, article_features, fit_end_date, config, svd_k)
        else:
            working_table = table

        feature_columns = model_feature_columns(model_id, config, svd_k)
        for params in logreg_combos:
            _, val_pred, _ = fit_predict(working_table, feature_columns, train_idx, val_idx, params, random_state)
            y_val = working_table.iloc[val_idx]["y"].to_numpy()
            accuracy = accuracy_score(y_val, val_pred)
            macro_f1 = f1_score(y_val, val_pred, average="macro")
            record = {
                "model_id": model_id, "svd_k": svd_k, **params,
                "val_accuracy": accuracy, "val_macro_f1": macro_f1,
            }
            records.append(record)
            if best is None or (accuracy, macro_f1) > (best["val_accuracy"], best["val_macro_f1"]):
                best = record
    return best, records


def refit_and_predict_test(model_id: str, table: pd.DataFrame, article_features: pd.DataFrame, splits: dict, config: dict, best_config: dict):
    """Refit the winning config on train+val, predict test once (section 3.3)."""
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    train_val_idx = np.concatenate([train_idx, val_idx])
    random_state = config["models"]["random_state"]
    svd_k = best_config.get("svd_k")
    logreg_params = {k: best_config[k] for k in ("C", "penalty", "class_weight", "solver")}

    vectorizer = svd = None
    if uses_tfidf(model_id):
        fit_end_date = table["date"].iloc[train_val_idx].max()
        working_table, vectorizer, svd = attach_tfidf_svd(table, article_features, fit_end_date, config, svd_k)
    else:
        working_table = table

    feature_columns = model_feature_columns(model_id, config, svd_k)
    pipeline, test_pred, test_proba = fit_predict(
        working_table, feature_columns, train_val_idx, test_idx, logreg_params, random_state
    )
    return {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "test_pred": test_pred,
        "test_proba": test_proba,
        "vectorizer": vectorizer,
        "svd": svd,
        "working_table": working_table,
    }
