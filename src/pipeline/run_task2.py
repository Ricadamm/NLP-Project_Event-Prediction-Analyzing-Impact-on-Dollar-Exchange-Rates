"""Orchestrate the full Task 2 pipeline: features, modeling, evaluation.

Mirrors src/pipeline/run_task1.py's shape: each stage's output is written to
disk, a JSON manifest records what ran, and a set of built-in checks
(docs/task2_plan.md, section 4) fail the run rather than silently producing a
leaky or inconsistent result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import warnings

# This machine's BLAS backend emits spurious, verified-harmless "encountered
# in matmul" RuntimeWarnings from sklearn's sparse@dense products during
# TF-IDF/SVD fitting (see src/features/tfidf_features.py for the direct
# finiteness check). Suppressed once here so a full run's output stays
# readable; src/features/tfidf_features.py still hard-fails if the output is
# ever actually non-finite.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.build_daily_features import build_daily_features
from src.features.tfidf_features import top_terms_per_component
from src.modeling import baselines
from src.modeling.dataset import build_modeling_table
from src.modeling.evaluate import (
    binomial_test_vs_rate,
    block_bootstrap_ci,
    classification_metrics,
    mcnemar_test,
    walk_forward_evaluate,
)
from src.modeling.train import ALL_MODEL_IDS, NLP_MODEL_IDS, refit_and_predict_test, tune_model, uses_tfidf

ROOT = Path(__file__).resolve().parents[2]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_leakage_checks(table: pd.DataFrame, splits: dict) -> dict:
    checks = {}

    expected_y = (table["next_return_pct"] > 0).astype(int)
    checks["target_matches_next_day_sign"] = bool((table["y"] == expected_y).all())

    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    checks["no_duplicate_rows_across_splits"] = bool(len(all_idx) == len(set(all_idx.tolist())))
    checks["train_before_val"] = bool(train_idx.max() < val_idx.min())
    checks["val_before_test"] = bool(val_idx.max() < test_idx.min())
    checks["purge_gap_train_val"] = int(val_idx.min() - train_idx.max())
    checks["purge_gap_val_test"] = int(test_idx.min() - val_idx.max())
    checks["purge_gaps_at_least_one_day"] = bool(checks["purge_gap_train_val"] >= 2 and checks["purge_gap_val_test"] >= 2)

    return checks


def plot_test_predictions(test_predictions: pd.DataFrame, model_id: str, figures_dir: Path) -> Path:
    """Predicted probability vs. actual direction over the test period (section 3.6)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    dates = pd.to_datetime(test_predictions["date"])
    proba = test_predictions[f"{model_id}_proba"]
    y_true = test_predictions["y_true"]

    ax.plot(dates, proba, color="#2b6cb0", linewidth=1.2, label=f"{model_id} predicted P(up)")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    up_days = dates[y_true == 1]
    down_days = dates[y_true == 0]
    ax.scatter(up_days, [1.03] * len(up_days), marker="|", color="#2f855a", s=40, label="actual: up")
    ax.scatter(down_days, [-0.03] * len(down_days), marker="|", color="#c53030", s=40, label="actual: down")
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel("Predicted P(next day up)")
    ax.set_title(f"{model_id}: predicted probability vs. realized direction, test period")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = figures_dir / f"{model_id.lower()}_test_predictions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(test_metrics: dict, figures_dir: Path) -> Path:
    """Bar chart of test accuracy and macro-F1 for every model (section 3.6)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_ids = ["B0", "B1", "B2", "M1", "M2", "M3", "M4"]
    accuracy = [test_metrics[m]["accuracy"] for m in model_ids]
    macro_f1 = [test_metrics[m]["macro_f1"] for m in model_ids]

    x = np.arange(len(model_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, accuracy, width, label="Directional accuracy", color="#2b6cb0")
    ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#dd6b20")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(model_ids)
    ax.set_ylim(0, max(accuracy + macro_f1) * 1.2)
    ax.set_title("Test-set accuracy and macro-F1 by model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = figures_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_task2(config_path: str | Path = ROOT / "config/task2.yaml") -> dict:
    manifest = {"pipeline": "task2_nlp_features_and_modeling", "status": "running", "stages": {}}

    feature_report = build_daily_features(config_path)
    manifest["stages"]["daily_features"] = {
        "status": "passed" if feature_report["counts_reconcile"] else "failed",
        "article_rows": feature_report["article_rows"],
    }
    if not feature_report["counts_reconcile"]:
        raise RuntimeError("Daily feature counts do not reconcile with Task 1 output")

    table, splits, config = build_modeling_table(config_path)
    article_features = pd.read_csv(
        Path(config["outputs"]["interim_dir"]) / "cnbc_headline_features.csv",
        parse_dates=["effective_trade_date"],
    )

    leakage_checks = run_leakage_checks(table, splits)
    manifest["stages"]["leakage_checks"] = leakage_checks
    if not all(v is True or isinstance(v, int) for v in leakage_checks.values()) or not (
        leakage_checks["target_matches_next_day_sign"]
        and leakage_checks["no_duplicate_rows_across_splits"]
        and leakage_checks["train_before_val"]
        and leakage_checks["val_before_test"]
        and leakage_checks["purge_gaps_at_least_one_day"]
    ):
        raise RuntimeError(f"Leakage check failed: {leakage_checks}")

    reports_dir = Path(config["outputs"]["reports_dir"])
    processed_dir = Path(config["outputs"]["processed_dir"])

    # --- Baselines (B0, B1) ---------------------------------------------
    y_train = table.iloc[splits["train_idx"]]["y"]
    b0_label = baselines.majority_class_rate(y_train)
    b0_rate = max(y_train.mean(), 1 - y_train.mean())

    test_predictions = pd.DataFrame({"date": table["date"].iloc[splits["test_idx"]].to_numpy(), "y_true": table["y"].iloc[splits["test_idx"]].to_numpy()})
    test_predictions["B0_pred"] = baselines.predict_majority_class(b0_label, len(splits["test_idx"]))
    test_predictions["B1_pred"] = baselines.predict_persistence(table, splits["test_idx"])

    # --- Tune B2, M1-M4 on validation ------------------------------------
    tuning_records = []
    best_configs = {}
    for model_id in ALL_MODEL_IDS:
        best, records = tune_model(model_id, table, article_features, splits, config)
        best_configs[model_id] = best
        tuning_records.extend(records)
    tuning_log = pd.DataFrame.from_records(tuning_records)
    _atomic_write(reports_dir / "task2_tuning_log.csv", tuning_log.to_csv(index=False, lineterminator="\n"))

    # --- Refit winners on train+val, predict test once -------------------
    test_metrics, svd_top_terms, coefficients = {}, {}, {}
    fitted = {}
    for model_id in ALL_MODEL_IDS:
        result = refit_and_predict_test(model_id, table, article_features, splits, config, best_configs[model_id])
        fitted[model_id] = result
        test_predictions[f"{model_id}_pred"] = result["test_pred"]
        test_predictions[f"{model_id}_proba"] = result["test_proba"]
        if uses_tfidf(model_id):
            svd_top_terms[model_id] = top_terms_per_component(result["vectorizer"], result["svd"])
        logreg = result["pipeline"].named_steps["logreg"]
        coefficients[model_id] = dict(zip(result["feature_columns"], logreg.coef_[0].tolist()))

    y_test = test_predictions["y_true"].to_numpy()
    for model_id in ("B0", "B1", *ALL_MODEL_IDS):
        pred = test_predictions[f"{model_id}_pred"].to_numpy()
        test_metrics[model_id] = classification_metrics(y_test, pred)
        test_metrics[model_id]["binomial_vs_b0"] = binomial_test_vs_rate(y_test, pred, b0_rate)
        test_metrics[model_id]["bootstrap"] = block_bootstrap_ci(
            y_test, pred,
            config["evaluation"]["bootstrap"]["block_size"],
            config["evaluation"]["bootstrap"]["n_resamples"],
            config["evaluation"]["bootstrap"]["random_state"],
        )

    # --- Primary significance test: M4 (full model) vs B2 (price-only) ---
    mcnemar_m4_vs_b2 = mcnemar_test(y_test, test_predictions["B2_pred"], test_predictions["M4_pred"])

    best_nlp_model = max(NLP_MODEL_IDS, key=lambda m: test_metrics[m]["accuracy"])
    mcnemar_best_nlp_vs_b2 = mcnemar_test(y_test, test_predictions["B2_pred"], test_predictions[f"{best_nlp_model}_pred"])

    decision = {
        "primary_comparison": "M4_vs_B2",
        "m4_beats_b2_accuracy": bool(test_metrics["M4"]["accuracy"] > test_metrics["B2"]["accuracy"]),
        "m4_beats_b2_macro_f1": bool(test_metrics["M4"]["macro_f1"] > test_metrics["B2"]["macro_f1"]),
        "mcnemar_m4_vs_b2_p_value": mcnemar_m4_vs_b2["p_value"],
        "best_nlp_model_by_test_accuracy": best_nlp_model,
        "mcnemar_best_nlp_vs_b2_p_value": mcnemar_best_nlp_vs_b2["p_value"],
    }
    decision["news_adds_predictive_value"] = bool(
        decision["m4_beats_b2_accuracy"] and decision["m4_beats_b2_macro_f1"] and mcnemar_m4_vs_b2["p_value"] < 0.05
    )
    decision["verdict"] = (
        "News adds predictive value at this sample size."
        if decision["news_adds_predictive_value"]
        else "No evidence of added value at this sample size (see docs/task2_plan.md section 3.5)."
    )

    # --- Walk-forward robustness check (supplementary, not for selection) -
    walk_forward = walk_forward_evaluate(table, article_features, config, best_configs, model_ids=("B2", "M4"))
    _atomic_write(reports_dir / "task2_walk_forward.csv", walk_forward.to_csv(index=False, lineterminator="\n"))
    walk_forward_summary = (
        walk_forward.groupby("model_id")["accuracy"].agg(["mean", "std", "count"]).to_dict(orient="index")
        if not walk_forward.empty else {}
    )

    _atomic_write(processed_dir / "task2_test_predictions.csv", test_predictions.to_csv(index=False, lineterminator="\n"))
    _atomic_write(reports_dir / "svd_top_terms.json", json.dumps(svd_top_terms, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(reports_dir / "task2_coefficients.json", json.dumps(coefficients, ensure_ascii=False, indent=2) + "\n")

    figures_dir = reports_dir / "figures"
    plot_test_predictions(test_predictions, "M4", figures_dir)
    plot_test_predictions(test_predictions, "B2", figures_dir)
    plot_model_comparison(test_metrics, figures_dir)

    results = {
        "splits": {k: v for k, v in splits.items() if not k.endswith("_idx")},
        "split_sizes": {
            "train": int(len(splits["train_idx"])), "validation": int(len(splits["val_idx"])), "test": int(len(splits["test_idx"])),
        },
        "class_balance": {
            "train": float(y_train.mean()),
            "validation": float(table.iloc[splits["val_idx"]]["y"].mean()),
            "test": float(y_test.mean()),
        },
        "b0_majority_label": b0_label,
        "best_configs": {k: {kk: vv for kk, vv in v.items()} for k, v in best_configs.items()},
        "test_metrics": test_metrics,
        "decision": decision,
        "walk_forward_summary": walk_forward_summary,
        "leakage_checks": leakage_checks,
        "feature_report_summary": {
            "lm_coverage_rate": feature_report["lm_coverage_rate"],
            "category_vs_lm_neg_correlation": feature_report["category_vs_lm_neg_correlation"],
        },
    }
    _atomic_write(reports_dir / "task2_results.json", json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    manifest["stages"]["tuning"] = {"status": "passed", "configs_tried": len(tuning_records)}
    manifest["stages"]["final_test_evaluation"] = {"status": "passed", "decision": decision["verdict"]}
    manifest["status"] = "passed"
    _atomic_write(reports_dir / "task2_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    return {"manifest": manifest, "results": results, "fitted": fitted, "test_predictions": test_predictions, "table": table, "splits": splits}


def main() -> int:
    outcome = run_task2()
    print(json.dumps(outcome["results"]["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
