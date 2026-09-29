"""Build article-level and daily NLP features from cnbc_headlines_aligned.csv.

Produces the fit-free half of the feature set (docs/task2_plan.md, section
1.5): geopolitical category membership (reused from Task 1), Loughran-McDonald
tone, and news-volume counts, aggregated to the JISDOR trading-day calendar.

TF-IDF/SVD is deliberately NOT built here: it must be fit on a training
window only (src/features/tfidf_features.py), so it is built inside the
modeling step (src/modeling/train.py) instead of once, up front.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import yaml

from src.features.lm_sentiment import load_lexicon, score_dataframe
from src.features.text_preprocess import normalize_title

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


def load_config(path: str | Path = ROOT / "config/task2.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def add_category_dummies(headlines: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    parsed = headlines["matched_categories"].apply(json.loads)
    for category in categories:
        headlines[f"cat_{category}"] = parsed.apply(lambda cats, c=category: int(c in cats))
    return headlines


def build_article_features(config: dict) -> tuple[pd.DataFrame, dict]:
    headlines = pd.read_csv(config["inputs"]["headlines_aligned"], parse_dates=["effective_trade_date"])
    n_raw = len(headlines)

    headlines["normalized_title"] = headlines["title"].apply(
        lambda t: normalize_title(t, config["text_preprocessing"]["strip_prefixes"])
    )

    # docs/task2_plan.md section 1.2 calls for dropping exact duplicate titles
    # on the same trading day. In this dataset that is only 6 rows (3 pairs)
    # across 10,997 articles -- checked, not assumed. Dropping them would
    # break the harder invariant in section 4 (daily counts must reconcile
    # exactly with the already-committed cnbc_jisdor_daily.csv), for a change
    # too small to matter to any downstream feature. They are kept and only
    # counted here for transparency.
    duplicate_mask = headlines.duplicated(subset=["effective_trade_date", "normalized_title"], keep="first")
    n_duplicates = int(duplicate_mask.sum())

    headlines = add_category_dummies(headlines, config["categories"])

    lexicon = load_lexicon(config["inputs"]["lm_dictionary"])
    lm_scores = score_dataframe(
        headlines["normalized_title"],
        lexicon,
        config["lm_sentiment"]["negation_words"],
        config["lm_sentiment"]["negation_window"],
    )
    headlines = pd.concat([headlines, lm_scores], axis=1)

    # Known-overlap check (docs/task2_plan.md, section 1.4): some taxonomy
    # keywords are themselves LM Negative words, so lm_neg should correlate
    # at least weakly, but not so strongly it is just re-deriving the counts.
    category_lm_neg_correlation = {
        category: float(headlines["lm_neg"].corr(headlines[f"cat_{category}"]))
        for category in config["categories"]
    }

    report = {
        "input_rows": n_raw,
        "duplicate_titles_found_same_day_kept_for_reconciliation": n_duplicates,
        "article_rows": len(headlines),
        "lm_dictionary_path": str(config["inputs"]["lm_dictionary"]),
        "lm_coverage_rate": float(headlines["lm_has_hit"].mean()),
        "lm_zero_token_headlines": int((headlines["n_tokens"] == 0).sum()),
        "category_article_counts": {
            category: int(headlines[f"cat_{category}"].sum()) for category in config["categories"]
        },
        "category_vs_lm_neg_correlation": category_lm_neg_correlation,
    }
    return headlines, report


def aggregate_daily(headlines: pd.DataFrame, jisdor_daily: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    grouped = headlines.groupby("effective_trade_date")
    daily = grouped.agg(
        news_count=("article_id", "count"),
        lm_neg=("lm_neg", "mean"),
        lm_pos=("lm_pos", "mean"),
        lm_unc=("lm_unc", "mean"),
        lm_net=("lm_net", "mean"),
    )
    for category in categories:
        cat_col = f"cat_{category}"
        daily[f"{category}_count"] = grouped[cat_col].sum()
        cat_rows = headlines[headlines[cat_col] == 1]
        cat_daily = cat_rows.groupby("effective_trade_date")[["lm_net", "lm_neg"]].mean()
        daily[f"{category}_lm_net"] = cat_daily["lm_net"]
        daily[f"{category}_lm_neg"] = cat_daily["lm_neg"]

    daily = daily.reset_index().rename(columns={"effective_trade_date": "date"})

    calendar = jisdor_daily[["date"]].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    merged = calendar.merge(daily, on="date", how="left")

    count_columns = ["news_count"] + [f"{c}_count" for c in categories]
    mean_columns = [c for c in merged.columns if c not in count_columns + ["date"]]
    merged[count_columns] = merged[count_columns].fillna(0).astype(int)
    merged[mean_columns] = merged[mean_columns].fillna(0.0)
    return merged


def build_daily_features(config_path: str | Path = ROOT / "config/task2.yaml") -> dict:
    config = load_config(config_path)
    article_features, report = build_article_features(config)

    jisdor_daily = pd.read_csv(config["inputs"]["jisdor_daily"], parse_dates=["date"])
    daily_features = aggregate_daily(article_features, jisdor_daily, config["categories"])

    zero_news_days = int((daily_features["news_count"] == 0).sum())
    report["jisdor_trading_days"] = len(jisdor_daily)
    report["trading_days_with_zero_news"] = zero_news_days
    report["trading_days_with_news"] = len(jisdor_daily) - zero_news_days

    # Reconciliation against the already-committed Task 1 daily counts: this
    # feature table's raw counts must match exactly (docs/task2_plan.md, section 4).
    reconciliation = daily_features.merge(
        jisdor_daily[["date", "news_count"] + [f"{c}_count" for c in config["categories"]]],
        on="date", suffixes=("_features", "_task1"),
    )
    mismatches = {}
    for column in ["news_count"] + [f"{c}_count" for c in config["categories"]]:
        diff = reconciliation[f"{column}_features"] - reconciliation[f"{column}_task1"]
        mismatches[column] = int((diff != 0).sum())
    report["count_reconciliation_mismatches_by_column"] = mismatches
    report["counts_reconcile"] = all(v == 0 for v in mismatches.values())

    interim_dir = Path(config["outputs"]["interim_dir"])
    processed_dir = Path(config["outputs"]["processed_dir"])
    article_path = interim_dir / "cnbc_headline_features.csv"
    daily_path = processed_dir / "nlp_daily_features.csv"
    report_path = processed_dir / "nlp_feature_report.json"

    drop_cols = ["matched_categories"]
    _atomic_write(article_path, article_features.drop(columns=drop_cols).to_csv(index=False, lineterminator="\n"))
    _atomic_write(daily_path, daily_features.to_csv(index=False, lineterminator="\n"))
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    if not report["counts_reconcile"]:
        raise RuntimeError(f"Daily feature counts do not reconcile with Task 1 output: {mismatches}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/task2.yaml")
    args = parser.parse_args()
    report = build_daily_features(args.config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
