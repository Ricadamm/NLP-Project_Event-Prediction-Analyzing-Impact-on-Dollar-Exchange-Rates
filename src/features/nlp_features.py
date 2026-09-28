"""
Task 2 - NLP feature extraction: aligned CNBC headlines -> daily NLP features + JISDOR target.

Input : data/processed/cnbc_headlines_aligned.csv  (already classified + aligned to trade day)
Output: data/processed/model_ready_daily.csv       (one row per trading day, features + target)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pysentiment2 as ps

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data/processed/cnbc_headlines_aligned.csv"
OUTPUT_CSV = ROOT / "data/processed/model_ready_daily.csv"

CATEGORIES = [
    "armed_conflict", "sanctions", "trade_conflict",
    "energy_geopolitics", "political_instability", "monetary_geoeconomic",
]


def lm_scores(lm: ps.LM, title: str) -> pd.Series:
    tokens = lm.tokenize(title)
    score = lm.get_score(tokens)
    return pd.Series({
        "lm_positive": score["Positive"],
        "lm_negative": score["Negative"],
        "lm_tone": score["Polarity"],               # overall tone: positive vs negative
        "lm_opinion_ratio": score["Subjectivity"],   # how much of the headline is sentiment-loaded
    })


def build_features(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    # 1. Load data
    df = pd.read_csv(input_csv)
    df["title"] = df["title"].astype(str)
    df["matched_categories"] = df["matched_categories"].apply(ast.literal_eval)

    # one True/False column per category, from the already-provided labels
    for cat in CATEGORIES:
        df[cat] = df["matched_categories"].apply(lambda cats: cat in cats)

    # 2. TF-IDF -> single intensity number per headline
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(df["title"])
    df["tfidf_intensity"] = np.asarray(tfidf_matrix.sum(axis=1)).ravel()

    # 3. Loughran-McDonald -> sentiment scores per headline
    lm = ps.LM()
    df = df.join(df["title"].apply(lambda t: lm_scores(lm, t)))

    # 4. Aggregate to one row per trading day, per category
    feature_cols = ["tfidf_intensity", "lm_positive", "lm_negative", "lm_tone", "lm_opinion_ratio"]

    daily = df.groupby("effective_trade_date").agg(
        news_count=("title", "count"),
        jisdor=("jisdor", "first"),
        return_pct=("return_pct", "first"),
    ).reset_index().rename(columns={"effective_trade_date": "date"})
    daily = daily.set_index("date")

    for cat in CATEGORIES:
        sub = df[df[cat]]
        cat_daily = sub.groupby("effective_trade_date").agg(
            {**{c: "mean" for c in feature_cols}, "title": "count"}
        )
        cat_daily = cat_daily.rename(columns={"title": "count", **{c: f"{cat}_{c}" for c in feature_cols}})
        cat_daily = cat_daily.rename(columns={"count": f"{cat}_count"})
        daily = daily.join(cat_daily)

    daily = daily.reset_index()

    # 5. Target label: direction of return (UP=1 / DOWN=0)
    daily["direction"] = (daily["return_pct"] > 0).astype(int)

    # 6. Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)
    return daily


if __name__ == "__main__":
    result = build_features()
    print(f"rows: {len(result)}, columns: {len(result.columns)}")
    print(f"saved to: {OUTPUT_CSV}")
    print(result.head())
