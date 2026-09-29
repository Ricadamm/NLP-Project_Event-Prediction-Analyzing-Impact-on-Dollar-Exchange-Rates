"""
Task 2 - Baseline evaluation + Combined model (historical rates + NLP features).

Saves train/val/test splits to data/ and prints evaluation for:
  1. Naive baselines (majority class, persistence)
  2. Combined model (Logistic Regression on rate lags + NLP features)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "processed" / "model_ready_daily.csv"
SPLIT_DIR = ROOT / "data" / "splits"

NLP_COLS = [
    "news_count",
    "armed_conflict_tfidf_intensity", "armed_conflict_lm_tone", "armed_conflict_count",
    "sanctions_tfidf_intensity", "sanctions_lm_tone", "sanctions_count",
    "trade_conflict_tfidf_intensity", "trade_conflict_lm_tone", "trade_conflict_count",
    "energy_geopolitics_tfidf_intensity", "energy_geopolitics_lm_tone", "energy_geopolitics_count",
    "political_instability_tfidf_intensity", "political_instability_lm_tone", "political_instability_count",
    "monetary_geoeconomic_tfidf_intensity", "monetary_geoeconomic_lm_tone", "monetary_geoeconomic_count",
]

RATE_LAGS = [1, 2, 3, 5]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in RATE_LAGS:
        df[f"return_lag_{lag}"] = df["return_pct"].shift(lag)
        df[f"jisdor_lag_{lag}"] = df["jisdor"].shift(lag)
    df["jisdor_ma_5"] = df["jisdor"].rolling(5).mean()
    df["return_ma_5"] = df["return_pct"].rolling(5).mean()
    return df


def main():
    # 1. Load and prepare
    df = pd.read_csv(DATA_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = add_lag_features(df)
    df = df.dropna(subset=["return_pct"]).reset_index(drop=True)

    rate_cols = [c for c in df.columns if c.startswith(("return_lag", "jisdor_lag", "jisdor_ma", "return_ma"))]

    # 2. Chronological 70/15/15 split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"train: {len(train)} rows  ({train['date'].iloc[0].date()} -> {train['date'].iloc[-1].date()})")
    print(f"val:   {len(val)} rows  ({val['date'].iloc[0].date()} -> {val['date'].iloc[-1].date()})")
    print(f"test:  {len(test)} rows  ({test['date'].iloc[0].date()} -> {test['date'].iloc[-1].date()})")

    # 3. Save splits
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR / "val.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)
    print(f"\nSplits saved to {SPLIT_DIR}/")

    y_train = train["direction"]
    y_val = val["direction"]
    y_test = test["direction"]

    # 4. Naive baselines
    majority = y_train.mode()[0]
    pred_majority = np.full(len(y_test), majority)

    pred_persist = df["direction"].shift(1).iloc[val_end:].values
    persist_valid = ~np.isnan(pred_persist)

    print("\n" + "=" * 50)
    print("NAIVE BASELINES (on test set)")
    print("=" * 50)

    print(f"\nMajority class (always {majority}):")
    print(f"  Accuracy : {accuracy_score(y_test, pred_majority):.4f}")
    print(f"  F1 macro : {f1_score(y_test, pred_majority, average='macro'):.4f}")

    print(f"\nPersistence (predict = yesterday):")
    print(f"  Accuracy : {accuracy_score(y_test[persist_valid], pred_persist[persist_valid]):.4f}")
    print(f"  F1 macro : {f1_score(y_test[persist_valid], pred_persist[persist_valid], average='macro'):.4f}")

    # 5. Combined model: Logistic Regression on rate lags + NLP features
    feature_cols = rate_cols + NLP_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_train = train[feature_cols].fillna(0)
    X_val = val[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    pred_val = model.predict(X_val_s)
    pred_test = model.predict(X_test_s)

    print("\n" + "=" * 50)
    print("COMBINED MODEL - Logistic Regression (rate lags + NLP)")
    print("=" * 50)

    print("\nValidation set:")
    print(f"  Accuracy : {accuracy_score(y_val, pred_val):.4f}")
    print(f"  F1 macro : {f1_score(y_val, pred_val, average='macro'):.4f}")

    print("\nTest set:")
    print(f"  Accuracy : {accuracy_score(y_test, pred_test):.4f}")
    print(f"  F1 macro : {f1_score(y_test, pred_test, average='macro'):.4f}")

    print("\nTest classification report:")
    print(classification_report(y_test, pred_test, target_names=["DOWN (0)", "UP (1)"]))

    # 6. Feature importance
    coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
    print("Top 10 features by |coefficient|:")
    for feat, coef in coefs.head(10).items():
        print(f"  {feat:45s}  {coef:+.4f}")


if __name__ == "__main__":
    main()
