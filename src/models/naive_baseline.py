"""
Task 2 - Naive baseline models for JISDOR direction prediction.

Two naive baselines (no news features, no modeling - just simple rules):
  1. Majority class: always predict the most common direction seen in training.
  2. Persistence: predict today's direction will be the same as yesterday's.

Input : data/processed/model_ready_daily.csv
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_CSV = Path(r"C:\Users\MSI\Downloads\model_ready_daily.csv")

# 1. Load data, sorted chronologically
df = pd.read_csv(DATA_CSV)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Chronological 70/15/15 split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"val:   {len(val)} rows ({val['date'].min().date()} to {val['date'].max().date()})")
print(f"test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")

# 3. Baseline 1: majority class (learned from train only)
majority_class = train["direction"].mode()[0]
pred_majority = [majority_class] * len(test)

# 4. Baseline 2: persistence (predict same as previous day's actual direction)
# shift(1) on the full df keeps the join to "yesterday" correct even across the train/test boundary
df["pred_persistence"] = df["direction"].shift(1)
pred_persistence = df.loc[test.index, "pred_persistence"]
# first row of the whole dataset has no "yesterday" - drop it if it fell into test
valid = pred_persistence.notna()

y_true = test["direction"]

# 5. Evaluate
print("\n--- Majority class baseline ---")
print(f"always predicts: {majority_class}")
print(f"accuracy: {accuracy_score(y_true, pred_majority):.3f}")
print(f"f1 (macro): {f1_score(y_true, pred_majority, average='macro'):.3f}")

print("\n--- Persistence baseline (predict = yesterday's direction) ---")
print(f"accuracy: {accuracy_score(y_true[valid], pred_persistence[valid]):.3f}")
print(f"f1 (macro): {f1_score(y_true[valid], pred_persistence[valid], average='macro'):.3f}")

print(f"\ntest set direction distribution:\n{y_true.value_counts(normalize=True)}")
