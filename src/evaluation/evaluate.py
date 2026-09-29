from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"

train = pd.read_csv(SPLITS_DIR / "train.csv")
val = pd.read_csv(SPLITS_DIR / "val.csv")
test = pd.read_csv(SPLITS_DIR / "test.csv")

y_train = train["direction"]
y_test = test["direction"]

print(f"train: {len(train)} rows, val: {len(val)} rows, test: {len(test)} rows")
print(f"test class balance: {y_test.value_counts().to_dict()}")

# majority class baseline
majority = y_train.mode()[0]
pred_majority = np.full(len(y_test), majority)

print("\n--- Majority Class ---")
print(f"accuracy:  {accuracy_score(y_test, pred_majority):.4f}")
print(f"f1 macro:  {f1_score(y_test, pred_majority, average='macro'):.4f}")

# persistence baseline
full = pd.concat([train, val, test], ignore_index=True)
pred_persist = full["direction"].shift(1).iloc[len(train) + len(val):].values
valid = ~np.isnan(pred_persist)

print("\n--- Persistence ---")
print(f"accuracy:  {accuracy_score(y_test[valid], pred_persist[valid]):.4f}")
print(f"f1 macro:  {f1_score(y_test[valid], pred_persist[valid], average='macro'):.4f}")

# combined model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rate_cols = [c for c in train.columns if c.startswith(("return_lag", "jisdor_lag", "jisdor_ma", "return_ma"))]
nlp_cols = [c for c in train.columns if c.endswith(("_tfidf_intensity", "_lm_tone", "_count")) and c != "news_count"]
feature_cols = rate_cols + nlp_cols

X_train = train[feature_cols].fillna(0)
X_test = test[feature_cols].fillna(0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
pred_lr = model.predict(X_test_s)

print("\n--- Combined LR (rate + NLP) ---")
print(f"accuracy:  {accuracy_score(y_test, pred_lr):.4f}")
print(f"f1 macro:  {f1_score(y_test, pred_lr, average='macro'):.4f}")
print()
print(classification_report(y_test, pred_lr, target_names=["DOWN", "UP"]))
