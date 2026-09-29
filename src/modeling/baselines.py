"""B0 (majority class) and B1 (persistence) baselines (docs/task2_plan.md, section 2.3).

Both are parameter-light enough that "fitting" just means reading a rate or a
column off the training split; there is nothing here that could leak future
information if used correctly, but callers must still only compute B0's rate
from `train_idx`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def majority_class_rate(y_train: pd.Series) -> int:
    """B0: the majority class label learned from the training split."""
    return int(y_train.mean() >= 0.5)


def predict_majority_class(majority_label: int, n: int) -> np.ndarray:
    return np.full(n, majority_label, dtype=int)


def predict_persistence(table: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """B1: predict tomorrow's direction = today's realized direction.

    `ret_lag_0` is today's own (already realized) return, so this uses no
    future information.
    """
    return (table["ret_lag_0"].iloc[idx] > 0).astype(int).to_numpy()
