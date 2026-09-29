"""Assemble the modeling table: price features, target, NLP features, splits.

Everything here uses only information available by trading day `t` for the
prediction made at `t` (docs/task2_plan.md, sections 2.1-2.2). The chronological
70/15/15 split with a 1-day purge is computed once from the JISDOR calendar
and reused by every model, per section 3.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config/task2.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_price_features(jisdor_daily: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Lagged returns, momentum, volatility, MA distance, day-of-week, target.

    `return_pct_t` is the JISDOR return already realized by day t (Task 1's
    `cnbc_jisdor_daily.csv`), so `lag_returns=[0,1,2,3,4]` means "today's
    return through 4 trading days ago" -- all known at time t.
    """
    df = jisdor_daily.sort_values("date").reset_index(drop=True).copy()
    price_cfg = config["price_features"]

    for lag in price_cfg["lag_returns"]:
        df[f"ret_lag_{lag}"] = df["return_pct"].shift(lag)

    for window in price_cfg["momentum_windows"]:
        df[f"momentum_{window}"] = df["return_pct"].rolling(window).mean()

    vol_window = price_cfg["volatility_window"]
    df[f"volatility_{vol_window}"] = df["return_pct"].rolling(vol_window).std()

    ma_window = price_cfg["ma_distance_window"]
    moving_average = df["jisdor"].rolling(ma_window).mean()
    df[f"ma_distance_{ma_window}"] = df["jisdor"] / moving_average - 1

    # JISDOR trades weekdays only, so dayofweek is always in {0..4}; drop_first
    # avoids the dummy-variable trap (the 5 dummies would otherwise sum to 1).
    dow = pd.get_dummies(df["date"].dt.dayofweek, prefix="dow", drop_first=True)
    df = pd.concat([df, dow], axis=1)

    # Target: next-day direction. y_t uses return_pct at t+1 (docs/task2_plan.md, section 2.1).
    df["next_return_pct"] = df["return_pct"].shift(-1)
    df["y"] = (df["next_return_pct"] > 0).astype(int)
    df.loc[df["next_return_pct"].isna(), "y"] = np.nan

    warm_up = price_cfg["warm_up_days"]
    df = df.iloc[warm_up:].reset_index(drop=True)
    df = df.loc[df["next_return_pct"].notna()].reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    return df


def merge_nlp_features(modeling_table: pd.DataFrame, nlp_daily: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Left-join NLP features and log1p-transform the volume counts."""
    merged = modeling_table.merge(nlp_daily, on="date", how="left", suffixes=("", "_nlp"))
    count_columns = ["news_count"] + [f"{c}_count" for c in categories]
    for column in count_columns:
        merged[f"log_{column}"] = np.log1p(merged[column])
    return merged


def compute_split_bounds(dates: pd.Series, config: dict) -> dict:
    """Chronological 70/15/15 index boundaries with a 1-day purge at each edge."""
    n = len(dates)
    split_cfg = config["split"]
    train_end = int(n * split_cfg["train_fraction"])
    val_end = train_end + int(n * split_cfg["validation_fraction"])
    purge = split_cfg["purge_days"]

    train_idx = np.arange(0, train_end - purge)
    val_idx = np.arange(train_end, val_end - purge)
    test_idx = np.arange(val_end, n)

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_end_date": str(dates.iloc[train_end - 1].date()),
        "val_end_date": str(dates.iloc[val_end - 1].date()),
        "test_start_date": str(dates.iloc[val_end].date()),
        "test_end_date": str(dates.iloc[-1].date()),
    }


def price_feature_columns(config: dict) -> list[str]:
    price_cfg = config["price_features"]
    columns = [f"ret_lag_{lag}" for lag in price_cfg["lag_returns"]]
    columns += [f"momentum_{w}" for w in price_cfg["momentum_windows"]]
    columns += [f"volatility_{price_cfg['volatility_window']}"]
    columns += [f"ma_distance_{price_cfg['ma_distance_window']}"]
    columns += ["dow_1", "dow_2", "dow_3", "dow_4"]  # dow_0 (Monday) is the dropped reference
    return columns


def count_feature_columns(categories: list[str]) -> list[str]:
    return ["log_news_count"] + [f"log_{c}_count" for c in categories]


def lm_feature_columns(categories: list[str]) -> list[str]:
    overall = ["lm_neg", "lm_pos", "lm_unc", "lm_net"]
    per_category = [f"{c}_lm_net" for c in categories] + [f"{c}_lm_neg" for c in categories]
    return overall + per_category


def build_modeling_table(config_path: str | Path = ROOT / "config/task2.yaml") -> tuple[pd.DataFrame, dict, dict]:
    config = load_config(config_path)
    jisdor_daily = pd.read_csv(config["inputs"]["jisdor_daily"], parse_dates=["date"])
    nlp_daily = pd.read_csv(config["outputs"]["processed_dir"] + "/nlp_daily_features.csv", parse_dates=["date"])

    priced = build_price_features(jisdor_daily, config)
    table = merge_nlp_features(priced, nlp_daily, config["categories"])
    for column in ["dow_1", "dow_2", "dow_3", "dow_4"]:
        if column not in table.columns:
            table[column] = 0
        table[column] = table[column].fillna(0).astype(int)

    price_rows_with_nan = table[price_feature_columns(config)].isna().any(axis=1).sum()
    if price_rows_with_nan:
        raise RuntimeError(f"{price_rows_with_nan} rows still have NaN price features after warm-up drop")

    splits = compute_split_bounds(table["date"], config)
    return table, splits, config
