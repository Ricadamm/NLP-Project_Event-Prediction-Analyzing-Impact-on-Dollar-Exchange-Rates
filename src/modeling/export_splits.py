"""Export the chronological train/validation/test splits to data/splits/.

This script does not define a new split. It calls the same
`build_modeling_table` / `compute_split_bounds` used by
`src/pipeline/run_task2.py`, so the exported rows are exactly the rows every
model was tuned and evaluated on (docs/task2_plan.md, section 3.1).

What the files contain:
- identifiers and the target: `date`, `jisdor`, `return_pct`,
  `next_return_pct`, `y` (1 if the next trading day's return is > 0);
- price features (lagged returns, momentum, volatility, MA distance,
  day-of-week dummies);
- fit-free NLP features: news/category counts (raw and log1p) and
  Loughran-McDonald tone (overall and per category).

What they deliberately do NOT contain: TF-IDF/SVD components. Those are
learned from text, so they are refit on each training window inside the
modeling step (train only while tuning, train+validation for the final model)
and cannot be stored as one static column without leaking later headlines
into earlier rows.

The 1-day purge rows at each split boundary are excluded from every file,
exactly as in modeling.

Usage:
    python -m src.modeling.export_splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.modeling.dataset import (
    build_modeling_table,
    count_feature_columns,
    lm_feature_columns,
    price_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data/splits"


def split_columns(config: dict) -> list[str]:
    """Columns written to every split file, in a readable order."""
    categories = config["categories"]
    identifiers = ["date", "jisdor", "return_pct", "next_return_pct", "y"]
    raw_counts = ["news_count"] + [f"{c}_count" for c in categories]
    return (
        identifiers
        + price_feature_columns(config)
        + raw_counts
        + count_feature_columns(categories)
        + lm_feature_columns(categories)
    )


def export_splits(
    config_path: str | Path = ROOT / "config/task2.yaml",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    table, splits, config = build_modeling_table(config_path)
    columns = split_columns(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"source": "src.modeling.dataset.build_modeling_table", "splits": {}}
    for name, idx_key in (("train", "train_idx"), ("validation", "val_idx"), ("test", "test_idx")):
        frame = table.iloc[splits[idx_key]][columns].copy()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        frame.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
        summary["splits"][name] = {
            "rows": len(frame),
            "start_date": frame["date"].iloc[0],
            "end_date": frame["date"].iloc[-1],
            "share_next_day_up": round(float(frame["y"].mean()), 4),
        }

    summary["purged_boundary_dates"] = [
        str(table["date"].iloc[i].date())
        for i in range(len(table))
        if i not in set(splits["train_idx"]) | set(splits["val_idx"]) | set(splits["test_idx"])
    ]
    summary["columns"] = columns
    summary["not_included"] = (
        "TF-IDF/SVD components: refit per training window inside src/modeling/train.py "
        "(see docs/task2_plan.md, section 1.3)."
    )

    # Consistency check against the last full pipeline run, if one exists:
    # the exported split sizes must match what the models were evaluated on.
    results_path = ROOT / "reports/task2_results.json"
    if results_path.exists():
        recorded = json.loads(results_path.read_text(encoding="utf-8"))["split_sizes"]
        exported = {name: info["rows"] for name, info in summary["splits"].items()}
        if recorded != exported:
            raise RuntimeError(
                f"Exported split sizes {exported} differ from reports/task2_results.json {recorded}"
            )
        summary["matches_reports_task2_results"] = True

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/task2.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = export_splits(args.config, args.output_dir)
    print(json.dumps(summary["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
