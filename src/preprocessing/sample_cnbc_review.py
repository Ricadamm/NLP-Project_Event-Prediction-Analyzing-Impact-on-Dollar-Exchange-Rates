"""Create a deterministic, category-balanced CNBC candidate review sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/cnbc_jisdor_aligned_v2.csv"
DEFAULT_OUTPUT = ROOT / "data/interim/news/cnbc_manual_review_sample.csv"
DEFAULT_REPORT = ROOT / "data/interim/news/cnbc_manual_review_sample_report.json"
COLUMNS = [
    "article_id", "published_at_wib", "title", "url", "section",
    "matched_categories", "matched_keywords", "filter_reason",
    "human_relevant", "human_primary_category", "human_notes",
]


def _decode_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return []


def _candidate(value) -> bool:
    return value is True or (isinstance(value, str) and value.casefold() == "true")


def _stable_key(article_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{article_id}".encode("utf-8")).hexdigest()


def _category_sequence(rows: list[dict], seed: int) -> deque:
    """Round-robin a category across its publication-date/section strata."""
    strata = defaultdict(list)
    for row in rows:
        publication_date = str(row.get("published_at_wib", ""))[:10] or str(row.get("canonical_url_date", ""))
        section = str(row.get("section", "")) or "(missing)"
        strata[(publication_date, section)].append(row)
    queues = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda row: _stable_key(str(row["article_id"]), seed))
        queues.append(deque(ordered))
    result = deque()
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return result


def sample_candidates(frame: pd.DataFrame, n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    required = {"article_id", "is_geopolitical_candidate", "matched_categories"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input is missing columns: {sorted(required - set(frame.columns))}")
    candidates = [row for row in frame.to_dict(orient="records") if _candidate(row["is_geopolitical_candidate"])]
    category_groups = defaultdict(list)
    for row in candidates:
        categories = sorted(_decode_list(row.get("matched_categories")))
        primary = categories[0] if categories else "(uncategorized)"
        category_groups[primary].append(row)
    category_queues = {
        category: _category_sequence(rows, seed)
        for category, rows in sorted(category_groups.items())
    }
    selected = []
    target = min(n, len(candidates))
    while len(selected) < target:
        progressed = False
        for category in sorted(category_queues):
            queue = category_queues[category]
            if queue and len(selected) < target:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    output_rows = []
    for row in selected:
        output_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "published_at_wib": row.get("published_at_wib", ""),
                "title": row.get("title", ""),
                "url": row.get("normalized_url", row.get("url", "")),
                "section": row.get("section", ""),
                "matched_categories": row.get("matched_categories", "[]"),
                "matched_keywords": row.get("matched_keywords", "[]"),
                "filter_reason": row.get("filter_reason", ""),
                "human_relevant": "",
                "human_primary_category": "",
                "human_notes": "",
            }
        )
    sample = pd.DataFrame(output_rows, columns=COLUMNS)
    sampled_category_counts = Counter(
        sorted(_decode_list(row["matched_categories"]))[0]
        if _decode_list(row["matched_categories"]) else "(uncategorized)"
        for row in output_rows
    )
    report = {
        "candidate_population": len(candidates),
        "requested_sample_size": n,
        "sample_size": len(sample),
        "all_candidates_included": len(candidates) <= n,
        "random_seed": seed,
        "sampling_method": (
            "Assign each candidate to its alphabetically first matched category; balance categories "
            "round-robin; within each category round-robin publication-date/section strata; order "
            "within strata by SHA-256(seed:article_id)."
        ),
        "sampled_primary_category_counts": dict(sorted(sampled_category_counts.items())),
        "human_columns_blank": all(
            sample[column].eq("").all()
            for column in ("human_relevant", "human_primary_category", "human_notes")
        ) if not sample.empty else True,
    }
    return sample, report


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sample_file(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    *,
    n: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(input_path, keep_default_na=False)
    sample, report = sample_candidates(frame, n=n, seed=seed)
    report.update({"input_path": str(Path(input_path)), "output_path": str(Path(output_path))})
    _atomic_text(Path(output_path), sample.to_csv(index=False, lineterminator="\n"))
    _atomic_text(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return sample, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    sample_file(args.input, args.output, args.report, n=args.n, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
