"""Resumable CNBC historical archive proof-of-concept (maximum seven days).

This collector discovers CNBC title/URL metadata from one official site-map
page per calendar day.  It has no full-range default and rejects ranges longer
than seven days, making an accidental five-year CNBC run impossible here.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import pandas as pd
import yaml

from .cnbc_client import CnbcClient, CnbcError, archive_url


ROOT = Path(__file__).resolve().parents[2]
PILOT_START = date(2021, 9, 1)
PILOT_END = date(2021, 9, 7)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("archive"), dict):
        raise ValueError("CNBC config requires an archive mapping")
    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("CNBC config requires a validation mapping")
    max_days = validation.get("max_days", 7)
    if isinstance(max_days, bool) or not isinstance(max_days, int) or not 1 <= max_days <= 7:
        raise ValueError("validation.max_days must be an integer from 1 through 7")
    return config


def validate_range(start_date: date, end_date: date, max_days: int = 7) -> int:
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("start_date and end_date must be dates")
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")
    day_count = (end_date - start_date).days + 1
    if day_count > max_days:
        raise ValueError(
            f"CNBC proof-of-concept is limited to {max_days} days; requested {day_count}"
        )
    return day_count


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    base = root.resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else str(resolved)


def _next_attempt(directory: Path) -> int:
    attempts = []
    for path in directory.glob("archive.attempt-*.json"):
        try:
            attempts.append(int(path.stem.rsplit("-", 1)[1]))
        except (ValueError, IndexError):
            continue
    return max(attempts, default=0) + 1


def collect(
    config: dict,
    start_date: date,
    end_date: date,
    *,
    raw_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    client=None,
    force: bool = False,
    report_only: bool = False,
) -> dict:
    """Collect or audit a CNBC range and return a reproducibility report."""
    if force and report_only:
        raise ValueError("force cannot be combined with report_only")
    if not isinstance(config, dict) or not isinstance(config.get("archive"), dict):
        raise ValueError("CNBC config requires archive settings")
    validation = config.get("validation", {})
    max_days = validation.get("max_days", 7)
    day_count = validate_range(start_date, end_date, max_days)
    raw_dir = Path(raw_dir or ROOT / "data/raw/news/cnbc")
    report_path = Path(report_path or ROOT / "data/interim/news/cnbc_pilot_report.json")
    config_hash = _fingerprint(config)
    own_client = client is None and not report_only
    worker = None if report_only else (client or CnbcClient(config["archive"]))
    completed_days = []
    failed_days = []
    records_by_archive_day = {}
    cache_hits = 0
    raw_records = 0
    try:
        for offset in range(day_count):
            day = start_date + timedelta(days=offset)
            archive_directory = raw_dir / str(day.year) / day.isoformat()
            checkpoint_path = raw_dir / "_checkpoints" / f"{day.isoformat()}.json"
            checkpoint = None
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            cached_path = None
            if checkpoint and checkpoint.get("status") == "completed":
                raw_file = checkpoint.get("raw_file")
                candidate = (raw_dir / raw_file).resolve() if isinstance(raw_file, str) else None
                if (
                    checkpoint.get("config_sha256") == config_hash
                    and candidate is not None
                    and candidate.is_relative_to(raw_dir.resolve())
                    and candidate.exists()
                ):
                    cached_path = candidate
            if cached_path is None and report_only:
                attempts = sorted(archive_directory.glob("archive.attempt-*.json"))
                cached_path = attempts[-1] if attempts else None
            if cached_path is not None and not force:
                envelope = json.loads(cached_path.read_text(encoding="utf-8"))
                if envelope.get("status") == "completed":
                    count = len(envelope.get("records", []))
                    completed_days.append(day.isoformat())
                    raw_records += count
                    records_by_archive_day[day.isoformat()] = count
                    cache_hits += 1
                    continue
            if report_only:
                failed_days.append(
                    {"date": day.isoformat(), "error": "No completed cached archive attempt"}
                )
                continue

            attempt = _next_attempt(archive_directory)
            identity = {
                "archive_date": day.isoformat(),
                "archive_url": archive_url(day, config["archive"].get("base_url", "https://www.cnbc.com/site-map/articles")),
                "config_sha256": config_hash,
            }
            try:
                page = worker.fetch_day(day)
                envelope = {
                    **identity,
                    "status": "completed",
                    "attempt": attempt,
                    "retrieved_at_utc": page.retrieved_at_utc,
                    "response_sha256": page.response_sha256,
                    "response_bytes": page.response_bytes,
                    "reported_total_count": page.total_count,
                    "records": page.records,
                }
                completed_days.append(day.isoformat())
                raw_records += len(page.records)
                records_by_archive_day[day.isoformat()] = len(page.records)
            except CnbcError as exc:
                envelope = {
                    **identity,
                    "status": "failed",
                    "attempt": attempt,
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "records": [],
                }
                failed_days.append({"date": day.isoformat(), "error": str(exc)})
            raw_path = archive_directory / f"archive.attempt-{attempt:04d}.json"
            _atomic_json(raw_path, envelope)
            _atomic_json(
                checkpoint_path,
                {
                    "archive_date": day.isoformat(),
                    "status": envelope["status"],
                    "attempt": attempt,
                    "config_sha256": config_hash,
                    "raw_file": _relative(raw_path, raw_dir),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    finally:
        if own_client and worker is not None:
            worker.close()

    stats = getattr(worker, "stats", {}) if worker is not None else {}
    report = {
        "pipeline": "cnbc_historical_archive_poc",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "end_date_inclusive": True,
        "maximum_allowed_days": max_days,
        "requested_days": day_count,
        "completed_days": completed_days,
        "failed_days": failed_days,
        "raw_records": raw_records,
        "records_by_archive_day": records_by_archive_day,
        "checkpoint_cache_hits": cache_hits,
        "request_count_this_run": stats.get("request_count", 0),
        "retry_count_this_run": stats.get("retry_count", 0),
        "collection_complete": len(completed_days) == day_count and not failed_days,
        "report_only": report_only,
        "config_sha256": config_hash,
        "configuration": config,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "raw_directory": _relative(raw_dir, ROOT),
        "timestamp_semantics": (
            "Daily archive discovery only; intraday publication timestamps are not acquired"
        ),
        "warnings": [
            "CNBC site-map results are archive discoveries, not an independently proven exhaustive corpus.",
            "Article bodies are not fetched by this proof-of-concept.",
            "This collector rejects ranges longer than seven days.",
        ],
    }
    _atomic_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/cnbc.yaml")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use the fixed September 1-7, 2021 validation range",
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--enrich-existing", action="store_true",
        help="Skip discovery and enrich URLs already in cnbc_news_clean.csv",
    )
    parser.add_argument("--force-enrichment", action="store_true")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/raw/news/cnbc/article_metadata")
    parser.add_argument("--enriched-output", type=Path, default=ROOT / "data/interim/news/cnbc_news_enriched.csv")
    parser.add_argument("--enrichment-report", type=Path, default=ROOT / "data/interim/news/cnbc_enrichment_report.json")
    args = parser.parse_args(argv)
    if args.force and args.report_only:
        parser.error("--force cannot be combined with --report-only")
    config = load_config(args.config)
    if args.pilot:
        args.start_date = args.start_date or date.fromisoformat(config["validation"]["start_date"])
        args.end_date = args.end_date or date.fromisoformat(config["validation"]["end_date"])
    if args.start_date is None or args.end_date is None:
        parser.error("Supply --start-date and --end-date, or --pilot; there is no full-range default")
    try:
        validate_range(args.start_date, args.end_date, config["validation"]["max_days"])
        if args.enrich_existing:
            from .enrich_cnbc import enrich_file

            input_path = ROOT / "data/interim/news/cnbc_news_clean.csv"
            existing = pd.read_csv(input_path, keep_default_na=False)
            dates = pd.to_datetime(existing["publication_date"], errors="raise").dt.date
            if any(day < args.start_date or day > args.end_date for day in dates):
                raise ValueError("Existing CNBC pilot contains publication dates outside the requested enrichment range")
            _, enrichment = enrich_file(
                input_path, args.enriched_output, args.enrichment_report,
                config=config["article_metadata"], cache_dir=args.metadata_dir,
                force=args.force_enrichment,
            )
            print(json.dumps({key: enrichment[key] for key in (
                "total_pilot_articles", "exact_publisher_timestamps", "missing_timestamps",
                "failed_fetches", "cache_hits",
            )}, indent=2))
            return 0
        report = collect(
            config,
            args.start_date,
            args.end_date,
            raw_dir=args.raw_dir,
            report_path=args.report,
            force=args.force,
            report_only=args.report_only,
        )
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "start_date",
                    "end_date",
                    "requested_days",
                    "raw_records",
                    "collection_complete",
                )
            },
            indent=2,
        )
    )
    return 0 if report["collection_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
