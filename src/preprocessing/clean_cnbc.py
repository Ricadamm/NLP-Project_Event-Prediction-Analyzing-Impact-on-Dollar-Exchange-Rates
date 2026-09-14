"""Clean CNBC daily archive discovery records into a date-level news table.

CNBC's archive payload supplies titles and canonical URLs, but no intraday
publication timestamp.  The cleaner extracts and validates the calendar date
encoded by CNBC in each canonical URL and labels that provenance explicitly;
it never fabricates midnight or another publication time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [
    "article_id",
    "original_url",
    "normalized_url",
    "title",
    "publication_date",
    "published_at_utc",
    "published_at_wib",
    "publisher",
    "source_type",
    "discovered_for_date",
    "archive_url",
    "retrieved_at_utc",
    "timestamp_semantics",
    "source_record_count",
    "quality_flags",
    "retrieval_provenance",
]


def clean_title(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_cnbc_url(value) -> str | None:
    """Return a conservative canonical CNBC URL or ``None`` when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if any(character.isspace() or ord(character) < 32 for character in candidate):
        return None
    try:
        parsed = urlsplit(candidate)
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if (parsed.hostname or "").lower() not in {"cnbc.com", "www.cnbc.com"}:
        return None
    if not parsed.path.startswith("/"):
        return None
    # Scheme/host case and fragments are not article identity. CNBC archive
    # URLs do not require query parameters, which are commonly tracking data.
    return urlunsplit(("https", "www.cnbc.com", parsed.path, "", ""))


def publication_date_from_url(url: str) -> date | None:
    path = urlsplit(url).path
    if not path.endswith(".html"):
        return None
    # Standard stories begin with the date, while CNBC also publishes valid
    # advertorial and slash-containing slug paths around the same date segment.
    match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", path)
    if not match:
        return None
    try:
        parsed = date(*(int(part) for part in match.groups()))
    except ValueError:
        return None
    return parsed


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def clean_records(
    records: Iterable[Mapping],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be supplied together")
    if start_date is not None and end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    groups: dict[str, list[dict]] = defaultdict(list)
    quarantine = []
    scope_exclusions = []
    issues = Counter()
    raw_count = 0
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("Each CNBC record must be a mapping")
        raw_count += 1
        record = dict(raw)
        normalized = normalize_cnbc_url(record.get("url"))
        title = clean_title(record.get("title"))
        pub_date = publication_date_from_url(normalized) if normalized else None
        discovered_raw = record.get("discovered_for_date")
        try:
            discovered = date.fromisoformat(discovered_raw) if isinstance(discovered_raw, str) else None
        except ValueError:
            discovered = None
        flags = []
        if normalized is None:
            flags.append("invalid_url")
            issues["invalid_urls"] += 1
        if pub_date is None:
            flags.append("missing_url_publication_date")
            issues["missing_url_publication_dates"] += 1
        if not title:
            flags.append("missing_title")
            issues["missing_titles"] += 1
        if discovered is None:
            flags.append("invalid_discovery_date")
            issues["invalid_discovery_dates"] += 1
        if pub_date is not None and discovered is not None and pub_date != discovered:
            # CNBC archive pages use updatedDate internally, so a canonical URL
            # date mismatch is possible and must remain visible.
            flags.append("url_date_differs_from_archive_date")
            issues["url_date_mismatches"] += 1
        if normalized is None or pub_date is None:
            quarantine.append({"reason_codes": sorted(flags), "record": record})
            continue
        if start_date is not None and not start_date <= pub_date <= end_date:
            flags.append("publication_date_outside_requested_range")
            issues["out_of_scope_publication_dates"] += 1
            scope_exclusions.append({"reason_codes": sorted(flags), "record": record})
            continue
        groups[normalized].append(
            {
                "record": record,
                "title": title,
                "publication_date": pub_date,
                "discovered": discovered,
                "flags": flags,
                "canonical": _canonical_json(record),
            }
        )

    rows = []
    for normalized, observations in sorted(groups.items()):
        observations.sort(key=lambda item: item["canonical"])
        first = observations[0]["record"]
        title = next((item["title"] for item in observations if item["title"]), "")
        flags = sorted({flag for item in observations for flag in item["flags"]})
        discovered = sorted(
            {item["discovered"].isoformat() for item in observations if item["discovered"]}
        )
        rows.append(
            {
                "article_id": _article_id(normalized),
                "original_url": next(
                    (item["record"].get("url") for item in observations if item["record"].get("url")),
                    "",
                ),
                "normalized_url": normalized,
                "title": title,
                "publication_date": observations[0]["publication_date"].isoformat(),
                "published_at_utc": "",
                "published_at_wib": "",
                "publisher": "CNBC",
                "source_type": first.get("__typename", ""),
                "discovered_for_date": _canonical_json(discovered),
                "archive_url": first.get("archive_url", ""),
                "retrieved_at_utc": first.get("retrieved_at_utc", ""),
                "timestamp_semantics": "date_only_from_cnbc_canonical_url",
                "source_record_count": len(observations),
                "quality_flags": _canonical_json(flags),
                "retrieval_provenance": _canonical_json(
                    [item["record"] for item in observations]
                ),
            }
        )

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["publication_date", "normalized_url"]).reset_index(drop=True)
    duplicate_count = raw_count - len(frame) - len(quarantine) - len(scope_exclusions)
    report = {
        "raw_records": raw_count,
        "clean_articles": len(frame),
        "duplicate_records": duplicate_count,
        "quarantined_records": len(quarantine),
        "scope_excluded_records": len(scope_exclusions),
        "invalid_urls": issues["invalid_urls"],
        "missing_url_publication_dates": issues["missing_url_publication_dates"],
        "missing_titles": issues["missing_titles"],
        "invalid_discovery_dates": issues["invalid_discovery_dates"],
        "url_date_mismatches": issues["url_date_mismatches"],
        "requested_start_date": start_date.isoformat() if start_date else None,
        "requested_end_date": end_date.isoformat() if end_date else None,
        "earliest_publication_date": frame["publication_date"].min() if not frame.empty else None,
        "latest_publication_date": frame["publication_date"].max() if not frame.empty else None,
        "timestamp_semantics": (
            "CNBC archive discovery provides no intraday publication time; publication_date "
            "is parsed from the canonical CNBC URL and timestamps remain empty"
        ),
        "counts_reconcile": raw_count == (
            len(frame) + duplicate_count + len(quarantine) + len(scope_exclusions)
        ),
        "count_definitions": {
            "clean_articles": "Unique valid canonical URLs inside the requested publication-date range",
            "duplicate_records": "In-scope valid occurrences minus unique normalized URLs",
            "quarantined_records": "Invalid or undated records; excludes valid out-of-scope records",
            "scope_excluded_records": "Valid records outside the requested publication-date range",
        },
        "quarantine_records": sorted(quarantine + scope_exclusions, key=_canonical_json),
    }
    return frame, report


def read_raw_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(
        file for file in path.rglob("*.json")
        if "_checkpoints" not in file.parts and "article_metadata" not in file.parts
    )
    latest: dict[str, tuple[int, Path, dict]] = {}
    for file in files:
        envelope = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise ValueError(f"Expected JSON object: {file}")
        day = str(envelope.get("archive_date", file))
        match = re.search(r"\.attempt-(\d+)\.json$", file.name)
        attempt = int(match.group(1)) if match else 0
        if day not in latest or attempt > latest[day][0]:
            latest[day] = (attempt, file, envelope)
    records = []
    for _, file, envelope in sorted(latest.values(), key=lambda item: str(item[1])):
        if envelope.get("status") != "completed":
            continue
        batch = envelope.get("records")
        if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
            raise ValueError(f"Completed CNBC envelope has invalid records: {file}")
        records.extend(batch)
    return records


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


def write_clean_outputs(
    records: Iterable[Mapping],
    output_path: str | Path,
    report_path: str | Path,
    quarantine_path: str | Path | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    output_path, report_path = Path(output_path), Path(report_path)
    quarantine_path = Path(quarantine_path) if quarantine_path else output_path.with_name(
        "cnbc_news_quarantine.json"
    )
    resolved = {path.resolve() for path in (output_path, report_path, quarantine_path)}
    if len(resolved) != 3:
        raise ValueError("CNBC CSV, report, and quarantine paths must be different")
    frame, report = clean_records(records, start_date=start_date, end_date=end_date)
    report.update(
        {
            "output_path": str(output_path),
            "quarantine_path": str(quarantine_path),
        }
    )
    _atomic_text(output_path, frame.to_csv(index=False, lineterminator="\n"))
    _atomic_text(
        quarantine_path,
        json.dumps(report["quarantine_records"], ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return frame, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/news/cnbc")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/interim/news/cnbc_news_clean.csv"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/interim/news/cnbc_news_cleaning_report.json",
    )
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args(argv)
    raw = args.input.resolve()
    targets = [args.output.resolve(), args.report.resolve()]
    if args.quarantine:
        targets.append(args.quarantine.resolve())
    if any(target == raw or (raw.is_dir() and target.is_relative_to(raw)) for target in targets):
        parser.error("CNBC clean outputs must be outside the raw input path")
    write_clean_outputs(read_raw_records(args.input), args.output, args.report, args.quarantine,
                        start_date=args.start_date, end_date=args.end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
