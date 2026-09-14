"""Clean GDELT candidate metadata without making relevance decisions.

``published_at_utc`` and ``published_at_wib`` retain the requested downstream
schema, but represent GDELT's ``seendate`` (first seen/indexed time), NOT a
verified publisher publication time. For repeated URLs we use the earliest
valid first-seen timestamp and keep every source record in JSON provenance.

Raw files are read only. Invalid URLs are quarantined, not silently discarded
or combined into a shared missing-URL article. Missing titles and invalid
timestamps on otherwise valid URLs remain in the output with quality flags.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import tempfile
from typing import Any
import unicodedata
from urllib.parse import urlsplit

import pandas as pd


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TRACKING_KEYS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
)
COLUMNS = [
    "article_id", "original_url", "normalized_url", "title",
    "published_at_utc", "published_at_wib", "domain", "language",
    "sourcecountry", "categories", "socialimage", "retrieved_at_utc",
    "url_mobile", "seendate", "timestamp_semantics", "original_urls",
    "source_record_count", "quality_flags", "retrieval_provenance",
]
IDENTITY_FIELDS = ("query", "topic", "domain", "logical_start", "logical_end", "api_config")


def normalize_url(value: Any) -> str | None:
    """Validate an HTTP(S) URL and remove only five known UTM query keys.

    Leading/trailing whitespace is trimmed; the original value is preserved
    separately. Keep scheme/host case, ports, path escapes, fragments, and the
    order, duplicates and spelling of remaining query parameters. In particular
    do not parse and re-encode the query: that can change article identity.
    Encoded parameter names are not decoded to infer tracking parameters.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127
           for character in candidate):
        return None
    if re.search(r"%(?![0-9A-Fa-f]{2})", candidate):
        return None
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        # Accessing port also validates a malformed or out-of-range port.
        _ = parsed.port
        if "\\" in parsed.netloc or parsed.hostname.startswith("."):
            return None
    except ValueError:
        return None

    before_fragment, fragment_marker, fragment = candidate.partition("#")
    base, query_marker, query = before_fragment.partition("?")
    if query_marker:
        parts = query.split("&")
        retained = [part for part in parts
                    if part.partition("=")[0].lower() not in TRACKING_KEYS]
        if len(retained) != len(parts):
            before_fragment = base + ("?" + "&".join(retained) if retained else "")
    return before_fragment + fragment_marker + fragment


def article_id(normalized_url: str) -> str:
    """Stable SHA-256 identity; callers must first validate/normalize the URL."""
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()


def clean_title(value: Any) -> str:
    """NFC and whitespace normalization only; retain case, numbers/punctuation."""
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split())


def parse_timestamp(value: Any) -> pd.Timestamp:
    """Accept compact GDELT UTC or aware ISO timestamps; reject naive values."""
    try:
        if isinstance(value, str):
            value = value.strip()
            if re.fullmatch(r"\d{8}T\d{6}Z", value):
                parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
            else:
                parsed = datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            parsed = value
        else:
            return pd.NaT
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return pd.NaT
        return pd.Timestamp(parsed).tz_convert("UTC").as_unit("ns")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _json_safe(value: Any) -> Any:
    """Keep JSON records intact; represent Python missing scalars as JSON null."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, datetime) and not pd.isna(value):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _text(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value or "(missing)" for value in values).items()))


def clean_records(records: Iterable[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict]:
    """Return one row per valid normalized URL and a calculated QA report.

    Results are independent of input order. Scalar metadata is selected from
    the first nonempty value in earliest-valid-seendate / canonical-JSON order.
    Categories are a sorted JSON list. ``retrieval_provenance`` is a sorted
    JSON list of EVERY source record, including identical repeated retrievals.
    ``quality_flags`` aggregates problems in any source occurrence.

    Counts reconcile as raw_returned_records = unique_articles +
    duplicate_records + quarantined_records. Missing URLs are a subset of
    invalid_urls; missing timestamps are a subset of invalid_timestamps. Raw
    category counts and unique-article language/domain counts are distinguished
    by the report's count_definitions field.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    raw_records: list[dict] = []
    quarantine: list[dict] = []
    issues = Counter()
    raw_categories: list[str] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("Each news record must be a mapping (JSON object)")
        record = _json_safe(raw)
        raw_records.append(record)
        normalized = normalize_url(record.get("url"))
        title = clean_title(record.get("title"))
        timestamp = parse_timestamp(record.get("seendate"))
        category = _text(record.get("query_category")).strip()
        raw_categories.append(category)
        flags = []
        if not _text(record.get("url")):
            issues["missing_urls"] += 1
            flags.append("missing_url")
        if normalized is None:
            issues["invalid_urls"] += 1
            flags.append("invalid_url")
        if not title:
            issues["missing_titles"] += 1
            flags.append("missing_title")
        if not _text(record.get("seendate")):
            issues["missing_timestamps"] += 1
            flags.append("missing_timestamp")
        if pd.isna(timestamp):
            issues["invalid_timestamps"] += 1
            flags.append("invalid_timestamp")
        if flags:
            issues["records_with_quality_issues"] += 1
        if normalized is None:
            quarantine.append({"reason_codes": sorted(flags), "record": record})
        else:
            groups[normalized].append({
                "record": record, "title": title, "timestamp": timestamp,
                "category": category, "flags": flags, "canonical": _json(record),
            })

    rows = []
    for normalized, observations in sorted(groups.items()):
        observations.sort(key=lambda item: (
            pd.isna(item["timestamp"]),
            item["timestamp"].value if not pd.isna(item["timestamp"]) else 0,
            item["canonical"],
        ))

        def first_text(field: str) -> str:
            return next((_text(item["record"].get(field)) for item in observations
                         if _text(item["record"].get(field))), "")

        categories = sorted({item["category"] for item in observations if item["category"]})
        flags = sorted({flag for item in observations for flag in item["flags"]})
        row = {field: first_text(field) for field in (
            "domain", "language", "sourcecountry", "socialimage",
            "retrieved_at_utc", "url_mobile", "seendate",
        )}
        row.update({
            "article_id": article_id(normalized),
            "original_url": first_text("url"),
            "normalized_url": normalized,
            "title": next((item["title"] for item in observations if item["title"]), ""),
            "published_at_utc": observations[0]["timestamp"],
            "categories": _json(categories),
            "timestamp_semantics": "gdelt_first_seen",
            "original_urls": _json(sorted({item["record"]["url"] for item in observations})),
            "source_record_count": len(observations),
            "quality_flags": _json(flags),
            "retrieval_provenance": _json([
                item["record"] for item in sorted(observations, key=lambda item: item["canonical"])
            ]),
        })
        rows.append(row)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    # Pin nanosecond resolution even for an empty frame; pandas may otherwise
    # choose platform/default second resolution and make the output schema vary.
    frame["published_at_utc"] = pd.to_datetime(
        frame["published_at_utc"], utc=True
    ).astype("datetime64[ns, UTC]")
    frame["published_at_wib"] = frame["published_at_utc"].dt.tz_convert("Asia/Jakarta")
    frame = frame.sort_values(["published_at_utc", "normalized_url"], na_position="last").reset_index(drop=True)
    times = frame["published_at_utc"].dropna()
    quarantine.sort(key=_json)
    report = {
        "raw_returned_records": len(raw_records),
        "unique_articles": len(frame),
        "duplicate_records": len(raw_records) - len(quarantine) - len(frame),
        "quarantined_records": len(quarantine),
        **{name: issues[name] for name in (
            "missing_urls", "invalid_urls", "missing_titles", "missing_timestamps",
            "invalid_timestamps", "records_with_quality_issues",
        )},
        "unique_articles_with_quality_issues": int((frame["quality_flags"] != "[]").sum()),
        "records_by_category": _counts(raw_categories),
        "unique_articles_by_category": _counts(
            category for categories in frame["categories"] for category in json.loads(categories)
        ),
        "records_by_language": _counts(frame["language"]),
        "records_by_domain": _counts(frame["domain"]),
        "raw_records_by_language": _counts(_text(record.get("language")) for record in raw_records),
        "raw_records_by_domain": _counts(_text(record.get("domain")) for record in raw_records),
        "earliest_article": times.min().isoformat() if not times.empty else None,
        "latest_article": times.max().isoformat() if not times.empty else None,
        "timestamp_semantics": "GDELT seendate is first-seen/indexed time, not verified publication time",
        "count_definitions": {
            "records_by_category": "Raw retrieval occurrences, including quarantined records",
            "records_by_language": "Deduplicated valid-URL articles; deterministic representative metadata",
            "records_by_domain": "Deduplicated valid-URL articles; deterministic representative metadata",
            "duplicate_records": "Valid-URL occurrences minus unique normalized URLs",
            "invalid_urls": "Includes missing_urls; every invalid-URL occurrence is quarantined separately",
            "invalid_timestamps": "Includes missing_timestamps; counted over raw retrieval occurrences",
            "quality_flags": "Union of issues observed in any source occurrence of the URL",
        },
        "quarantine_records": quarantine,
    }
    return frame, report


def _read_envelope(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        envelope = json.load(handle)
    if not isinstance(envelope, dict):
        raise ValueError(f"Expected a collector JSON object: {path}")
    return envelope


def _terminal_records(envelope: dict, path: Path) -> list[dict]:
    if envelope.get("status") not in {"completed", "saturated"}:
        return []
    batch = envelope.get("records")
    if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
        raise ValueError(f"Terminal collector envelope must contain a records list: {path}")
    return batch


def read_raw_records(path: str | Path) -> list[dict]:
    """Read active completed/saturated collector attempts recursively.

    Checkpoint ``raw_file`` references are authoritative when present. Without
    a checkpoint (for example after cloning tracked raw samples), choose the
    greatest numbered attempt for each full query/window/config identity.
    A latest failed/pending/split attempt never revives an older completion.
    Only split parents activate their child windows; this also prevents stale
    children from reappearing after a forced parent re-fetch stops splitting.

    An explicit file reads that particular envelope. Directory input includes
    all query/config groups found below it, without applying a study-date
    filter. The collector itself supplies records scoped to its requested run.
    Malformed active evidence raises an error instead of silently losing data.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return _terminal_records(_read_envelope(path), path)

    selected: dict[str, tuple[int, Path, dict]] = {}
    for file in sorted(path.rglob("*.json")):
        if "_checkpoints" in file.relative_to(path).parts:
            continue
        envelope = _read_envelope(file)
        identity = envelope.get("identity")
        if identity is not None and not isinstance(identity, dict):
            raise ValueError(f"Collector identity must be a JSON object: {file}")
        # Identity-free envelopes can still be inspected as standalone samples.
        key = _json(identity) if identity is not None else str(file)
        match = re.search(r"\.attempt-(\d+)\.json$", file.name)
        attempt = int(match.group(1)) if match else 0
        candidate = (attempt, file, envelope)
        if key not in selected or (attempt, str(file)) > (selected[key][0], str(selected[key][1])):
            selected[key] = candidate

    checkpoint_dir = path / "_checkpoints"
    for checkpoint in sorted(checkpoint_dir.glob("*.json")):
        state = _read_envelope(checkpoint)
        if not all(field in state for field in IDENTITY_FIELDS):
            raise ValueError(f"Checkpoint lacks full logical-window identity: {checkpoint}")
        identity = {field: state[field] for field in IDENTITY_FIELDS}
        status = state.get("status")
        if status not in {"completed", "saturated", "split", "failed", "pending"}:
            raise ValueError(f"Invalid checkpoint status: {checkpoint}")
        key = _json(identity)
        if status in {"failed", "pending"}:
            # An interrupted newest attempt may not have produced a raw file.
            envelope = {"identity": identity, "status": status, "records": []}
            selected[key] = (int(state.get("attempt", 0)), checkpoint, envelope)
            continue
        raw_file = state.get("raw_file")
        if not isinstance(raw_file, str):
            raise ValueError(f"Checkpoint lacks raw_file reference: {checkpoint}")
        referenced = (path / raw_file).resolve()
        if not referenced.is_relative_to(path):
            raise ValueError(f"Checkpoint raw_file escapes input directory: {checkpoint}")
        envelope = _read_envelope(referenced)
        if envelope.get("identity") != identity or envelope.get("status") != status:
            raise ValueError(f"Checkpoint/raw identity or status mismatch: {checkpoint}")
        selected[key] = (int(state.get("attempt", 0)), referenced, envelope)

    # Group related window trees; query, source, topic and complete API config
    # remain part of identity, so independently configured runs never collide.
    contexts: dict[str, list[tuple]] = defaultdict(list)
    ungrouped = []
    for _, file, envelope in selected.values():
        identity = envelope.get("identity", {})
        if not all(field in identity for field in IDENTITY_FIELDS):
            ungrouped.append((file, envelope))
            continue
        start = parse_timestamp(identity["logical_start"])
        end = parse_timestamp(identity["logical_end"])
        if pd.isna(start) or pd.isna(end) or end <= start:
            raise ValueError(f"Invalid collector logical window: {file}")
        context = _json({key: value for key, value in identity.items()
                         if key not in {"logical_start", "logical_end"}})
        contexts[context].append((start, end, file, envelope))

    active = list(ungrouped)
    for windows in contexts.values():
        ancestors = []
        for start, end, file, envelope in sorted(windows, key=lambda item: (item[0].value, -item[1].value)):
            ancestors = [(left, right, status) for left, right, status in ancestors if right > start]
            blocked = any(left <= start and end <= right and status != "split"
                          for left, right, status in ancestors)
            if not blocked:
                active.append((file, envelope))
            ancestors.append((start, end, envelope.get("status")))
    return [record for file, envelope in sorted(active, key=lambda item: str(item[0]))
            for record in _terminal_records(envelope, file)]


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _report_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def write_clean_outputs(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    report_path: str | Path,
    quarantine_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Write CSV, QA JSON and lossless quarantine JSON without touching raw data."""
    output_path, report_path = Path(output_path), Path(report_path)
    quarantine_path = Path(quarantine_path) if quarantine_path else output_path.with_name("gdelt_news_quarantine.json")
    paths = [output_path.resolve(), report_path.resolve(), quarantine_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("CSV, report and quarantine output paths must be different")
    frame, report = clean_records(records)
    report.update({"output_path": _report_path(output_path), "quarantine_path": _report_path(quarantine_path)})
    _write_atomic(output_path, frame.to_csv(index=False, lineterminator="\n"))
    _write_atomic(quarantine_path, json.dumps(report["quarantine_records"], ensure_ascii=False, indent=2) + "\n")
    _write_atomic(report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    LOGGER.info("News cleaning: raw=%d unique=%d duplicates=%d quarantined=%d output=%s",
                report["raw_returned_records"], report["unique_articles"],
                report["duplicate_records"], report["quarantined_records"], output_path)
    return frame, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/news/gdelt",
                        help="Collector envelope JSON file or directory (recursive)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/news/gdelt_news_all_candidates.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "data/interim/news/gdelt_news_all_candidates_report.json")
    parser.add_argument("--quarantine", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.quarantine is None:
        args.quarantine = args.output.with_name(f"{args.output.stem}_quarantine.json")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw_path = args.input.resolve()
    output_paths = [args.output.resolve(), args.report.resolve(),
                    args.quarantine.resolve()]
    if any(target == raw_path or (raw_path.is_dir() and target.is_relative_to(raw_path))
           for target in output_paths):
        parser.error("Clean output paths must be outside the raw input file/directory")
    write_clean_outputs(read_raw_records(args.input), args.output, args.report, args.quarantine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
