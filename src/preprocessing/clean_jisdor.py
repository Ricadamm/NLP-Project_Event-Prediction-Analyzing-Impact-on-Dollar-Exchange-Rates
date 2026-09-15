"""Extract and validate the Bank Indonesia workbook without altering observations.

Only wholly blank rows below the named table header are ignored. Every other
row must have a valid date and finite positive rate; duplicate dates fail. Excel
dates must be datetime cells or date strings, never unformatted serial numbers.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
from numbers import Number
import os
from pathlib import Path
import re
import tempfile

import pandas as pd


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/interim/jisdor/jisdor_clean.csv"
DEFAULT_REPORT = REPOSITORY_ROOT / "data/interim/jisdor/jisdor_cleaning_report.json"
EXPECTED = {"row_count": 1202, "start_date": "2021-09-01", "end_date": "2026-09-01", "duplicate_dates": 0}
POLICIES = {
    "blank_rows": "Ignore only entirely blank rows below the header.",
    "invalid_or_missing_values": "Fail; retain source row and raw values in this report.",
    "duplicates": "Fail on every duplicate date, including identical observations.",
    "date_parsing": "pandas parsing of datetime cells or complete YYYY-MM-DD / MM/DD/YYYY strings with optional time; reject numeric serials and partial dates; normalize daily.",
    "rate_parsing": "Keep numeric cells unchanged; strip Rp/IDR and whitespace in strings; groups of three after dot/comma are thousands; otherwise accept dot/comma decimals.",
    "financial_values": "Require finite positive rates; no imputation, rescaling, interpolation, or outlier removal.",
    "expected_values": "Comparison only; discrepancies do not modify or reject otherwise valid observations.",
}


class JisdorValidationError(ValueError):
    """A failed source validation; report contains diagnostic records."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


def _blank(value) -> bool:
    return bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())


def _display(value):
    if _blank(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def locate_workbook(root: Path = REPOSITORY_ROOT) -> Path:
    candidates = sorted(
        path for path in root.rglob("*.xlsx")
        if not path.name.startswith("~$")
        and not {".git", ".venv", "venv", "env", "__pycache__"}.intersection(path.parts)
    )
    if len(candidates) != 1:
        raise JisdorValidationError(
            f"Expected one XLSX workbook, found {len(candidates)}; select explicitly with --input. "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def extract_table(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Find one row with unique Tanggal and Kurs names, regardless of positions."""
    candidates = []
    for sheet_name, frame in sheets.items():
        for index, row in enumerate(frame.itertuples(index=False, name=None)):
            names = [str(value).strip().casefold() if not _blank(value) else "" for value in row]
            if "tanggal" in names and "kurs" in names:
                if names.count("tanggal") != 1 or names.count("kurs") != 1:
                    raise JisdorValidationError(f"Ambiguous columns in {sheet_name!r}, row {index + 1}.")
                candidates.append((sheet_name, index, names.index("tanggal"), names.index("kurs")))
    if len(candidates) != 1:
        raise JisdorValidationError(f"Expected one named Tanggal/Kurs table; found {len(candidates)}.")
    sheet_name, header, date_column, rate_column = candidates[0]
    frame = sheets[sheet_name]
    rows = []
    blank_count = 0
    for index in range(header + 1, len(frame)):
        row = frame.iloc[index]
        if all(_blank(value) for value in row):
            blank_count += 1
            continue
        rows.append({"source_row": index + 1, "date_raw": row.iloc[date_column], "jisdor_raw": row.iloc[rate_column]})
    metadata = {
        "sheet_names": list(sheets), "selected_sheet": sheet_name,
        "header_row": header + 1, "raw_sheet_rows": len(frame),
        "raw_rows": len(rows), "blank_rows_ignored": blank_count,
        "header_rows_skipped": header + 1,
        "empty_columns": [int(index + 1) for index in range(frame.shape[1]) if all(_blank(value) for value in frame.iloc[header:, index])],
    }
    return pd.DataFrame(rows, columns=["source_row", "date_raw", "jisdor_raw"]), metadata


def parse_date(value) -> pd.Timestamp:
    if _blank(value):
        return pd.NaT
    if isinstance(value, Number) or isinstance(value, bool):
        raise ValueError("Numeric dates/Excel serials require a real date cell or explicit date string")
    if isinstance(value, str):
        value = value.strip().lstrip("'")
        if not re.match(r"^(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4})(?:$|[ T]\d{1,2}:\d{2})", value):
            raise ValueError("Date string must include full YYYY-MM-DD or MM/DD/YYYY calendar date")
        if "/" in value.split()[0] and int(value.split("/", 1)[0]) > 12:
            raise ValueError("Slash dates use month/day/year, so month must be at most 12")
    parsed = pd.to_datetime(value, errors="raise", dayfirst=False)
    if pd.isna(parsed):
        raise ValueError("Unparseable date")
    if parsed.tzinfo is not None:
        raise ValueError("JISDOR observation dates must not contain a timezone")
    return parsed.normalize()


def parse_rate(value) -> float:
    if _blank(value):
        return float("nan")
    if isinstance(value, bool):
        raise ValueError("Boolean rate is invalid")
    if isinstance(value, str):
        value = value.strip().lstrip("'")
        value = re.sub(r"^(?:Rp\.?|IDR)\s*", "", value, flags=re.IGNORECASE)
        if re.search(r"\s", value):
            if not re.fullmatch(r"[+-]?\d{1,3}(?:\s+\d{3})+(?:[.,]\d{1,2})?", value):
                raise ValueError("Internal whitespace must separate three-digit groups")
            value = re.sub(r"\s+", "", value)
        if re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})+", value):
            separators = re.findall(r"[.,]", value)
            if len(set(separators)) != 1:
                raise ValueError("Mixed thousands separators")
            value = value.replace(separators[0], "")
        elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d{1,2}", value):
            value = value.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d{1,2}", value):
            value = value.replace(",", "")
        elif re.fullmatch(r"[+-]?\d+,\d{1,2}", value):
            value = value.replace(",", ".")
    rate = float(pd.to_numeric(value, errors="raise"))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Rate must be finite and positive")
    return rate


def validate_table(raw: pd.DataFrame, metadata: dict | None = None) -> tuple[pd.DataFrame, dict]:
    report = dict(metadata or {})
    records = []
    invalid = []
    missing_dates = missing_rates = invalid_dates = invalid_rates = 0
    for row in raw.to_dict(orient="records"):
        problems = []
        parsed_date = pd.NaT
        parsed_rate = float("nan")
        if _blank(row["date_raw"]):
            missing_dates += 1
            problems.append("missing date")
        else:
            try:
                parsed_date = parse_date(row["date_raw"])
            except (ValueError, TypeError, OverflowError) as exc:
                invalid_dates += 1
                problems.append(f"invalid date: {exc}")
        if _blank(row["jisdor_raw"]):
            missing_rates += 1
            problems.append("missing jisdor")
        else:
            try:
                parsed_rate = parse_rate(row["jisdor_raw"])
            except (ValueError, TypeError, OverflowError) as exc:
                invalid_rates += 1
                problems.append(f"invalid jisdor: {exc}")
        records.append({"source_row": row["source_row"], "date": parsed_date, "jisdor": parsed_rate})
        if problems:
            invalid.append({"source_row": row["source_row"], "date_raw": _display(row["date_raw"]), "jisdor_raw": _display(row["jisdor_raw"]), "issues": problems})
    frame = pd.DataFrame(records, columns=["source_row", "date", "jisdor"])
    duplicates = frame.loc[frame["date"].notna() & frame.duplicated("date", keep=False)]
    report.update({
        "raw_rows": len(raw), "row_count": len(frame), "clean_rows": 0,
        "missing_dates": missing_dates, "missing_jisdor": missing_rates,
        "invalid_dates": invalid_dates, "invalid_jisdor": invalid_rates,
        "duplicate_dates": int(frame.loc[frame["date"].notna()].duplicated("date").sum()),
        "duplicate_records": [
            {"source_row": int(row.source_row), "date": row.date.strftime("%Y-%m-%d"), "jisdor": _display(row.jisdor)}
            for row in duplicates.itertuples(index=False)
        ],
        "invalid_records": invalid,
    })
    if frame.empty or invalid or not duplicates.empty:
        raise JisdorValidationError("JISDOR validation failed: empty table, invalid/missing values, or duplicate dates; see QA records.", report)
    clean = frame[["date", "jisdor"]].sort_values("date", kind="stable").reset_index(drop=True)
    if all(float(value).is_integer() for value in clean["jisdor"]) and clean["jisdor"].max() < 2**63:
        clean["jisdor"] = clean["jisdor"].astype("int64")
    report.update({
        "clean_rows": len(clean), "start_date": clean["date"].min().strftime("%Y-%m-%d"),
        "end_date": clean["date"].max().strftime("%Y-%m-%d"),
        "jisdor_dtype": str(clean["jisdor"].dtype),
        "chronologically_sorted": bool(clean["date"].is_monotonic_increasing),
        "min_jisdor": float(clean["jisdor"].min()), "max_jisdor": float(clean["jisdor"].max()),
    })
    report["expected_comparison"] = {key: {"expected": expected, "actual": report[key], "matches": report[key] == expected} for key, expected in EXPECTED.items()}
    report["expected_discrepancies"] = [key for key, comparison in report["expected_comparison"].items() if not comparison["matches"]]
    return clean, report


def clean_workbook(input_path: str | Path | None = None, output_path: str | Path = DEFAULT_OUTPUT,
                   report_path: str | Path = DEFAULT_REPORT, sheet: str | None = None) -> tuple[pd.DataFrame, dict]:
    output_path, report_path = Path(output_path).resolve(), Path(report_path).resolve()
    if output_path == report_path or (input_path and Path(input_path).resolve() in (output_path, report_path)):
        raise ValueError("Input, CSV, and report paths must be distinct")
    if output_path.suffix.lower() != ".csv" or report_path.suffix.lower() != ".json":
        raise ValueError("Output paths must use .csv and .json extensions")
    report = {
        "status": "running", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Bank Indonesia JISDOR USD/IDR", "policies": POLICIES,
        "output_path": output_path.relative_to(REPOSITORY_ROOT).as_posix() if output_path.is_relative_to(REPOSITORY_ROOT) else str(output_path),
        "output_valid": False,
    }
    _atomic_text(report_path, json.dumps(report, indent=2) + "\n")
    # A failed or interrupted rerun cannot leave an older CSV masquerading as its result.
    output_path.unlink(missing_ok=True)
    try:
        source = Path(input_path).resolve() if input_path else locate_workbook()
        if source in (output_path, report_path):
            raise ValueError("Input and output paths must be distinct")
        report["source_path"] = source.relative_to(REPOSITORY_ROOT).as_posix() if source.is_relative_to(REPOSITORY_ROOT) else str(source)
        report["source_sha256"] = _sha256(source)
        sheets = pd.read_excel(source, sheet_name=None if sheet is None else [sheet], header=None, dtype=object, keep_default_na=False, engine="openpyxl")
        raw, metadata = extract_table(sheets)
        report.update(metadata)
        clean, statistics = validate_table(raw, metadata)
        report.update(statistics)
        if _sha256(source) != report["source_sha256"]:
            raise JisdorValidationError("Source workbook changed during cleaning")
        _atomic_text(output_path, clean.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n"))
        report.update({"status": "passed", "output_valid": True, "output_sha256": _sha256(output_path)})
        _atomic_text(report_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as exc:
        if isinstance(exc, JisdorValidationError):
            report.update(exc.report)
        report.update({"status": "failed", "output_valid": False, "error": str(exc)})
        report.pop("output_sha256", None)
        output_path.unlink(missing_ok=True)
        _atomic_text(report_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise JisdorValidationError(str(exc), report) from exc
    LOGGER.info("JISDOR CLEANING SUMMARY: raw=%s clean=%s range=%s..%s duplicates=%s missing_dates=%s missing_jisdor=%s dtype=%s sorted=%s output=%s",
                report["raw_rows"], report["clean_rows"], report["start_date"], report["end_date"],
                report["duplicate_dates"], report["missing_dates"], report["missing_jisdor"],
                report["jisdor_dtype"], report["chronologically_sorted"], output_path)
    if report["expected_discrepancies"]:
        LOGGER.warning("Expected QA differs for %s; actual source observations preserved", report["expected_discrepancies"])
    return clean, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Explicit XLSX path; otherwise locate one repository workbook")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sheet", help="Explicit sheet name when the workbook contains multiple candidate tables")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        clean_workbook(args.input, args.output, args.report, args.sheet)
    except (JisdorValidationError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
