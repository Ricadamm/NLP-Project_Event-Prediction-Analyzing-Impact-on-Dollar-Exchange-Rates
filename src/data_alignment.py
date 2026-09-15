"""Temporally align cleaned news with daily Bank Indonesia JISDOR rates.

Rules are deterministic and forward-moving. Time-aware news published before
16:00 WIB is eligible for that calendar day's JISDOR observation; news at or
after the cutoff is eligible from the following calendar day. Weekends,
holidays, and other missing JISDOR dates map to the next available observation.
CNBC archive rows are date-only, so they use their validated canonical-URL date
without an invented time and are labelled separately in the output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEWS = ROOT / "data/interim/news/cnbc_news_clean.csv"
DEFAULT_JISDOR = ROOT / "data/interim/jisdor/jisdor_clean.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/cnbc_jisdor_aligned.csv"
DEFAULT_REPORT = ROOT / "data/processed/cnbc_jisdor_alignment_report.json"


class AlignmentError(ValueError):
    """Input data or alignment policy is invalid."""


def parse_market_close(value: str | time) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise AlignmentError("market close must be a local wall-clock time")
        return value
    if not isinstance(value, str):
        raise AlignmentError("market close must use HH:MM or HH:MM:SS")
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            pass
    raise AlignmentError("market close must use HH:MM or HH:MM:SS")


def validate_jisdor(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not {"date", "jisdor"}.issubset(frame.columns):
        raise AlignmentError("JISDOR input must contain date and jisdor columns")
    clean = frame[["date", "jisdor"]].copy()
    parsed = pd.to_datetime(clean["date"], errors="coerce")
    if parsed.isna().any():
        raise AlignmentError("JISDOR dates must all be valid")
    if getattr(parsed.dt, "tz", None) is not None:
        raise AlignmentError("JISDOR dates must be timezone-naive daily observations")
    if (parsed != parsed.dt.normalize()).any():
        raise AlignmentError("JISDOR dates must not contain intraday times")
    rates = pd.to_numeric(clean["jisdor"], errors="coerce")
    if rates.isna().any() or not rates.map(math.isfinite).all() or (rates <= 0).any():
        raise AlignmentError("JISDOR rates must be finite positive numbers")
    if parsed.duplicated().any():
        raise AlignmentError("JISDOR dates must be unique")
    clean["date"] = parsed.dt.date
    clean["jisdor"] = rates
    return clean.sort_values("date", kind="stable").reset_index(drop=True)


def _aware_timestamp(value) -> pd.Timestamp | None:
    if value is None or not isinstance(value, (datetime, pd.Timestamp, str)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.tz_convert("Asia/Jakarta")


def _date_only(value) -> date | None:
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def align_news_to_jisdor(
    news: pd.DataFrame,
    jisdor: pd.DataFrame,
    *,
    market_close_wib: str | time = "16:00",
) -> tuple[pd.DataFrame, dict]:
    """Return news rows augmented with the next eligible JISDOR observation."""
    if not isinstance(news, pd.DataFrame):
        raise AlignmentError("news input must be a DataFrame")
    rates = validate_jisdor(jisdor)
    if rates.empty:
        raise AlignmentError("JISDOR input must not be empty")
    cutoff = parse_market_close(market_close_wib)
    trading_dates = rates["date"].tolist()
    rate_by_date = dict(zip(rates["date"], rates["jisdor"]))
    output_rows = []
    reasons = Counter()
    for source in news.to_dict(orient="records"):
        timestamp = _aware_timestamp(source.get("published_at_utc"))
        if timestamp is None:
            timestamp = _aware_timestamp(source.get("published_at_wib"))
        pub_date = _date_only(source.get("publication_date"))
        event_date = eligible = basis = None
        after_close = False
        if timestamp is not None:
            event_date = timestamp.date()
            after_close = timestamp.time().replace(tzinfo=None) >= cutoff
            eligible = event_date + timedelta(days=1) if after_close else event_date
            basis = "timestamp_wib"
        elif pub_date is not None:
            event_date = pub_date
            eligible = event_date
            basis = "publication_date_only"

        aligned_date = None
        if eligible is not None:
            aligned_date = next((day for day in trading_dates if day >= eligible), None)

        row = dict(source)
        if event_date is None:
            reason = "invalid_or_missing_publication_time"
        elif aligned_date is None:
            reason = "beyond_jisdor_range"
        elif basis == "publication_date_only":
            reason = (
                "date_only_same_trading_day"
                if aligned_date == eligible
                else "date_only_next_trading_day"
            )
        elif after_close:
            reason = (
                "after_market_close"
                if aligned_date == eligible
                else "after_market_close_next_trading_day"
            )
        else:
            reason = "same_trading_day" if aligned_date == eligible else "next_trading_day"
        reasons[reason] += 1
        row.update(
            {
                "event_date_wib": event_date.isoformat() if event_date else "",
                "alignment_basis": basis or "",
                "eligible_jisdor_date": eligible.isoformat() if eligible else "",
                "jisdor_date": aligned_date.isoformat() if aligned_date else "",
                "jisdor": rate_by_date.get(aligned_date, pd.NA),
                "alignment_lag_calendar_days": (
                    (aligned_date - event_date).days
                    if aligned_date is not None and event_date is not None
                    else pd.NA
                ),
                "alignment_reason": reason,
            }
        )
        output_rows.append(row)

    extra_columns = [
        "event_date_wib", "alignment_basis", "eligible_jisdor_date", "jisdor_date",
        "jisdor", "alignment_lag_calendar_days", "alignment_reason",
    ]
    aligned = pd.DataFrame(output_rows, columns=list(news.columns) + extra_columns)
    unaligned_reasons = {"invalid_or_missing_publication_time", "beyond_jisdor_range"}
    aligned_count = sum(
        value for reason, value in reasons.items() if reason not in unaligned_reasons
    )
    report = {
        "news_rows": len(news),
        "aligned_rows": aligned_count,
        "unaligned_rows": len(news) - aligned_count,
        "alignment_reason_counts": dict(sorted(reasons.items())),
        "market_close_wib": cutoff.isoformat(),
        "timezone": "Asia/Jakarta",
        "jisdor_rows": len(rates),
        "jisdor_start_date": rates["date"].min().isoformat(),
        "jisdor_end_date": rates["date"].max().isoformat(),
        "policy": {
            "before_market_close": "same calendar date when JISDOR exists",
            "at_or_after_market_close": "next calendar date, then next available JISDOR date",
            "non_trading_day": "next available JISDOR date; no interpolation",
            "date_only_news": "calendar date, then next available JISDOR date; no time invented",
            "out_of_range": "retain row with empty JISDOR fields",
        },
    }
    return aligned, report


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


def align_files(
    news_path: str | Path = DEFAULT_NEWS,
    jisdor_path: str | Path = DEFAULT_JISDOR,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    *,
    market_close_wib: str | time = "16:00",
) -> tuple[pd.DataFrame, dict]:
    paths = [Path(path).resolve() for path in (news_path, jisdor_path, output_path, report_path)]
    if len(set(paths)) != 4:
        raise AlignmentError("News, JISDOR, output, and report paths must be distinct")
    news = pd.read_csv(paths[0], keep_default_na=False)
    jisdor = pd.read_csv(paths[1])
    aligned, report = align_news_to_jisdor(news, jisdor, market_close_wib=market_close_wib)
    report.update({"news_input": str(paths[0]), "jisdor_input": str(paths[1]), "output_path": str(paths[2])})
    _atomic_text(paths[2], aligned.to_csv(index=False, lineterminator="\n"))
    _atomic_text(paths[3], json.dumps(report, indent=2, sort_keys=True) + "\n")
    return aligned, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--jisdor", type=Path, default=DEFAULT_JISDOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--market-close-wib", default="16:00")
    args = parser.parse_args(argv)
    try:
        align_files(args.news, args.jisdor, args.output, args.report,
                    market_close_wib=args.market_close_wib)
    except (AlignmentError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
