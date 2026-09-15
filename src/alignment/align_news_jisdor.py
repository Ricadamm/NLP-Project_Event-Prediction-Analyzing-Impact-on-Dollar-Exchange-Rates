"""Strictly align exact CNBC timestamps to actual JISDOR observations."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, time
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml

from src.data_alignment import AlignmentError, validate_jisdor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEWS = ROOT / "data/interim/news/cnbc_news_candidates.csv"
DEFAULT_JISDOR = ROOT / "data/interim/jisdor/jisdor_clean.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/cnbc_jisdor_aligned_v2.csv"
DEFAULT_REPORT = ROOT / "data/processed/cnbc_alignment_report.json"
DEFAULT_DAILY_OUTPUT = ROOT / "data/processed/cnbc_jisdor_daily.csv"
CATEGORY_COUNT_COLUMNS = {
    "armed_conflict": "armed_conflict_count",
    "sanctions": "sanctions_count",
    "trade_conflict": "trade_conflict_count",
    "energy_geopolitics": "energy_geopolitics_count",
    "political_instability": "political_instability_count",
    "monetary_geoeconomic": "monetary_geoeconomic_count",
}


def _parse_cutoff(value: str) -> time:
    if not isinstance(value, str):
        raise AlignmentError("cutoff_wib must be HH:MM or HH:MM:SS text")
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            pass
    raise AlignmentError("cutoff_wib must be HH:MM or HH:MM:SS text")


def validate_alignment_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise AlignmentError("alignment config must be a mapping")
    timezone_name = config.get("timezone")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError) as exc:
        raise AlignmentError(f"Unknown alignment timezone: {timezone_name!r}") from exc
    default_cutoff = config.get("default_cutoff_wib")
    _parse_cutoff(default_cutoff)
    regimes = config.get("cutoff_regimes", [])
    if not isinstance(regimes, list):
        raise AlignmentError("alignment.cutoff_regimes must be a list")
    normalized = []
    for regime in regimes:
        if not isinstance(regime, Mapping):
            raise AlignmentError("Each cutoff regime must be a mapping")
        try:
            start = date.fromisoformat(str(regime["start_date"]))
            end = date.fromisoformat(str(regime["end_date"]))
        except (KeyError, ValueError) as exc:
            raise AlignmentError("Cutoff regimes require valid start_date and end_date") from exc
        if end < start:
            raise AlignmentError("Cutoff regime end_date precedes start_date")
        cutoff = str(regime.get("cutoff_wib", ""))
        _parse_cutoff(cutoff)
        normalized.append({**dict(regime), "start_date": start, "end_date": end, "cutoff_wib": cutoff})
    for index, left in enumerate(normalized):
        for right in normalized[index + 1:]:
            if max(left["start_date"], right["start_date"]) <= min(left["end_date"], right["end_date"]):
                raise AlignmentError("Cutoff regimes must not overlap")
    return {
        "timezone": timezone_name,
        "default_cutoff_wib": default_cutoff,
        "assumption": config.get("assumption", "Research configuration; not inferred from data."),
        "cutoff_regimes": normalized,
    }


def cutoff_for_date(day: date, config: Mapping) -> tuple[time, str]:
    validated = validate_alignment_config(config)
    for regime in validated["cutoff_regimes"]:
        if regime["start_date"] <= day <= regime["end_date"]:
            return _parse_cutoff(regime["cutoff_wib"]), "date_specific_regime"
    return _parse_cutoff(validated["default_cutoff_wib"]), "default_research_assumption"


def jisdor_features(jisdor: pd.DataFrame) -> pd.DataFrame:
    """Add unscaled movement values without inserting calendar observations."""
    rates = validate_jisdor(jisdor)
    features = rates.copy()
    features["previous_jisdor"] = features["jisdor"].shift(1)
    features["change_idr"] = features["jisdor"] - features["previous_jisdor"]
    features["return_pct"] = ((features["jisdor"] / features["previous_jisdor"]) - 1) * 100
    return features


def _decoded_categories(value) -> set[str]:
    if isinstance(value, list):
        decoded = value
    elif isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = []
    else:
        decoded = []
    return {str(item) for item in decoded} if isinstance(decoded, list) else set()


def aggregate_jisdor_daily(
    aligned_news: pd.DataFrame,
    jisdor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate selected articles while retaining every actual JISDOR date."""
    features = jisdor_features(jisdor)
    daily = features.copy()
    daily["date"] = daily["date"].map(lambda value: value.isoformat())
    daily["news_count"] = 0
    for column in CATEGORY_COUNT_COLUMNS.values():
        daily[column] = 0
    by_date = {day: index for index, day in enumerate(daily["date"])}
    counted_articles = 0
    for row in aligned_news.to_dict(orient="records"):
        day = str(row.get("effective_trade_date", ""))
        if day not in by_date:
            continue
        index = by_date[day]
        daily.at[index, "news_count"] += 1
        counted_articles += 1
        for category in _decoded_categories(row.get("matched_categories")):
            column = CATEGORY_COUNT_COLUMNS.get(category)
            if column:
                daily.at[index, column] += 1
    count_columns = ["news_count", *CATEGORY_COUNT_COLUMNS.values()]
    daily[count_columns] = daily[count_columns].astype(int)
    trading_days_with_news = int(daily["news_count"].gt(0).sum())
    report = {
        "jisdor_trading_days": len(daily),
        "trading_days_with_candidate_news": trading_days_with_news,
        "trading_days_with_zero_candidate_news": len(daily) - trading_days_with_news,
        "aligned_candidate_articles_counted": counted_articles,
        "news_count_sum": int(daily["news_count"].sum()),
        "counts_reconcile": int(daily["news_count"].sum()) == counted_articles,
        "policy": "All and only actual JISDOR observations are retained; no weekend or holiday rows are synthesized.",
    }
    return daily, report


def write_daily_output(
    aligned_news_path: str | Path,
    jisdor_path: str | Path,
    output_path: str | Path = DEFAULT_DAILY_OUTPUT,
) -> tuple[pd.DataFrame, dict]:
    aligned = pd.read_csv(aligned_news_path, keep_default_na=False)
    jisdor = pd.read_csv(jisdor_path)
    daily, report = aggregate_jisdor_daily(aligned, jisdor)
    report.update(
        {
            "aligned_news_input": str(Path(aligned_news_path)),
            "jisdor_input": str(Path(jisdor_path)),
            "output_path": str(Path(output_path)),
        }
    )
    _atomic_text(Path(output_path), daily.to_csv(index=False, lineterminator="\n"))
    return daily, report


def _exact_local_timestamp(row: Mapping, timezone_name: str) -> pd.Timestamp | None:
    if row.get("timestamp_status") != "exact_publisher_timestamp":
        return None
    for field in ("published_at_utc", "published_at_wib", "published_at_original"):
        value = row.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            parsed = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError):
            continue
        if not pd.isna(parsed) and parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.tz_convert(timezone_name)
    return None


def align_exact_news_to_jisdor(
    news: pd.DataFrame,
    jisdor: pd.DataFrame,
    alignment_config: Mapping,
) -> tuple[pd.DataFrame, dict]:
    """Map exact timestamps to the first still-available actual JISDOR fixing."""
    config = validate_alignment_config(alignment_config)
    features = jisdor_features(jisdor)
    trading_dates = features["date"].tolist()
    feature_by_date = {row.date: row for row in features.itertuples(index=False)}
    rows = []
    reasons = Counter()
    effective_counts = Counter()
    exact_count = 0
    for source in news.to_dict(orient="records"):
        local = _exact_local_timestamp(source, config["timezone"])
        effective = None
        reason = None
        cutoff_text = ""
        cutoff_source = ""
        if local is None:
            reason = "missing_exact_timestamp_unaligned"
        else:
            exact_count += 1
            day = local.date()
            cutoff, cutoff_source = cutoff_for_date(day, config)
            cutoff_text = cutoff.isoformat()
            if day in feature_by_date:
                if local.time().replace(tzinfo=None) < cutoff:
                    effective = day
                    reason = "same_day_before_cutoff"
                else:
                    effective = next((candidate for candidate in trading_dates if candidate > day), None)
                    reason = "after_cutoff_next_trade_day" if effective else "beyond_jisdor_range_unaligned"
            else:
                effective = next((candidate for candidate in trading_dates if candidate >= day), None)
                if effective is None:
                    reason = "beyond_jisdor_range_unaligned"
                elif day.weekday() >= 5:
                    reason = "weekend_next_trade_day"
                else:
                    reason = "non_jisdor_day_next_trade_day"
        movement = feature_by_date.get(effective)
        row = dict(source)
        row.update(
            {
                "effective_trade_date": effective.isoformat() if effective else "",
                "alignment_reason": reason,
                "alignment_cutoff_wib": cutoff_text,
                "alignment_cutoff_source": cutoff_source,
                "alignment_timestamp_used": local.isoformat() if local is not None else "",
                "previous_jisdor": movement.previous_jisdor if movement is not None else pd.NA,
                "jisdor": movement.jisdor if movement is not None else pd.NA,
                "change_idr": movement.change_idr if movement is not None else pd.NA,
                "return_pct": movement.return_pct if movement is not None else pd.NA,
            }
        )
        rows.append(row)
        reasons[reason] += 1
        if effective:
            effective_counts[effective.isoformat()] += 1
    alignment_columns = [
        "effective_trade_date", "alignment_reason", "alignment_cutoff_wib",
        "alignment_cutoff_source", "alignment_timestamp_used", "previous_jisdor",
        "jisdor", "change_idr", "return_pct",
    ]
    aligned = pd.DataFrame(
        rows,
        columns=list(news.columns) + [
            column for column in alignment_columns if column not in news.columns
        ],
    )
    aligned_count = int(aligned["effective_trade_date"].ne("").sum()) if not aligned.empty else 0
    missing_previous = int(
        aligned.loc[aligned["effective_trade_date"].ne(""), "previous_jisdor"].isna().sum()
    ) if not aligned.empty else 0
    expected_reasons = (
        "same_day_before_cutoff", "after_cutoff_next_trade_day",
        "weekend_next_trade_day", "non_jisdor_day_next_trade_day",
        "missing_exact_timestamp_unaligned", "beyond_jisdor_range_unaligned",
    )
    report = {
        "total_articles": len(news),
        "articles_with_exact_timestamps": exact_count,
        "articles_aligned": aligned_count,
        "articles_unaligned": len(news) - aligned_count,
        "alignment_reason_counts": {reason: reasons[reason] for reason in expected_reasons},
        "same_day_before_cutoff": reasons["same_day_before_cutoff"],
        "after_cutoff_next_trade_day": reasons["after_cutoff_next_trade_day"],
        "weekend_next_trade_day": reasons["weekend_next_trade_day"],
        "non_jisdor_day_next_trade_day": reasons["non_jisdor_day_next_trade_day"],
        "missing_exact_timestamp_unaligned": reasons["missing_exact_timestamp_unaligned"],
        "effective_trade_date_counts": dict(sorted(effective_counts.items())),
        "timezone": config["timezone"],
        "default_cutoff_wib": _parse_cutoff(config["default_cutoff_wib"]).isoformat(),
        "cutoff_assumption": config["assumption"],
        "cutoff_regimes": [
            {**regime, "start_date": regime["start_date"].isoformat(), "end_date": regime["end_date"].isoformat()}
            for regime in config["cutoff_regimes"]
        ],
        "jisdor_observation_count": len(features),
        "jisdor_start_date": min(trading_dates).isoformat(),
        "jisdor_end_date": max(trading_dates).isoformat(),
        "financial_feature_definitions": {
            "previous_jisdor": "previous actual JISDOR observation",
            "change_idr": "jisdor - previous_jisdor",
            "return_pct": "((jisdor / previous_jisdor) - 1) * 100",
        },
        "aligned_rows_missing_previous_jisdor": missing_previous,
        "counts_reconcile": aligned_count + (len(news) - aligned_count) == len(news),
        "policy": "No exact publisher timestamp means no strict alignment; no midnight substitution or interpolation.",
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
    cnbc_config_path: str | Path = ROOT / "config/cnbc.yaml",
) -> tuple[pd.DataFrame, dict]:
    news = pd.read_csv(news_path, keep_default_na=False)
    jisdor = pd.read_csv(jisdor_path)
    full_config = yaml.safe_load(Path(cnbc_config_path).read_text(encoding="utf-8"))
    aligned, report = align_exact_news_to_jisdor(news, jisdor, full_config["alignment"])
    report.update({"news_input": str(Path(news_path)), "jisdor_input": str(Path(jisdor_path)), "output_path": str(Path(output_path))})
    _atomic_text(Path(output_path), aligned.to_csv(index=False, lineterminator="\n"))
    _atomic_text(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return aligned, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--jisdor", type=Path, default=DEFAULT_JISDOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cnbc-config", type=Path, default=ROOT / "config/cnbc.yaml")
    args = parser.parse_args(argv)
    align_files(args.news, args.jisdor, args.output, args.report,
                cnbc_config_path=args.cnbc_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
