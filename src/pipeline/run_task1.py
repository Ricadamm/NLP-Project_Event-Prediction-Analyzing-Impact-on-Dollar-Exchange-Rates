"""Orchestrate resumable CNBC pilot or full historical Task 1 collection."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from src.acquisition.collect_cnbc import (
    STUDY_END, STUDY_START, collect, load_config, validate_range,
)
from src.acquisition.enrich_cnbc import enrich_file
from src.alignment.align_news_jisdor import align_files, write_daily_output
from src.preprocessing.clean_cnbc import read_raw_records, write_clean_outputs
from src.preprocessing.clean_jisdor import clean_workbook
from src.preprocessing.filter_cnbc import filter_file
from src.preprocessing.prefilter_cnbc import prefilter_file
from src.preprocessing.sample_cnbc_review import sample_validation_files


ROOT = Path(__file__).resolve().parents[2]
PILOT_START = date(2021, 9, 1)
PILOT_END = date(2021, 9, 7)


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


def run_task1(
    *,
    source: str = "cnbc",
    start_date: date = PILOT_START,
    end_date: date = PILOT_END,
    config_path: str | Path = ROOT / "config/cnbc.yaml",
    taxonomy_path: str | Path = ROOT / "config/geopolitical_topics.yaml",
    raw_dir: str | Path = ROOT / "data/raw/news/cnbc",
    interim_dir: str | Path = ROOT / "data/interim",
    processed_dir: str | Path = ROOT / "data/processed",
    jisdor_input: str | Path = ROOT / "data/raw/Informasi Kurs Jisdor.xlsx",
    manifest_path: str | Path | None = None,
    refresh_discovery: bool = False,
    full_range: bool = False,
    force_enrichment: bool = False,
    enrichment_workers: int = 1,
    cache_only: bool = False,
    clean_jisdor: bool = True,
    sample_size: int = 100,
    reject_sample_size: int = 100,
    sample_seed: int = 42,
) -> dict:
    if source != "cnbc":
        raise ValueError("This orchestrator supports --source cnbc only; GDELT remains independent")
    if (
        isinstance(enrichment_workers, bool)
        or not isinstance(enrichment_workers, int)
        or enrichment_workers < 1
    ):
        raise ValueError("enrichment_workers must be a positive integer")
    config = load_config(config_path)
    study_start = date.fromisoformat(
        config["validation"].get("study_start_date", STUDY_START.isoformat())
    )
    study_end = date.fromisoformat(
        config["validation"].get("study_end_date", STUDY_END.isoformat())
    )
    validate_range(
        start_date, end_date, config["validation"]["max_days"],
        full_range=full_range, study_start=study_start, study_end=study_end,
    )
    raw_dir, interim_dir, processed_dir = Path(raw_dir), Path(interim_dir), Path(processed_dir)
    news_dir, jisdor_dir = interim_dir / "news", interim_dir / "jisdor"
    mode = "full_range" if full_range else "pilot"
    manifest_path = Path(manifest_path) if manifest_path else interim_dir / (
        "task1_cnbc_full_run_report.json" if full_range else "task1_cnbc_v2_run_report.json"
    )
    manifest = {
        "pipeline": (
            "task1_cnbc_full_historical" if full_range
            else "task1_cnbc_timestamp_enriched_pilot"
        ),
        "mode": mode,
        "status": "running",
        "source": source,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "GDELT": "not invoked or modified; retained as an independent fallback",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)
    try:
        jisdor_csv = jisdor_dir / "jisdor_clean.csv"
        if clean_jisdor:
            _, report = clean_workbook(
                jisdor_input, jisdor_csv, jisdor_dir / "jisdor_cleaning_report.json"
            )
            manifest["stages"]["jisdor_validation"] = {
                "status": report["status"], "rows": report["clean_rows"], "output": str(jisdor_csv)
            }
        else:
            if not jisdor_csv.exists():
                raise FileNotFoundError(f"Missing cached JISDOR data: {jisdor_csv}")
            manifest["stages"]["jisdor_validation"] = {"status": "cached", "output": str(jisdor_csv)}

        acquisition_report_path = news_dir / (
            "cnbc_full_discovery_report.json" if full_range else
            ("cnbc_pilot_report.json" if refresh_discovery else "cnbc_pilot_cached_report.json")
        )
        acquisition = collect(
            config, start_date, end_date, raw_dir=raw_dir,
            report_path=acquisition_report_path,
            # Refresh means resume missing/failed work. Completed days remain
            # immutable cache hits; it is deliberately not a force-refetch.
            force=False, report_only=not refresh_discovery,
            full_range=full_range, progress_every=100,
        )
        if not acquisition["collection_complete"]:
            raise RuntimeError("Existing CNBC daily archive evidence is incomplete")
        manifest["stages"]["discovery"] = {
            "status": "resumed" if refresh_discovery else "reused",
            "raw_records": acquisition["raw_records"],
            "completed_days": acquisition["completed_days"],
            "output": str(acquisition_report_path),
        }

        article_index_csv = news_dir / "cnbc_article_index.csv"
        cleaned, cleaning = write_clean_outputs(
            read_raw_records(raw_dir, start_date=start_date, end_date=end_date),
            article_index_csv,
            news_dir / "cnbc_article_index_report.json",
            news_dir / "cnbc_article_index_quarantine.json",
            start_date=start_date, end_date=end_date,
        )
        manifest["stages"]["article_index"] = {
            "status": "passed", "rows": len(cleaned), "output": str(article_index_csv)
        }

        prefiltered_csv = news_dir / "cnbc_prefiltered.csv"
        queue_csv = news_dir / "cnbc_enrichment_queue.csv"
        prefiltered, queue, prefiltering = prefilter_file(
            article_index_csv, prefiltered_csv, queue_csv,
            news_dir / "cnbc_prefilter_report.json",
            taxonomy_path=taxonomy_path, cnbc_config_path=config_path,
        )
        manifest["stages"]["title_prefilter"] = {
            "status": "passed", "input_rows": len(prefiltered),
            "candidate_rows": len(queue), "output": str(prefiltered_csv),
            "enrichment_queue": str(queue_csv),
        }

        enriched_csv = news_dir / "cnbc_news_enriched.csv"
        enriched, enrichment = enrich_file(
            queue_csv, enriched_csv, news_dir / "cnbc_enrichment_report.json",
            config=config["article_metadata"], cache_dir=raw_dir / "article_metadata",
            force=force_enrichment, cache_only=cache_only,
            enrichment_workers=enrichment_workers,
        )
        manifest["stages"]["article_metadata_enrichment"] = {
            "status": "passed_with_unresolved" if enrichment["missing_timestamps"] else "passed",
            "rows": len(enriched),
            "exact_timestamps": enrichment["exact_publisher_timestamps"],
            "missing_timestamps": enrichment["missing_timestamps"],
            "failed_fetches": enrichment["failed_fetches"],
            "enrichment_workers": enrichment["enrichment_workers"],
            "output": str(enriched_csv),
        }

        candidate_csv = news_dir / "cnbc_news_candidates.csv"
        candidates, filtering = filter_file(
            enriched_csv, candidate_csv, news_dir / "cnbc_filtering_report.json",
            taxonomy_path=taxonomy_path, cnbc_config_path=config_path,
            candidates_only=True,
        )
        manifest["stages"]["geopolitical_filtering"] = {
            "status": "passed", "rows": len(candidates),
            "candidate_articles": filtering["candidate_articles"],
            "output": str(candidate_csv),
        }

        aligned_csv = processed_dir / "cnbc_jisdor_aligned.csv"
        aligned, alignment = align_files(
            candidate_csv, jisdor_csv, aligned_csv,
            processed_dir / "cnbc_alignment_report.json",
            cnbc_config_path=config_path,
        )
        manifest["stages"]["strict_jisdor_alignment"] = {
            "status": "passed_with_unaligned" if alignment["articles_unaligned"] else "passed",
            "rows": len(aligned), "aligned": alignment["articles_aligned"],
            "unaligned": alignment["articles_unaligned"], "output": str(aligned_csv),
        }

        daily_csv = processed_dir / "cnbc_jisdor_daily.csv"
        daily, daily_report = write_daily_output(aligned_csv, jisdor_csv, daily_csv)
        manifest["stages"]["trading_day_aggregation"] = {
            "status": "passed", "rows": len(daily), "output": str(daily_csv),
            "days_with_news": daily_report["trading_days_with_candidate_news"],
        }

        sample, sampling = sample_validation_files(
            candidate_csv, prefiltered_csv, news_dir / "cnbc_manual_review_sample.csv",
            news_dir / "cnbc_manual_review_sample_report.json",
            candidate_n=sample_size, reject_n=reject_sample_size, seed=sample_seed,
        )
        manifest["stages"]["manual_review_sample"] = {
            "status": "passed", "sample_size": len(sample),
            "seed": sampling["random_seed"],
            "output": str(news_dir / "cnbc_manual_review_sample.csv"),
        }

        qa = {
            "pipeline": manifest["pipeline"],
            "mode": mode,
            "requested_start_date": start_date.isoformat(),
            "requested_end_date": end_date.isoformat(),
            "requested_calendar_days": acquisition["requested_days"],
            "archive_days_completed": len(acquisition["completed_days"]),
            "archive_days_failed": len(acquisition["failed_days"]),
            "archive_days_pending": acquisition["requested_days"] - len(acquisition["completed_days"]) - len(acquisition["failed_days"]),
            "raw_sitemap_records": acquisition["raw_records"],
            "unique_urls": len(cleaned),
            "prefilter_candidate_count": prefiltering["prefilter_candidates"],
            "prefilter_reject_count": prefiltering["prefilter_rejects"],
            "prefilter_rate_percent": prefiltering["prefilter_rate_percent"],
            "article_metadata_fetches": enrichment["successful_http_fetches_this_run"] + enrichment["failed_fetch_attempts_this_run"],
            "article_metadata_fetches_this_run": enrichment["successful_http_fetches_this_run"] + enrichment["failed_fetch_attempts_this_run"],
            "article_metadata_http_requests_this_run": enrichment["http_request_count_this_run"],
            "enrichment_workers": enrichment["enrichment_workers"],
            "metadata_cache_hits": enrichment["cache_hits"],
            "metadata_failures": enrichment["failed_fetches"],
            "exact_timestamp_count": enrichment["exact_publisher_timestamps"],
            "missing_timestamp_count": enrichment["missing_timestamps"],
            "final_geopolitical_candidate_count": filtering["candidate_articles"],
            "candidate_percentage_of_discoveries": filtering["candidate_articles"] / len(cleaned) * 100 if len(cleaned) else 0.0,
            "candidate_percentage": filtering["candidate_articles"] / len(cleaned) * 100 if len(cleaned) else 0.0,
            "category_counts": filtering["counts_by_matched_category"],
            "section_counts": filtering["candidate_counts_by_section"],
            "aligned_articles": alignment["articles_aligned"],
            "unaligned_articles": alignment["articles_unaligned"],
            "same_day_before_cutoff": alignment["same_day_before_cutoff"],
            "after_cutoff_next_trade_day": alignment["after_cutoff_next_trade_day"],
            "weekend_next_trade_day": alignment["weekend_next_trade_day"],
            "non_jisdor_day_next_trade_day": alignment["non_jisdor_day_next_trade_day"],
            "jisdor_trading_days": daily_report["jisdor_trading_days"],
            "trading_days_with_candidate_news": daily_report["trading_days_with_candidate_news"],
            "trading_days_with_zero_candidate_news": daily_report["trading_days_with_zero_candidate_news"],
            "counts_reconcile": {
                "archive_days": acquisition["requested_days"] == len(acquisition["completed_days"]) + len(acquisition["failed_days"]),
                "discovery_index": acquisition["raw_records"] == len(cleaned) + cleaning["duplicate_records"] + cleaning["quarantined_records"] + cleaning["scope_excluded_records"],
                "prefilter": len(cleaned) == prefiltering["prefilter_candidates"] + prefiltering["prefilter_rejects"],
                "metadata": len(enriched) == enrichment["exact_publisher_timestamps"] + enrichment["missing_timestamps"],
                "metadata_queue": len(enriched) == enrichment["cache_hits"] + enrichment["successful_http_fetches_this_run"] + enrichment["failed_fetch_attempts_this_run"],
                "final_filter": len(enriched) == filtering["candidate_articles"] + filtering["non_candidate_articles"],
                "alignment": len(candidates) == alignment["articles_aligned"] + alignment["articles_unaligned"],
                "trading_days": len(daily) == daily_report["trading_days_with_candidate_news"] + daily_report["trading_days_with_zero_candidate_news"],
            },
            "limitations": [
                "Articles rejected by the title prefilter are not publisher-metadata enriched.",
                "The complete article index is retained so rejected rows can be audited and queued later.",
                "The deterministic filters are candidate-retrieval rules, not human relevance labels.",
                "The 15:00 WIB cutoff is a configured research assumption.",
            ],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        qa["all_counts_reconcile"] = all(qa["counts_reconcile"].values())
        qa_path = processed_dir / (
            "cnbc_full_collection_report.json" if full_range else "cnbc_pilot_pipeline_report.json"
        )
        _atomic_json(qa_path, qa)
        manifest["stages"]["production_qa"] = {
            "status": "passed" if qa["all_counts_reconcile"] else "failed",
            "output": str(qa_path),
        }
        if not qa["all_counts_reconcile"]:
            raise RuntimeError("CNBC production QA counts do not reconcile")
        manifest["status"] = "passed"
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(manifest_path, manifest)
        raise
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["cnbc"], default="cnbc")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--pilot", action="store_true", help="Use the configured September 1-7, 2021 pilot")
    parser.add_argument(
        "--full-range", action="store_true",
        help="Explicitly allow a date range within 2021-09-01 through 2026-09-01",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/cnbc.yaml")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "config/geopolitical_topics.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/news/cnbc")
    parser.add_argument("--interim-dir", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--jisdor-input", type=Path, default=ROOT / "data/raw/Informasi Kurs Jisdor.xlsx")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--force-enrichment", action="store_true")
    parser.add_argument(
        "--enrichment-workers", type=int, default=1,
        help="Parallel CNBC article metadata workers (recommended production value: 4)",
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-jisdor-clean", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--reject-sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.pilot and args.full_range:
        parser.error("--pilot and --full-range are mutually exclusive")
    config = load_config(args.config)
    if args.pilot:
        args.start_date = args.start_date or date.fromisoformat(config["validation"]["start_date"])
        args.end_date = args.end_date or date.fromisoformat(config["validation"]["end_date"])
    if args.start_date is None or args.end_date is None:
        parser.error("Supply both dates or --pilot; there is no full-range default")
    try:
        result = run_task1(
            source=args.source, start_date=args.start_date, end_date=args.end_date,
            config_path=args.config, taxonomy_path=args.taxonomy,
            raw_dir=args.raw_dir, interim_dir=args.interim_dir,
            processed_dir=args.processed_dir, jisdor_input=args.jisdor_input,
            manifest_path=args.manifest, refresh_discovery=args.refresh_discovery,
            full_range=args.full_range,
            force_enrichment=args.force_enrichment,
            enrichment_workers=args.enrichment_workers, cache_only=args.cache_only,
            clean_jisdor=not args.skip_jisdor_clean,
            sample_size=args.sample_size, reject_sample_size=args.reject_sample_size,
            sample_seed=args.sample_seed,
        )
    except Exception as exc:
        print(f"Task 1 CNBC {'full-range' if args.full_range else 'pilot'} failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
