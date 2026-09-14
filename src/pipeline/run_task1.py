"""Orchestrate the bounded CNBC timestamp-enriched Task 1 pilot."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from src.acquisition.collect_cnbc import collect, load_config, validate_range
from src.acquisition.enrich_cnbc import enrich_file
from src.alignment.align_news_jisdor import align_files
from src.preprocessing.clean_cnbc import read_raw_records, write_clean_outputs
from src.preprocessing.clean_jisdor import clean_workbook
from src.preprocessing.filter_cnbc import filter_file
from src.preprocessing.sample_cnbc_review import sample_file


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
    manifest_path: str | Path = ROOT / "data/interim/task1_cnbc_v2_run_report.json",
    refresh_discovery: bool = False,
    force_enrichment: bool = False,
    cache_only: bool = False,
    clean_jisdor: bool = True,
    sample_size: int = 100,
    sample_seed: int = 42,
) -> dict:
    if source != "cnbc":
        raise ValueError("This v2 orchestrator supports --source cnbc only; GDELT remains independent")
    config = load_config(config_path)
    validate_range(start_date, end_date, config["validation"]["max_days"])
    raw_dir, interim_dir, processed_dir = Path(raw_dir), Path(interim_dir), Path(processed_dir)
    news_dir, jisdor_dir = interim_dir / "news", interim_dir / "jisdor"
    manifest_path = Path(manifest_path)
    manifest = {
        "pipeline": "task1_cnbc_timestamp_enriched_pilot",
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
            "cnbc_pilot_report.json" if refresh_discovery else "cnbc_pilot_cached_report.json"
        )
        acquisition = collect(
            config, start_date, end_date, raw_dir=raw_dir,
            report_path=acquisition_report_path,
            force=refresh_discovery, report_only=not refresh_discovery,
        )
        if not acquisition["collection_complete"]:
            raise RuntimeError("Existing CNBC daily archive evidence is incomplete")
        manifest["stages"]["pilot_discovery"] = {
            "status": "refreshed" if refresh_discovery else "reused",
            "raw_records": acquisition["raw_records"],
            "completed_days": acquisition["completed_days"],
            "output": str(acquisition_report_path),
        }

        clean_csv = news_dir / "cnbc_news_clean.csv"
        cleaned, cleaning = write_clean_outputs(
            read_raw_records(raw_dir), clean_csv,
            news_dir / "cnbc_news_cleaning_report.json",
            news_dir / "cnbc_news_quarantine.json",
            start_date=start_date, end_date=end_date,
        )
        manifest["stages"]["pilot_cleaning"] = {
            "status": "passed", "rows": len(cleaned), "output": str(clean_csv)
        }

        enriched_csv = news_dir / "cnbc_news_enriched.csv"
        enriched, enrichment = enrich_file(
            clean_csv, enriched_csv, news_dir / "cnbc_enrichment_report.json",
            config=config["article_metadata"], cache_dir=raw_dir / "article_metadata",
            force=force_enrichment, cache_only=cache_only,
        )
        manifest["stages"]["article_metadata_enrichment"] = {
            "status": "passed_with_unresolved" if enrichment["missing_timestamps"] else "passed",
            "rows": len(enriched),
            "exact_timestamps": enrichment["exact_publisher_timestamps"],
            "missing_timestamps": enrichment["missing_timestamps"],
            "failed_fetches": enrichment["failed_fetches"],
            "output": str(enriched_csv),
        }

        candidate_csv = news_dir / "cnbc_news_candidates.csv"
        candidates, filtering = filter_file(
            enriched_csv, candidate_csv, news_dir / "cnbc_filtering_report.json",
            taxonomy_path=taxonomy_path, cnbc_config_path=config_path,
        )
        manifest["stages"]["geopolitical_filtering"] = {
            "status": "passed", "rows": len(candidates),
            "candidate_articles": filtering["candidate_articles"],
            "output": str(candidate_csv),
        }

        aligned_csv = processed_dir / "cnbc_jisdor_aligned_v2.csv"
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

        sample, sampling = sample_file(
            aligned_csv, news_dir / "cnbc_manual_review_sample.csv",
            news_dir / "cnbc_manual_review_sample_report.json",
            n=sample_size, seed=sample_seed,
        )
        manifest["stages"]["manual_review_sample"] = {
            "status": "passed", "sample_size": len(sample),
            "seed": sampling["random_seed"],
            "output": str(news_dir / "cnbc_manual_review_sample.csv"),
        }
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
    parser.add_argument("--config", type=Path, default=ROOT / "config/cnbc.yaml")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "config/geopolitical_topics.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/news/cnbc")
    parser.add_argument("--interim-dir", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--jisdor-input", type=Path, default=ROOT / "data/raw/Informasi Kurs Jisdor.xlsx")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/interim/task1_cnbc_v2_run_report.json")
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--force-enrichment", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-jisdor-clean", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args(argv)
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
            force_enrichment=args.force_enrichment, cache_only=args.cache_only,
            clean_jisdor=not args.skip_jisdor_clean,
            sample_size=args.sample_size, sample_seed=args.sample_seed,
        )
    except Exception as exc:
        print(f"Task 1 CNBC v2 pilot failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

