"""Resumable, bounded GDELT candidate collection; no modeling or alignment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import sys
import tempfile

import yaml

from .gdelt_client import GdeltClient, GdeltError, build_query, split_window

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)
UTC = timezone.utc
STAT_KEYS = ("request_count", "http_429_count", "retry_count")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def entirely_outside_request(records: list[dict], params: dict) -> bool:
    """Conclude wrong scope only when every record has a valid outside timestamp.

    Missing/invalid timestamps and mixed in/out-of-window responses do not
    establish that the API ignored the whole request. Their ordinary QA remains
    the cleaner's responsibility, and all raw values are retained unchanged.
    """
    if not records:
        return False
    lower = datetime.strptime(params["startdatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    upper = datetime.strptime(params["enddatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    for record in records:
        try:
            seen = datetime.strptime(record.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return False
        if seen.strftime("%Y%m%dT%H%M%SZ") != record.get("seendate"):
            return False
        if lower <= seen <= upper:
            return False
    return True


def load_settings(config_path: Path, topics_path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    taxonomy = yaml.safe_load(topics_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(taxonomy, dict) or not taxonomy:
        raise ValueError("Config and taxonomy must be nonempty YAML mappings")
    for name, topic in taxonomy.items():
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid category name: {name!r}")
        if not isinstance(topic, dict) or not isinstance(topic.get("keywords"), list):
            raise ValueError(f"Category {name!r} requires a keywords list")
    return config, taxonomy


class WindowCollector:
    """Each logical window has one checkpoint and immutable attempt JSON files.

    Parent saturation responses are kept as raw evidence. Only terminal windows
    feed cleaning. A resumed split node traverses its children without redoing
    successfully completed requests. A failed/pending node is retried.
    """

    def __init__(self, client, api_config: dict, raw_dir: Path, force: bool = False,
                 report_only: bool = False):
        if force and report_only:
            raise ValueError("--force cannot be combined with --report-only")
        self.client = client
        self.api_config = api_config
        self.raw_dir = Path(raw_dir)
        self.force = force
        self.report_only = report_only
        self.nodes: dict[str, dict] = {}
        self.new_stats = Counter()
        self.cache_hits = 0
        self.minimum = api_config.get("minimum_window_minutes", 15)

    def _state_from_raw_attempts(self, identity: dict, key: str,
                                start: datetime, end: datetime) -> dict:
        """Reconstruct absent checkpoints in memory for an offline fresh clone.

        Latest numbered evidence controls status; earlier attempts contribute
        only counters. A malformed matching file makes the window pending,
        rather than silently selecting older successful evidence.
        """
        directory = self.raw_dir / str(start.year) / start.strftime("%Y-%m-%d") / identity["topic"]
        files = list(directory.glob(f"{key}.attempt-*.json"))
        if not files:
            return {}
        try:
            params = self.client.request_parameters(identity["query"], start, end)
            attempts = []
            for file in files:
                match = re.fullmatch(rf"{key}\.attempt-(\d+)\.json", file.name)
                number = int(match[1]) if match else 0
                if number < 1 or file.name != f"{key}.attempt-{number:04}.json":
                    raise ValueError(f"Invalid raw attempt filename: {file.name}")
                envelope = read_json(file)
                records = envelope.get("records")
                stats = envelope.get("attempt_stats")
                count = envelope.get("returned_count", 0)
                if (envelope.get("identity") != identity
                        or envelope.get("status") not in {"pending", "failed", "completed", "saturated", "split"}
                        or not isinstance(records, list)
                        or not isinstance(stats, dict)
                        or any(type(stats.get(name)) is not int or stats[name] < 0 for name in STAT_KEYS)
                        or type(count) is not int or count != len(records)
                        or any(not isinstance(record, dict)
                               or record.get("requested_start") != params["startdatetime"]
                               or record.get("requested_end") != params["enddatetime"]
                               or record.get("query_category") != identity["topic"]
                               or record.get("query_string") != identity["query"]
                               or record.get("logical_start") != identity["logical_start"]
                               or record.get("logical_end") != identity["logical_end"]
                               for record in records)):
                    raise ValueError(f"Invalid raw attempt identity/schema: {file.name}")
                attempts.append((number, file, envelope))
            number, file, latest = max(attempts, key=lambda attempt: attempt[0])
            state = {**identity, "attempt": number, "status": latest["status"],
                     "raw_file": file.relative_to(self.raw_dir).as_posix(),
                     "returned_count": latest.get("returned_count", 0),
                     "stats": {name: sum(envelope["attempt_stats"][name] for _, _, envelope in attempts)
                               for name in STAT_KEYS},
                     "all_raw_response_records": sum(envelope.get("returned_count", 0) for _, _, envelope in attempts),
                     "reconstructed_from_raw_attempts": True}
            for name in ("error", "response_scope_mismatch"):
                if name in latest:
                    state[name] = latest[name]
            LOG.info("Reconstructed offline checkpoint %s from %s raw attempts", key, len(attempts))
            return state
        except (KeyError, TypeError, ValueError, OSError) as error:
            LOG.warning("Cannot reconstruct offline checkpoint %s: %s", key, error)
            return {**identity, "status": "pending", "error": str(error),
                    "raw_attempt_reconstruction_error": str(error)}

    def visit(self, topic: str, domain: str, query: str,
              start: datetime, end: datetime) -> list[dict]:
        identity = {"query": query, "topic": topic, "domain": domain,
                    "logical_start": timestamp(start), "logical_end": timestamp(end),
                    "api_config": self.api_config}
        key = fingerprint(identity)
        checkpoint = self.raw_dir / "_checkpoints" / f"{key}.json"
        previous = {}
        if checkpoint.exists():
            try:
                previous = read_json(checkpoint)
            except (ValueError, OSError):
                LOG.warning("Unreadable checkpoint %s; %s", checkpoint,
                            "pending offline" if self.report_only else "fetching again")
        elif self.report_only:
            previous = self._state_from_raw_attempts(identity, key, start, end)
        state = previous
        cached = False
        if not self.force and state.get("status") in {"completed", "saturated", "split"}:
            try:
                envelope = read_json(self.raw_dir / state["raw_file"])
                expected_params = self.client.request_parameters(query, start, end)
                if (envelope.get("identity") != identity or
                        envelope.get("status") != state["status"] or
                        not isinstance(envelope.get("records"), list) or
                        any(not isinstance(record, dict) or
                            record.get("requested_start") != expected_params["startdatetime"] or
                            record.get("requested_end") != expected_params["enddatetime"] or
                            record.get("query_category") != topic or
                            record.get("query_string") != query
                            for record in envelope["records"])):
                    raise ValueError("Checkpoint/raw identity mismatch")
                if (len(envelope["records"]) >= self.client.max_records
                        and entirely_outside_request(envelope["records"], expected_params)):
                    # Old checkpoints may predate the scope guard. Revalidate
                    # once through HTTP instead of traversing a wrong-period tree.
                    raise ValueError("Cached capped response is entirely outside the requested period")
                cached = True
                self.cache_hits += 1
                LOG.info("Resume %s %s %s -> %s", topic, domain, timestamp(start), state["status"])
            except (KeyError, ValueError, OSError):
                LOG.warning("Missing/invalid cached raw file for %s; %s", key,
                            "pending offline" if self.report_only else "fetching again")
        if not cached and self.report_only:
            # Offline auditing never manufactures a completed empty response,
            # retries failures, or mutates the evidence being audited.
            state = {**previous, **identity,
                     "status": "failed" if previous.get("status") == "failed" else "pending"}
            if state["status"] == "pending":
                state.setdefault("error", "No valid cached response is available; collection is required for this window")
            self.nodes[key] = state
            return []
        if not cached:
            attempt = int(previous.get("attempt", 0)) + 1
            relative = Path(str(start.year)) / start.strftime("%Y-%m-%d") / topic / f"{key}.attempt-{attempt:04}.json"
            # Never overwrite raw evidence, even if a checkpoint disappeared.
            while (self.raw_dir / relative).exists():
                attempt += 1
                relative = relative.with_name(f"{key}.attempt-{attempt:04}.json")
            state = {**identity, "attempt": attempt, "status": "pending",
                     "stats": dict(previous.get("stats", {})), "raw_file": relative.as_posix(),
                     "all_raw_response_records": previous.get("all_raw_response_records", previous.get("returned_count", 0))}
            atomic_json(checkpoint, state)
            before = dict(self.client.stats)
            envelope = {"identity": identity, "status": "pending", "records": []}
            try:
                LOG.info("Collect %s %s [%s, %s)", topic, domain, timestamp(start), timestamp(end))
                params = self.client.request_parameters(query, start, end)
                envelope["request_parameters"] = params
                articles = self.client.request(query, start, end)
                retrieved = timestamp(datetime.now(UTC))
                provenance = {"query_category": topic, "query_string": query,
                              "requested_start": params["startdatetime"],
                              "requested_end": params["enddatetime"],
                              "logical_start": timestamp(start), "logical_end": timestamp(end),
                              "retrieved_at_utc": retrieved}
                records = [{**article, **provenance} for article in articles]
                # Preserve a received payload before evaluating it, including a
                # failed wrong-period response that must never feed cleaning.
                envelope.update(records=records, retrieved_at_utc=retrieved, returned_count=len(records))
                state["returned_count"] = len(records)
                state["all_raw_response_records"] += len(records)
                saturated = len(articles) >= self.client.max_records
                if saturated and entirely_outside_request(records, params):
                    state["response_scope_mismatch"] = True
                    envelope["response_scope_mismatch"] = True
                    raise GdeltError(
                        f"All {len(records)} capped response records fall outside actual HTTP request bounds; "
                        "retained as failed raw evidence without splitting. Resume this window later."
                    )
                children = split_window(start, end, self.minimum) if saturated else None
                status = "split" if children else ("saturated" if saturated else "completed")
                envelope.update(status=status, records=records, request_parameters=params,
                                retrieved_at_utc=retrieved, returned_count=len(records))
                state.update(status=status, returned_count=len(records), error=None)
                if status == "split":
                    LOG.warning("SPLIT %s [%s, %s): %s results", topic, start, end, len(records))
                elif status == "saturated":
                    LOG.critical("POTENTIALLY INCOMPLETE minimum window: %s [%s, %s)", topic, start, end)
            except GdeltError as error:
                state.update(status="failed", error=str(error))
                envelope.update(status="failed", error=str(error))
                LOG.error("Failed %s %s [%s, %s): %s", topic, domain, start, end, error)
            finally:
                delta = {name: self.client.stats.get(name, 0) - before.get(name, 0) for name in STAT_KEYS}
                self.new_stats.update(delta)
                state["stats"] = {name: state["stats"].get(name, 0) + delta[name] for name in STAT_KEYS}
                envelope["attempt_stats"] = delta
                # A KeyboardInterrupt leaves pending evidence and a retryable checkpoint.
                atomic_json(self.raw_dir / relative, envelope)
                atomic_json(checkpoint, state)
                LOG.info("Saved %s (%s)", self.raw_dir / relative, state["status"])
        self.nodes[key] = state
        if state["status"] == "split":
            children = split_window(start, end, self.minimum)
            if children is None:
                raise ValueError("Cached split node cannot split with its recorded configuration")
            return [record for left, right in children
                    for record in self.visit(topic, domain, query, left, right)]
        return envelope["records"] if state["status"] in {"completed", "saturated"} else []


def collect(config: dict, taxonomy: dict, start_date: date, end_date: date, *,
            topics: list[str] | None = None, domains: list[str] | None = None,
            pilot: bool = False, force: bool = False, raw_dir: Path | None = None,
            output_dir: Path | None = None, client=None, report_only: bool = False) -> dict:
    if force and report_only:
        raise ValueError("--force cannot be combined with --report-only")
    from src.preprocessing.clean_news import write_clean_outputs

    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date (both are inclusive)")
    if pilot and (end_date - start_date).days >= 7:
        raise ValueError("A pilot may include at most seven days")
    study = config["study"]
    if start_date < date.fromisoformat(str(study["start_date"])) or end_date > date.fromisoformat(str(study["end_date"])):
        raise ValueError("Requested dates are outside the configured study range")
    selected_topics = list(dict.fromkeys(topics if topics is not None else taxonomy))
    selected_domains = list(dict.fromkeys(domains if domains is not None else config["sources"]["domains"]))
    if not selected_topics or not selected_domains:
        raise ValueError("At least one topic and source domain are required")
    unknown = set(selected_topics) - set(taxonomy)
    if unknown:
        raise ValueError(f"Unknown topics: {sorted(unknown)}")
    language = config["language"]["source_language"]
    queries = {(topic, domain): build_query(taxonomy[topic]["keywords"], domain, language)
               for topic in selected_topics for domain in selected_domains}
    raw_dir = Path(raw_dir or ROOT / "data/raw/news/gdelt")
    output_dir = Path(output_dir or ROOT / "data/interim/news")
    client = client if client is not None else GdeltClient(config["api"])
    worker = WindowCollector(client, config["api"], raw_dir, force, report_only)
    lower = datetime.combine(start_date, datetime.min.time(), UTC)
    upper = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), UTC)
    jobs = [(topic, domain, queries[topic, domain], lower + timedelta(days=i), lower + timedelta(days=i + 1))
            for i in range((end_date - start_date).days + 1)
            for topic in selected_topics for domain in selected_domains]
    failure_limit = config.get("collection", {}).get("max_consecutive_failed_jobs", 3)
    if not isinstance(failure_limit, int) or failure_limit < 1:
        raise ValueError("collection.max_consecutive_failed_jobs must be a positive integer")
    records, pending = [], []
    consecutive_failures = 0
    for index, (topic, domain, query, start, end) in enumerate(jobs):
        failed_before = sum(state["status"] == "failed" for state in worker.nodes.values())
        records.extend(worker.visit(topic, domain, query, start, end))
        failed_after = sum(state["status"] == "failed" for state in worker.nodes.values())
        consecutive_failures = consecutive_failures + 1 if failed_after > failed_before else 0
        if not report_only and consecutive_failures >= failure_limit:
            LOG.error("Stopping after %s consecutive failed jobs; remaining jobs stay pending", failure_limit)
            pending = [{"topic": t, "domain": d, "query": q, "logical_start": timestamp(s),
                        "logical_end": timestamp(e), "status": "pending"}
                       for t, d, q, s, e in jobs[index + 1:]]
            break
    # Padding prevents boundary gaps; preserve spillover in raw, count its
    # deliberate exclusion from the requested study period in QA.
    in_scope, spillover, scope_mismatches = [], [], []
    for record in records:
        try:
            seen = datetime.strptime(record.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except (TypeError, ValueError):
            in_scope.append(record)  # Cleaner flags invalid timestamps, never invents one.
        else:
            (in_scope if lower <= seen < upper else spillover).append(record)
            # Distinguish deliberate one-second boundary overlap from an API
            # that ignored historical bounds and returned another period.
            requested_start = datetime.strptime(record["requested_start"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            requested_end = datetime.strptime(record["requested_end"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            if seen < requested_start or seen > requested_end:
                scope_mismatches.append({"url": record.get("url"), "seendate": record.get("seendate"),
                                         "requested_start": record["requested_start"],
                                         "requested_end": record["requested_end"],
                                         "query_category": record.get("query_category")})
    clean_path = output_dir / "gdelt_news_clean.csv"
    _, cleaning = write_clean_outputs(in_scope, clean_path, output_dir / "gdelt_news_cleaning_report.json")
    states = list(worker.nodes.values())
    pending.extend(state for state in states if state["status"] == "pending")
    failed = [state for state in states if state["status"] == "failed"]
    scope_failed = [state for state in failed if state.get("response_scope_mismatch")]
    saturated = [state for state in states if state["status"] == "saturated"]
    totals = {name: sum(state.get("stats", {}).get(name, 0) for state in states) for name in STAT_KEYS}
    snapshot = {"config": config, "taxonomy": taxonomy}
    report = {**cleaning, "pilot": pilot, "report_only": report_only, "pilot_start": start_date.isoformat(),
              "pilot_end": end_date.isoformat(), "end_date_inclusive": True,
              "topics_queried": selected_topics, "domains_queried": selected_domains,
              "source_language": language, "planned_daily_jobs": len(jobs),
              "raw_returned_records": len(records), "records_sent_to_cleaner": len(in_scope),
              "all_raw_response_records": sum(state.get("all_raw_response_records", state.get("returned_count", 0)) for state in states),
              "excluded_boundary_records": len(spillover),
              "unexpected_out_of_window_records": scope_mismatches,
              "failed_windows": failed, "saturated_minimum_size_windows": saturated,
              "pending_jobs": pending, "collection_complete": not (failed or saturated or pending or scope_mismatches),
              "request_windows_completed": not (failed or pending),
              "unsaturated_retrieval": not saturated,
              "timestamp_scope_consistent": not (scope_mismatches or scope_failed),
              "exhaustive_retrieval_claimed": False, "checkpoint_cache_hits": worker.cache_hits,
              "window_status_counts": dict(Counter(state["status"] for state in states)),
              **totals, "this_run_http": {name: worker.new_stats[name] for name in STAT_KEYS},
              "config_sha256": fingerprint(snapshot), "configuration": snapshot,
              "generated_at_utc": timestamp(datetime.now(UTC)), "python_version": platform.python_version(),
              "clean_output": str(clean_path.relative_to(ROOT)) if clean_path.is_relative_to(ROOT) else str(clean_path),
              "timestamp_semantics": "GDELT seendate is first-seen/index time, not verified article publication time",
              "warnings": ["Retrieved articles are candidates; relevance and archive coverage are unvalidated."]}
    report["count_definitions"]["raw_returned_records"] = (
        "Terminal response occurrences only (split parents excluded): unique_articles + "
        "duplicate_records + quarantined_records + excluded_boundary_records")
    report["count_definitions"]["records_sent_to_cleaner"] = "In-scope occurrences, including invalid timestamps for explicit QA"
    report["count_definitions"]["all_raw_response_records"] = (
        "Cumulative saved response occurrences across attempts for visited checkpoint windows, "
        "including split parents and failed wrong-period responses; excludes unrelated/orphaned windows")
    if failed or pending:
        report["warnings"].append("Pilot/collection is incomplete: failed or pending requests are not zero-news observations.")
    if saturated:
        report["warnings"].append("Minimum-size saturated intervals may be truncated.")
    if scope_mismatches:
        report["warnings"].append("API timestamp scope mismatch: some returned seendate values fall outside submitted bounds. Precise DOC boundary/index-time semantics remain unverified; raw timestamps are retained without adjustment.")
    if scope_failed:
        report["warnings"].append("Capped responses entirely outside actual request bounds were saved as failed, retryable raw evidence; splitting was suppressed to avoid repeatedly querying an ignored period.")
    if not records:
        report["warnings"].append("No candidate records were available to clean; an empty CSV does not establish historical absence.")
    report_path = output_dir / ("gdelt_pilot_report.json" if pilot else "gdelt_collection_report.json")
    atomic_json(report_path, report)
    LOG.info("Saved collection QA %s; complete=%s", report_path, report["collection_complete"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/gdelt.yaml")
    parser.add_argument("--topics-config", type=Path, default=ROOT / "config/geopolitical_topics.yaml")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat, help="Inclusive last UTC calendar date")
    parser.add_argument("--topics", nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--pilot", action="store_true", help="Default to 2021-09-01 through 2021-09-07; maximum seven days")
    parser.add_argument("--force", action="store_true", help="Intentionally re-fetch, preserving earlier raw attempt files")
    parser.add_argument("--report-only", action="store_true", help="Rebuild scoped CSV and QA from cached checkpoints, with no HTTP or raw/checkpoint writes")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/gdelt_collection.log")
    args = parser.parse_args(argv)
    if args.force and args.report_only:
        parser.error("--force cannot be combined with --report-only")
    if args.pilot:
        args.start_date = args.start_date or date(2021, 9, 1)
        args.end_date = args.end_date or date(2021, 9, 7)
    if args.start_date is None or args.end_date is None:
        parser.error("Supply --start-date AND --end-date, or --pilot; no full-range default run")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        config, taxonomy = load_settings(args.config, args.topics_config)
        report = collect(config, taxonomy, args.start_date, args.end_date, topics=args.topics,
                         domains=args.domains, pilot=args.pilot, force=args.force,
                         raw_dir=args.raw_dir, output_dir=args.output_dir, report_only=args.report_only)
    except (ValueError, KeyError, OSError, yaml.YAMLError) as error:
        LOG.error("Collection configuration/storage error: %s", error)
        return 1
    summary_keys = ("pilot_start", "pilot_end", "raw_returned_records", "unique_articles",
                    "duplicate_records", "http_429_count", "retry_count", "collection_complete")
    print(json.dumps({key: report.get(key) for key in summary_keys}, indent=2))
    return 0 if report["collection_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
