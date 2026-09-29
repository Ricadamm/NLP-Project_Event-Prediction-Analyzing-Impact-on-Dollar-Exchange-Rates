"""Collector persistence and orchestration tests; never use live HTTP."""

from datetime import date, datetime, timedelta, timezone
import json

import pytest

from src.acquisition import collect_gdelt as collector
from src.acquisition.gdelt_client import GdeltError


UTC = timezone.utc
START = datetime(2021, 9, 1, tzinfo=UTC)
END = START + timedelta(days=1)
QUERY = "(war) domain:reuters.com sourcelang:english"


class FakeClient:
    """Scripted logical requests, with the same public contract as GdeltClient."""

    def __init__(self, results=None, max_records=250, extra_stats=None):
        self.results = iter(results) if results is not None else None
        self.max_records = max_records
        self.calls = []
        self.stats = {name: 0 for name in collector.STAT_KEYS}
        self.extra_stats = extra_stats or {}

    def request_parameters(self, query, start, end):
        return {
            "query": query, "mode": "artlist", "format": "json",
            "maxrecords": self.max_records, "sort": "dateasc",
            "startdatetime": (start - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S"),
            "enddatetime": (end + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S"),
        }

    def request(self, query, start, end):
        self.calls.append((query, start, end))
        self.stats["request_count"] += 1
        for key, value in self.extra_stats.items():
            self.stats[key] += value
        action = next(self.results) if self.results is not None else []
        if isinstance(action, BaseException):
            raise action
        return action(query, start, end) if callable(action) else action


def article(name="one", seen="20210901T120000Z"):
    return {"url": f"https://www.reuters.com/world/{name}", "title": f"News {name}",
            "seendate": seen, "domain": "reuters.com", "language": "English",
            "sourcecountry": "United States", "socialimage": ""}


@pytest.fixture
def config():
    return {
        "api": {"max_records": 250, "minimum_window_minutes": 15},
        "study": {"start_date": "2021-09-01", "end_date": "2026-09-01"},
        "sources": {"domains": ["reuters.com"]},
        "language": {"source_language": "english"},
    }


@pytest.fixture
def taxonomy():
    return {"armed_conflict": {"keywords": ["war"]}}


def run(tmp_path, config, taxonomy, client, **kwargs):
    return collector.collect(
        config, taxonomy, date(2021, 9, 1), kwargs.pop("end_date", date(2021, 9, 1)),
        raw_dir=tmp_path / "raw", output_dir=tmp_path / "clean", client=client, **kwargs,
    )


def checkpoints(raw_dir):
    return [json.loads(path.read_text()) for path in sorted((raw_dir / "_checkpoints").glob("*.json"))]


def test_resume_skips_completed_and_preserves_provenance(tmp_path):
    first_client = FakeClient([[article()]])
    first = collector.WindowCollector(first_client, {"minimum_window_minutes": 15}, tmp_path)
    records = first.visit("armed_conflict", "reuters.com", QUERY, START, END)
    assert records[0]["query_category"] == "armed_conflict"
    assert records[0]["requested_start"] == "20210831235959"
    assert records[0]["logical_start"] == "2021-09-01T00:00:00Z"
    raw_state = checkpoints(tmp_path)[0]
    raw_path = tmp_path / raw_state["raw_file"]
    original = raw_path.read_bytes()
    resumed_client = FakeClient([])
    resumed = collector.WindowCollector(resumed_client, first.api_config, tmp_path)
    assert resumed.visit("armed_conflict", "reuters.com", QUERY, START, END) == records
    assert not resumed_client.calls
    assert resumed.cache_hits == 1
    assert raw_path.read_bytes() == original


def test_split_resume_retries_failed_leaf_without_replaying_success(tmp_path):
    settings = {"minimum_window_minutes": 720}
    first = collector.WindowCollector(
        FakeClient([[article("parent1"), article("parent2")], [article("left")], GdeltError("temporary")], max_records=2),
        settings, tmp_path,
    )
    assert [record["title"] for record in first.visit("war", "reuters.com", QUERY, START, END)] == ["News left"]
    assert {state["status"] for state in first.nodes.values()} == {"split", "completed", "failed"}
    second_client = FakeClient([[article("right", "20210901T150000Z")]], max_records=2)
    second = collector.WindowCollector(second_client, settings, tmp_path)
    records = second.visit("war", "reuters.com", QUERY, START, END)
    assert [record["title"] for record in records] == ["News left", "News right"]
    assert [(start, end) for _, start, end in second_client.calls] == [(START + timedelta(hours=12), END)]
    assert second.cache_hits == 2
    assert sum(state["stats"]["request_count"] for state in second.nodes.values()) == 4
    assert second.new_stats["request_count"] == 1


def test_force_retains_immutable_attempt_files(tmp_path):
    settings = {"minimum_window_minutes": 15}
    first = collector.WindowCollector(FakeClient([[article("old")]]), settings, tmp_path)
    first.visit("war", "reuters.com", QUERY, START, END)
    old_file = tmp_path / checkpoints(tmp_path)[0]["raw_file"]
    old_bytes = old_file.read_bytes()
    forced = collector.WindowCollector(FakeClient([[article("new")]]), settings, tmp_path, force=True)
    assert forced.visit("war", "reuters.com", QUERY, START, END)[0]["title"] == "News new"
    assert old_file.read_bytes() == old_bytes
    assert checkpoints(tmp_path)[0]["attempt"] == 2
    assert len(list(tmp_path.rglob("*.attempt-*.json"))) == 2


def test_lost_checkpoint_does_not_overwrite_raw_evidence(tmp_path):
    settings = {"minimum_window_minutes": 15}
    collector.WindowCollector(FakeClient([[article("old")]]), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    old_file = tmp_path / checkpoints(tmp_path)[0]["raw_file"]
    original = old_file.read_bytes()
    next((tmp_path / "_checkpoints").glob("*.json")).unlink()
    collector.WindowCollector(FakeClient([[article("new")]]), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    assert old_file.read_bytes() == original
    assert checkpoints(tmp_path)[0]["attempt"] == 2


@pytest.mark.parametrize("changed", ["query", "api"])
def test_changed_query_or_api_configuration_does_not_reuse_cache(tmp_path, changed):
    settings = {"minimum_window_minutes": 15, "max_records": 250}
    collector.WindowCollector(FakeClient([[]]), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    other_settings = {**settings, "max_records": 200} if changed == "api" else settings
    other_query = QUERY.replace("war", "invasion") if changed == "query" else QUERY
    client = FakeClient([[]])
    worker = collector.WindowCollector(client, other_settings, tmp_path)
    worker.visit("war", "reuters.com", other_query, START, END)
    assert len(client.calls) == 1
    assert worker.cache_hits == 0
    assert len(checkpoints(tmp_path)) == 2


def test_interrupt_saves_pending_state_and_resume_retries(tmp_path):
    settings = {"minimum_window_minutes": 15}
    interrupted = collector.WindowCollector(FakeClient([KeyboardInterrupt()]), settings, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        interrupted.visit("war", "reuters.com", QUERY, START, END)
    pending = checkpoints(tmp_path)[0]
    assert pending["status"] == "pending"
    assert json.loads((tmp_path / pending["raw_file"]).read_text())["status"] == "pending"
    client = FakeClient([[article()]])
    resumed = collector.WindowCollector(client, settings, tmp_path)
    assert len(resumed.visit("war", "reuters.com", QUERY, START, END)) == 1
    assert checkpoints(tmp_path)[0]["attempt"] == 2
    assert checkpoints(tmp_path)[0]["status"] == "completed"
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("corruption", ["missing", "invalid_json", "wrong_identity", "wrong_records_type", "invalid_record_entry"])
def test_missing_or_malformed_cached_raw_is_refetched(tmp_path, corruption):
    settings = {"minimum_window_minutes": 15}
    collector.WindowCollector(FakeClient([[article("old")]]), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    raw_file = tmp_path / checkpoints(tmp_path)[0]["raw_file"]
    if corruption == "missing":
        raw_file.unlink()
    elif corruption == "invalid_json":
        raw_file.write_text("{broken")
    else:
        envelope = json.loads(raw_file.read_text())
        if corruption == "wrong_identity":
            envelope["identity"]["query"] = "different"
        elif corruption == "wrong_records_type":
            envelope["records"] = {}
        else:
            envelope["records"] = [None]
        raw_file.write_text(json.dumps(envelope))
    client = FakeClient([[article("recovered")]])
    worker = collector.WindowCollector(client, settings, tmp_path)
    records = worker.visit("war", "reuters.com", QUERY, START, END)
    assert records[0]["title"] == "News recovered"
    assert len(client.calls) == 1
    assert checkpoints(tmp_path)[0]["attempt"] == 2


@pytest.mark.parametrize("contents", ["{broken", "[]"])
def test_malformed_checkpoint_is_recovered(tmp_path, contents):
    settings = {"minimum_window_minutes": 15}
    collector.WindowCollector(FakeClient([[]]), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    next((tmp_path / "_checkpoints").glob("*.json")).write_text(contents)
    client = FakeClient([[]])
    collector.WindowCollector(client, settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    assert len(client.calls) == 1
    assert checkpoints(tmp_path)[0]["attempt"] == 2


def test_atomic_json_failure_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint.json"
    collector.atomic_json(output, {"status": "pending"})
    original = output.read_bytes()
    def interrupted_replace(*args, **kwargs):
        raise OSError("simulated storage interruption")
    monkeypatch.setattr(type(output), "replace", interrupted_replace)
    with pytest.raises(OSError, match="interruption"):
        collector.atomic_json(output, {"status": "completed"})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_minimum_saturation_saved_and_reported_incomplete(tmp_path, config, taxonomy):
    config["api"].update(max_records=2, minimum_window_minutes=1440)
    report = run(tmp_path, config, taxonomy, FakeClient([[article("one"), article("two")]], max_records=2))
    assert report["raw_returned_records"] == report["unique_articles"] == 2
    assert not report["collection_complete"]
    assert len(report["saturated_minimum_size_windows"]) == 1
    assert not report["failed_windows"]
    assert not report["exhaustive_retrieval_claimed"]


def test_wrong_year_capped_response_fails_once_retains_raw_and_remains_retryable(tmp_path, config, taxonomy):
    wrong_period = [article(str(index), "20260901T120000Z") for index in range(250)]
    client = FakeClient([wrong_period])
    report = run(tmp_path, config, taxonomy, client)
    assert len(client.calls) == 1
    assert report["window_status_counts"] == {"failed": 1}
    assert report["all_raw_response_records"] == 250
    assert report["raw_returned_records"] == report["unique_articles"] == 0
    assert not report["timestamp_scope_consistent"]
    assert not report["collection_complete"]
    state = checkpoints(tmp_path / "raw")[0]
    envelope = json.loads((tmp_path / "raw" / state["raw_file"]).read_text())
    assert state["response_scope_mismatch"] and envelope["response_scope_mismatch"]
    assert envelope["status"] == "failed"
    assert len(envelope["records"]) == 250
    assert envelope["records"][0]["seendate"] == "20260901T120000Z"
    assert envelope["records"][0]["requested_start"] == "20210831235959"
    resumed = run(tmp_path, config, taxonomy, FakeClient([[article("recovered")]]))
    assert resumed["collection_complete"]
    assert resumed["all_raw_response_records"] == 251
    assert resumed["this_run_http"]["request_count"] == 1
    assert checkpoints(tmp_path / "raw")[0]["attempt"] == 2
    assert len(list((tmp_path / "raw").rglob("*.attempt-*.json"))) == 2


def test_normal_in_window_cap_splits_and_all_raw_count_includes_parent(tmp_path, config, taxonomy):
    config["api"].update(max_records=2, minimum_window_minutes=720)
    client = FakeClient([
        [article("parent-left", "20210901T030000Z"), article("parent-right", "20210901T180000Z")],
        [article("left", "20210901T030000Z")], [article("right", "20210901T180000Z")],
    ], max_records=2)
    report = run(tmp_path, config, taxonomy, client)
    assert len(client.calls) == 3
    assert report["window_status_counts"] == {"split": 1, "completed": 2}
    assert report["all_raw_response_records"] == 4
    assert report["raw_returned_records"] == report["unique_articles"] == 2
    assert report["collection_complete"]


def test_legacy_wrong_period_split_cache_is_refetched_without_traversing_children(tmp_path):
    settings = {"minimum_window_minutes": 720}
    initial = collector.WindowCollector(FakeClient([
        [article("one"), article("two")], [], [],
    ], max_records=2), settings, tmp_path)
    initial.visit("war", "reuters.com", QUERY, START, END)
    parent = next(state for state in checkpoints(tmp_path) if state["status"] == "split")
    parent_file = tmp_path / parent["raw_file"]
    envelope = json.loads(parent_file.read_text())
    for record in envelope["records"]:
        record["seendate"] = "20260901T120000Z"
    parent_file.write_text(json.dumps(envelope))
    wrong_period = [article("one", "20260901T120000Z"), article("two", "20260901T120000Z")]
    client = FakeClient([wrong_period], max_records=2)
    resumed = collector.WindowCollector(client, settings, tmp_path)
    assert resumed.visit("war", "reuters.com", QUERY, START, END) == []
    assert len(client.calls) == len(resumed.nodes) == 1
    assert next(iter(resumed.nodes.values()))["status"] == "failed"
    assert parent_file.exists()


def test_scope_guard_requires_exact_valid_gdelt_timestamps():
    params = FakeClient().request_parameters(QUERY, START, END)
    assert collector.entirely_outside_request([article("wrong", "20260901T120000Z")], params)
    assert not collector.entirely_outside_request([article("short", "20260901T1Z")], params)
    assert not collector.entirely_outside_request([article("missing", None)], params)


@pytest.mark.parametrize("outside_seen", ["20210902T001500Z", "invalid-date"])
def test_mixed_or_invalid_timestamp_cap_is_retained_for_ordinary_qa(tmp_path, config, taxonomy, outside_seen):
    config["api"].update(max_records=2, minimum_window_minutes=1440)
    client = FakeClient([[article("inside"), article("other", outside_seen)]], max_records=2)
    report = run(tmp_path, config, taxonomy, client)
    assert len(client.calls) == 1
    assert report["window_status_counts"] == {"saturated": 1}
    assert not report["failed_windows"]
    assert report["all_raw_response_records"] == report["raw_returned_records"] == 2
    if outside_seen == "invalid-date":
        assert report["invalid_timestamps"] == 1
    else:
        assert len(report["unexpected_out_of_window_records"]) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_valid_zero_results_distinguished_from_request_failure(tmp_path, config, taxonomy, failed):
    report = run(tmp_path, config, taxonomy, FakeClient([GdeltError("HTTP unavailable") if failed else []]))
    assert report["raw_returned_records"] == report["unique_articles"] == 0
    assert report["collection_complete"] is (not failed)
    assert len(report["failed_windows"]) == int(failed)
    assert (tmp_path / "clean/gdelt_news_clean.csv").exists()


def test_seven_inclusive_days_and_topic_domain_iteration(tmp_path, config, taxonomy):
    taxonomy["sanctions"] = {"keywords": ["sanction", "asset freeze"]}
    config["sources"]["domains"] = ["reuters.com", "example.com"]
    client = FakeClient()
    report = run(tmp_path, config, taxonomy, client, end_date=date(2021, 9, 7), pilot=True)
    assert report["planned_daily_jobs"] == len(client.calls) == 28
    assert {start.date() for _, start, _ in client.calls} == {date(2021, 9, day) for day in range(1, 8)}
    assert max(end for _, _, end in client.calls) == datetime(2021, 9, 8, tzinfo=UTC)
    assert all(end - start == timedelta(days=1) for _, start, end in client.calls)
    assert report["topics_queried"] == ["armed_conflict", "sanctions"]
    assert report["domains_queried"] == ["reuters.com", "example.com"]
    assert all("sourcelang:english" in query for query, _, _ in client.calls)
    assert any('"asset freeze"' in query for query, _, _ in client.calls)


def test_circuit_breaker_records_remaining_jobs_as_pending(tmp_path, config, taxonomy):
    config["collection"] = {"max_consecutive_failed_jobs": 2}
    client = FakeClient([GdeltError("unavailable"), GdeltError("unavailable")])
    report = run(tmp_path, config, taxonomy, client, end_date=date(2021, 9, 7), pilot=True)
    assert len(client.calls) == 2
    assert len(report["failed_windows"]) == 2
    assert len(report["pending_jobs"]) == 5
    assert report["planned_daily_jobs"] == 7
    assert not report["collection_complete"]
    recovered = run(tmp_path, config, taxonomy, FakeClient(), end_date=date(2021, 9, 7), pilot=True)
    assert recovered["collection_complete"]
    assert recovered["request_count"] == 9
    assert recovered["this_run_http"]["request_count"] == 7


def test_boundary_spillover_preserved_raw_counted_and_excluded_from_clean(tmp_path, config, taxonomy):
    returned = [article("before", "20210831T235959Z"), article("start", "20210901T000000Z"),
                article("last", "20210901T235959Z"), article("end", "20210902T000000Z")]
    report = run(tmp_path, config, taxonomy, FakeClient([returned]))
    assert report["raw_returned_records"] == 4
    assert report["excluded_boundary_records"] == 2
    assert report["records_sent_to_cleaner"] == report["unique_articles"] == 2
    assert not report["unexpected_out_of_window_records"]
    assert report["collection_complete"]
    state = checkpoints(tmp_path / "raw")[0]
    assert len(json.loads((tmp_path / "raw" / state["raw_file"]).read_text())["records"]) == 4


@pytest.mark.parametrize("seen", ["20260801T120000Z", "20210831T235958Z", "20210902T000002Z"])
def test_ignored_http_window_is_reported_incomplete_even_with_successful_http(tmp_path, config, taxonomy, seen):
    report = run(tmp_path, config, taxonomy, FakeClient([[article("wrong-period", seen)]]))
    assert report["raw_returned_records"] == 1
    assert report["records_sent_to_cleaner"] == report["unique_articles"] == 0
    assert report["excluded_boundary_records"] == 1
    assert len(report["unexpected_out_of_window_records"]) == 1
    assert not report["collection_complete"]
    assert not report["failed_windows"]
    assert any("scope mismatch" in warning.lower() for warning in report["warnings"])
    # Resume must preserve the scientific warning even without new HTTP traffic.
    resumed = run(tmp_path, config, taxonomy, FakeClient([]))
    assert not resumed["collection_complete"]
    assert resumed["unexpected_out_of_window_records"] == report["unexpected_out_of_window_records"]
    assert resumed["this_run_http"]["request_count"] == 0


def test_http_counters_persist_on_cached_resume(tmp_path, config, taxonomy):
    first = run(tmp_path, config, taxonomy, FakeClient([[article()]], extra_stats={"request_count": 1, "http_429_count": 1, "retry_count": 1}))
    second = run(tmp_path, config, taxonomy, FakeClient([]))
    for key, expected in {"request_count": 2, "http_429_count": 1, "retry_count": 1}.items():
        assert first[key] == second[key] == expected
        assert second["this_run_http"][key] == 0
    assert second["window_status_counts"] == {"completed": 1}
    assert second["checkpoint_cache_hits"] == 1


def test_default_cli_never_starts_full_range(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No-date CLI must not start collection")
    monkeypatch.setattr(collector, "collect", forbidden)
    with pytest.raises(SystemExit) as exit_info:
        collector.main([])
    assert exit_info.value.code == 2


def test_pilot_over_seven_days_rejected_before_requests(tmp_path, config, taxonomy):
    client = FakeClient([])
    with pytest.raises(ValueError, match="at most seven days"):
        run(tmp_path, config, taxonomy, client, end_date=date(2021, 9, 8), pilot=True)
    assert not client.calls


def test_cli_pilot_uses_seven_day_default(monkeypatch, tmp_path, config, taxonomy):
    captured = {}
    monkeypatch.setattr(collector, "load_settings", lambda *_: (config, taxonomy))
    def fake_collect(config, taxonomy, start, end, **kwargs):
        captured.update(start=start, end=end, **kwargs)
        return {"collection_complete": True}
    monkeypatch.setattr(collector, "collect", fake_collect)
    assert collector.main(["--pilot", "--log-file", str(tmp_path / "run.log")]) == 0
    assert captured["start"] == date(2021, 9, 1)
    assert captured["end"] == date(2021, 9, 7)
    assert captured["pilot"]


def raw_snapshot(path):
    return {file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob("*") if file.is_file()}


def test_report_only_missing_data_is_pending_without_requests_or_raw_files(tmp_path, config, taxonomy):
    client = FakeClient([])
    raw = tmp_path / "raw"
    worker = collector.WindowCollector(client, config["api"], raw, report_only=True)
    assert worker.visit("armed_conflict", "reuters.com", QUERY, START, END) == []
    assert not raw.exists()
    assert next(iter(worker.nodes.values()))["status"] == "pending"
    report = run(tmp_path, config, taxonomy, client, report_only=True,
                 end_date=date(2021, 9, 7), pilot=True)
    assert not client.calls
    assert not raw.exists()
    assert report["report_only"]
    assert report["planned_daily_jobs"] == len(report["pending_jobs"]) == 7
    assert not report["collection_complete"]
    assert not report["request_windows_completed"]
    assert report["this_run_http"] == dict.fromkeys(collector.STAT_KEYS, 0)
    assert (tmp_path / "clean/gdelt_news_clean.csv").exists()


def test_report_only_rebuilds_completed_scope_without_mutating_evidence(tmp_path, config, taxonomy):
    first = run(tmp_path, config, taxonomy, FakeClient([[article("one"), article("one")]],
                                                    extra_stats={"http_429_count": 1, "retry_count": 1}))
    original = raw_snapshot(tmp_path / "raw")
    clean_bytes = (tmp_path / "clean/gdelt_news_clean.csv").read_bytes()
    (tmp_path / "clean/gdelt_news_clean.csv").unlink()
    client = FakeClient([])
    rebuilt = run(tmp_path, config, taxonomy, client, report_only=True)
    assert not client.calls
    assert rebuilt["report_only"] and rebuilt["collection_complete"]
    assert rebuilt["raw_returned_records"] == first["raw_returned_records"] == 2
    assert rebuilt["unique_articles"] == rebuilt["duplicate_records"] == 1
    assert rebuilt["request_count"] == first["request_count"]
    assert rebuilt["http_429_count"] == rebuilt["retry_count"] == 1
    assert rebuilt["this_run_http"] == dict.fromkeys(collector.STAT_KEYS, 0)
    assert raw_snapshot(tmp_path / "raw") == original
    assert (tmp_path / "clean/gdelt_news_clean.csv").read_bytes() == clean_bytes


def test_report_only_preserves_failed_state_and_ignores_network_circuit_breaker(tmp_path, config, taxonomy):
    config["collection"] = {"max_consecutive_failed_jobs": 1}
    run(tmp_path, config, taxonomy, FakeClient([GdeltError("historical failure")]),
        end_date=date(2021, 9, 7), pilot=True)
    original = raw_snapshot(tmp_path / "raw")
    client = FakeClient([])
    report = run(tmp_path, config, taxonomy, client, end_date=date(2021, 9, 7),
                 pilot=True, report_only=True)
    assert not client.calls
    assert len(report["failed_windows"]) == 1
    assert report["failed_windows"][0]["error"] == "historical failure"
    assert len(report["pending_jobs"]) == 6
    assert report["window_status_counts"] == {"failed": 1, "pending": 6}
    assert report["request_count"] == 1
    assert raw_snapshot(tmp_path / "raw") == original


@pytest.mark.parametrize("corruption", ["missing", "malformed"])
def test_report_only_invalid_cache_stays_pending_without_repairing_raw(tmp_path, config, taxonomy, corruption):
    run(tmp_path, config, taxonomy, FakeClient([[article()]]))
    state = checkpoints(tmp_path / "raw")[0]
    file = tmp_path / "raw" / state["raw_file"]
    if corruption == "missing":
        file.unlink()
    else:
        file.write_text("{invalid")
    original = raw_snapshot(tmp_path / "raw")
    client = FakeClient([])
    report = run(tmp_path, config, taxonomy, client, report_only=True)
    assert not client.calls
    assert len(report["pending_jobs"]) == 1
    assert report["raw_returned_records"] == 0
    assert not report["collection_complete"]
    assert raw_snapshot(tmp_path / "raw") == original


def test_report_only_limits_selected_dates_topics_and_domains(tmp_path, config, taxonomy):
    taxonomy["sanctions"] = {"keywords": ["sanction"]}
    config["sources"]["domains"].append("example.com")
    def records_for_window(query, start, end):
        suffix = "war" if "(war)" in query else "sanctions"
        suffix += "-reuters" if "domain:reuters.com" in query else "-example"
        return [article(f"{start.day}-{suffix}", start.strftime("%Y%m%dT120000Z"))]
    run(tmp_path, config, taxonomy, FakeClient([records_for_window] * 8), end_date=date(2021, 9, 2))
    original = raw_snapshot(tmp_path / "raw")
    client = FakeClient([])
    report = run(tmp_path, config, taxonomy, client, topics=["sanctions"],
                 domains=["reuters.com"], report_only=True)
    assert report["planned_daily_jobs"] == report["raw_returned_records"] == report["unique_articles"] == 1
    assert report["request_count"] == 1
    assert report["collection_complete"]
    clean_text = (tmp_path / "clean/gdelt_news_clean.csv").read_text()
    assert "1-sanctions-reuters" in clean_text
    assert "2-sanctions" not in clean_text and "1-war" not in clean_text
    assert not client.calls
    assert raw_snapshot(tmp_path / "raw") == original


def test_report_only_traverses_cached_split_tree_and_marks_missing_leaf_pending(tmp_path):
    settings = {"minimum_window_minutes": 720}
    collector.WindowCollector(FakeClient([
        [article("parent1"), article("parent2")], [article("left")], [article("right")],
    ], max_records=2), settings, tmp_path).visit("war", "reuters.com", QUERY, START, END)
    right_state = next(state for state in checkpoints(tmp_path)
                       if state["logical_start"] == "2021-09-01T12:00:00Z")
    (tmp_path / right_state["raw_file"]).unlink()
    original = raw_snapshot(tmp_path)
    client = FakeClient([], max_records=2)
    worker = collector.WindowCollector(client, settings, tmp_path, report_only=True)
    records = worker.visit("war", "reuters.com", QUERY, START, END)
    assert [record["title"] for record in records] == ["News left"]
    assert sorted(state["status"] for state in worker.nodes.values()) == ["completed", "pending", "split"]
    assert not client.calls
    assert raw_snapshot(tmp_path) == original


def test_force_and_report_only_rejected_before_io(tmp_path, config, taxonomy):
    with pytest.raises(ValueError, match="cannot be combined"):
        run(tmp_path, config, taxonomy, FakeClient([]), force=True, report_only=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        collector.WindowCollector(FakeClient([]), config["api"], tmp_path / "raw", force=True, report_only=True)
    log_file = tmp_path / "logs/run.log"
    with pytest.raises(SystemExit) as failure:
        collector.main(["--pilot", "--force", "--report-only", "--log-file", str(log_file)])
    assert failure.value.code == 2
    assert not list(tmp_path.iterdir())


def test_cli_forwards_report_only(monkeypatch, tmp_path, config, taxonomy):
    captured = {}
    monkeypatch.setattr(collector, "load_settings", lambda *_: (config, taxonomy))
    def fake_collect(*args, **kwargs):
        captured.update(kwargs)
        return {"collection_complete": False}
    monkeypatch.setattr(collector, "collect", fake_collect)
    assert collector.main(["--pilot", "--report-only", "--log-file", str(tmp_path / "run.log")]) == 2
    assert captured["report_only"]


def copy_raw_without_checkpoints(source, destination):
    for file in source.rglob("*.attempt-*.json"):
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())


def test_fresh_clone_report_only_reconstructs_latest_records_and_cumulative_counts(tmp_path, config, taxonomy):
    original_root, clone = tmp_path / "original", tmp_path / "clone"
    run(original_root, config, taxonomy,
        FakeClient([[article("old")]], extra_stats={"request_count": 1, "http_429_count": 1, "retry_count": 1}))
    expected = run(original_root, config, taxonomy, FakeClient([[article("new"), article("new")]]), force=True)
    expected_csv = (original_root / "clean/gdelt_news_clean.csv").read_bytes()
    copy_raw_without_checkpoints(original_root / "raw", clone / "raw")
    original = raw_snapshot(clone / "raw")
    client = FakeClient([])
    report = run(clone, config, taxonomy, client, report_only=True)
    assert report["collection_complete"]
    assert report["raw_returned_records"] == 2
    assert report["unique_articles"] == report["duplicate_records"] == 1
    for key in (*collector.STAT_KEYS, "all_raw_response_records"):
        assert report[key] == expected[key]
    assert report["all_raw_response_records"] == 3
    assert report["request_count"] == 3
    assert report["this_run_http"] == dict.fromkeys(collector.STAT_KEYS, 0)
    assert (clone / "clean/gdelt_news_clean.csv").read_bytes() == expected_csv
    assert not client.calls
    assert not (clone / "raw/_checkpoints").exists()
    assert raw_snapshot(clone / "raw") == original


def test_fresh_clone_latest_failure_blocks_older_success(tmp_path, config, taxonomy):
    original_root, clone = tmp_path / "original", tmp_path / "clone"
    run(original_root, config, taxonomy, FakeClient([[article("old")]]))
    expected = run(original_root, config, taxonomy, FakeClient([GdeltError("latest attempt failed")]), force=True)
    copy_raw_without_checkpoints(original_root / "raw", clone / "raw")
    original = raw_snapshot(clone / "raw")
    client = FakeClient([])
    report = run(clone, config, taxonomy, client, report_only=True)
    assert report["raw_returned_records"] == report["unique_articles"] == 0
    assert not report["collection_complete"]
    assert len(report["failed_windows"]) == 1
    assert report["failed_windows"][0]["error"] == "latest attempt failed"
    assert report["failed_windows"][0]["attempt"] == 2
    assert report["all_raw_response_records"] == expected["all_raw_response_records"] == 1
    assert report["request_count"] == expected["request_count"] == 2
    assert not client.calls
    assert raw_snapshot(clone / "raw") == original


def test_fresh_clone_reconstructs_split_parent_and_children(tmp_path, config, taxonomy):
    config["api"].update(max_records=2, minimum_window_minutes=720)
    original_root, clone = tmp_path / "original", tmp_path / "clone"
    expected = run(original_root, config, taxonomy, FakeClient([
        [article("parent-left", "20210901T030000Z"), article("parent-right", "20210901T180000Z")],
        [article("left", "20210901T030000Z")], [article("right", "20210901T180000Z")],
    ], max_records=2))
    copy_raw_without_checkpoints(original_root / "raw", clone / "raw")
    original = raw_snapshot(clone / "raw")
    client = FakeClient([], max_records=2)
    report = run(clone, config, taxonomy, client, report_only=True)
    assert report["collection_complete"]
    assert report["window_status_counts"] == {"split": 1, "completed": 2}
    assert report["raw_returned_records"] == expected["raw_returned_records"] == 2
    assert report["all_raw_response_records"] == expected["all_raw_response_records"] == 4
    assert report["request_count"] == expected["request_count"] == 3
    assert not client.calls
    assert raw_snapshot(clone / "raw") == original


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_explicit_checkpoint_blocks_raw_reconstruction(tmp_path, config, taxonomy, status):
    run(tmp_path, config, taxonomy, FakeClient([[article("old")]]))
    file = next((tmp_path / "raw/_checkpoints").glob("*.json"))
    state = json.loads(file.read_text())
    state.update(status=status, error="authoritative state")
    file.write_text(json.dumps(state))
    original = raw_snapshot(tmp_path / "raw")
    report = run(tmp_path, config, taxonomy, FakeClient([]), report_only=True)
    assert report["raw_returned_records"] == 0
    assert not report["collection_complete"]
    assert report["window_status_counts"] == {status: 1}
    assert raw_snapshot(tmp_path / "raw") == original


@pytest.mark.parametrize("corruption", ["filename", "identity", "stats", "records", "json"])
def test_malformed_matching_raw_attempt_is_explicitly_pending(tmp_path, config, taxonomy, corruption):
    original_root, clone = tmp_path / "original", tmp_path / "clone"
    run(original_root, config, taxonomy, FakeClient([[article()]]))
    copy_raw_without_checkpoints(original_root / "raw", clone / "raw")
    file = next((clone / "raw").rglob("*.attempt-*.json"))
    if corruption == "filename":
        file.rename(file.with_name(file.name.replace("0001", "bad")))
    elif corruption == "json":
        file.write_text("{broken")
    else:
        envelope = json.loads(file.read_text())
        if corruption == "identity":
            envelope["identity"]["query"] = "wrong query"
        elif corruption == "stats":
            envelope["attempt_stats"]["request_count"] = "unknown"
        else:
            envelope["records"] = [None]
        file.write_text(json.dumps(envelope))
    original = raw_snapshot(clone / "raw")
    client = FakeClient([])
    report = run(clone, config, taxonomy, client, report_only=True)
    assert not report["collection_complete"]
    assert report["raw_returned_records"] == 0
    assert len(report["pending_jobs"]) == 1
    assert report["pending_jobs"][0]["raw_attempt_reconstruction_error"]
    assert not client.calls
    assert raw_snapshot(clone / "raw") == original
