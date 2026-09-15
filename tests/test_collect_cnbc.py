from datetime import date, datetime, timezone
import json

import pytest

from src.acquisition.cnbc_client import ArchivePage, CnbcRequestError
from src.acquisition.collect_cnbc import collect, validate_range


class Client:
    def __init__(self):
        self.calls = []
        self.stats = {"request_count": 0, "retry_count": 0}

    def fetch_day(self, day):
        self.calls.append(day)
        self.stats["request_count"] += 1
        url = f"https://www.cnbc.com/{day:%Y/%m/%d}/story.html"
        return ArchivePage(
            day,
            "https://archive.example/",
            [{"title": str(day), "url": url, "discovered_for_date": day.isoformat()}],
            1,
            datetime.now(timezone.utc).isoformat(),
            "0" * 64,
            100,
        )


class SecondPageFailureClient:
    def __init__(self):
        self.stats = {"request_count": 2, "retry_count": 0}
        self.last_partial_records = [
            {"title": f"Story {index}", "url": f"https://www.cnbc.com/{index}.html"}
            for index in range(250)
        ]
        self.last_reported_total_count = 260
        self.last_attempt_evidence = [
            {"page": 1, "http_status": 200, "result_count": 250},
            {"page": 2, "http_status": 503},
        ]

    def fetch_day(self, day):
        raise CnbcRequestError("CNBC HTTP 503 on page 2; exhausted 0 retries")


def config():
    return {
        "archive": {"base_url": "https://www.cnbc.com/site-map/articles"},
        "validation": {"start_date": "2021-09-01", "end_date": "2021-09-07", "max_days": 7},
    }


def test_range_guard_prevents_five_year_or_eight_day_run():
    assert validate_range(date(2021, 9, 1), date(2021, 9, 7)) == 7
    with pytest.raises(ValueError, match="limited to 7 days"):
        validate_range(date(2021, 9, 1), date(2021, 9, 8))
    with pytest.raises(ValueError, match="limited to 7 days"):
        validate_range(date(2021, 9, 1), date(2026, 9, 1))


def test_collects_seven_daily_pages_and_resumes_without_http(tmp_path):
    client = Client()
    raw = tmp_path / "raw"
    report_path = tmp_path / "report.json"
    report = collect(config(), date(2021, 9, 1), date(2021, 9, 7),
                     raw_dir=raw, report_path=report_path, client=client)
    assert report["collection_complete"]
    assert report["raw_records"] == 7
    assert len(client.calls) == 7
    assert len(list(raw.rglob("archive.attempt-0001.json"))) == 7
    assert json.loads(report_path.read_text())["maximum_allowed_days"] == 7

    second = Client()
    resumed = collect(config(), date(2021, 9, 1), date(2021, 9, 7),
                      raw_dir=raw, report_path=report_path, client=second)
    assert resumed["checkpoint_cache_hits"] == 7
    assert second.calls == []


def test_report_only_does_not_acquire_missing_days(tmp_path):
    report = collect(config(), date(2021, 9, 1), date(2021, 9, 1),
                     raw_dir=tmp_path / "raw", report_path=tmp_path / "report.json",
                     report_only=True)
    assert not report["collection_complete"]
    assert report["request_count_this_run"] == 0


def test_second_page_failure_keeps_day_failed_and_never_checkpoints_partial_completion(tmp_path):
    raw = tmp_path / "raw"
    report = collect(
        config(), date(2021, 9, 1), date(2021, 9, 1),
        raw_dir=raw, report_path=tmp_path / "report.json",
        client=SecondPageFailureClient(),
    )
    attempt = json.loads(next(raw.rglob("archive.attempt-0001.json")).read_text())
    checkpoint = json.loads((raw / "_checkpoints" / "2021-09-01.json").read_text())
    assert not report["collection_complete"]
    assert report["completed_days"] == []
    assert attempt["status"] == checkpoint["status"] == "failed"
    assert attempt["records"] == []
    assert len(attempt["partial_records"]) == 250
    assert attempt["reported_total_count"] == 260


def test_collector_refuses_partial_page_labeled_as_complete(tmp_path):
    class PartialClient(Client):
        def fetch_day(self, day):
            page = super().fetch_day(day)
            return ArchivePage(
                page.archive_date, page.archive_url, page.records, 2,
                page.retrieved_at_utc, page.response_sha256, page.response_bytes,
            )

    raw = tmp_path / "raw"
    report = collect(
        config(), date(2021, 9, 1), date(2021, 9, 1),
        raw_dir=raw, report_path=tmp_path / "report.json", client=PartialClient(),
    )
    checkpoint = json.loads((raw / "_checkpoints" / "2021-09-01.json").read_text())
    assert not report["collection_complete"]
    assert checkpoint["status"] == "failed"
