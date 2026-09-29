"""Offline metadata-cleaning tests; no live GDELT or publisher requests."""

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from src.preprocessing.clean_news import (
    COLUMNS,
    article_id,
    clean_records,
    clean_title,
    main,
    normalize_url,
    parse_timestamp,
    read_raw_records,
    write_clean_outputs,
)


def record(**overrides):
    return {
        "url": "https://www.reuters.com/world/story?utm_source=feed",
        "title": "  Oil\tprices rise  5%: OPEC's decision!  ",
        "seendate": "20220101T183000Z",
        "domain": "reuters.com",
        "language": "English",
        "sourcecountry": "United Kingdom",
        "query_category": "energy_geopolitics",
        "query_string": '(OPEC OR "oil supply") domain:reuters.com sourcelang:english',
        "requested_start": "2022-01-01T00:00:00+00:00",
        "requested_end": "2022-01-02T00:00:00+00:00",
        "retrieved_at_utc": "2026-09-09T00:00:00+00:00",
        **overrides,
    }


def test_utc_wib_conversion_and_date_boundary():
    frame, report = clean_records([record()])
    row = frame.iloc[0]
    assert row.published_at_utc.isoformat() == "2022-01-01T18:30:00+00:00"
    assert row.published_at_wib.isoformat() == "2022-01-02T01:30:00+07:00"
    assert str(frame.published_at_utc.dt.tz) == "UTC"
    assert str(frame.published_at_wib.dt.tz) == "Asia/Jakarta"
    assert row.timestamp_semantics == "gdelt_first_seen"
    assert "not verified publication" in report["timestamp_semantics"]


@pytest.mark.parametrize("value", [
    "2022-01-01", "2022-01-01T12:00:00", "20220230T000000Z",
    "20220101T000000", "bad", "", None, 20220101,
])
def test_invalid_or_naive_timestamps_are_not_assumed_utc(value):
    assert pd.isna(parse_timestamp(value))


def test_aware_iso_offset_normalizes_to_utc():
    assert parse_timestamp("2022-01-02T01:30:00+07:00").isoformat() == "2022-01-01T18:30:00+00:00"
    assert parse_timestamp("20220101T000000Z").isoformat() == "2022-01-01T00:00:00+00:00"


def test_title_cleaning_preserves_semantics():
    assert clean_title(" \tCafe\u0301\u00a0 Prices\nUP 10% — OPEC's $5! ") == "Café Prices UP 10% — OPEC's $5!"
    assert clean_title("x\u200dy") == "x\u200dy"  # Do not delete Unicode joiners.
    assert clean_title(None) == ""


@pytest.mark.parametrize(("original", "expected"), [
    ("https://Example.COM:443/A%2fb/?z=2&a=1&a=2&utm_source=x#Part", "https://Example.COM:443/A%2fb/?z=2&a=1&a=2#Part"),
    ("https://example.com/a?utm_campaign=c&x=a+b&x=a%20b&utm_content=c", "https://example.com/a?x=a+b&x=a%20b"),
    ("https://example.com/?utm_medium=x", "https://example.com/"),
    ("https://example.com/?utm_term=x&utm_source=y", "https://example.com/"),
    ("https://example.com/a?b=2&&a=1&", "https://example.com/a?b=2&&a=1&"),
    ("https://example.com/a?", "https://example.com/a?"),
    ("https://example.com/a?article=7#utm_source=x", "https://example.com/a?article=7#utm_source=x"),
    ("https://example.com/a?%75tm_source=x", "https://example.com/a?%75tm_source=x"),
    ("https://example.com/a?ref=article&utm_id=identity", "https://example.com/a?ref=article&utm_id=identity"),
    ("https://example.com/a/%2F/b", "https://example.com/a/%2F/b"),
    (" https://example.com/a ", "https://example.com/a"),
])
def test_url_normalization_is_non_destructive(original, expected):
    assert normalize_url(original) == expected
    assert normalize_url(expected) == expected


@pytest.mark.parametrize("value", [
    None, "", "  ", "not a URL", "/relative/path", "ftp://example.com/a",
    "https:///path", "https://example.com:wrong/a", "https://example.com:99999/a",
    "https://exa\nmple.com/a", "https://example.com/a b", "https://[bad/a",
    "https://example.com/%ZZ", 42,
])
def test_invalid_urls_are_detected(value):
    assert normalize_url(value) is None


def test_dedup_categories_provenance_ids_and_input_order_are_deterministic():
    sources = [
        record(),
        record(url="https://www.reuters.com/world/story?utm_campaign=other",
               title="Other title", query_category="sanctions",
               seendate="20220101T173000Z", retrieved_at_utc="2026-09-09T01:00:00+00:00"),
        record(),  # Preserve even identical repeated retrievals in provenance.
    ]
    untouched = deepcopy(sources)
    forward, report = clean_records(sources)
    reverse, reverse_report = clean_records(reversed(sources))
    pd.testing.assert_frame_equal(forward, reverse)
    assert report == reverse_report
    assert sources == untouched
    row = forward.iloc[0]
    assert json.loads(row.categories) == ["energy_geopolitics", "sanctions"]
    assert row.published_at_utc.isoformat() == "2022-01-01T17:30:00+00:00"
    assert row.title == "Other title"
    assert row.original_url in {source["url"] for source in sources}
    assert len(json.loads(row.original_urls)) == 2
    assert len(json.loads(row.retrieval_provenance)) == 3
    assert json.loads(row.retrieval_provenance).count(sources[0]) == 2
    expected_id = hashlib.sha256(b"https://www.reuters.com/world/story").hexdigest()
    assert row.article_id == article_id(row.normalized_url) == expected_id
    assert report["raw_returned_records"] == 3
    assert report["unique_articles"] == 1
    assert report["duplicate_records"] == 2
    assert report["records_by_category"] == {"energy_geopolitics": 2, "sanctions": 1}
    assert report["unique_articles_by_category"] == {"energy_geopolitics": 1, "sanctions": 1}


def test_url_identity_quirks_remain_distinct():
    urls = ["https://example.com/a%2Fb", "https://example.com/a/b",
            "https://example.com/a?x=1&y=2", "https://example.com/a?y=2&x=1",
            "https://example.com/a#one", "https://example.com/a#two"]
    frame, report = clean_records([record(url=url) for url in urls])
    assert frame.article_id.nunique() == len(urls)
    assert report["duplicate_records"] == 0


def test_quarantine_and_quality_counts_preserve_problem_records():
    sources = [record(), record(title=None, seendate="bad"),
               record(url=None), record(url=None),
               record(url="not a URL", title="", seendate=None),
               record(url="https://example.com/keep", title=None, seendate="naive")]
    frame, report = clean_records(sources)
    assert report["raw_returned_records"] == 6
    assert report["unique_articles"] == 2
    assert report["duplicate_records"] == 1
    assert report["quarantined_records"] == 3
    assert report["missing_urls"] == 2
    assert report["invalid_urls"] == 3
    assert report["missing_titles"] == 3
    assert report["missing_timestamps"] == 1
    assert report["invalid_timestamps"] == 3
    assert report["records_with_quality_issues"] == 5
    assert report["raw_returned_records"] == sum(report[key] for key in (
        "unique_articles", "duplicate_records", "quarantined_records"))
    assert len(report["quarantine_records"]) == 3
    assert sum(item["record"]["url"] is None for item in report["quarantine_records"]) == 2
    kept = frame.loc[frame.normalized_url == "https://example.com/keep"].iloc[0]
    assert kept.title == ""
    assert pd.isna(kept.published_at_utc)
    assert json.loads(kept.quality_flags) == ["invalid_timestamp", "missing_title"]


def test_empty_records_have_stable_schema_and_zero_counts():
    frame, report = clean_records([])
    assert list(frame.columns) == COLUMNS
    assert frame.empty
    assert str(frame.published_at_utc.dtype) == "datetime64[ns, UTC]"
    assert str(frame.published_at_wib.dtype) == "datetime64[ns, Asia/Jakarta]"
    assert report["raw_returned_records"] == report["unique_articles"] == 0
    assert report["duplicate_records"] == report["invalid_timestamps"] == 0
    assert report["earliest_article"] is None
    assert report["records_by_category"] == {}


def test_recursive_reader_only_reads_terminal_envelopes(tmp_path):
    nested = tmp_path / "2022" / "01"
    nested.mkdir(parents=True)
    for status in ["completed", "saturated", "split", "failed", "pending"]:
        (nested / f"{status}.json").write_text(json.dumps({
            "status": status, "records": [record(query_category=status)]
        }), encoding="utf-8")
    assert [item["query_category"] for item in read_raw_records(tmp_path)] == ["completed", "saturated"]


def test_reader_does_not_silently_skip_broken_terminal_envelope(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"status":"completed","records":null}', encoding="utf-8")
    with pytest.raises(ValueError, match="records list"):
        read_raw_records(path)


def identity(**overrides):
    return {
        "query": "war domain:reuters.com sourcelang:english", "topic": "armed_conflict",
        "domain": "reuters.com", "logical_start": "2022-01-01T00:00:00Z",
        "logical_end": "2022-01-02T00:00:00Z", "api_config": {"max_records": 250},
        **overrides,
    }


def write_attempt(root, name, attempt, status="completed", window=None, title="Current"):
    path = root / "2022" / f"{name}.attempt-{attempt:04}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"identity": window or identity(), "status": status, "records": [record(title=title)]}
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def write_checkpoint(root, raw, attempt, status, window=None):
    path = root / "_checkpoints" / "window.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**(window or identity()), "attempt": attempt,
                                "status": status, "raw_file": raw.relative_to(root).as_posix()}), encoding="utf-8")
    return path


def test_reader_uses_checkpoint_referenced_attempt_and_skips_checkpoint_records(tmp_path):
    first = write_attempt(tmp_path, "window", 1, title="Checkpoint choice")
    write_attempt(tmp_path, "window", 2, title="Orphan newer file")
    write_checkpoint(tmp_path, first, 1, "completed")
    assert [item["title"] for item in read_raw_records(tmp_path)] == ["Checkpoint choice"]


@pytest.mark.parametrize("status", ["failed", "pending", "split"])
def test_latest_attempt_never_revives_superseded_completion(tmp_path, status):
    write_attempt(tmp_path, "window", 1, title="Old completion")
    write_attempt(tmp_path, "window", 2, status=status)
    assert read_raw_records(tmp_path) == []


def test_reader_without_checkpoint_selects_latest_numbered_attempt_per_full_identity(tmp_path):
    write_attempt(tmp_path, "window", 2, title="Old")
    write_attempt(tmp_path, "window", 10, title="Latest")
    write_attempt(tmp_path, "other_config", 1, title="Other config", window=identity(api_config={"max_records": 20}))
    assert sorted(item["title"] for item in read_raw_records(tmp_path)) == ["Latest", "Other config"]


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_new_checkpoint_failure_or_interrupt_excludes_old_completion_without_raw_file(tmp_path, status):
    write_attempt(tmp_path, "window", 1)
    absent = tmp_path / "2022" / "window.attempt-0002.json"
    write_checkpoint(tmp_path, absent, 2, status)
    assert read_raw_records(tmp_path) == []


def test_terminal_checkpoint_with_missing_raw_evidence_raises(tmp_path):
    absent = tmp_path / "2022" / "window.attempt-0001.json"
    write_checkpoint(tmp_path, absent, 1, "completed")
    with pytest.raises(FileNotFoundError):
        read_raw_records(tmp_path)


def test_force_parent_completion_excludes_old_child_attempts(tmp_path):
    write_attempt(tmp_path, "parent", 1, status="split")
    write_attempt(tmp_path, "child", 1, title="Old child", window=identity(logical_end="2022-01-01T12:00:00Z"))
    assert [item["title"] for item in read_raw_records(tmp_path)] == ["Old child"]
    write_attempt(tmp_path, "parent", 2, title="New parent")
    assert [item["title"] for item in read_raw_records(tmp_path)] == ["New parent"]


def test_report_paths_within_repository_are_relative(tmp_path, monkeypatch):
    import src.preprocessing.clean_news as cleaner
    monkeypatch.setattr(cleaner, "ROOT", tmp_path)
    output = tmp_path / "data" / "interim" / "news" / "clean.csv"
    _, report = write_clean_outputs([], output, output.with_suffix(".json"))
    assert report["output_path"] == "data/interim/news/clean.csv"
    assert report["quarantine_path"] == "data/interim/news/gdelt_news_quarantine.json"


def test_cli_writes_csv_report_quarantine_and_preserves_raw(tmp_path):
    raw = tmp_path / "raw" / "response.json"
    raw.parent.mkdir()
    raw.write_text(json.dumps({"status": "completed", "records": [record(), record(url=None)]}), encoding="utf-8")
    before = raw.read_bytes()
    output = tmp_path / "interim" / "gdelt_news_clean.csv"
    report_path = tmp_path / "interim" / "report.json"
    assert main(["--input", str(raw.parent), "--output", str(output), "--report", str(report_path)]) == 0
    assert raw.read_bytes() == before
    saved = pd.read_csv(output)
    assert len(saved) == 1
    assert saved.iloc[0].published_at_wib == "2022-01-02 01:30:00+07:00"
    assert json.loads(report_path.read_text())["quarantined_records"] == 1
    quarantine = json.loads((output.parent / "gdelt_news_clean_quarantine.json").read_text())
    assert quarantine[0]["record"]["url"] is None
    assert not list(output.parent.glob("*.tmp"))


def test_standalone_defaults_preserve_scoped_collector_outputs(tmp_path, monkeypatch):
    import src.preprocessing.clean_news as cleaner
    monkeypatch.setattr(cleaner, "ROOT", tmp_path)
    raw = tmp_path / "data/raw/news/gdelt/response.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"status": "completed", "records": [record()]}), encoding="utf-8")
    interim = tmp_path / "data/interim/news"
    interim.mkdir(parents=True)
    collector_files = [interim / name for name in (
        "gdelt_news_clean.csv", "gdelt_news_cleaning_report.json",
        "gdelt_news_quarantine.json", "gdelt_pilot_report.json",
    )]
    for path in collector_files:
        path.write_text("scoped collector evidence", encoding="utf-8")
    assert main([]) == 0
    assert all(path.read_text() == "scoped collector evidence" for path in collector_files)
    assert (interim / "gdelt_news_all_candidates.csv").exists()
    assert (interim / "gdelt_news_all_candidates_report.json").exists()
    assert (interim / "gdelt_news_all_candidates_quarantine.json").exists()


def test_outputs_cannot_overwrite_each_other(tmp_path):
    path = tmp_path / "output.csv"
    with pytest.raises(ValueError, match="must be different"):
        write_clean_outputs([], path, path)


def test_cli_prevents_overwriting_raw_input(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text('{"status":"completed","records":[]}', encoding="utf-8")
    before = raw.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--input", str(raw), "--output", str(raw)])
    assert error.value.code == 2
    assert raw.read_bytes() == before
