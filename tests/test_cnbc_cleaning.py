import json
from datetime import date

from src.preprocessing.clean_cnbc import COLUMNS, clean_records, publication_date_from_url


def record(url="https://www.cnbc.com/2021/09/01/story.html", **updates):
    value = {
        "__typename": "cnbcnewsstory",
        "title": "  A   CNBC story  ",
        "url": url,
        "discovered_for_date": "2021-09-01",
        "archive_url": "https://www.cnbc.com/site-map/articles/2021/September/1/",
        "retrieved_at_utc": "2026-09-14T00:00:00+00:00",
    }
    value.update(updates)
    return value


def test_clean_deduplicates_and_keeps_date_only_semantics():
    frame, report = clean_records([record(), record(title="A second title")])
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 1
    assert frame.iloc[0].title == "A CNBC story"
    assert frame.iloc[0].publication_date == "2021-09-01"
    assert frame.iloc[0].published_at_utc == ""
    assert frame.iloc[0].timestamp_semantics == "date_only_from_cnbc_canonical_url"
    assert frame.iloc[0].source_record_count == 2
    assert len(json.loads(frame.iloc[0].retrieval_provenance)) == 2
    assert report["duplicate_records"] == 1


def test_bad_domain_and_undated_url_are_quarantined():
    frame, report = clean_records([
        record("https://example.com/2021/09/01/story.html"),
        record("https://www.cnbc.com/story.html"),
    ])
    assert frame.empty
    assert report["quarantined_records"] == 2
    assert report["invalid_urls"] == 1
    assert report["missing_url_publication_dates"] == 2


def test_date_parser_accepts_slashes_in_slug_and_advertorial_prefix():
    assert publication_date_from_url(
        "https://www.cnbc.com/2021/09/01/another-9/11-story.html"
    ) == date(2021, 9, 1)
    assert publication_date_from_url(
        "https://www.cnbc.com/advertorial/2021/09/03/story.html"
    ) == date(2021, 9, 3)


def test_archive_updated_date_mismatch_is_visible_not_dropped():
    frame, report = clean_records([record(discovered_for_date="2021-09-02")])
    assert len(frame) == 1
    assert report["url_date_mismatches"] == 1
    assert "url_date_differs_from_archive_date" in frame.iloc[0].quality_flags


def test_requested_publication_scope_excludes_but_preserves_provenance():
    frame, report = clean_records(
        [record("https://www.cnbc.com/2021/08/31/old.html"), record()],
        start_date=date(2021, 9, 1), end_date=date(2021, 9, 7),
    )
    assert frame.publication_date.tolist() == ["2021-09-01"]
    assert report["scope_excluded_records"] == 1
    assert report["duplicate_records"] == 0
    assert any(
        "publication_date_outside_requested_range" in item["reason_codes"]
        for item in report["quarantine_records"]
    )
