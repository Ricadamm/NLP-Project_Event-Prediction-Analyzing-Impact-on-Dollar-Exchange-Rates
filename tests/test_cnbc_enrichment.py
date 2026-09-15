import json
from collections import Counter
import threading
import time

import pandas as pd
import pytest

from src.acquisition.enrich_cnbc import (
    CnbcArticleError, enrich_dataframe, extract_article_metadata,
    normalize_publisher_timestamp,
)


def test_jsonld_timestamp_has_priority_and_converts_timezone():
    html = """
    <html><head>
      <meta property="article:published_time" content="2021-09-03T10:00:00+00:00">
      <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"NewsArticle",
       "headline":"Test", "datePublished":"2021-09-03T09:12:45-04:00",
       "articleSection":"Politics", "keywords":["sanctions","world"],
       "author":[{"@type":"Person","name":"A. Reporter"}]}
      </script>
    </head></html>
    """
    result = extract_article_metadata(html, "https://www.cnbc.com/2021/09/03/test.html")
    assert result["published_at_original"] == "2021-09-03T09:12:45-04:00"
    assert result["published_at_utc"] == "2021-09-03T13:12:45+00:00"
    assert result["published_at_wib"] == "2021-09-03T20:12:45+07:00"
    assert result["timestamp_source"] == "jsonld.datePublished"
    assert result["timestamp_status"] == "exact_publisher_timestamp"
    assert result["section"] == "Politics"
    assert result["section_source"] == "jsonld.articleSection"
    assert result["authors"] == ["A. Reporter"]


def test_meta_timestamp_and_section_fallback():
    html = """
    <html><head>
      <meta property="article:published_time" content="2021-09-03T13:12:45+0000">
      <meta property="article:section" content="World">
      <meta name="keywords" content="military, diplomacy">
    </head></html>
    """
    result = extract_article_metadata(html, "https://www.cnbc.com/2021/09/03/test.html")
    assert result["timestamp_source"] == "meta.article:published_time"
    assert result["published_at_wib"] == "2021-09-03T20:12:45+07:00"
    assert result["section"] == "World"
    assert result["section_source"] == "meta.article:section"
    assert result["keywords"] == ["diplomacy", "military"]


def test_missing_timestamp_is_not_invented_and_url_section_is_last_fallback():
    result = extract_article_metadata(
        "<html><head><title>No structured time</title></head></html>",
        "https://www.cnbc.com/advertorial/2021/09/03/test.html",
    )
    assert result["published_at_utc"] is None
    assert result["published_at_wib"] is None
    assert result["timestamp_status"] == "missing_publisher_timestamp"
    assert result["section"] == "advertorial"
    assert result["section_source"] == "canonical_url_path_fallback"


def test_naive_timestamp_is_rejected():
    assert normalize_publisher_timestamp("2021-09-03T12:00:00") is None


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.stats = {"request_count": 0, "retry_count": 0}

    def fetch(self, url):
        self.calls.append(url)
        self.stats["request_count"] += 1
        if self.fail:
            raise CnbcArticleError("HTTP 403", 403)
        return {
            "http_status": 200, "published_at_original": "2021-09-03T12:00:00Z",
            "published_at_utc": "2021-09-03T12:00:00+00:00",
            "published_at_wib": "2021-09-03T19:00:00+07:00",
            "timestamp_source": "jsonld.datePublished",
            "timestamp_status": "exact_publisher_timestamp", "section": "Politics",
            "section_source": "jsonld.articleSection", "keywords": [],
            "publisher_categories": [], "authors": [],
        }


def pilot():
    return pd.DataFrame([{
        "article_id": "a" * 64,
        "normalized_url": "https://www.cnbc.com/2021/09/03/test.html",
        "publication_date": "2021-09-03",
        "discovered_for_date": '["2021-09-03"]',
    }])


def multi_article_pilot(count=8):
    return pd.DataFrame([
        {
            "article_id": f"article-{index}",
            "normalized_url": f"https://www.cnbc.com/2021/09/03/story-{index}.html",
            "publication_date": "2021-09-03",
            "discovered_for_date": '["2021-09-03"]',
        }
        for index in range(count)
    ])


def test_successful_cache_is_reused_without_refetch(tmp_path):
    first = FakeClient()
    enriched, report = enrich_dataframe(pilot(), cache_dir=tmp_path, client=first, progress_every=0)
    assert report["successful_http_fetches"] == 1
    assert report["cache_hits"] == 0
    second = FakeClient()
    reproduced, cached_report = enrich_dataframe(pilot(), cache_dir=tmp_path, client=second, progress_every=0)
    assert second.calls == []
    assert cached_report["cache_hits"] == 1
    assert reproduced.iloc[0].published_at_utc == enriched.iloc[0].published_at_utc


def test_workers_one_retains_serial_client_behavior(tmp_path):
    client = FakeClient()
    enriched, report = enrich_dataframe(
        multi_article_pilot(3), cache_dir=tmp_path, client=client,
        enrichment_workers=1, progress_every=0,
    )
    assert client.calls == multi_article_pilot(3).normalized_url.tolist()
    assert enriched.article_id.tolist() == ["article-0", "article-1", "article-2"]
    assert report["enrichment_workers"] == 1


class ParallelClient(FakeClient):
    def __init__(self, barrier, client_number, calls, calls_lock, fail_index=None):
        super().__init__()
        self.barrier = barrier
        self.client_number = client_number
        self.all_calls = calls
        self.calls_lock = calls_lock
        self.fail_index = fail_index
        self.first_call = True

    def fetch(self, url):
        index = int(url.rsplit("-", 1)[1].split(".", 1)[0])
        with self.calls_lock:
            self.all_calls.append((index, self.client_number, threading.get_ident()))
        self.stats["request_count"] += 1
        if self.first_call:
            self.first_call = False
            self.barrier.wait(timeout=2)
        # Deliberately complete in a different order from the input.
        time.sleep((8 - index) * 0.002)
        if index == self.fail_index:
            raise CnbcArticleError("HTTP 503", 503)
        return {
            "http_status": 200,
            "published_at_original": f"2021-09-03T12:00:0{index}Z",
            "published_at_utc": f"2021-09-03T12:00:0{index}+00:00",
            "published_at_wib": f"2021-09-03T19:00:0{index}+07:00",
            "timestamp_source": "jsonld.datePublished",
            "timestamp_status": "exact_publisher_timestamp",
            "section": "Politics",
            "section_source": "jsonld.articleSection",
            "keywords": [], "publisher_categories": [], "authors": [],
        }


def test_workers_four_are_isolated_resilient_atomic_and_ordered(tmp_path):
    barrier = threading.Barrier(4)
    calls = []
    calls_lock = threading.Lock()
    clients = []

    def client_factory():
        client = ParallelClient(
            barrier, len(clients), calls, calls_lock, fail_index=3
        )
        clients.append(client)
        return client

    source = multi_article_pilot()
    enriched, report = enrich_dataframe(
        source, cache_dir=tmp_path, client_factory=client_factory,
        enrichment_workers=4, progress_every=0,
    )

    assert report["enrichment_workers"] == 4
    assert len(clients) == 4
    assert len({thread_id for _, _, thread_id in calls}) == 4
    assert Counter(index for index, _, _ in calls) == Counter(range(8))
    assert enriched.article_id.tolist() == source.article_id.tolist()
    assert enriched.iloc[3].article_metadata_status == "failed"
    assert (enriched.drop(index=3).article_metadata_status == "completed").all()
    for article_id in source.article_id:
        cached = json.loads((tmp_path / f"{article_id}.json").read_text())
        assert cached["article_id"] == article_id
    assert list(tmp_path.glob("*.tmp")) == []


def test_parallel_cache_preflight_does_not_create_clients_or_submit_requests(tmp_path):
    first = FakeClient()
    source = multi_article_pilot(4)
    enrich_dataframe(source, cache_dir=tmp_path, client=first, progress_every=0)

    def forbidden_factory():
        pytest.fail("a client must not be created when every cache is completed")

    reproduced, report = enrich_dataframe(
        source, cache_dir=tmp_path, client_factory=forbidden_factory,
        enrichment_workers=4, force=True, progress_every=0,
    )
    assert reproduced.article_id.tolist() == source.article_id.tolist()
    assert report["cache_hits"] == 4
    assert report["http_request_count_this_run"] == 0


def test_duplicate_article_ids_are_rejected_before_scheduling(tmp_path):
    source = multi_article_pilot(2)
    source["article_id"] = [1, "1"]
    client = FakeClient()
    with pytest.raises(ValueError, match="article_id values must be unique"):
        enrich_dataframe(source, cache_dir=tmp_path, client=client, progress_every=0)
    assert client.calls == []


def test_fetch_failure_preserves_article_and_status(tmp_path):
    enriched, report = enrich_dataframe(
        pilot(), cache_dir=tmp_path, client=FakeClient(fail=True), progress_every=0
    )
    assert len(enriched) == 1
    assert enriched.iloc[0].timestamp_status == "unresolved"
    assert enriched.iloc[0].timestamp_resolution_detail == "fetch_failed"
    assert enriched.iloc[0].published_at_utc is None
    assert report["failed_fetches"] == 1
    cached = json.loads((tmp_path / f"{'a' * 64}.json").read_text())
    assert cached["status"] == "failed"
    assert cached["http_status"] == 403
