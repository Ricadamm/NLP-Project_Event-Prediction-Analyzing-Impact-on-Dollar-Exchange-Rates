from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.acquisition.cnbc_client import ArchivePage
from src.acquisition.collect_cnbc import collect, validate_range
from src.acquisition.enrich_cnbc import enrich_dataframe
from src.alignment.align_news_jisdor import aggregate_jisdor_daily
from src.preprocessing.filter_cnbc import filter_dataframe
from src.preprocessing.prefilter_cnbc import prefilter_dataframe


TAXONOMY = {
    "armed_conflict": {"keywords": ["war", "military"]},
    "sanctions": {"keywords": ["sanctions", "asset freeze"]},
    "trade_conflict": {"keywords": ["trade war"]},
    "energy_geopolitics": {"keywords": ["oil embargo"]},
    "political_instability": {"keywords": ["political crisis"]},
    "monetary_geoeconomic": {"keywords": ["Federal Reserve"]},
}


def test_full_range_is_explicit_and_study_bounded():
    assert validate_range(date(2021, 9, 1), date(2021, 9, 7)) == 7
    with pytest.raises(ValueError, match="limited to 7 days"):
        validate_range(date(2021, 9, 1), date(2021, 9, 30))
    assert validate_range(
        date(2021, 9, 1), date(2021, 9, 30), full_range=True
    ) == 30
    assert validate_range(
        date(2021, 9, 1), date(2026, 9, 1), full_range=True
    ) == 1827
    with pytest.raises(ValueError, match="must be within"):
        validate_range(date(2021, 8, 31), date(2021, 9, 2), full_range=True)
    with pytest.raises(ValueError, match="must be within"):
        validate_range(date(2026, 8, 31), date(2026, 9, 2), full_range=True)
    with pytest.raises(ValueError, match="must be within"):
        validate_range(date(2021, 8, 31), date(2021, 8, 31))
    with pytest.raises(ValueError, match="must be within"):
        validate_range(date(2026, 9, 2), date(2026, 9, 2))


class ArchiveClient:
    def __init__(self):
        self.calls = []
        self.stats = {"request_count": 0, "retry_count": 0}

    def fetch_day(self, day):
        self.calls.append(day)
        self.stats["request_count"] += 1
        return ArchivePage(
            archive_date=day,
            archive_url=f"https://archive.example/{day}",
            records=[{
                "title": "Daily title",
                "url": f"https://www.cnbc.com/{day:%Y/%m/%d}/daily.html",
                "discovered_for_date": day.isoformat(),
            }],
            total_count=1,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            response_sha256="0" * 64,
            response_bytes=100,
        )


def _collection_config():
    return {
        "archive": {"base_url": "https://www.cnbc.com/site-map/articles"},
        "validation": {
            "max_days": 7,
            "study_start_date": "2021-09-01",
            "study_end_date": "2026-09-01",
        },
    }


def test_thirty_day_collection_uses_mock_and_resumes_completed_days(tmp_path):
    first = ArchiveClient()
    collect(
        _collection_config(), date(2021, 9, 1), date(2021, 9, 30),
        raw_dir=tmp_path / "raw", report_path=tmp_path / "report.json",
        client=first, full_range=True, progress_every=0,
    )
    assert len(first.calls) == 30
    second = ArchiveClient()
    report = collect(
        _collection_config(), date(2021, 9, 1), date(2021, 9, 30),
        raw_dir=tmp_path / "raw", report_path=tmp_path / "report.json",
        client=second, full_range=True, progress_every=0,
    )
    assert second.calls == []
    assert report["checkpoint_cache_hits"] == 30


def _discoveries():
    return pd.DataFrame([
        {
            "article_id": "candidate",
            "title": "Military sanctions agreed after war talks",
            "normalized_url": "https://www.cnbc.com/2021/09/01/story.html",
            "publication_date": "2021-09-01",
            "discovered_for_date": '["2021-09-01"]',
        },
        {
            "article_id": "reject",
            "title": "An award for the best smartwatch",
            "normalized_url": "https://www.cnbc.com/2021/09/01/award-smartwatch.html",
            "publication_date": "2021-09-01",
            "discovered_for_date": '["2021-09-01"]',
        },
    ])


class ArticleClient:
    def __init__(self):
        self.calls = []
        self.stats = {"request_count": 0, "retry_count": 0}

    def fetch(self, url):
        self.calls.append(url)
        self.stats["request_count"] += 1
        return {
            "http_status": 200,
            "published_at_original": "2021-09-01T01:00:00Z",
            "published_at_utc": "2021-09-01T01:00:00+00:00",
            "published_at_wib": "2021-09-01T08:00:00+07:00",
            "timestamp_source": "jsonld.datePublished",
            "timestamp_status": "exact_publisher_timestamp",
            "section": "World",
            "section_source": "jsonld.articleSection",
            "keywords": ["sanctions"],
            "publisher_categories": ["World", "sanctions"],
            "article_type": "NewsArticle",
            "authors": ["Reporter"],
        }


def test_prefilter_queue_is_boundary_aware_and_only_queue_is_enriched(tmp_path):
    filtered, queue, report = prefilter_dataframe(_discoveries(), TAXONOMY)
    assert filtered.prefilter_candidate.tolist() == [True, False]
    assert queue.article_id.tolist() == ["candidate"]
    assert report["prefilter_candidates"] == 1
    client = ArticleClient()
    enriched, _ = enrich_dataframe(
        queue, cache_dir=tmp_path / "metadata", client=client, progress_every=0
    )
    assert enriched.article_id.tolist() == ["candidate"]
    assert client.calls == ["https://www.cnbc.com/2021/09/01/story.html"]


def test_prefilter_phrase_and_word_boundaries():
    frame = _discoveries().copy()
    frame.loc[0, "title"] = "Officials order an asset freeze"
    filtered, queue, _ = prefilter_dataframe(frame, TAXONOMY)
    assert queue.article_id.tolist() == ["candidate"]
    assert "asset freeze" in filtered.iloc[0].prefilter_keywords
    assert not bool(filtered.iloc[1].prefilter_candidate)  # war does not match award/smartwatch


def test_final_filter_is_deterministic():
    enriched = _discoveries().iloc[[0]].assign(
        publisher_keywords='["sanctions"]', section="World", subsection="",
        publisher_categories='["World"]',
    )
    first, first_report = filter_dataframe(enriched, TAXONOMY, high_value_sections=["world"])
    second, second_report = filter_dataframe(enriched, TAXONOMY, high_value_sections=["world"])
    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report


def test_daily_aggregation_keeps_zero_news_actual_trading_days():
    jisdor = pd.DataFrame({
        "date": ["2021-09-01", "2021-09-02", "2021-09-03"],
        "jisdor": [14280, 14290, 14270],
    })
    aligned = pd.DataFrame([{
        "article_id": "one",
        "effective_trade_date": "2021-09-02",
        "matched_categories": '["armed_conflict", "sanctions"]',
    }])
    daily, report = aggregate_jisdor_daily(aligned, jisdor)
    assert daily.news_count.tolist() == [0, 1, 0]
    assert daily.armed_conflict_count.tolist() == [0, 1, 0]
    assert daily.sanctions_count.tolist() == [0, 1, 0]
    assert report["trading_days_with_zero_candidate_news"] == 2
