import pandas as pd
import pytest

from src.data_alignment import AlignmentError, align_news_to_jisdor


def rates():
    return pd.DataFrame({
        "date": ["2021-09-01", "2021-09-02", "2021-09-03", "2021-09-06", "2021-09-07"],
        "jisdor": [14284, 14281, 14261, 14239, 14195],
    })


def test_date_only_weekend_moves_to_next_jisdor_without_invented_time():
    news = pd.DataFrame([{"article_id": "a", "publication_date": "2021-09-04"}])
    aligned, report = align_news_to_jisdor(news, rates())
    assert aligned.iloc[0].alignment_basis == "publication_date_only"
    assert aligned.iloc[0].jisdor_date == "2021-09-06"
    assert aligned.iloc[0].jisdor == 14239
    assert aligned.iloc[0].alignment_reason == "date_only_next_trading_day"
    assert aligned.iloc[0].alignment_lag_calendar_days == 2
    assert report["aligned_rows"] == 1


def test_utc_timestamp_is_converted_to_wib_and_cutoff_is_inclusive():
    news = pd.DataFrame([
        {"article_id": "before", "published_at_utc": "2021-09-01T08:59:00Z"},
        {"article_id": "cutoff", "published_at_utc": "2021-09-01T09:00:00Z"},
        {"article_id": "friday-after", "published_at_utc": "2021-09-03T10:00:00Z"},
    ])
    aligned, _ = align_news_to_jisdor(news, rates(), market_close_wib="16:00")
    assert aligned.jisdor_date.tolist() == ["2021-09-01", "2021-09-02", "2021-09-06"]
    assert aligned.alignment_reason.tolist() == [
        "same_trading_day", "after_market_close", "after_market_close_next_trading_day"
    ]


def test_missing_and_out_of_range_rows_are_retained_unaligned():
    news = pd.DataFrame([
        {"article_id": "missing", "publication_date": ""},
        {"article_id": "late", "publication_date": "2021-09-08"},
    ])
    aligned, report = align_news_to_jisdor(news, rates())
    assert len(aligned) == 2
    assert aligned.alignment_reason.tolist() == [
        "invalid_or_missing_publication_time", "beyond_jisdor_range"
    ]
    assert report["unaligned_rows"] == 2


def test_invalid_jisdor_is_rejected_without_interpolation():
    with pytest.raises(AlignmentError):
        align_news_to_jisdor(pd.DataFrame(), pd.DataFrame({"date": ["2021-09-01"], "jisdor": [float("inf")]}))

