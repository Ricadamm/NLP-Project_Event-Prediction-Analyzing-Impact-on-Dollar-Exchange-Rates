import pandas as pd

from src.alignment.align_news_jisdor import align_exact_news_to_jisdor, jisdor_features


CONFIG = {
    "timezone": "Asia/Jakarta",
    "default_cutoff_wib": "15:00",
    "assumption": "test assumption",
    "cutoff_regimes": [],
}


def rates():
    return pd.DataFrame({
        "date": ["2021-09-02", "2021-09-03", "2021-09-06", "2021-09-07"],
        "jisdor": [14281, 14261, 14239, 14195],
    })


def exact(article_id, timestamp):
    return {
        "article_id": article_id,
        "timestamp_status": "exact_publisher_timestamp",
        "published_at_utc": timestamp,
    }


def test_cutoff_weekend_holiday_and_missing_timestamp_rules():
    news = pd.DataFrame([
        exact("before", "2021-09-03T07:59:00Z"),       # 14:59 WIB
        exact("after", "2021-09-03T08:00:00Z"),        # 15:00 WIB, Friday
        exact("saturday", "2021-09-04T03:00:00Z"),
        exact("sunday", "2021-09-05T03:00:00Z"),
        exact("holiday", "2021-09-01T03:00:00Z"),      # absent actual date
        {"article_id": "missing", "timestamp_status": "missing_publisher_timestamp",
         "published_at_utc": ""},
    ])
    aligned, report = align_exact_news_to_jisdor(news, rates(), CONFIG)
    assert aligned.effective_trade_date.tolist() == [
        "2021-09-03", "2021-09-06", "2021-09-06", "2021-09-06", "2021-09-02", ""
    ]
    assert aligned.alignment_reason.tolist() == [
        "same_day_before_cutoff", "after_cutoff_next_trade_day",
        "weekend_next_trade_day", "weekend_next_trade_day",
        "non_jisdor_day_next_trade_day", "missing_exact_timestamp_unaligned",
    ]
    assert report["articles_aligned"] == 5
    assert report["articles_unaligned"] == 1
    assert aligned.iloc[1].alignment_cutoff_wib == "15:00:00"


def test_financial_movements_use_previous_actual_observation():
    features = jisdor_features(rates())
    monday = features.loc[features.date.astype(str) == "2021-09-06"].iloc[0]
    assert monday.previous_jisdor == 14261
    assert monday.change_idr == -22
    assert abs(monday.return_pct - (((14239 / 14261) - 1) * 100)) < 1e-12

