import pandas as pd

from src.preprocessing.sample_cnbc_review import COLUMNS, sample_candidates


def frame():
    rows = []
    for index in range(30):
        category = "armed_conflict" if index < 20 else "sanctions"
        rows.append({
            "article_id": f"id-{index}", "is_geopolitical_candidate": True,
            "matched_categories": f'["{category}"]', "matched_keywords": '["war"]',
            "published_at_wib": f"2021-09-{(index % 7) + 1:02d}T10:00:00+07:00",
            "title": f"Title {index}", "normalized_url": f"https://www.cnbc.com/{index}.html",
            "section": "World" if index % 2 else "Politics", "filter_reason": "title_keyword_match",
        })
    rows.append({"article_id": "not-candidate", "is_geopolitical_candidate": False,
                 "matched_categories": "[]"})
    return pd.DataFrame(rows)


def test_sample_is_deterministic_balanced_and_has_blank_labels():
    first, report = sample_candidates(frame(), n=12, seed=42)
    second, _ = sample_candidates(frame(), n=12, seed=42)
    assert list(first.columns) == COLUMNS
    assert first.article_id.tolist() == second.article_id.tolist()
    assert len(first) == 12
    assert report["sampled_primary_category_counts"] == {"armed_conflict": 6, "sanctions": 6}
    assert report["human_columns_blank"]
    assert first[["human_relevant", "human_primary_category", "human_notes"]].eq("").all().all()


def test_all_candidates_are_included_when_population_is_small():
    sample, report = sample_candidates(frame().head(3), n=100, seed=42)
    assert len(sample) == 3
    assert report["sample_size"] == 3

