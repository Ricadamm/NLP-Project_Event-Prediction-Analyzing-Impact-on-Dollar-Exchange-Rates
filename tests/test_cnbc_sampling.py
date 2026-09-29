import pandas as pd

from src.preprocessing.sample_cnbc_review import (
    COLUMNS, sample_candidates, sample_review_sets,
)


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


def test_review_sample_includes_prefilter_rejects_with_blank_labels():
    prefiltered = pd.DataFrame([
        {
            "article_id": f"reject-{index}",
            "prefilter_candidate": False,
            "prefilter_categories": "[]",
            "prefilter_keywords": "[]",
            "prefilter_reason": "no_title_or_url_prefilter_match",
            "title": f"Reject {index}",
            "normalized_url": f"https://www.cnbc.com/{2021 + index}/reject.html",
            "publication_date": f"{2021 + index}-01-01",
            "discovered_for_date": f'["{2021 + index}-01-01"]',
        }
        for index in range(3)
    ])
    sample, report = sample_review_sets(
        frame(), prefiltered, candidate_n=4, reject_n=2, seed=42
    )
    assert sample.sample_group.value_counts().to_dict() == {
        "final_candidate": 4, "prefilter_reject": 2,
    }
    rejects = sample.loc[sample.sample_group == "prefilter_reject"]
    assert rejects["section"].eq("").all()
    assert sample[["human_relevant", "human_primary_category", "human_notes"]].eq("").all().all()
    assert report["prefilter_reject_sample_size"] == 2
