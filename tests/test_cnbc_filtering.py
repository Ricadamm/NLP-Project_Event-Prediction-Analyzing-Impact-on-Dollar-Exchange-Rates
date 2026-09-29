import json

import pandas as pd

from src.preprocessing.filter_cnbc import filter_dataframe, match_taxonomy


TAXONOMY = {
    "armed_conflict": {"keywords": ["war", "military"]},
    "sanctions": {"keywords": ["sanctions", "asset freeze"]},
}


def test_word_boundaries_do_not_match_fragments():
    assert match_taxonomy("An award for military service", TAXONOMY) == {
        "armed_conflict": ["military"]
    }
    assert match_taxonomy("An award ceremony", TAXONOMY) == {}


def test_phrase_and_multi_category_matches_are_preserved():
    frame = pd.DataFrame([{
        "article_id": "a", "title": "Military sanctions and an asset freeze",
        "keywords": "[]", "section": "Politics", "subsection": "",
        "publisher_categories": "[]",
    }])
    result, report = filter_dataframe(frame, TAXONOMY, high_value_sections=["politics"])
    row = result.iloc[0]
    assert bool(row.is_geopolitical_candidate)
    assert json.loads(row.matched_categories) == ["armed_conflict", "sanctions"]
    assert set(json.loads(row.matched_keywords)) == {"military", "sanctions", "asset freeze"}
    assert row.filter_reason == "title_keyword_match"
    assert report["candidate_articles"] == 1


def test_publisher_keyword_can_select_but_section_prior_alone_cannot():
    frame = pd.DataFrame([
        {"article_id": "keyword", "title": "Central update", "keywords": '["sanctions"]',
         "section": "Business", "subsection": "", "publisher_categories": "[]"},
        {"article_id": "section", "title": "Ordinary update", "keywords": "[]",
         "section": "Politics", "subsection": "", "publisher_categories": "[]"},
        {"article_id": "none", "title": "Consumer product launch", "keywords": "[]",
         "section": "Retail", "subsection": "", "publisher_categories": "[]"},
    ])
    result, report = filter_dataframe(frame, TAXONOMY, high_value_sections=["politics"])
    assert result.is_geopolitical_candidate.tolist() == [True, False, False]
    assert result.filter_reason.tolist() == [
        "publisher_keyword_match", "section_prior_only_insufficient", "no_taxonomy_match"
    ]
    assert report["candidate_articles"] + report["non_candidate_articles"] == 3

