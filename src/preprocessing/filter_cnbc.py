"""Apply the existing geopolitical taxonomy to enriched CNBC metadata.

This is an explainable candidate-retrieval rule, not ground-truth relevance.
Every input article is retained and receives a machine-readable decision trail.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/interim/news/cnbc_news_enriched.csv"
DEFAULT_OUTPUT = ROOT / "data/interim/news/cnbc_news_candidates.csv"
DEFAULT_REPORT = ROOT / "data/interim/news/cnbc_filtering_report.json"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_list(value) -> list[str]:
    if isinstance(value, list):
        decoded = value
    elif isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [value]
    else:
        decoded = []
    if not isinstance(decoded, list):
        return []
    return [str(item).strip() for item in decoded if str(item).strip()]


def keyword_pattern(keyword: str) -> re.Pattern:
    """Compile a case-insensitive phrase matcher with token boundaries."""
    words = keyword.strip().split()
    if not words:
        raise ValueError("Taxonomy keywords must be non-empty strings")
    phrase = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<!\w){phrase}(?!\w)", flags=re.IGNORECASE)


def validate_taxonomy(taxonomy: Mapping) -> dict[str, list[str]]:
    if not isinstance(taxonomy, Mapping) or not taxonomy:
        raise ValueError("Geopolitical taxonomy must be a non-empty mapping")
    result = {}
    for category, definition in taxonomy.items():
        keywords = definition.get("keywords") if isinstance(definition, Mapping) else None
        if not isinstance(category, str) or not isinstance(keywords, list) or not keywords:
            raise ValueError(f"Invalid taxonomy category: {category!r}")
        if any(not isinstance(keyword, str) or not keyword.strip() for keyword in keywords):
            raise ValueError(f"Invalid keyword in taxonomy category {category!r}")
        result[category] = list(keywords)
    return result


def match_taxonomy(text: str, taxonomy: Mapping) -> dict[str, list[str]]:
    """Return every category and configured keyword matched in ``text``."""
    validated = validate_taxonomy(taxonomy)
    value = text if isinstance(text, str) else ""
    matches = {}
    for category, keywords in validated.items():
        found = [keyword for keyword in keywords if keyword_pattern(keyword).search(value)]
        if found:
            matches[category] = found
    return matches


def _section_prior(row: Mapping, high_value_sections: list[str]) -> list[str]:
    fields = [row.get("section", ""), row.get("subsection", "")]
    fields.extend(_decode_list(row.get("publisher_categories")))
    text = " | ".join(str(value) for value in fields if value)
    matches = []
    for prior in high_value_sections:
        if keyword_pattern(prior).search(text):
            matches.append(prior)
    return sorted(set(matches), key=str.casefold)


def filter_dataframe(
    enriched: pd.DataFrame,
    taxonomy: Mapping,
    *,
    high_value_sections: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Label all rows using title and publisher-keyword taxonomy matches."""
    validate_taxonomy(taxonomy)
    if not isinstance(enriched, pd.DataFrame) or "title" not in enriched.columns:
        raise ValueError("Enriched CNBC data must contain a title column")
    priors = high_value_sections or []
    rows = []
    category_counts = Counter()
    reason_counts = Counter()
    for source in enriched.to_dict(orient="records"):
        title_matches = match_taxonomy(str(source.get("title", "")), taxonomy)
        publisher_keyword_values = _decode_list(source.get("keywords"))
        publisher_matches = match_taxonomy(" | ".join(publisher_keyword_values), taxonomy)
        section_matches = _section_prior(source, priors)
        categories = sorted(set(title_matches) | set(publisher_matches))
        matched_keywords = sorted(
            {
                keyword
                for matches in (title_matches, publisher_matches)
                for values in matches.values()
                for keyword in values
            },
            key=str.casefold,
        )
        title_keyword_count = len({item for values in title_matches.values() for item in values})
        publisher_keyword_count = len({item for values in publisher_matches.values() for item in values})
        score = title_keyword_count * 3 + publisher_keyword_count * 2 + (1 if section_matches else 0)
        candidate = bool(categories)
        if title_matches and publisher_matches:
            reason = "title_and_publisher_keyword_match"
        elif title_matches:
            reason = "title_keyword_match"
        elif publisher_matches:
            reason = "publisher_keyword_match"
        elif section_matches:
            reason = "section_prior_only_insufficient"
        else:
            reason = "no_taxonomy_match"
        row = dict(source)
        row.update(
            {
                "is_geopolitical_candidate": candidate,
                "matched_categories": _json(categories),
                "matched_keywords": _json(matched_keywords),
                "filter_score": score,
                "filter_reason": reason,
                "title_category_matches": _json(title_matches),
                "publisher_keyword_category_matches": _json(publisher_matches),
                "matched_section_priors": _json(section_matches),
            }
        )
        rows.append(row)
        reason_counts[reason] += 1
        if candidate:
            category_counts.update(categories)
    filtered = pd.DataFrame(rows)
    candidate_mask = filtered["is_geopolitical_candidate"] if not filtered.empty else pd.Series(dtype=bool)
    candidate_count = int(candidate_mask.sum()) if not filtered.empty else 0
    candidate_sections = Counter(
        str(value) if value not in (None, "") and not pd.isna(value) else "(missing)"
        for value in filtered.loc[candidate_mask, "section"]
    ) if candidate_count and "section" in filtered.columns else Counter()
    report = {
        "total_articles": len(filtered),
        "candidate_articles": candidate_count,
        "non_candidate_articles": len(filtered) - candidate_count,
        "candidate_rate_percent": (candidate_count / len(filtered) * 100) if len(filtered) else 0.0,
        "counts_by_matched_category": {
            category: category_counts[category] for category in sorted(validate_taxonomy(taxonomy))
        },
        "candidate_counts_by_section": dict(sorted(candidate_sections.items(), key=lambda item: (-item[1], item[0]))),
        "counts_by_filter_reason": dict(sorted(reason_counts.items())),
        "taxonomy": validate_taxonomy(taxonomy),
        "section_priors": priors,
        "rule": (
            "Candidate iff at least one existing taxonomy keyword matches the title or a publisher keyword tag. "
            "Section priors add one score point but never create a candidate alone."
        ),
        "score": "3 per title keyword + 2 per publisher keyword + 1 for any configured section prior",
        "counts_reconcile": candidate_count + (len(filtered) - candidate_count) == len(filtered),
    }
    return filtered, report


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def filter_file(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    *,
    taxonomy_path: str | Path = ROOT / "config/geopolitical_topics.yaml",
    cnbc_config_path: str | Path = ROOT / "config/cnbc.yaml",
) -> tuple[pd.DataFrame, dict]:
    enriched = pd.read_csv(input_path, keep_default_na=False)
    taxonomy = yaml.safe_load(Path(taxonomy_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(cnbc_config_path).read_text(encoding="utf-8"))
    priors = config.get("filtering", {}).get("high_value_sections", [])
    filtered, report = filter_dataframe(enriched, taxonomy, high_value_sections=priors)
    report.update({"input_path": str(Path(input_path)), "output_path": str(Path(output_path)), "taxonomy_path": str(Path(taxonomy_path))})
    _atomic_text(Path(output_path), filtered.to_csv(index=False, lineterminator="\n"))
    _atomic_text(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return filtered, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "config/geopolitical_topics.yaml")
    parser.add_argument("--cnbc-config", type=Path, default=ROOT / "config/cnbc.yaml")
    args = parser.parse_args(argv)
    filter_file(args.input, args.output, args.report, taxonomy_path=args.taxonomy,
                cnbc_config_path=args.cnbc_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
