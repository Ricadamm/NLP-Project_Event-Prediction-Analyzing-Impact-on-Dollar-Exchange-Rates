"""Recall-oriented CNBC title prefilter applied before page enrichment.

The prefilter is an HTTP queueing rule, not the final geopolitical decision.
It uses the project taxonomy plus a small, documented CNBC vocabulary section
from ``config/cnbc.yaml`` and retains every discovery with its decision trail.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit

import pandas as pd
import yaml

from src.preprocessing.filter_cnbc import keyword_pattern, validate_taxonomy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/interim/news/cnbc_article_index.csv"
DEFAULT_OUTPUT = ROOT / "data/interim/news/cnbc_prefiltered.csv"
DEFAULT_QUEUE = ROOT / "data/interim/news/cnbc_enrichment_queue.csv"
DEFAULT_REPORT = ROOT / "data/interim/news/cnbc_prefilter_report.json"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validated_additions(value: Mapping | None, categories: set[str]) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("prefilter.additional_keywords must be a mapping")
    result = {}
    for category, keywords in value.items():
        if category not in categories:
            raise ValueError(f"Prefilter category is absent from the taxonomy: {category!r}")
        if not isinstance(keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip() for keyword in keywords
        ):
            raise ValueError(f"Invalid prefilter additions for {category!r}")
        result[category] = list(keywords)
    return result


def _matches(text: str, definitions: Mapping[str, list[str]]) -> dict[str, list[str]]:
    found = {}
    for category, keywords in definitions.items():
        matches = [keyword for keyword in keywords if keyword_pattern(keyword).search(text)]
        if matches:
            found[category] = matches
    return found


def _url_text(value: str) -> str:
    try:
        path = unquote(urlsplit(value).path)
    except (TypeError, ValueError):
        return ""
    return re.sub(r"[-_/]+", " ", path)


def prefilter_dataframe(
    discoveries: pd.DataFrame,
    taxonomy: Mapping,
    *,
    additional_keywords: Mapping | None = None,
    use_url_path: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Label every discovery and return the candidate-only enrichment queue."""
    required = {"article_id", "title", "normalized_url"}
    if not isinstance(discoveries, pd.DataFrame) or not required.issubset(discoveries.columns):
        missing = sorted(required - set(getattr(discoveries, "columns", [])))
        raise ValueError(f"CNBC article index is missing columns: {missing}")
    taxonomy_terms = validate_taxonomy(taxonomy)
    additions = _validated_additions(additional_keywords, set(taxonomy_terms))
    rows = []
    category_counts = Counter()
    reason_counts = Counter()
    for source in discoveries.to_dict(orient="records"):
        title = str(source.get("title", ""))
        path_text = _url_text(str(source.get("normalized_url", ""))) if use_url_path else ""
        title_taxonomy = _matches(title, taxonomy_terms)
        title_additions = _matches(title, additions)
        path_taxonomy = _matches(path_text, taxonomy_terms)
        path_additions = _matches(path_text, additions)
        all_matches = (title_taxonomy, title_additions, path_taxonomy, path_additions)
        categories = sorted({category for matches in all_matches for category in matches})
        keywords = sorted(
            {keyword for matches in all_matches for values in matches.values() for keyword in values},
            key=str.casefold,
        )
        reasons = []
        if title_taxonomy:
            reasons.append("title_taxonomy_match")
        if title_additions:
            reasons.append("title_config_addition_match")
        if path_taxonomy:
            reasons.append("url_path_taxonomy_match")
        if path_additions:
            reasons.append("url_path_config_addition_match")
        reason = "+".join(reasons) if reasons else "no_title_or_url_prefilter_match"
        row = dict(source)
        row.update(
            {
                "prefilter_candidate": bool(categories),
                "prefilter_categories": _json(categories),
                "prefilter_keywords": _json(keywords),
                "prefilter_reason": reason,
            }
        )
        rows.append(row)
        reason_counts[reason] += 1
        if categories:
            category_counts.update(categories)
    columns = list(discoveries.columns) + [
        "prefilter_candidate", "prefilter_categories", "prefilter_keywords", "prefilter_reason"
    ]
    filtered = pd.DataFrame(rows, columns=columns)
    queue = filtered.loc[filtered["prefilter_candidate"]].reset_index(drop=True)
    candidate_count = len(queue)
    report = {
        "total_discoveries": len(filtered),
        "prefilter_candidates": candidate_count,
        "prefilter_rejects": len(filtered) - candidate_count,
        "prefilter_rate_percent": candidate_count / len(filtered) * 100 if len(filtered) else 0.0,
        "counts_by_category": {
            category: category_counts[category] for category in sorted(taxonomy_terms)
        },
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "taxonomy_keywords": taxonomy_terms,
        "configured_additional_keywords": additions,
        "use_url_path": use_url_path,
        "rule": (
            "Recall-oriented queue rule: candidate when a boundary-aware taxonomy or configured "
            "additional term matches the title or normalized URL path. This is not the final filter."
        ),
        "limitation": (
            "Articles rejected by the title prefilter are not publisher-metadata enriched. "
            "The complete article index is retained for later recall audits."
        ),
        "counts_reconcile": candidate_count + (len(filtered) - candidate_count) == len(filtered),
    }
    return filtered, queue, report


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


def prefilter_file(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    queue_path: str | Path = DEFAULT_QUEUE,
    report_path: str | Path = DEFAULT_REPORT,
    *,
    taxonomy_path: str | Path = ROOT / "config/geopolitical_topics.yaml",
    cnbc_config_path: str | Path = ROOT / "config/cnbc.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    discoveries = pd.read_csv(input_path, keep_default_na=False)
    taxonomy = yaml.safe_load(Path(taxonomy_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(cnbc_config_path).read_text(encoding="utf-8"))
    settings = config.get("prefilter", {})
    filtered, queue, report = prefilter_dataframe(
        discoveries,
        taxonomy,
        additional_keywords=settings.get("additional_keywords", {}),
        use_url_path=settings.get("use_url_path", True),
    )
    report.update(
        {
            "input_path": str(Path(input_path)),
            "output_path": str(Path(output_path)),
            "queue_path": str(Path(queue_path)),
            "taxonomy_path": str(Path(taxonomy_path)),
        }
    )
    _atomic_text(Path(output_path), filtered.to_csv(index=False, lineterminator="\n"))
    _atomic_text(Path(queue_path), queue.to_csv(index=False, lineterminator="\n"))
    _atomic_text(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return filtered, queue, report

