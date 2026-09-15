"""Fetch and cache publisher metadata for prefiltered CNBC candidates.

Only URLs present in the supplied enrichment queue are visited. Successful HTTP
responses are reduced to structured metadata and stored as one JSON record per
article ID; raw HTML is not retained. Missing publisher timestamps stay missing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/interim/news/cnbc_enrichment_queue.csv"
DEFAULT_CACHE = ROOT / "data/raw/news/cnbc/article_metadata"
DEFAULT_OUTPUT = ROOT / "data/interim/news/cnbc_news_enriched.csv"
DEFAULT_REPORT = ROOT / "data/interim/news/cnbc_enrichment_report.json"
PARSER_VERSION = 1


class CnbcArticleError(RuntimeError):
    """A CNBC article request failed after the bounded policy."""

    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


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


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _types(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def normalize_publisher_timestamp(value) -> tuple[str, str, str] | None:
    """Return original, UTC ISO, and WIB ISO values for an aware timestamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    original = value.strip()
    try:
        parsed = pd.Timestamp(original)
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    utc = parsed.tz_convert("UTC").isoformat()
    wib = parsed.tz_convert("Asia/Jakarta").isoformat()
    return original, utc, wib


def _parse_embedded_assignments(html: str) -> list[dict]:
    states = []
    for marker in ("window.__c_data=", "window.__s_data="):
        start = html.find(marker)
        if start < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(html[start + len(marker):].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            states.append(value)
    return states


def _first_structured_value(states: list[dict], keys: tuple[str, ...]):
    wanted = {key.casefold() for key in keys}
    for state in states:
        for node in _walk_json(state):
            for key, value in node.items():
                if str(key).casefold() in wanted and value not in (None, "", []):
                    return value, str(key)
    return None, None


def _meta_values(soup: BeautifulSoup, names: tuple[str, ...]) -> list[tuple[str, str]]:
    wanted = {name.casefold() for name in names}
    values = []
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name") or tag.get("itemprop")
        content = tag.get("content")
        if isinstance(key, str) and key.casefold() in wanted and isinstance(content, str) and content.strip():
            values.append((content.strip(), key))
    return values


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, list):
        values = value
    else:
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()}, key=str.casefold)


def _author_names(value) -> list[str]:
    if not isinstance(value, list):
        value = [value]
    names = []
    for author in value:
        if isinstance(author, str):
            name = author
        elif isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = ""
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names), key=str.casefold)


def extract_article_metadata(html: str, url: str) -> dict:
    """Extract timestamps and publisher categories using a fixed hierarchy."""
    if not isinstance(html, str) or not html:
        raise ValueError("article HTML must be non-empty text")
    soup = BeautifulSoup(html, "lxml")
    jsonld_nodes = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            value = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        jsonld_nodes.extend(_walk_json(value))
    article_nodes = [
        node for node in jsonld_nodes
        if any(kind in {"NewsArticle", "Article"} for kind in _types(node.get("@type")))
    ]
    article_nodes.sort(
        key=lambda node: 0 if "NewsArticle" in _types(node.get("@type")) else 1
    )
    primary = article_nodes[0] if article_nodes else {}

    timestamp = None
    timestamp_source = None
    for node in article_nodes:
        normalized = normalize_publisher_timestamp(node.get("datePublished"))
        if normalized:
            timestamp = normalized
            timestamp_source = "jsonld.datePublished"
            primary = node
            break
    if timestamp is None:
        for value, key in _meta_values(
            soup,
            (
                "article:published_time", "og:article:published_time", "datePublished",
                "parsely-pub-date", "sailthru.date", "pubdate",
            ),
        ):
            normalized = normalize_publisher_timestamp(value)
            if normalized:
                timestamp = normalized
                timestamp_source = f"meta.{key}"
                break
    states = _parse_embedded_assignments(html)
    if timestamp is None:
        value, key = _first_structured_value(
            states, ("datePublished", "publishedDate", "publishDate", "published_at")
        )
        normalized = normalize_publisher_timestamp(value)
        if normalized:
            timestamp = normalized
            timestamp_source = f"cnbc_structured.{key}"
    if timestamp is None:
        for tag in soup.find_all("time", attrs={"datetime": True}):
            context = " ".join(tag.parent.get_text(" ", strip=True).split()).casefold()
            if "publish" not in context:
                continue
            normalized = normalize_publisher_timestamp(tag.get("datetime"))
            if normalized:
                timestamp = normalized
                timestamp_source = "rendered.time.datetime"
                break

    section = primary.get("articleSection") if isinstance(primary.get("articleSection"), str) else None
    section_source = "jsonld.articleSection" if section else None
    meta_sections = _meta_values(soup, ("article:section", "section"))
    subsection = meta_sections[0][0] if meta_sections else None
    if not section and subsection:
        section, section_source = subsection, f"meta.{meta_sections[0][1]}"
    if not section:
        value, key = _first_structured_value(states, ("articleSection", "sectionName", "section"))
        if isinstance(value, str) and value.strip():
            section, section_source = value.strip(), f"cnbc_structured.{key}"
    if not section:
        path_parts = [part for part in urlsplit(url).path.split("/") if part]
        if path_parts and not path_parts[0].isdigit() and path_parts[0] not in {"video"}:
            section, section_source = path_parts[0], "canonical_url_path_fallback"

    keywords = _string_list(primary.get("keywords"))
    if not keywords:
        meta_keywords = _meta_values(soup, ("news_keywords", "keywords"))
        keywords = _string_list(meta_keywords[0][0]) if meta_keywords else []
    categories = sorted(
        {item for item in [section, subsection, *keywords] if isinstance(item, str) and item.strip()},
        key=str.casefold,
    )
    headline = primary.get("headline") if isinstance(primary.get("headline"), str) else None
    if not headline:
        headline_meta = _meta_values(soup, ("og:title", "twitter:title"))
        headline = headline_meta[0][0] if headline_meta else None
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical = canonical_tag.get("href") if canonical_tag else None

    return {
        "headline": headline,
        "canonical_url": canonical,
        "published_at_original": timestamp[0] if timestamp else None,
        "published_at_utc": timestamp[1] if timestamp else None,
        "published_at_wib": timestamp[2] if timestamp else None,
        "timestamp_source": timestamp_source,
        "timestamp_status": "exact_publisher_timestamp" if timestamp else "missing_publisher_timestamp",
        "section": section.strip() if isinstance(section, str) else None,
        "subsection": subsection.strip() if isinstance(subsection, str) else None,
        "section_source": section_source,
        "article_type": next(iter(_types(primary.get("@type"))), None),
        "keywords": keywords,
        "publisher_categories": categories,
        "authors": _author_names(primary.get("author")),
        "metadata_source": timestamp_source.split(".", 1)[0] if timestamp_source else None,
    }


def _number(config: Mapping, key: str, default: float, minimum: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    return value


class CnbcArticleClient:
    """Polite per-worker article client with bounded retry behavior."""

    def __init__(self, config: Mapping, session=None, sleep=time.sleep, monotonic=time.monotonic):
        self.config = dict(config)
        self.timeout_seconds = _number(config, "timeout_seconds", 30, 0.001)
        retries = config.get("max_retries", 3)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.max_retries = retries
        self.initial_retry_delay_seconds = _number(config, "initial_retry_delay_seconds", 2, 0)
        self.max_retry_delay_seconds = _number(config, "max_retry_delay_seconds", 30, 0)
        self.request_delay_seconds = _number(config, "request_delay_seconds", 2, 0)
        self.user_agent = config.get("user_agent", "NLP-Project-CNBC-POC/1.0")
        self.session = session or requests.Session()
        self._owns_session = session is None
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_finished = None
        self.stats = {"request_count": 0, "retry_count": 0}

    def fetch(self, url: str) -> dict:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in {"cnbc.com", "www.cnbc.com"}:
            raise CnbcArticleError(f"Refusing non-CNBC URL: {url}")
        last_failure = "unknown request failure"
        for attempt in range(self.max_retries + 1):
            if self._last_finished is not None:
                remaining = self.request_delay_seconds - (self._monotonic() - self._last_finished)
                if remaining > 0:
                    self._sleep(remaining)
            self.stats["request_count"] += 1
            response = None
            try:
                try:
                    response = self.session.get(
                        url,
                        timeout=self.timeout_seconds,
                        headers={"Accept": "text/html", "User-Agent": self.user_agent},
                    )
                finally:
                    self._last_finished = self._monotonic()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_failure = f"transport failure: {exc}"
            except requests.RequestException as exc:
                raise CnbcArticleError(f"request failed: {exc}") from exc
            else:
                if response.status_code == 200:
                    result = extract_article_metadata(response.text, url)
                    result.update({"http_status": 200, "response_bytes": len(response.content)})
                    return result
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    last_failure = f"HTTP {response.status_code}"
                else:
                    raise CnbcArticleError(f"HTTP {response.status_code}", response.status_code)
            if attempt >= self.max_retries:
                status = response.status_code if response is not None else None
                raise CnbcArticleError(
                    f"{last_failure}; exhausted {self.max_retries} retries", status
                )
            self.stats["retry_count"] += 1
            delay = min(self.initial_retry_delay_seconds * (2**attempt), self.max_retry_delay_seconds)
            if delay:
                self._sleep(delay)
        raise AssertionError("unreachable retry state")

    def close(self):
        if self._owns_session:
            self.session.close()


def _comparison(source_dates: list[str], published_original: str | None) -> str:
    if not source_dates or not published_original:
        return "missing_comparison_source"
    normalized = normalize_publisher_timestamp(published_original)
    if not normalized:
        return "missing_comparison_source"
    published_date = pd.Timestamp(normalized[0]).date().isoformat()
    return "exact_match" if published_date in source_dates else "different_date"


def _discovery_dates(value) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
    else:
        decoded = value
    if isinstance(decoded, str):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return []
    valid = []
    for item in decoded:
        try:
            valid.append(date.fromisoformat(str(item)).isoformat())
        except ValueError:
            continue
    return sorted(set(valid))


def _completed_cache(cache_path: Path, url: str) -> dict | None:
    """Return a compatible completed cache record, ignoring invalid files."""
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict):
        return None
    if (
        cached.get("status") == "completed"
        and cached.get("url") == url
        and cached.get("parser_version") == PARSER_VERSION
    ):
        return cached
    return None


def _fetch_and_cache(
    article_id: str,
    url: str,
    cache_path: Path,
    client,
) -> dict:
    """Fetch one article and atomically persist its success or failure record."""
    try:
        extracted = client.fetch(url)
        metadata = {
            "status": "completed",
            "article_id": article_id,
            "url": url,
            "parser_version": PARSER_VERSION,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            **extracted,
        }
    except CnbcArticleError as exc:
        metadata = {
            "status": "failed",
            "article_id": article_id,
            "url": url,
            "parser_version": PARSER_VERSION,
            "http_status": exc.http_status,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
    except Exception as exc:
        # A malformed response or an unexpected client error is isolated to the
        # article so the rest of a production queue can continue.
        metadata = {
            "status": "failed",
            "article_id": article_id,
            "url": url,
            "parser_version": PARSER_VERSION,
            "http_status": None,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    _atomic_json(cache_path, metadata)
    return metadata


def enrich_dataframe(
    pilot: pd.DataFrame,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    client=None,
    client_factory=None,
    config: Mapping | None = None,
    enrichment_workers: int = 1,
    force: bool = False,
    cache_only: bool = False,
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict]:
    required = {"article_id", "normalized_url", "publication_date", "discovered_for_date"}
    if not required.issubset(pilot.columns):
        raise ValueError(f"Pilot input is missing columns: {sorted(required - set(pilot.columns))}")
    if pilot["article_id"].map(str).duplicated().any():
        raise ValueError("Pilot article_id values must be unique")
    if (
        isinstance(enrichment_workers, bool)
        or not isinstance(enrichment_workers, int)
        or enrichment_workers < 1
    ):
        raise ValueError("enrichment_workers must be a positive integer")
    if client is not None and client_factory is not None:
        raise ValueError("Supply either client or client_factory, not both")
    if client is not None and enrichment_workers != 1:
        raise ValueError(
            "A shared client is only supported with enrichment_workers=1; "
            "use client_factory for parallel enrichment"
        )
    cache_dir = Path(cache_dir)
    sources = pilot.to_dict(orient="records")
    metadata_by_index: dict[int, dict] = {}
    pending: list[tuple[int, str, str, Path]] = []
    cache_hits = live_successes = failed_fetch_attempts = 0
    completed = 0

    def report_progress() -> None:
        if progress_every and (
            completed % progress_every == 0 or completed == len(sources)
        ):
            print(f"CNBC candidate enrichment: {completed}/{len(sources)}", flush=True)

    # Cache preflight deliberately happens before executor submission. Completed
    # compatible records never enter the worker queue.
    for index, source in enumerate(sources):
        article_id = str(source["article_id"])
        url = str(source["normalized_url"])
        cache_path = cache_dir / f"{article_id}.json"
        # A valid completed record is immutable resume evidence, including when
        # --force is present. Failed, missing, or incompatible records are still
        # eligible for a new attempt.
        metadata = _completed_cache(cache_path, url)
        if metadata is None:
            pending.append((index, article_id, url, cache_path))
        else:
            metadata_by_index[index] = metadata
            cache_hits += 1
            completed += 1
            report_progress()

    clients = []
    clients_lock = threading.Lock()
    own_clients = client is None

    def make_client():
        created = client_factory() if client_factory is not None else CnbcArticleClient(config or {})
        with clients_lock:
            clients.append(created)
        return created

    if cache_only:
        for index, _article_id, url, _cache_path in pending:
            metadata_by_index[index] = {
                "status": "failed", "url": url, "error": "cache_miss"
            }
            failed_fetch_attempts += 1
            completed += 1
            report_progress()
    elif enrichment_workers == 1:
        worker = client if client is not None else make_client()
        if client is not None:
            clients.append(client)
        try:
            for index, article_id, url, cache_path in pending:
                metadata = _fetch_and_cache(article_id, url, cache_path, worker)
                metadata_by_index[index] = metadata
                if metadata["status"] == "completed":
                    live_successes += 1
                else:
                    failed_fetch_attempts += 1
                completed += 1
                report_progress()
        finally:
            if own_clients:
                close = getattr(worker, "close", None)
                if close is not None:
                    close()
    else:
        thread_state = threading.local()

        def initialize_worker() -> None:
            thread_state.client = make_client()

        def run_pending(item: tuple[int, str, str, Path]) -> tuple[int, dict]:
            index, article_id, url, cache_path = item
            metadata = _fetch_and_cache(
                article_id, url, cache_path, thread_state.client
            )
            return index, metadata

        executor = ThreadPoolExecutor(
            max_workers=enrichment_workers,
            thread_name_prefix="cnbc-enrichment",
            initializer=initialize_worker,
        )
        pending_iterator = iter(pending)
        futures = {}
        try:
            # Keep only N futures submitted at a time. This bounds both HTTP
            # concurrency and the executor queue, making interruption/resume
            # prompt even for a multi-thousand-article production queue.
            for _ in range(enrichment_workers):
                item = next(pending_iterator, None)
                if item is None:
                    break
                futures[executor.submit(run_pending, item)] = item[0]
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    index, metadata = future.result()
                    metadata_by_index[index] = metadata
                    if metadata["status"] == "completed":
                        live_successes += 1
                    else:
                        failed_fetch_attempts += 1
                    completed += 1
                    report_progress()
                    item = next(pending_iterator, None)
                    if item is not None:
                        futures[executor.submit(run_pending, item)] = item[0]
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        finally:
            if own_clients:
                for worker in clients:
                    close = getattr(worker, "close", None)
                    if close is not None:
                        close()

    rows = []
    for index, source in enumerate(sources):
        metadata = metadata_by_index[index]
        row = dict(source)
        discoveries = _discovery_dates(source.get("discovered_for_date"))
        canonical_date = str(source.get("publication_date", ""))
        raw_timestamp_status = metadata.get("timestamp_status", "fetch_failed")
        timestamp_status = (
            "exact_publisher_timestamp"
            if raw_timestamp_status == "exact_publisher_timestamp"
            else "unresolved"
        )
        row.update(
            {
                "sitemap_date": discoveries[0] if len(discoveries) == 1 else "",
                "sitemap_dates": _json(discoveries),
                "canonical_url_date": canonical_date,
                "published_at_original": metadata.get("published_at_original"),
                "published_at_utc": metadata.get("published_at_utc"),
                "published_at_wib": metadata.get("published_at_wib"),
                "timestamp_source": metadata.get("timestamp_source"),
                "timestamp_status": timestamp_status,
                "timestamp_resolution_detail": raw_timestamp_status,
                "sitemap_vs_published_date_match": _comparison(discoveries, metadata.get("published_at_original")),
                "url_date_vs_published_date_match": _comparison([canonical_date] if canonical_date else [], metadata.get("published_at_original")),
                "section": metadata.get("section"),
                "subsection": metadata.get("subsection"),
                "section_source": metadata.get("section_source"),
                "article_type": metadata.get("article_type"),
                "keywords": _json(metadata.get("keywords", [])),
                "publisher_keywords": _json(metadata.get("keywords", [])),
                "publisher_categories": _json(metadata.get("publisher_categories", [])),
                "authors": _json(metadata.get("authors", [])),
                "article_metadata_status": metadata.get("status"),
                "article_metadata_error": metadata.get("error"),
            }
        )
        row["timestamp_semantics"] = (
            "exact_publisher_timestamp" if metadata.get("timestamp_status") == "exact_publisher_timestamp"
            else "missing_publisher_timestamp"
        )
        rows.append(row)

    if rows:
        enriched = pd.DataFrame(rows)
    else:
        enriched = pilot.copy()
        for column in (
            "sitemap_date", "sitemap_dates", "canonical_url_date",
            "published_at_original", "published_at_utc", "published_at_wib",
            "timestamp_source", "timestamp_status", "timestamp_resolution_detail",
            "sitemap_vs_published_date_match",
            "url_date_vs_published_date_match", "section", "subsection", "section_source",
            "article_type", "keywords", "publisher_keywords", "publisher_categories",
            "authors", "article_metadata_status", "article_metadata_error",
            "timestamp_semantics",
        ):
            enriched[column] = pd.Series(dtype="object")
    timestamp_counts = Counter(enriched["timestamp_source"].fillna("(missing)"))
    section_counts = Counter(enriched["section"].fillna("(missing)"))
    section_source_counts = Counter(enriched["section_source"].fillna("(missing)"))
    sitemap_comparison = Counter(enriched["sitemap_vs_published_date_match"])
    url_comparison = Counter(enriched["url_date_vs_published_date_match"])
    comparison_values = ("exact_match", "different_date", "missing_comparison_source")
    exact_count = int((enriched["timestamp_status"] == "exact_publisher_timestamp").sum())
    section_count = int(enriched["section"].notna().sum())
    successful_records = int((enriched["article_metadata_status"] == "completed").sum())
    failed_records = len(enriched) - successful_records
    stats = {
        "request_count": sum(getattr(worker, "stats", {}).get("request_count", 0) for worker in clients),
        "retry_count": sum(getattr(worker, "stats", {}).get("retry_count", 0) for worker in clients),
    }
    report = {
        "total_candidate_articles": len(pilot),
        # Backward-compatible report key retained for the validated pilot.
        "total_pilot_articles": len(pilot),
        "successful_http_fetches": successful_records,
        "successful_http_fetches_this_run": live_successes,
        "failed_fetches": failed_records,
        "failed_fetch_attempts_this_run": failed_fetch_attempts,
        "cache_hits": cache_hits,
        "enrichment_workers": enrichment_workers,
        "http_request_count_this_run": stats.get("request_count", 0),
        "http_retry_count_this_run": stats.get("retry_count", 0),
        "exact_publisher_timestamps": exact_count,
        "missing_timestamps": len(pilot) - exact_count,
        "timestamp_source_counts": dict(sorted(timestamp_counts.items())),
        "section_extraction_success": section_count,
        "missing_sections": len(pilot) - section_count,
        "section_source_counts": dict(sorted(section_source_counts.items())),
        "section_counts": dict(sorted(section_counts.items(), key=lambda item: (-item[1], item[0]))),
        "sitemap_vs_published_date_counts": {
            key: sitemap_comparison[key] for key in comparison_values
        },
        "url_date_vs_published_date_counts": {
            key: url_comparison[key] for key in comparison_values
        },
        "date_comparison_basis": "Calendar date in the publisher timestamp's original timezone offset",
        "cache_only": cache_only,
        "force_enrichment_requested": force,
        "timestamp_hierarchy": [
            "jsonld.datePublished", "structured publication meta tag",
            "CNBC structured page metadata", "rendered publication time",
        ],
        "parser_version": PARSER_VERSION,
        "counts_reconcile": exact_count + (len(pilot) - exact_count) == len(pilot),
    }
    return enriched, report


def enrich_file(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    *,
    config: Mapping | None = None,
    cache_dir: str | Path = DEFAULT_CACHE,
    force: bool = False,
    cache_only: bool = False,
    enrichment_workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    pilot = pd.read_csv(input_path, keep_default_na=False)
    enriched, report = enrich_dataframe(
        pilot, cache_dir=cache_dir, config=config, force=force,
        cache_only=cache_only, enrichment_workers=enrichment_workers,
    )
    report.update({"input_path": str(Path(input_path)), "output_path": str(Path(output_path)), "cache_directory": str(Path(cache_dir))})
    _atomic_text(Path(output_path), enriched.to_csv(index=False, lineterminator="\n"))
    _atomic_json(Path(report_path), report)
    return enriched, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=ROOT / "config/cnbc.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--enrichment-workers", type=int, default=1)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    enrich_file(args.input, args.output, args.report, config=config["article_metadata"],
                cache_dir=args.cache_dir, force=args.force, cache_only=args.cache_only,
                enrichment_workers=args.enrichment_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
