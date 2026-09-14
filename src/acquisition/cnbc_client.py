"""Bounded client for CNBC's server-rendered historical article site map.

The daily site-map page embeds the result of CNBC's own date search in
``window.__c_data``.  This client retrieves one page per day and extracts only
the embedded title/URL discovery records.  It deliberately does not crawl
article bodies, bypass access controls, or infer an intraday publication time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import time
from urllib.parse import urlsplit

import requests


DEFAULT_BASE_URL = "https://www.cnbc.com/site-map/articles"
STATE_MARKER = "window.__c_data="


class CnbcError(RuntimeError):
    """Base class for a CNBC acquisition failure."""


class CnbcRequestError(CnbcError):
    """A transport or HTTP failure prevented archive retrieval."""


class CnbcResponseError(CnbcError):
    """The archive response did not contain a valid complete result payload."""


@dataclass(frozen=True)
class ArchivePage:
    """Validated result of one CNBC daily archive request."""

    archive_date: date
    archive_url: str
    records: list[dict]
    total_count: int
    retrieved_at_utc: str
    response_sha256: str
    response_bytes: int


def archive_url(day: date, base_url: str = DEFAULT_BASE_URL) -> str:
    """Return CNBC's date-specific article site-map URL."""
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError("day must be a datetime.date")
    return f"{base_url.rstrip('/')}/{day.year}/{day.strftime('%B')}/{day.day}/"


def _find_search_payload(value) -> list[dict]:
    matches: list[dict] = []
    if isinstance(value, dict):
        pagination = value.get("pagination")
        results = value.get("results")
        if isinstance(pagination, dict) and isinstance(results, list) and "totalCount" in pagination:
            matches.append(value)
        for child in value.values():
            matches.extend(_find_search_payload(child))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_find_search_payload(child))
    return matches


def parse_archive_html(html: str) -> tuple[list[dict], int]:
    """Extract title/URL rows and total count from a CNBC archive HTML page.

    JSON decoding starts at the assignment marker rather than using a greedy
    regular expression, so later script assignments cannot be consumed.
    """
    if not isinstance(html, str) or not html:
        raise CnbcResponseError("CNBC archive response is empty")
    marker_index = html.find(STATE_MARKER)
    if marker_index < 0:
        raise CnbcResponseError("CNBC archive response has no window.__c_data payload")
    payload_start = marker_index + len(STATE_MARKER)
    try:
        state, _ = json.JSONDecoder().raw_decode(html[payload_start:].lstrip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise CnbcResponseError("CNBC embedded archive state is invalid JSON") from exc
    matches = _find_search_payload(state)
    if len(matches) != 1:
        raise CnbcResponseError(
            f"Expected one CNBC archive search payload, found {len(matches)}"
        )
    search = matches[0]
    total = search["pagination"]["totalCount"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise CnbcResponseError("CNBC archive totalCount must be a non-negative integer")
    records = search["results"]
    if any(not isinstance(record, dict) for record in records):
        raise CnbcResponseError("CNBC archive results must be JSON objects")
    if len(records) != total:
        raise CnbcResponseError(
            f"CNBC archive page is incomplete: received {len(records)} of {total} results"
        )
    return records, total


def _number(config: Mapping, key: str, default: float, minimum: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    return value


def _integer(config: Mapping, key: str, default: int, minimum: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


class CnbcClient:
    """Serial CNBC archive client with bounded retries and request spacing."""

    def __init__(self, config: Mapping, session=None, sleep=time.sleep, monotonic=time.monotonic):
        if not isinstance(config, Mapping):
            raise ValueError("CNBC archive config must be a mapping")
        self.config = dict(config)
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL)
        parsed = urlsplit(self.base_url) if isinstance(self.base_url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("archive.base_url must be an HTTP(S) URL without query or fragment")
        self.timeout_seconds = _number(self.config, "timeout_seconds", 60, 0.001)
        self.max_retries = _integer(self.config, "max_retries", 3, 0)
        self.initial_retry_delay_seconds = _number(
            self.config, "initial_retry_delay_seconds", 2, 0
        )
        self.max_retry_delay_seconds = _number(
            self.config, "max_retry_delay_seconds", 30, 0
        )
        self.minimum_request_interval_seconds = _number(
            self.config, "minimum_request_interval_seconds", 1, 0
        )
        self.max_results_per_day = _integer(
            self.config, "max_results_per_day", 250, 1
        )
        self.user_agent = self.config.get(
            "user_agent", "NLP-Project-CNBC-POC/1.0 (historical research)"
        )
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("archive.user_agent must be a non-empty string")
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_attempt_finished: float | None = None
        self.stats = {"request_count": 0, "retry_count": 0}

    def _wait_for_slot(self) -> None:
        if self._last_attempt_finished is None:
            return
        remaining = self.minimum_request_interval_seconds - (
            self._monotonic() - self._last_attempt_finished
        )
        if remaining > 0:
            self._sleep(remaining)

    def fetch_day(self, day: date) -> ArchivePage:
        url = archive_url(day, self.base_url)
        last_failure = "unknown failure"
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            self.stats["request_count"] += 1
            try:
                try:
                    response = self.session.get(
                        url,
                        timeout=self.timeout_seconds,
                        headers={"Accept": "text/html", "User-Agent": self.user_agent},
                    )
                finally:
                    self._last_attempt_finished = self._monotonic()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_failure = f"CNBC transport failure for {url}: {exc}"
            except requests.RequestException as exc:
                raise CnbcRequestError(f"CNBC request failed for {url}: {exc}") from exc
            else:
                if response.status_code == 200:
                    records, total = parse_archive_html(response.text)
                    if total > self.max_results_per_day:
                        raise CnbcResponseError(
                            f"CNBC archive has {total} results, exceeding configured "
                            f"max_results_per_day={self.max_results_per_day}"
                        )
                    retrieved = datetime.now(timezone.utc).isoformat()
                    enriched = []
                    for raw in records:
                        record = dict(raw)
                        record.update(
                            {
                                "discovered_for_date": day.isoformat(),
                                "archive_url": url,
                                "retrieved_at_utc": retrieved,
                            }
                        )
                        enriched.append(record)
                    content = response.content
                    return ArchivePage(
                        archive_date=day,
                        archive_url=url,
                        records=enriched,
                        total_count=total,
                        retrieved_at_utc=retrieved,
                        response_sha256=hashlib.sha256(content).hexdigest(),
                        response_bytes=len(content),
                    )
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    excerpt = " ".join(response.text.split())[:300]
                    last_failure = f"CNBC HTTP {response.status_code} for {url}: {excerpt}"
                else:
                    raise CnbcRequestError(f"CNBC HTTP {response.status_code} for {url}")
            if attempt >= self.max_retries:
                raise CnbcRequestError(
                    f"{last_failure}; exhausted {self.max_retries} retries"
                )
            self.stats["retry_count"] += 1
            delay = min(
                self.initial_retry_delay_seconds * (2**attempt),
                self.max_retry_delay_seconds,
            )
            if delay:
                self._sleep(delay)
        raise AssertionError("unreachable CNBC retry state")

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

