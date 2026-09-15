"""Bounded client for CNBC's server-rendered historical article site map.

The daily site-map page embeds the result of CNBC's own date search in
``window.__c_data``.  When that first page reports more than 250 results,
CNBC's SiteMap component retrieves the remaining pages through its GraphQL
``searchResults`` query.  This client mirrors that publisher-defined flow and
extracts only title/URL discovery records.  It deliberately does not crawl
article bodies, bypass access controls, or infer an intraday publication time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import math
import time
from urllib.parse import urlsplit

import requests


DEFAULT_BASE_URL = "https://www.cnbc.com/site-map/articles"
DEFAULT_GRAPHQL_URL = "https://webql-redesign.cnbcfm.com/graphql"
CNBC_PAGE_SIZE = 250
STATE_MARKER = "window.__c_data="
SEARCH_KEY_PREFIX = "search("
SEARCH_QUERY = """query searchResults($fromDate: String!, $toDate: String!, $page: Int!) {
  search(partner: "cnbc03", pageSize: 250, sortBy: updatedDate,
         dateType: updatedDate, fromDate: $fromDate, toDate: $toDate, page: $page) {
    pagination { totalCount }
    results { title url }
  }
}"""


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
    page_evidence: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class _EmbeddedArchivePage:
    records: list[dict]
    total_count: int
    request_metadata: dict | None


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


def _find_request_metadata(value, target: dict) -> list[dict]:
    """Find Apollo cache-key arguments associated with ``target`` by identity."""
    matches: list[dict] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if child is target and isinstance(key, str) and key.startswith(SEARCH_KEY_PREFIX):
                if key.endswith(")"):
                    try:
                        metadata = json.loads(key[len(SEARCH_KEY_PREFIX):-1])
                    except json.JSONDecodeError:
                        metadata = None
                    if isinstance(metadata, dict):
                        matches.append(metadata)
            matches.extend(_find_request_metadata(child, target))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_find_request_metadata(child, target))
    return matches


def _validate_search_payload(search: object) -> tuple[list[dict], int]:
    if not isinstance(search, dict):
        raise CnbcResponseError("CNBC archive search payload must be a JSON object")
    pagination = search.get("pagination")
    records = search.get("results")
    if not isinstance(pagination, dict) or "totalCount" not in pagination:
        raise CnbcResponseError("CNBC archive response has no pagination totalCount")
    total = pagination["totalCount"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise CnbcResponseError("CNBC archive totalCount must be a non-negative integer")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise CnbcResponseError("CNBC archive results must be JSON objects")
    return records, total


def _parse_embedded_archive_page(html: str) -> _EmbeddedArchivePage:
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
    records, total = _validate_search_payload(search)
    metadata_matches = _find_request_metadata(state, search)
    metadata = metadata_matches[0] if len(metadata_matches) == 1 else None
    return _EmbeddedArchivePage(records, total, metadata)


def parse_archive_html(html: str) -> tuple[list[dict], int]:
    """Extract title/URL rows and total count from a CNBC archive HTML page.

    JSON decoding starts at the assignment marker rather than using a greedy
    regular expression, so later script assignments cannot be consumed.
    """
    parsed = _parse_embedded_archive_page(html)
    records, total = parsed.records, parsed.total_count
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
        self.graphql_url = self.config.get("graphql_url", DEFAULT_GRAPHQL_URL)
        graphql = urlsplit(self.graphql_url) if isinstance(self.graphql_url, str) else None
        if (
            graphql is None
            or graphql.scheme not in {"http", "https"}
            or not graphql.hostname
            or graphql.query
            or graphql.fragment
        ):
            raise ValueError("archive.graphql_url must be an HTTP(S) URL without query or fragment")
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
        # ``max_results_per_day`` was the original single-page guard.  Retain
        # it as a backwards-compatible page-size ceiling rather than rejecting
        # valid days whose publisher-reported total spans multiple pages.
        page_size_ceiling = self.config.get("max_results_per_page")
        if page_size_ceiling is None:
            page_size_ceiling = self.config.get("max_results_per_day", CNBC_PAGE_SIZE)
        self.max_results_per_page = _integer(
            {"value": page_size_ceiling}, "value", CNBC_PAGE_SIZE, 1
        )
        self.max_pages_per_day = _integer(self.config, "max_pages_per_day", 100, 1)
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
        self.last_attempt_evidence: list[dict] = []
        self.last_partial_records: list[dict] = []
        self.last_reported_total_count: int | None = None

    def _wait_for_slot(self) -> None:
        if self._last_attempt_finished is None:
            return
        remaining = self.minimum_request_interval_seconds - (
            self._monotonic() - self._last_attempt_finished
        )
        if remaining > 0:
            self._sleep(remaining)

    def _request(
        self, method: str, url: str, *, page_number: int,
        evidence_context: dict | None = None, **kwargs,
    ):
        last_failure = "unknown failure"
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            self.stats["request_count"] += 1
            try:
                try:
                    response = getattr(self.session, method)(url, timeout=self.timeout_seconds, **kwargs)
                finally:
                    self._last_attempt_finished = self._monotonic()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_failure = f"CNBC transport failure for {url}: {exc}"
                self.last_attempt_evidence.append({
                    "page": page_number,
                    "request_url": url,
                    "request_method": method.upper(),
                    "request_attempt": attempt + 1,
                    "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "transport_error": str(exc),
                    **(evidence_context or {}),
                })
            except requests.RequestException as exc:
                raise CnbcRequestError(f"CNBC request failed for {url}: {exc}") from exc
            else:
                content = response.content
                evidence = {
                    "page": page_number,
                    "request_url": url,
                    "request_method": method.upper(),
                    "request_attempt": attempt + 1,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": response.status_code,
                    "response_sha256": hashlib.sha256(content).hexdigest(),
                    "response_bytes": len(content),
                    **(evidence_context or {}),
                }
                self.last_attempt_evidence.append(evidence)
                if response.status_code == 200:
                    return response, evidence
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

    @staticmethod
    def _deduplicate_records(records: list[dict]) -> list[dict]:
        unique: list[dict] = []
        seen_urls: set[str] = set()
        for record in records:
            url = record.get("url")
            if not isinstance(url, str) or not url.strip():
                raise CnbcResponseError(
                    "CNBC archive result has no stable non-empty URL for deduplication"
                )
            stable_url = url.strip()
            if stable_url not in seen_urls:
                seen_urls.add(stable_url)
                unique.append(record)
        return unique

    @staticmethod
    def _pagination_variables(day: date, metadata: dict | None) -> dict:
        if metadata is None:
            raise CnbcResponseError(
                "CNBC incomplete archive payload has no validated Apollo pagination metadata"
            )
        expected_date = day.strftime("%m-%d-%Y")
        expected = {
            "dateType": "updatedDate",
            "fromDate": expected_date,
            "page": 1,
            "pageSize": CNBC_PAGE_SIZE,
            "partner": "cnbc03",
            "sortBy": "updatedDate",
            "toDate": expected_date,
        }
        if metadata != expected:
            raise CnbcResponseError(
                "CNBC incomplete archive payload has unexpected pagination metadata: "
                f"{metadata!r}"
            )
        return {"fromDate": metadata["fromDate"], "toDate": metadata["toDate"]}

    @staticmethod
    def _parse_graphql_page(response, expected_total: int) -> tuple[list[dict], int]:
        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CnbcResponseError("CNBC pagination response is invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("errors"):
            raise CnbcResponseError("CNBC pagination response contains GraphQL errors")
        data = payload.get("data")
        records, total = _validate_search_payload(
            data.get("search") if isinstance(data, dict) else None
        )
        if total != expected_total:
            raise CnbcResponseError(
                f"CNBC archive totalCount changed during pagination: {expected_total} to {total}"
            )
        return records, total

    def fetch_day(self, day: date) -> ArchivePage:
        url = archive_url(day, self.base_url)
        self.last_attempt_evidence = []
        self.last_partial_records = []
        self.last_reported_total_count = None

        response, evidence = self._request(
            "get",
            url,
            page_number=1,
            evidence_context={"request_kind": "archive_html"},
            headers={"Accept": "text/html", "User-Agent": self.user_agent},
        )
        first_page = _parse_embedded_archive_page(response.text)
        total = first_page.total_count
        self.last_reported_total_count = total
        evidence.update({
            "result_count": len(first_page.records),
            "reported_total_count": total,
            "embedded_request_metadata": first_page.request_metadata,
        })
        records = list(first_page.records)
        self.last_partial_records = list(records)
        response_contents = [response.content]

        paginated = len(records) < total
        if paginated:
            variables = self._pagination_variables(day, first_page.request_metadata)
            if self.max_results_per_page < CNBC_PAGE_SIZE:
                raise CnbcResponseError(
                    f"CNBC pageSize={CNBC_PAGE_SIZE} exceeds configured "
                    f"max_results_per_page={self.max_results_per_page}"
                )
            if len(records) != CNBC_PAGE_SIZE:
                raise CnbcResponseError(
                    f"CNBC non-final page 1 returned {len(records)} rather than {CNBC_PAGE_SIZE} results"
                )
            page_count = math.ceil(total / CNBC_PAGE_SIZE)
            if page_count > self.max_pages_per_day:
                raise CnbcResponseError(
                    f"CNBC archive requires {page_count} pages, exceeding configured "
                    f"max_pages_per_day={self.max_pages_per_day}"
                )
            for page_number in range(2, page_count + 1):
                page_variables = {**variables, "page": page_number}
                page_response, page_evidence = self._request(
                    "post",
                    self.graphql_url,
                    page_number=page_number,
                    evidence_context={
                        "request_kind": "graphql",
                        "operation_name": "searchResults",
                        "request_variables": page_variables,
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "Referer": url,
                        "User-Agent": self.user_agent,
                    },
                    json={
                        "operationName": "searchResults",
                        "query": SEARCH_QUERY,
                        "variables": page_variables,
                    },
                )
                page_records, page_total = self._parse_graphql_page(page_response, total)
                page_evidence.update({
                    "result_count": len(page_records),
                    "reported_total_count": page_total,
                })
                records.extend(page_records)
                self.last_partial_records = list(records)
                response_contents.append(page_response.content)

        final_records = self._deduplicate_records(records) if paginated else records
        if len(final_records) != total:
            raise CnbcResponseError(
                "CNBC archive remains incomplete after pagination and URL deduplication: "
                f"received {len(records)} results, {len(final_records)} unique URLs, "
                f"expected {total}"
            )

        retrieved = datetime.now(timezone.utc).isoformat()
        enriched = []
        for raw in final_records:
            record = dict(raw)
            record.update(
                {
                    "discovered_for_date": day.isoformat(),
                    "archive_url": url,
                    "retrieved_at_utc": retrieved,
                }
            )
            enriched.append(record)
        digest = hashlib.sha256()
        for content in response_contents:
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        response_sha256 = (
            hashlib.sha256(response_contents[0]).hexdigest()
            if len(response_contents) == 1 else digest.hexdigest()
        )
        return ArchivePage(
            archive_date=day,
            archive_url=url,
            records=enriched,
            total_count=total,
            retrieved_at_utc=retrieved,
            response_sha256=response_sha256,
            response_bytes=sum(len(content) for content in response_contents),
            page_evidence=[dict(item) for item in self.last_attempt_evidence],
        )

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
