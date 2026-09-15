"""Small, serial GDELT DOC 2.0 client with bounded retries.

Windows are timezone-aware, second-resolution logical intervals [start, end).
The HTTP bounds are padded by one second by default because GDELT documents
them as "after" and "before" without a formal endpoint-inclusivity contract.
The raw returned records are preserved; the collector records both bounds and
the cleaning stage reports any records outside the logical study interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import logging
import math
import re
import time
from urllib.parse import urlsplit

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltError(RuntimeError):
    """A request failed; its window must remain eligible for resume."""


class GdeltRequestError(GdeltError):
    """An HTTP or transport error prevented retrieval."""


class GdeltResponseError(GdeltError):
    """GDELT returned an invalid or explanatory response instead of articles."""


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("GDELT timestamps must be timezone-aware datetime objects")
    if value.microsecond:
        raise ValueError("GDELT timestamps must have whole-second precision")
    return value.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Convert an aware timestamp to GDELT's UTC YYYYMMDDHHMMSS format."""
    return _utc(value).strftime("%Y%m%d%H%M%S")


def _window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("GDELT window end must be later than start")
    return start, end


def build_query(
    keywords: Sequence[str], source_domain: str, source_language: str = "english"
) -> str:
    """Quote phrases and retain the configured keyword order without expansion.

    Only literal keyword/phrase values are accepted. GDELT operators, quotes,
    brackets, and standalone Boolean words belong in code, not taxonomy values.
    """
    if isinstance(keywords, (str, bytes)) or not isinstance(keywords, Sequence) or not keywords:
        raise ValueError("keywords must be a non-empty sequence of literal strings")
    terms = []
    for keyword in keywords:
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Each keyword must be a non-empty string")
        if any(ord(character) < 32 for character in keyword):
            raise ValueError("Keywords must not contain control characters")
        term = " ".join(keyword.split())
        if term.upper() in {"OR", "AND", "NOT"} or any(
            not (character.isalnum() or character in " -'.") for character in term
        ):
            raise ValueError(f"Unsupported syntax in literal keyword: {keyword!r}")
        # GDELT also requires hyphenated words to be quoted.
        terms.append(f'"{term}"' if any(character in term for character in " -'.") else term)
    if not isinstance(source_domain, str) or not re.fullmatch(
        r"(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
        r"[A-Za-z](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
        source_domain,
    ):
        raise ValueError("source_domain must be a bare domain, without a URL or operators")
    if not isinstance(source_language, str) or not re.fullmatch(r"[A-Za-z]{2,30}", source_language):
        raise ValueError("source_language must be a language name or code without operators")
    return (
        f"({' OR '.join(terms)}) domain:{source_domain.lower()} "
        f"sourcelang:{source_language.lower()}"
    )


def split_window(
    start: datetime, end: datetime, min_minutes: int = 15
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]] | None:
    """Split without gaps, keeping each child at least the minimum length.

    The midpoint is rounded down to a whole minimum-window unit relative to
    start. Thus a daily tree reaches 15-minute leaves (45 splits to 15 + 30).
    Non-aligned input may retain a final remainder longer than the minimum.
    """
    start, end = _window(start, end)
    if isinstance(min_minutes, bool) or not isinstance(min_minutes, int) or min_minutes < 15:
        raise ValueError("min_minutes must be an integer of at least 15")
    unit = timedelta(minutes=min_minutes)
    units = (end - start) // unit
    if units < 2:
        return None
    midpoint = start + (units // 2) * unit
    return ((start, midpoint), (midpoint, end))


def _number(config: Mapping, key: str, default: float, minimum: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    return float(value)


def _integer(config: Mapping, key: str, default: int, minimum: int, maximum: int | None = None) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


class GdeltClient:
    """Serial requests with an idle gap after every completed HTTP attempt.

    max_retries counts retries after the initial attempt. The minimum request
    interval is measured from response/transport completion to the next start,
    so a slow response cannot consume the pause required by the service.
    """

    def __init__(self, api_config: Mapping, session=None, sleep=time.sleep, monotonic=time.monotonic):
        if not isinstance(api_config, Mapping):
            raise ValueError("api_config must be a mapping containing the API settings")
        self.config = dict(api_config)
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL)
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTP(S) URL")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) endpoint without query parameters")
        for key, required in (("mode", "artlist"), ("format", "json"), ("sort", "dateasc")):
            if self.config.get(key, required) != required:
                raise ValueError(f"Task 1 requires api.{key}: {required}")
        self.max_records = _integer(self.config, "max_records", 250, 1, 250)
        self.max_retries = _integer(self.config, "max_retries", 5, 0)
        self.minimum_window_minutes = _integer(self.config, "minimum_window_minutes", 15, 15)
        self.boundary_padding_seconds = _integer(self.config, "boundary_padding_seconds", 1, 0)
        self.timeout_seconds = _number(self.config, "timeout_seconds", 60, 0.001)
        self.initial_retry_delay_seconds = _number(self.config, "initial_retry_delay_seconds", 5, 0.001)
        self.max_retry_delay_seconds = _number(self.config, "max_retry_delay_seconds", 300, 0.001)
        self.request_delay_seconds = _number(self.config, "request_delay_seconds", 1, 0)
        self.minimum_request_interval_seconds = _number(
            self.config, "minimum_request_interval_seconds", 5, 5
        )
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_attempt_finished = None
        self.stats = {"http_429_count": 0, "retry_count": 0, "request_count": 0}

    def request_parameters(self, query: str, start: datetime, end: datetime) -> dict:
        """Return actual padded HTTP parameters; retain logical bounds separately."""
        start, end = _window(start, end)
        if end - start < timedelta(minutes=self.minimum_window_minutes):
            raise ValueError("GDELT logical request window is shorter than minimum_window_minutes")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("GDELT query must be a non-empty string")
        padding = timedelta(seconds=self.boundary_padding_seconds)
        return {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": self.max_records,
            "startdatetime": format_timestamp(start - padding),
            "enddatetime": format_timestamp(end + padding),
            "sort": "dateasc",
        }

    @staticmethod
    def _excerpt(response) -> str:
        return " ".join(str(getattr(response, "text", "")).split())[:400]

    def _wait_for_slot(self):
        if self._last_attempt_finished is not None:
            remaining = self.minimum_request_interval_seconds - (
                self._monotonic() - self._last_attempt_finished
            )
            if remaining > 0:
                self._sleep(remaining)

    def _retry_delay(self, attempt: int, response=None) -> float:
        # Cap arithmetic before exponentiation so a mistaken retry count cannot
        # construct an unbounded delay. Server Retry-After is never shortened.
        delay = min(self.initial_retry_delay_seconds, self.max_retry_delay_seconds)
        for _ in range(attempt):
            delay = min(delay * 2, self.max_retry_delay_seconds)
            if delay >= self.max_retry_delay_seconds:
                break
        retry_after = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
        if retry_after:
            try:
                requested = float(retry_after)
            except (TypeError, ValueError):
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=timezone.utc)
                    requested = (retry_date - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    requested = 0
            if math.isfinite(requested) and requested > self.max_retry_delay_seconds:
                raise GdeltRequestError(
                    f"GDELT Retry-After ({requested:g}s) exceeds max_retry_delay_seconds "
                    f"({self.max_retry_delay_seconds:g}s); resume this window later"
                )
            if math.isfinite(requested):
                delay = max(delay, requested)
        return delay

    def request(self, query: str, start: datetime, end: datetime) -> list[dict]:
        """Fetch once logically, retrying transport/429/5xx failures as configured.

        A valid {"articles": []} is a successful empty result. Missing articles,
        non-JSON explanatory errors, and invalid entries are failures, never
        converted to empty successes. Saturation is evaluated by the collector.
        """
        params = self.request_parameters(query, start, end)
        context = f"{format_timestamp(start)}..{format_timestamp(end)} query={query!r}"
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            self.stats["request_count"] += 1
            response = None
            try:
                try:
                    response = self.session.get(
                        self.base_url,
                        params=params,
                        timeout=self.timeout_seconds,
                        headers={"Accept": "application/json", "User-Agent": "NLP-Project-GDELT/1.0"},
                    )
                finally:
                    # requests.get reads the response body before returning.
                    # Capture completion even when a transport error is raised.
                    self._last_attempt_finished = self._monotonic()
            except (requests.Timeout, requests.ConnectionError) as exc:
                failure = f"GDELT transport failure for {context}: {exc}"
                LOGGER.warning("%s; attempt %s", failure, attempt + 1)
            except requests.RequestException as exc:
                LOGGER.error("GDELT request error for %s; attempt %s: %s", context, attempt + 1, exc)
                raise GdeltRequestError(f"GDELT request failed for {context}: {exc}") from exc
            else:
                status = response.status_code
                LOGGER.info("GDELT HTTP %s for %s; attempt %s", status, context, attempt + 1)
                if status == 429:
                    self.stats["http_429_count"] += 1
                if status == 429 or 500 <= status <= 599:
                    failure = f"GDELT HTTP {status} for {context}: {self._excerpt(response)}"
                elif status != 200:
                    raise GdeltRequestError(
                        f"GDELT HTTP {status} for {context}: {self._excerpt(response)}"
                    )
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise GdeltResponseError(
                            f"GDELT returned non-JSON for {context}: {self._excerpt(response)}"
                        ) from exc
                    if not isinstance(payload, dict) or "articles" not in payload:
                        raise GdeltResponseError(f"GDELT JSON has no articles array for {context}")
                    articles = payload["articles"]
                    if not isinstance(articles, list) or any(not isinstance(row, dict) for row in articles):
                        raise GdeltResponseError(f"GDELT articles must be an array of objects for {context}")
                    if len(articles) > self.max_records:
                        raise GdeltResponseError(f"GDELT exceeded requested maxrecords for {context}")
                    LOGGER.info("GDELT returned %s articles for %s", len(articles), context)
                    if self.request_delay_seconds:
                        self._sleep(self.request_delay_seconds)
                    return articles
            if attempt >= self.max_retries:
                raise GdeltRequestError(f"{failure}; exhausted {self.max_retries} retries")
            try:
                delay = self._retry_delay(attempt, response)
            except GdeltRequestError as exc:
                raise GdeltRequestError(f"{exc}; {context}") from exc
            self.stats["retry_count"] += 1
            LOGGER.warning("%s; retry %s/%s in %.1fs", failure, attempt + 1, self.max_retries, delay)
            self._sleep(delay)
        raise AssertionError("Unreachable retry state")

    fetch_window = request

    def close(self):
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
