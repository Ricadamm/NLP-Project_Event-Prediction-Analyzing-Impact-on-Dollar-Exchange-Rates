"""Offline checks for query safety, temporal coverage and failure accounting."""

from datetime import datetime, timedelta, timezone
import json

import pytest
import requests

from src.acquisition.gdelt_client import (
    GdeltClient,
    GdeltRequestError,
    GdeltResponseError,
    build_query,
    format_timestamp,
    split_window,
)


START = datetime(2021, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(days=1)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now


class Response:
    def __init__(self, payload=None, status=200, text=None, headers=None):
        self.status_code = status
        self.payload = {"articles": []} if payload is None else payload
        self.text = json.dumps(self.payload) if text is None else text
        self.headers = {} if headers is None else headers

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, outcomes, clock):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.clock = clock

    def get(self, url, **kwargs):
        self.calls.append((self.clock.now, url, kwargs))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_for(outcomes, **settings):
    clock = Clock()
    session = Session(outcomes, clock)
    client = GdeltClient(settings, session=session, sleep=clock.sleep, monotonic=clock.monotonic)
    return client, session, clock


def test_query_builder_preserves_configured_order_case_and_quotes_phrases():
    assert build_query(["war", "invasion", "Federal Reserve", "f-16"], "reuters.com") == (
        '(war OR invasion OR "Federal Reserve" OR "f-16") domain:reuters.com sourcelang:english'
    )
    assert build_query(["war"], "EXAMPLE.ORG", "French") == (
        "(war) domain:example.org sourcelang:french"
    )


@pytest.mark.parametrize("terms", [[], "war", [None], [""], ["OR"], ["(war OR coup)"],
                                   ['war" OR coup'], ["domain:evil.com"], ["war\ncoup"], ["war*"]])
def test_query_builder_rejects_operator_injection_and_invalid_terms(terms):
    with pytest.raises(ValueError):
        build_query(terms, "reuters.com")


@pytest.mark.parametrize("domain,language", [
    ("https://reuters.com", "english"), ("reuters.com OR evil.com", "english"),
    ("-reuters.com", "english"), ("reuters.com", "english OR french"),
])
def test_query_builder_rejects_source_operator_injection(domain, language):
    with pytest.raises(ValueError):
        build_query(["war"], domain, language)


def test_timestamp_and_request_parameters_convert_to_utc_and_preserve_boundary_overlap():
    wib = timezone(timedelta(hours=7))
    start_wib = datetime(2021, 9, 1, 7, tzinfo=wib)
    client, session, _ = client_for([Response()])
    query = build_query(["war"], "reuters.com")
    assert format_timestamp(start_wib) == "20210901000000"
    assert client.request(query, start_wib, END) == []
    _, url, kwargs = session.calls[0]
    assert url == "https://api.gdeltproject.org/api/v2/doc/doc"
    assert kwargs["params"] == {
        "query": query, "mode": "artlist", "format": "json", "maxrecords": 250,
        "sort": "dateasc", "startdatetime": "20210831235959", "enddatetime": "20210902000001",
    }
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["headers"]["User-Agent"]
    assert kwargs["timeout"] == 60


@pytest.mark.parametrize("start,end", [
    (START.replace(tzinfo=None), END), (START, END.replace(tzinfo=None)),
    (START, START), (END, START), (START, START + timedelta(minutes=14)),
    (START.replace(microsecond=1), END),
])
def test_invalid_windows_fail_before_network(start, end):
    client, session, _ = client_for([])
    with pytest.raises(ValueError):
        client.request("war", start, end)
    assert not session.calls
    assert client.stats["request_count"] == 0


def test_daily_split_tree_reaches_15_minute_leaves_without_gaps_or_overlaps():
    pending = [(START, END)]
    leaves = []
    while pending:
        interval = pending.pop()
        children = split_window(*interval)
        if children is None:
            leaves.append(interval)
        else:
            pending.extend(children)
    leaves.sort()
    assert len(leaves) == 96
    assert leaves[0][0] == START
    assert leaves[-1][1] == END
    assert all(stop - start == timedelta(minutes=15) for start, stop in leaves)
    assert all(left[1] == right[0] for left, right in zip(leaves, leaves[1:]))


def test_split_handles_odd_multiple_of_minimum_and_non_aligned_remainder():
    assert split_window(START, START + timedelta(minutes=45)) == (
        (START, START + timedelta(minutes=15)),
        (START + timedelta(minutes=15), START + timedelta(minutes=45)),
    )
    assert split_window(START, START + timedelta(minutes=29)) is None
    with pytest.raises(ValueError):
        split_window(START, END, min_minutes=1)


def test_retry_backoff_429_5xx_transport_and_success_stats():
    article = {"url": "https://reuters.com/a", "title": "Original title", "extra": {"raw": True}}
    payload = {"articles": [article]}
    client, session, clock = client_for([
        Response(status=429), Response(status=503), requests.Timeout("timeout"),
        requests.ConnectionError("offline"), Response(payload),
    ])
    result = client.fetch_window("war", START, END)
    assert result is payload["articles"]
    assert result[0] is article
    assert clock.sleeps == [5, 10, 20, 40, 1]
    assert client.stats == {"http_429_count": 1, "retry_count": 4, "request_count": 5}
    assert len(session.calls) == 5


def test_server_minimum_idle_gap_applies_after_success_and_permanent_failure():
    client, session, clock = client_for([Response(), Response(status=400), Response()])
    assert client.request("war", START, END) == []
    with pytest.raises(GdeltRequestError, match="HTTP 400"):
        client.request("war", START, END)
    assert client.request("war", START, END) == []
    assert [call[0] for call in session.calls] == [0, 5, 10]
    assert clock.sleeps == [1, 4, 5, 1]


@pytest.mark.parametrize("first_outcome", [Response(), requests.Timeout("slow timeout")])
def test_slow_response_or_failure_cannot_consume_minimum_idle_gap(first_outcome):
    clock = Clock()

    class SlowSession(Session):
        def __init__(self):
            super().__init__([first_outcome, Response()], clock)
            self.durations = iter([20, 30])
            self.finished = []

        def get(self, url, **kwargs):
            try:
                return super().get(url, **kwargs)
            finally:
                clock.now += next(self.durations)
                self.finished.append(clock.now)

    session = SlowSession()
    client = GdeltClient({"max_retries": 0}, session=session,
                         sleep=clock.sleep, monotonic=clock.monotonic)
    if isinstance(first_outcome, Exception):
        with pytest.raises(GdeltRequestError, match="slow timeout"):
            client.request("war", START, END)
    else:
        client.request("war", START, END)
    client.request("war", START, END)
    assert [call[0] for call in session.calls] == [0, 25]
    assert session.finished == [20, 55]
    assert session.calls[1][0] - session.finished[0] == 5


def test_retry_after_is_honored_and_never_shortened_to_retry_cap():
    client, session, clock = client_for([
        Response(status=429, headers={"Retry-After": "17"}), Response(),
    ])
    client.request("war", START, END)
    assert clock.sleeps == [17, 1]
    client, session, clock = client_for([
        Response(status=429, headers={"Retry-After": "301"}),
    ])
    with pytest.raises(GdeltRequestError, match="resume this window later"):
        client.request("war", START, END)
    assert len(session.calls) == 1
    assert client.stats["retry_count"] == 0


@pytest.mark.parametrize("outcome", [Response(status=429), Response(status=500),
                                     requests.Timeout("late"), requests.ConnectionError("offline")])
def test_retries_are_bounded_and_exhaustion_is_not_empty_success(outcome):
    client, session, _ = client_for([outcome, outcome], max_retries=1)
    with pytest.raises(GdeltRequestError, match="exhausted 1 retries"):
        client.request("war", START, END)
    assert len(session.calls) == 2
    assert client.stats["retry_count"] == 1
    assert client.stats["request_count"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_fail_without_retries(status):
    client, session, _ = client_for([Response(status=status, text="Invalid query")])
    with pytest.raises(GdeltRequestError, match=f"HTTP {status}.*Invalid query"):
        client.request("war", START, END)
    assert len(session.calls) == 1
    assert client.stats["retry_count"] == 0


@pytest.mark.parametrize("payload", [{}, [], {"articles": None}, {"articles": {}},
                                     {"articles": [None]}, {"articles": ["bad"]}])
def test_invalid_json_schema_is_a_failure(payload):
    client, session, _ = client_for([Response(payload)])
    with pytest.raises(GdeltResponseError):
        client.request("war", START, END)
    assert len(session.calls) == 1
    assert client.stats["retry_count"] == 0


def test_explanatory_plaintext_is_reported_instead_of_returned_as_empty():
    client, session, _ = client_for([
        Response(ValueError("not JSON"), text="Your query was too short or too long"),
    ])
    with pytest.raises(GdeltResponseError, match="non-JSON.*Your query was too short"):
        client.request("war", START, END)
    assert len(session.calls) == 1


def test_zero_retry_budget_and_invalid_query_do_not_issue_extra_requests():
    client, session, _ = client_for([Response(status=429)], max_retries=0)
    with pytest.raises(ValueError):
        client.request(" ", START, END)
    with pytest.raises(GdeltRequestError, match="exhausted 0 retries"):
        client.request("war", START, END)
    assert client.stats == {"http_429_count": 1, "retry_count": 0, "request_count": 1}


def test_max_records_and_padding_are_configurable_for_bounded_probes():
    client, _, _ = client_for([Response()], max_records=1, boundary_padding_seconds=0)
    params = client.request_parameters("war", START, END)
    assert params["maxrecords"] == 1
    assert params["startdatetime"] == "20210901000000"
    assert params["enddatetime"] == "20210902000000"


@pytest.mark.parametrize("setting,value", [
    ("max_records", 251), ("max_records", 0), ("max_records", True),
    ("max_retries", -1), ("timeout_seconds", 0), ("timeout_seconds", float("nan")),
    ("minimum_request_interval_seconds", 1), ("minimum_window_minutes", 14),
    ("boundary_padding_seconds", -1), ("mode", "timelinevol"),
    ("format", "html"), ("sort", "datedesc"), ("base_url", "not-a-url"),
])
def test_invalid_configuration_fails_before_network(setting, value):
    with pytest.raises(ValueError):
        GdeltClient({setting: value})


def test_non_transient_requests_exception_is_not_retried():
    client, session, _ = client_for([requests.RequestException("invalid transport setting")])
    with pytest.raises(GdeltRequestError, match="invalid transport setting"):
        client.request("war", START, END)
    assert len(session.calls) == 1


def test_response_over_the_requested_cap_is_rejected():
    client, _, _ = client_for([Response({"articles": [{}, {}]})], max_records=1)
    with pytest.raises(GdeltResponseError, match="exceeded requested maxrecords"):
        client.request("war", START, END)


def test_logs_record_status_and_count_for_success_and_http_failure(caplog):
    caplog.set_level("INFO", logger="src.acquisition.gdelt_client")
    client, _, _ = client_for([Response(status=503), Response({"articles": [{"url": "https://reuters.com/a"}]})])
    client.request("war", START, END)
    assert "GDELT HTTP 503" in caplog.text
    assert "GDELT HTTP 200" in caplog.text
    assert "GDELT returned 1 articles" in caplog.text
