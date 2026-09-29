from datetime import date
import json

import pytest
import requests

from src.acquisition.cnbc_client import (
    CnbcClient, CnbcRequestError, CnbcResponseError, archive_url, parse_archive_html,
)


def html(records, total=None, *, day="09-01-2021", include_metadata=True):
    metadata = {
        "dateType": "updatedDate",
        "fromDate": day,
        "page": 1,
        "pageSize": 250,
        "partner": "cnbc03",
        "sortBy": "updatedDate",
        "toDate": day,
    }
    search_key = f"search({json.dumps(metadata, separators=(',', ':'))})" if include_metadata else "search({})"
    state = {
        "ROOT_QUERY": {
            search_key: {
                "pagination": {"totalCount": len(records) if total is None else total},
                "results": records,
            }
        }
    }
    return f"<html><script>window.__c_data={json.dumps(state)};window.next=1</script></html>"


def graphql_page(records, total):
    return json.dumps({
        "data": {"search": {"pagination": {"totalCount": total}, "results": records}}
    })


def records(start, stop):
    return [
        {"title": f"Story {index}", "url": f"https://www.cnbc.com/story-{index}.html"}
        for index in range(start, stop)
    ]


class Response:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text
        self.content = text.encode()


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def config(**updates):
    base = {
        "max_retries": 1,
        "initial_retry_delay_seconds": 0,
        "max_retry_delay_seconds": 0,
        "minimum_request_interval_seconds": 0,
        "max_results_per_day": 250,
    }
    base.update(updates)
    return base


def test_archive_url_and_embedded_payload_parsing():
    records = [{"title": "One", "url": "https://www.cnbc.com/2021/09/01/one.html"}]
    assert archive_url(date(2021, 9, 1)).endswith("/2021/September/1/")
    parsed, total = parse_archive_html(html(records))
    assert parsed == records
    assert total == 1


@pytest.mark.parametrize("body", ["", "<html></html>", "window.__c_data={bad"])
def test_invalid_archive_payload_fails(body):
    with pytest.raises(CnbcResponseError):
        parse_archive_html(body)


def test_truncated_archive_payload_fails():
    with pytest.raises(CnbcResponseError, match="incomplete"):
        parse_archive_html(html([{"title": "one"}], total=2))


def test_client_retries_transport_and_enriches_records():
    session = Session([
        requests.ConnectionError("offline"),
        Response(text=html([{"title": "One", "url": "https://www.cnbc.com/2021/09/01/one.html"}])),
    ])
    client = CnbcClient(config(), session=session, sleep=lambda _: None)
    page = client.fetch_day(date(2021, 9, 1))
    assert page.total_count == 1
    assert page.records[0]["discovered_for_date"] == "2021-09-01"
    assert client.stats == {"request_count": 2, "retry_count": 1}
    assert session.calls[0][2]["headers"]["Accept"] == "text/html"


def test_client_does_not_retry_permanent_http_error():
    client = CnbcClient(config(), session=Session([Response(404, "missing")]))
    with pytest.raises(CnbcRequestError, match="404"):
        client.fetch_day(date(2021, 9, 1))


@pytest.mark.parametrize("count", [249, 250])
def test_complete_day_uses_one_archive_page(count):
    session = Session([Response(text=html(records(0, count)))])
    page = CnbcClient(config(), session=session, sleep=lambda _: None).fetch_day(
        date(2021, 9, 1)
    )
    assert page.total_count == count
    assert len(page.records) == count
    assert [call[0] for call in session.calls] == ["GET"]


def test_incomplete_first_page_retrieves_remaining_graphql_page():
    session = Session([
        Response(text=html(records(0, 250), total=260)),
        Response(text=graphql_page(records(250, 260), total=260)),
    ])
    page = CnbcClient(config(), session=session, sleep=lambda _: None).fetch_day(
        date(2021, 9, 1)
    )
    assert page.total_count == 260
    assert len(page.records) == 260
    assert [call[0] for call in session.calls] == ["GET", "POST"]
    post = session.calls[1]
    assert post[1] == "https://webql-redesign.cnbcfm.com/graphql"
    assert post[2]["json"]["operationName"] == "searchResults"
    assert post[2]["json"]["variables"] == {
        "fromDate": "09-01-2021", "toDate": "09-01-2021", "page": 2,
    }
    assert [item["result_count"] for item in page.page_evidence] == [250, 10]


def test_multi_page_results_are_deduplicated_by_stable_url():
    duplicate_and_remainder = [records(0, 250)[-1], *records(250, 260)]
    session = Session([
        Response(text=html(records(0, 250), total=260)),
        Response(text=graphql_page(duplicate_and_remainder, total=260)),
    ])
    page = CnbcClient(config(), session=session, sleep=lambda _: None).fetch_day(
        date(2021, 9, 1)
    )
    assert len(page.records) == page.total_count == 260
    assert len({record["url"] for record in page.records}) == 260


def test_pagination_refuses_to_guess_without_embedded_request_metadata():
    session = Session([
        Response(text=html(records(0, 250), total=260, include_metadata=False)),
    ])
    client = CnbcClient(config(), session=session, sleep=lambda _: None)
    with pytest.raises(CnbcResponseError, match="pagination metadata"):
        client.fetch_day(date(2021, 9, 1))
    assert [call[0] for call in session.calls] == ["GET"]


def test_deduplicated_result_count_must_equal_reported_total():
    duplicate_and_only_nine_new = [records(0, 250)[-1], *records(250, 259)]
    session = Session([
        Response(text=html(records(0, 250), total=260)),
        Response(text=graphql_page(duplicate_and_only_nine_new, total=260)),
    ])
    client = CnbcClient(config(), session=session, sleep=lambda _: None)
    with pytest.raises(CnbcResponseError, match="259 unique URLs, expected 260"):
        client.fetch_day(date(2021, 9, 1))


def test_second_page_failure_exhausts_retries_and_never_returns_partial_day():
    session = Session([
        Response(text=html(records(0, 250), total=260)),
        Response(503, "temporarily unavailable"),
        Response(503, "still unavailable"),
    ])
    client = CnbcClient(config(), session=session, sleep=lambda _: None)
    with pytest.raises(CnbcRequestError, match="exhausted 1 retries"):
        client.fetch_day(date(2021, 9, 1))
    assert [call[0] for call in session.calls] == ["GET", "POST", "POST"]
    assert len(client.last_partial_records) == 250
    assert client.last_reported_total_count == 260
