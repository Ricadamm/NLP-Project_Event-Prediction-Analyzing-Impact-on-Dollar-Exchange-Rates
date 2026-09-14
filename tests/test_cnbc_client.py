from datetime import date
import json

import pytest
import requests

from src.acquisition.cnbc_client import (
    CnbcClient, CnbcRequestError, CnbcResponseError, archive_url, parse_archive_html,
)


def html(records, total=None):
    state = {
        "ROOT_QUERY": {
            "search({})": {
                "pagination": {"totalCount": len(records) if total is None else total},
                "results": records,
            }
        }
    }
    return f"<html><script>window.__c_data={json.dumps(state)};window.next=1</script></html>"


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
        self.calls.append((url, kwargs))
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
    assert session.calls[0][1]["headers"]["Accept"] == "text/html"


def test_client_does_not_retry_permanent_http_error():
    client = CnbcClient(config(), session=Session([Response(404, "missing")]))
    with pytest.raises(CnbcRequestError, match="404"):
        client.fetch_day(date(2021, 9, 1))

