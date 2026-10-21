from __future__ import annotations

import pytest

from churnwise.net import (
    CODE_BUDGET,
    CODE_HTTP,
    CODE_PROTOCOL,
    CODE_RATE_LIMITED,
    CODE_TIMEOUT,
    CODE_TOO_LARGE,
    DiscoveryHttpError,
    HttpResponse,
    ResponseTooLargeError,
    TransportTimeout,
    TransportUnavailable,
    UnsafeUrlError,
)
from tests.helpers import json_response, make_client


def test_timeout_is_retried_then_succeeds_with_deterministic_backoff():
    routes = {
        "https://www.openml.org/x": [TransportTimeout("slow"), json_response({"ok": 1})]
    }
    client, transport, sleeps = make_client(routes, max_retries=2)
    response = client.get("https://www.openml.org/x")
    assert response.status == 200
    assert len(transport.calls) == 2
    assert sleeps == [0.5]


def test_exhausted_timeouts_raise_timeout_code():
    routes = {"https://www.openml.org/x": TransportTimeout("slow")}
    client, transport, _ = make_client(routes, max_retries=1)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_TIMEOUT
    assert len(transport.calls) == 2


def test_rate_limit_honours_retry_after_header():
    routes = {
        "https://www.openml.org/x": [
            json_response({"error": "slow down"}, status=429, headers={"Retry-After": "2"}),
            json_response({"ok": 1}),
        ]
    }
    client, _, sleeps = make_client(routes, max_retries=2)
    assert client.get("https://www.openml.org/x").status == 200
    assert sleeps == [2.0]


def test_persistent_rate_limit_is_reported_as_rate_limited():
    routes = {
        "https://www.openml.org/x": json_response({}, status=429, headers={"Retry-After": "0"})
    }
    client, _, _ = make_client(routes, max_retries=1)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_RATE_LIMITED


def test_client_error_is_not_retried():
    routes = {"https://www.openml.org/x": HttpResponse(status=404)}
    client, transport, sleeps = make_client(routes, max_retries=3)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_HTTP
    assert error.value.status == 404
    assert len(transport.calls) == 1
    assert sleeps == []


def test_unavailable_connection_is_retried_then_reported():
    routes = {"https://www.openml.org/x": TransportUnavailable("dns")}
    client, _, sleeps = make_client(routes, max_retries=2)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_TIMEOUT
    assert sleeps == [0.5, 1.0]


def test_response_size_cap_is_enforced():
    def oversized(url, timeout, max_bytes):
        raise ResponseTooLargeError("too large")

    client, _, _ = make_client({"https://www.openml.org/x": oversized})
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_TOO_LARGE


def test_request_budget_is_enforced():
    client, transport, _ = make_client(
        {"https://www.openml.org/x": json_response({"ok": 1})}, max_requests=1
    )
    client.get("https://www.openml.org/x")
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/x")
    assert error.value.code == CODE_BUDGET
    assert len(transport.calls) == 1


def test_redirect_is_followed_and_revalidated():
    routes = {
        "https://www.openml.org/a": HttpResponse(
            status=302, headers={"location": "https://www.openml.org/b"}
        ),
        "https://www.openml.org/b": json_response({"ok": 1}),
    }
    client, transport, _ = make_client(routes)
    assert client.get("https://www.openml.org/a").status == 200
    assert transport.calls == ["https://www.openml.org/a", "https://www.openml.org/b"]


def test_redirect_to_disallowed_host_is_rejected():
    routes = {
        "https://www.openml.org/a": HttpResponse(
            status=302, headers={"location": "https://evil.example/payload"}
        )
    }
    client, _, _ = make_client(routes)
    with pytest.raises(UnsafeUrlError):
        client.get("https://www.openml.org/a")


def test_redirect_loop_is_bounded():
    routes = {
        "https://www.openml.org/a": HttpResponse(
            status=302, headers={"location": "https://www.openml.org/a"}
        )
    }
    client, _, _ = make_client(routes)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get("https://www.openml.org/a")
    assert error.value.code == CODE_PROTOCOL


def test_unsafe_schemes_hosts_ports_and_credentials_are_rejected():
    client, _, _ = make_client({})
    for url in (
        "http://www.openml.org/x",
        "ftp://www.openml.org/x",
        "https://evil.example/x",
        "https://www.openml.org:8443/x",
        "https://user:pass@www.openml.org/x",
    ):
        with pytest.raises(UnsafeUrlError):
            client.get(url)


def test_invalid_json_is_a_protocol_error():
    routes = {"https://www.openml.org/x": HttpResponse(status=200, body=b"<html>not json")}
    client, _, _ = make_client(routes)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get_json("https://www.openml.org/x")
    assert error.value.code == CODE_PROTOCOL


def test_non_object_json_is_a_protocol_error():
    routes = {"https://www.openml.org/x": HttpResponse(status=200, body=b"[1, 2, 3]")}
    client, _, _ = make_client(routes)
    with pytest.raises(DiscoveryHttpError) as error:
        client.get_json("https://www.openml.org/x")
    assert error.value.code == CODE_PROTOCOL
