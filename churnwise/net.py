"""Bounded HTTPS client used by every discovery adapter.

Security posture (see docs/DISCOVERY.md):

* HTTPS only, explicit host allowlist, no embedded credentials, port 443 only.
* Redirects are followed manually (max 3) and every hop is re-validated.
* Response bodies are capped before they are buffered.
* Requests per client are capped, so a source cannot spend unbounded time.
* Timeouts and bounded exponential backoff; no retry on 4xx.

The transport is injectable so tests never touch the network.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlsplit

import requests

CODE_TIMEOUT = "CW-SRC-001"
CODE_HTTP = "CW-SRC-002"
CODE_PROTOCOL = "CW-SRC-003"
CODE_RATE_LIMITED = "CW-SRC-004"
CODE_BUDGET = "CW-SRC-005"
CODE_TOO_LARGE = "CW-SRC-006"
CODE_UNSAFE_URL = "CW-SRC-007"

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_REDIRECTS = 3


class DiscoveryHttpError(Exception):
    """A source-level network failure with a stable error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        url: str | None = None,
        body: bytes = b"",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status
        self.url = url
        self.body = body


class UnsafeUrlError(ValueError):
    """Raised when a URL violates the allowlist or scheme policy."""


class ResponseTooLargeError(Exception):
    pass


class TransportTimeout(Exception):
    pass


class TransportUnavailable(Exception):
    pass


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    url: str = ""

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


Transport = Callable[[str, float, int], HttpResponse]


class RequestsTransport:
    """Production transport. Performs one HTTP GET without following redirects."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    def __call__(self, url: str, timeout: float, max_bytes: int) -> HttpResponse:
        try:
            with requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json, text/plain;q=0.8",
                },
                allow_redirects=False,
                stream=True,
            ) as response:
                total = 0
                chunks: list[bytes] = []
                for chunk in response.iter_content(chunk_size=65_536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ResponseTooLargeError(
                            f"response exceeded {max_bytes} bytes"
                        )
                    chunks.append(chunk)
                return HttpResponse(
                    status=response.status_code,
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=b"".join(chunks),
                    url=url,
                )
        except requests.Timeout as exc:
            raise TransportTimeout(f"request timed out after {timeout}s") from exc
        except requests.ConnectionError as exc:
            raise TransportUnavailable("connection failed") from exc
        except requests.RequestException as exc:  # pragma: no cover - defensive
            raise TransportUnavailable(str(exc)) from exc


class HttpClient:
    """Allowlisted client with retries and a request budget."""

    def __init__(
        self,
        allowed_hosts: frozenset[str] | set[str] | tuple[str, ...],
        *,
        timeout_s: float = 12.0,
        max_retries: int = 2,
        max_bytes: int = 2_000_000,
        max_requests: int = 30,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self._transport: Transport = transport or RequestsTransport("ChurnWiseAI/0.3")
        self._sleep = sleep
        self._request_count = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    # -- public API ---------------------------------------------------------
    def get(self, url: str, *, timeout: float | None = None) -> HttpResponse:
        target = self._validate(url)
        redirects = 0
        while True:
            response = self._get_with_retries(target, timeout or self.timeout_s)
            if response.status in REDIRECT_STATUSES:
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise DiscoveryHttpError(
                        CODE_PROTOCOL,
                        f"too many redirects (>{MAX_REDIRECTS})",
                        status=response.status,
                        url=target,
                    )
                location = response.header("location")
                if not location:
                    raise DiscoveryHttpError(
                        CODE_PROTOCOL,
                        "redirect response without a Location header",
                        status=response.status,
                        url=target,
                    )
                target = self._validate(urljoin(target, location))
                continue
            return response

    def get_json(self, url: str, *, timeout: float | None = None) -> dict[str, Any]:
        response = self.get(url, timeout=timeout)
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiscoveryHttpError(
                CODE_PROTOCOL,
                "response was not valid UTF-8 JSON",
                status=response.status,
                url=url,
            ) from exc
        if not isinstance(payload, dict):
            raise DiscoveryHttpError(
                CODE_PROTOCOL,
                "expected a JSON object at the response root",
                status=response.status,
                url=url,
            )
        return payload

    # -- internals ----------------------------------------------------------
    def _get_with_retries(self, url: str, timeout: float) -> HttpResponse:
        last_error: DiscoveryHttpError | None = None
        for attempt in range(self.max_retries + 1):
            self._consume_budget(url)
            try:
                response = self._transport(url, timeout, self.max_bytes)
            except TransportTimeout:
                last_error = DiscoveryHttpError(
                    CODE_TIMEOUT, f"timed out after {timeout}s", retryable=True, url=url
                )
            except TransportUnavailable as exc:
                last_error = DiscoveryHttpError(
                    CODE_TIMEOUT,
                    f"connection failed: {exc}",
                    retryable=True,
                    url=url,
                )
            except ResponseTooLargeError as exc:
                raise DiscoveryHttpError(CODE_TOO_LARGE, str(exc), url=url) from exc
            else:
                if response.status in RETRYABLE_STATUSES and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response)
                    self._sleep(delay)
                    continue
                if response.status == 429:
                    raise DiscoveryHttpError(
                        CODE_RATE_LIMITED,
                        "source rate limit reached (HTTP 429)",
                        retryable=False,
                        status=429,
                        url=url,
                        body=response.body,
                    )
                if 400 <= response.status < 500:
                    raise DiscoveryHttpError(
                        CODE_HTTP,
                        f"source returned HTTP {response.status}",
                        status=response.status,
                        url=url,
                        body=response.body,
                    )
                if response.status >= 500:
                    raise DiscoveryHttpError(
                        CODE_HTTP,
                        f"source returned HTTP {response.status}",
                        retryable=True,
                        status=response.status,
                        url=url,
                        body=response.body,
                    )
                return response
            if attempt < self.max_retries:
                self._sleep(self._retry_delay(attempt, None))
                continue
        assert last_error is not None
        raise last_error

    def _retry_delay(self, attempt: int, response: HttpResponse | None) -> float:
        if response is not None:
            raw = response.header("retry-after").strip()
            if raw:
                try:
                    return min(max(float(raw), 0.0), 10.0)
                except ValueError:
                    pass
        return min(0.5 * (2**attempt), 5.0)

    def _consume_budget(self, url: str) -> None:
        if self._request_count >= self.max_requests:
            raise DiscoveryHttpError(
                CODE_BUDGET,
                f"request budget of {self.max_requests} exhausted",
                url=url,
            )
        self._request_count += 1

    def _validate(self, url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            raise UnsafeUrlError("URL must be a non-empty string")
        parts = urlsplit(url.strip())
        if parts.scheme != "https":
            raise UnsafeUrlError(f"only https URLs are allowed, got {parts.scheme or 'none'}")
        if parts.username or parts.password:
            raise UnsafeUrlError("URLs with embedded credentials are not allowed")
        if parts.port not in (None, 443):
            raise UnsafeUrlError(f"non-default port is not allowed: {parts.port}")
        host = (parts.hostname or "").lower()
        if host not in self.allowed_hosts:
            raise UnsafeUrlError(f"host {host!r} is not allowlisted")
        return url.strip()
