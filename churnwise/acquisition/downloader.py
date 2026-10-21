"""Streamed, allowlisted HTTPS acquisition.

The downloader writes to a caller-supplied temporary path, hashes while
streaming, and never buffers the whole artifact in memory. A failed or
interrupted transfer leaves no file behind for promotion.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Protocol
from urllib.parse import unquote, urljoin, urlsplit

import requests

from churnwise.acquisition.errors import (
    CATEGORY_INTEGRITY,
    CATEGORY_INTERRUPTED,
    CATEGORY_NETWORK,
    CATEGORY_PERMISSION,
    CATEGORY_STORAGE,
    CODE_ACCESS,
    CODE_INTEGRITY,
    CODE_INTERRUPTED,
    CODE_NETWORK,
    CODE_SIZE_LIMIT,
    CODE_STORAGE,
    AcquisitionError,
    failure,
)
from churnwise.net import TransportTimeout, TransportUnavailable

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_REDIRECTS = 3
CHUNK_SIZE = 64 * 1024


class DownloadStream(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def iter_chunks(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class DownloadTransport(Protocol):
    def open(self, url: str, timeout: float) -> DownloadStream: ...


class RequestsDownloadTransport:
    """Production transport; one GET without automatic redirects."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    def open(self, url: str, timeout: float) -> DownloadStream:
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": self._user_agent, "Accept": "*/*"},
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout as exc:
            raise TransportTimeout(f"timed out after {timeout}s") from exc
        except requests.ConnectionError as exc:
            raise TransportUnavailable("connection failed") from exc
        except requests.RequestException as exc:  # pragma: no cover - defensive
            raise TransportUnavailable(str(exc)) from exc

        return _RequestsStream(response)


class _RequestsStream:
    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self.status = response.status_code
        self.headers = {k.lower(): v for k, v in response.headers.items()}
        self.url = response.url or ""

    def iter_chunks(self) -> Iterator[bytes]:
        try:
            yield from self._response.iter_content(chunk_size=CHUNK_SIZE)
        except requests.Timeout as exc:
            raise TransportTimeout("timed out while streaming the response") from exc
        except requests.RequestException as exc:
            raise TransportUnavailable(f"connection failed while streaming: {exc}") from exc

    def close(self) -> None:
        self._response.close()


@dataclass(frozen=True)
class DownloadReceipt:
    sha256: str
    bytes: int
    content_type: str | None
    final_url: str
    filename: str
    request_count: int


def filename_from_url(url: str) -> str:
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1]).strip()
    return name or "dataset.bin"


class StreamingDownloader:
    def __init__(
        self,
        allowed_hosts: frozenset[str] | set[str] | tuple[str, ...],
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_bytes: int,
        transport: DownloadTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self._transport = transport or RequestsDownloadTransport("ChurnWiseAI/0.3 (dataset acquisition)")
        self._sleep = sleep
        self._request_count = 0

    def download(
        self,
        url: str,
        destination: os.PathLike[str] | str,
        *,
        expected_size: int | None = None,
    ) -> DownloadReceipt:
        """Download with bounded retries, including a mid-stream restart."""
        for attempt in range(self.max_retries + 1):
            try:
                return self._download_once(url, destination, expected_size=expected_size)
            except AcquisitionError as exc:
                if exc.code == CODE_INTERRUPTED and attempt < self.max_retries:
                    self._sleep(self._retry_delay(attempt, None))
                    continue
                raise
        raise AssertionError("unreachable")

    def _download_once(
        self,
        url: str,
        destination: os.PathLike[str] | str,
        *,
        expected_size: int | None = None,
    ) -> DownloadReceipt:
        destination = os.fspath(destination)
        target = self._validate_url(url)
        redirects = 0
        receipt: DownloadReceipt | None = None

        try:
            while True:
                response = self._open_with_retries(target)
                try:
                    if response.status in REDIRECT_STATUSES:
                        redirects += 1
                        if redirects > MAX_REDIRECTS:
                            raise failure(
                                CODE_ACCESS,
                                f"too many redirects (>{MAX_REDIRECTS})",
                                category=CATEGORY_PERMISSION,
                            )
                        location = _header(response.headers, "location")
                        if not location:
                            raise failure(
                                CODE_NETWORK,
                                "redirect response had no Location header",
                                category=CATEGORY_NETWORK,
                            )
                        target = self._validate_url(urljoin(target, location))
                        continue

                    self._raise_for_status(response.status)
                    content_length = _content_length(response.headers)
                    transfer_encoded = _header(response.headers, "content-encoding").lower() not in {
                        "",
                        "identity",
                    }
                    if content_length is not None and content_length > self.max_bytes:
                        raise failure(
                            CODE_SIZE_LIMIT,
                            f"declared size {content_length} exceeds cap {self.max_bytes}",
                            category=CATEGORY_STORAGE,
                        )
                    if (
                        expected_size is not None
                        and content_length is not None
                        and not transfer_encoded
                        and content_length != expected_size
                    ):
                        raise failure(
                            CODE_INTEGRITY,
                            f"declared size {content_length} does not match expected {expected_size}",
                            category=CATEGORY_INTEGRITY,
                        )

                    digest = hashlib.sha256()
                    written = 0
                    with open(destination, "wb") as handle:
                        try:
                            for chunk in response.iter_chunks():
                                if not chunk:
                                    continue
                                written += len(chunk)
                                if written > self.max_bytes:
                                    raise failure(
                                        CODE_SIZE_LIMIT,
                                        f"download exceeded cap of {self.max_bytes} bytes",
                                        category=CATEGORY_STORAGE,
                                    )
                                digest.update(chunk)
                                handle.write(chunk)
                        except (TransportTimeout, TransportUnavailable) as exc:
                            raise failure(
                                CODE_INTERRUPTED,
                                f"transfer was interrupted after {written} bytes: {exc}",
                                category=CATEGORY_INTERRUPTED,
                                retryable=True,
                            ) from exc
                        handle.flush()
                        os.fsync(handle.fileno())

                    if (
                        content_length is not None
                        and not transfer_encoded
                        and written != content_length
                    ):
                        raise failure(
                            CODE_INTERRUPTED,
                            f"transfer ended after {written} of {content_length} bytes",
                            category=CATEGORY_INTERRUPTED,
                            retryable=True,
                        )
                    if expected_size is not None and written != expected_size:
                        raise failure(
                            CODE_INTEGRITY,
                            f"downloaded {written} bytes but expected {expected_size}",
                            category=CATEGORY_INTEGRITY,
                        )

                    receipt = DownloadReceipt(
                        sha256=digest.hexdigest(),
                        bytes=written,
                        content_type=_header(response.headers, "content-type") or None,
                        final_url=target,
                        filename=filename_from_url(target),
                        request_count=self._request_count,
                    )
                    return receipt
                finally:
                    response.close()
        except OSError as exc:
            _unlink(destination)
            if getattr(exc, "errno", None) == 28:  # ENOSPC
                raise failure(
                    CODE_STORAGE,
                    f"insufficient disk space while writing the artifact: {exc}",
                    category=CATEGORY_STORAGE,
                ) from exc
            raise failure(
                CODE_STORAGE,
                f"could not write the artifact: {exc}",
                category=CATEGORY_STORAGE,
            ) from exc
        except AcquisitionError:
            _unlink(destination)
            raise
        except BaseException:
            _unlink(destination)
            raise

    # -- internals ----------------------------------------------------------
    def _open_with_retries(self, url: str) -> DownloadStream:
        last: AcquisitionError | None = None
        for attempt in range(self.max_retries + 1):
            self._request_count += 1
            try:
                response = self._transport.open(url, self.timeout_s)
            except TransportTimeout:
                last = failure(
                    CODE_NETWORK,
                    f"timed out after {self.timeout_s}s",
                    category=CATEGORY_NETWORK,
                    retryable=True,
                )
            except TransportUnavailable as exc:
                last = failure(
                    CODE_NETWORK,
                    f"connection failed: {exc}",
                    category=CATEGORY_NETWORK,
                    retryable=True,
                )
            else:
                if response.status in RETRYABLE_STATUSES and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response.headers)
                    response.close()
                    self._sleep(delay)
                    continue
                return response
            if attempt < self.max_retries:
                self._sleep(self._retry_delay(attempt, None))
                continue
        assert last is not None
        raise last

    def _retry_delay(self, attempt: int, headers: Mapping[str, str] | None) -> float:
        if headers:
            raw = _header(headers, "retry-after").strip()
            if raw:
                try:
                    return min(max(float(raw), 0.0), 10.0)
                except ValueError:
                    pass
        return min(0.5 * (2**attempt), 5.0)

    def _validate_url(self, url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            raise failure(CODE_ACCESS, "download URL is empty", category=CATEGORY_PERMISSION)
        try:
            parts = urlsplit(url.strip())
        except ValueError as exc:
            raise failure(
                CODE_ACCESS,
                "download URL is malformed",
                category=CATEGORY_PERMISSION,
            ) from exc
        if parts.scheme != "https":
            raise failure(
                CODE_ACCESS,
                f"only https URLs are allowed, got {parts.scheme or 'none'}",
                category=CATEGORY_PERMISSION,
            )
        if parts.username or parts.password:
            raise failure(
                CODE_ACCESS,
                "URLs with embedded credentials are not allowed",
                category=CATEGORY_PERMISSION,
            )
        try:
            port = parts.port
        except ValueError as exc:
            raise failure(
                CODE_ACCESS,
                "download URL has an invalid port",
                category=CATEGORY_PERMISSION,
            ) from exc
        if port not in (None, 443):
            raise failure(
                CODE_ACCESS,
                f"non-default port is not allowed: {port}",
                category=CATEGORY_PERMISSION,
            )
        if (parts.hostname or "").lower() not in self.allowed_hosts:
            raise failure(
                CODE_ACCESS,
                f"host {parts.hostname!r} is not allowlisted for this source",
                category=CATEGORY_PERMISSION,
            )
        return url.strip()

    def _raise_for_status(self, status: int) -> None:
        if 200 <= status < 300:
            return
        if status in (401, 403):
            raise failure(
                CODE_ACCESS,
                f"source denied access (HTTP {status})",
                category=CATEGORY_PERMISSION,
                http_status=status,
            )
        if status == 404:
            raise failure(
                CODE_NETWORK,
                "resource not found (HTTP 404)",
                category=CATEGORY_NETWORK,
                http_status=status,
            )
        if status == 429:
            raise failure(
                CODE_NETWORK,
                "source rate limit reached (HTTP 429)",
                category=CATEGORY_NETWORK,
                retryable=True,
                http_status=status,
            )
        raise failure(
            CODE_NETWORK,
            f"source returned HTTP {status}",
            category=CATEGORY_NETWORK,
            retryable=status >= 500,
            http_status=status,
        )


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return str(value or "")
    return ""


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw = _header(headers, "content-length").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass
