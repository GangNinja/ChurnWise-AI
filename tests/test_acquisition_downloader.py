from __future__ import annotations

import hashlib

import pytest

from churnwise.acquisition.downloader import filename_from_url
from churnwise.acquisition.errors import CODE_INTEGRITY, CODE_INTERRUPTED, CODE_NETWORK, CODE_SIZE_LIMIT
from churnwise.net import TransportTimeout
from tests.acquisition_helpers import FakeDownloadTransport, FakeStream, make_downloader

URL = "https://www.openml.org/data/v1/download/1"
CSV = b"a,b,churn\n1,2,no\n3,4,yes\n"


def test_direct_download_writes_bytes_and_hashes_them(tmp_path):
    transport = FakeDownloadTransport({URL: FakeStream(body=CSV, headers={"Content-Type": "text/csv"})})
    destination = tmp_path / "artifact.part"

    receipt = make_downloader(transport).download(URL, destination)

    assert receipt.bytes == len(CSV)
    assert receipt.sha256 == hashlib.sha256(CSV).hexdigest()
    assert receipt.content_type == "text/csv"
    assert receipt.filename == "1"
    assert destination.read_bytes() == CSV
    assert transport.calls == [URL]


def test_redirect_is_followed_only_after_revalidation(tmp_path):
    target = "https://www.openml.org/data/v1/download/2"
    transport = FakeDownloadTransport(
        {
            URL: FakeStream(status=302, headers={"Location": target}),
            target: FakeStream(body=CSV),
        }
    )

    receipt = make_downloader(transport).download(URL, tmp_path / "artifact.part")

    assert receipt.final_url == target
    assert receipt.filename == "2"
    assert transport.calls == [URL, target]


def test_redirect_to_unlisted_host_is_rejected_and_leaves_no_file(tmp_path):
    transport = FakeDownloadTransport(
        {URL: FakeStream(status=302, headers={"Location": "https://evil.example/file.csv"})}
    )
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination)

    assert getattr(raised.value, "code", None) == "CW-ACQ-006"
    assert not destination.exists()


def test_transient_timeout_is_retried_then_succeeds(tmp_path):
    transport = FakeDownloadTransport(
        {URL: [TransportTimeout("temporary"), FakeStream(body=CSV)]}
    )
    sleeps: list[float] = []

    receipt = make_downloader(transport, sleeps=sleeps).download(URL, tmp_path / "artifact.part")

    assert receipt.bytes == len(CSV)
    assert sleeps and len(transport.calls) == 2


def test_retry_after_is_honoured_for_429(tmp_path):
    transport = FakeDownloadTransport(
        {URL: [FakeStream(status=429, headers={"Retry-After": "2"}), FakeStream(body=CSV)]}
    )
    sleeps: list[float] = []

    make_downloader(transport, sleeps=sleeps).download(URL, tmp_path / "artifact.part")

    assert sleeps == [2.0]


def test_missing_resource_is_a_structured_network_failure(tmp_path):
    transport = FakeDownloadTransport({URL: FakeStream(status=404)})
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination)

    assert getattr(raised.value, "code", None) == CODE_NETWORK
    assert getattr(raised.value, "http_status", None) == 404
    assert not destination.exists()


def test_declared_size_over_cap_is_rejected_before_writing(tmp_path):
    transport = FakeDownloadTransport(
        {URL: FakeStream(body=CSV, headers={"Content-Length": str(len(CSV))})}
    )
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport, max_bytes=5).download(URL, destination)

    assert getattr(raised.value, "code", None) == CODE_SIZE_LIMIT
    assert not destination.exists()


def test_stream_cap_without_declared_length_leaves_no_partial_file(tmp_path):
    transport = FakeDownloadTransport({URL: FakeStream(body=CSV)})
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport, max_bytes=5).download(URL, destination)

    assert getattr(raised.value, "code", None) == CODE_SIZE_LIMIT
    assert not destination.exists()


def test_expected_size_mismatch_is_an_integrity_failure(tmp_path):
    transport = FakeDownloadTransport(
        {URL: FakeStream(body=CSV, headers={"Content-Length": str(len(CSV))})}
    )
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination, expected_size=len(CSV) + 1)

    assert getattr(raised.value, "code", None) == CODE_INTEGRITY
    assert not destination.exists()


def test_incomplete_transfer_is_rejected(tmp_path):
    transport = FakeDownloadTransport(
        {URL: FakeStream(body=CSV, headers={"Content-Length": str(len(CSV) + 10)})}
    )
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination)

    assert getattr(raised.value, "code", None) == CODE_INTERRUPTED
    assert not destination.exists()


def test_unwritable_destination_is_a_storage_failure(tmp_path):
    transport = FakeDownloadTransport({URL: FakeStream(body=CSV)})
    destination = tmp_path / "missing-directory" / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination)

    assert getattr(raised.value, "code", None) == "CW-ACQ-008"
    assert not destination.exists()


def test_content_encoding_does_not_use_compressed_length_for_expected_size(tmp_path):
    transport = FakeDownloadTransport(
        {URL: FakeStream(
            body=CSV,
            headers={"Content-Length": "999", "Content-Encoding": "gzip"},
        )}
    )

    receipt = make_downloader(transport).download(
        URL, tmp_path / "artifact.part", expected_size=len(CSV)
    )

    assert receipt.bytes == len(CSV)


def test_midstream_timeout_is_retried_and_can_recover(tmp_path):
    transport = FakeDownloadTransport(
        {URL: [FakeStream(body=CSV, fail_after=5), FakeStream(body=CSV)]}
    )
    sleeps: list[float] = []

    receipt = make_downloader(transport, sleeps=sleeps).download(
        URL, tmp_path / "artifact.part"
    )

    assert receipt.bytes == len(CSV)
    assert len(transport.calls) == 2
    assert sleeps


def test_midstream_timeout_leaves_no_partial_file(tmp_path):
    transport = FakeDownloadTransport({URL: FakeStream(body=CSV, fail_after=5)})
    destination = tmp_path / "artifact.part"

    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(URL, destination)

    assert getattr(raised.value, "code", None) == CODE_INTERRUPTED
    assert not destination.exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://www.openml.org/data/v1/download/1",
        "https://user:pass@www.openml.org/data/v1/download/1",
        "https://www.openml.org:8443/data/v1/download/1",
        "https://www.openml.org:not-a-port/data/v1/download/1",
        "https://[::1/data/v1/download/1",
        "https://evil.example/data.csv",
    ],
)
def test_unsafe_urls_are_rejected_before_any_request(tmp_path, url):
    transport = FakeDownloadTransport({})
    with pytest.raises(Exception) as raised:
        make_downloader(transport).download(url, tmp_path / "artifact.part")
    assert getattr(raised.value, "code", None) == "CW-ACQ-006"
    assert transport.calls == []


def test_filename_from_url_is_safe_and_decoded():
    assert filename_from_url("https://www.openml.org/data/v1/download/22125226") == "22125226"
    assert filename_from_url("https://example.org/a%20b.csv") == "a b.csv"
