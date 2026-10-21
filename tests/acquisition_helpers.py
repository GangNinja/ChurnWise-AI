"""Shared, network-free helpers for Phase 3 tests."""

from __future__ import annotations

import dataclasses
import io
import zipfile
from pathlib import Path
from typing import Any

from churnwise.acquisition.contracts import (
    DatasetManifest,
    ManifestAcquisition,
    ManifestFile,
    ManifestIntegrity,
    ManifestSource,
)
from churnwise.acquisition.downloader import StreamingDownloader
from churnwise.acquisition.provenance import candidate_fingerprint
from churnwise.config import Settings
from churnwise.net import TransportTimeout
from tests.helpers import make_candidate


@dataclasses.dataclass
class FakeStream:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    url: str = ""
    fail_after: int | None = None
    closed: bool = False

    def iter_chunks(self):
        if self.fail_after is None:
            yield self.body
            return
        if self.fail_after > 0:
            yield self.body[: self.fail_after]
        raise TransportTimeout("simulated mid-stream timeout")

    def close(self) -> None:
        self.closed = True


class FakeDownloadTransport:
    """Routes URLs to fresh FakeStream objects and records every request."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self._counts: dict[str, int] = {}

    def open(self, url: str, timeout: float) -> FakeStream:
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected URL requested in test: {url}")
        value = self.routes[url]
        if isinstance(value, list):
            index = self._counts.get(url, 0)
            self._counts[url] = index + 1
            value = value[min(index, len(value) - 1)]
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value()
        if isinstance(value, FakeStream):
            return dataclasses.replace(value)
        return FakeStream(body=value)


def make_downloader(
    transport: FakeDownloadTransport,
    *,
    hosts: tuple[str, ...] = ("www.openml.org",),
    max_bytes: int = 10_000_000,
    max_retries: int = 2,
    sleeps: list[float] | None = None,
) -> StreamingDownloader:
    return StreamingDownloader(
        hosts,
        timeout_s=5.0,
        max_retries=max_retries,
        max_bytes=max_bytes,
        transport=transport,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def make_acquisition_candidate(**overrides: Any):
    defaults: dict[str, Any] = {
        "candidate_id": "openml:46915",
        "source_id": "openml",
        "source_name": "OpenML",
        "source_dataset_id": "46915",
        "title": "churn",
        "landing_url": "https://www.openml.org/d/46915",
        "resource_url": "https://www.openml.org/data/v1/download/22125226",
        "format": "csv",
        "size_bytes": None,
        "license_name": "MIT License",
        "license_status": "verified",
        "target_column": "CustomerChurned",
    }
    defaults.update(overrides)
    return make_candidate(**defaults)


def make_acquisition_settings(
    tmp_path: Path,
    *,
    offline: bool = False,
    allow_unverified_license: bool = False,
    max_download_bytes: int = 10_000_000,
    max_extracted_bytes: int = 10_000_000,
    max_archive_members: int = 20,
) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        report_dir=tmp_path / "reports",
        offline=offline,
        allow_unverified_license=allow_unverified_license,
        max_download_bytes=max_download_bytes,
        max_extracted_bytes=max_extracted_bytes,
        max_archive_members=max_archive_members,
        download_timeout_s=30.0,
    )


def csv_bytes(rows: int = 3) -> bytes:
    lines = ["customer_id,tenure,monthly_charge,contract,churn"]
    for index in range(rows):
        lines.append(f"{index},{index + 1},{20.0 + index},monthly,{'yes' if index % 2 else 'no'}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def arff_bytes() -> bytes:
    return b"""@relation churn
@attribute tenure numeric
@attribute contract {monthly,yearly}
@attribute churn {no,yes}
@data
1,monthly,no
2,yearly,yes
"""


def parquet_bytes() -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "tenure": [1, 2, 3],
            "contract": ["monthly", "yearly", "monthly"],
            "churn": ["no", "yes", "no"],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def zip_bytes(
    name: str = "churn.csv",
    content: bytes | None = None,
    *,
    extra: list[tuple[str, bytes, int]] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content if content is not None else csv_bytes())
        for member_name, member_content, external_attr in extra or []:
            info = zipfile.ZipInfo(member_name)
            info.external_attr = external_attr
            archive.writestr(info, member_content)
    return buffer.getvalue()


def make_manifest(
    candidate,
    *,
    sha256: str,
    bytes_: int,
    cache_path: str,
    raw_path: str,
    extracted_path: str | None = None,
    extracted_sha256: str | None = None,
    extracted_files: list[str] | None = None,
    archive_members: list[dict[str, Any]] | None = None,
    retrieved_at: str = "2026-10-20T00:00:00Z",
    run_id: str = "20261020T000000Z-test0001",
    dataset_sha256: str | None = None,
    seed: int = 42,
    code_version: str = "0.3.0",
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=candidate.candidate_id,
        title=candidate.title,
        run_id=run_id,
        dataset_sha256=dataset_sha256 or sha256,
        seed=seed,
        code_version=code_version,
        source=ManifestSource(
            provider=candidate.source_name,
            source_id=candidate.source_id,
            dataset_id=candidate.source_dataset_id,
            landing_url=candidate.landing_url,
            download_url=candidate.resource_url or candidate.landing_url,
            license_name=candidate.license_name,
            license_url=candidate.license_url,
            license_status=candidate.license_status.value,
            publisher=candidate.publisher,
            reported_version=candidate.published_at,
        ),
        retrieved_at=retrieved_at,
        file=ManifestFile(
            filename="artifact.csv",
            bytes=bytes_,
            sha256=sha256,
            content_type="text/csv",
            format=candidate.format,
            cache_path=cache_path,
            raw_path=raw_path,
            extracted_path=extracted_path,
            extracted_sha256=extracted_sha256,
            extracted_files=list(extracted_files or []),
        ),
        integrity=ManifestIntegrity(
            verified=True,
            method="sha256_recomputed",
            expected_size=candidate.size_bytes,
            archive_members=list(archive_members or []),
        ),
        acquisition=ManifestAcquisition(
            method="https_download", attempt=1, duration_ms=1, status="success"
        ),
        discovery={
            "candidate": candidate.to_dict(),
            "candidate_fingerprint": candidate_fingerprint(candidate),
        },
    )


def symlink_external_attr() -> int:
    import stat

    return (stat.S_IFLNK | 0o777) << 16
