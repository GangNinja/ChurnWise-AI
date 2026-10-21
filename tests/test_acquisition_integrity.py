from __future__ import annotations

import hashlib
import zipfile

import pytest

from churnwise.acquisition.errors import (
    CODE_CORRUPT_ARCHIVE,
    CODE_SIZE_LIMIT,
    CODE_UNEXPECTED_CONTENT,
    CODE_UNSAFE_ARCHIVE,
)
from churnwise.acquisition.integrity import (
    extract_archive,
    inspect_archive,
    inspect_download,
    sha256_file,
)
from tests.acquisition_helpers import (
    arff_bytes,
    csv_bytes,
    parquet_bytes,
    symlink_external_attr,
    zip_bytes,
)


def write(tmp_path, name: str, content: bytes):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_sha256_matches_hashlib(tmp_path):
    path = write(tmp_path, "data.csv", csv_bytes())
    assert sha256_file(path) == hashlib.sha256(csv_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("name", "content", "declared", "detected"),
    [
        ("data.csv", csv_bytes(), "csv", "csv"),
        ("data.tsv", b"a\tb\tchurn\n1\t2\tno\n3\t4\tyes\n", "tsv", "tsv"),
        ("data.arff", arff_bytes(), "arff", "arff"),
        ("data.parquet", parquet_bytes(), "parquet", "parquet"),
        ("data.json", b'[{"churn": "no"}, {"churn": "yes"}]', "json", "json"),
    ],
)
def test_supported_tabular_signatures_are_detected(tmp_path, name, content, declared, detected):
    inspection = inspect_download(
        write(tmp_path, name, content), declared_format=declared, content_type=None
    )
    assert inspection.format == detected
    assert inspection.is_archive is False


def test_html_error_page_with_200_is_rejected(tmp_path):
    path = write(tmp_path, "data.csv", b"<!doctype html><html><body>error</body></html>")
    with pytest.raises(Exception) as raised:
        inspect_download(path, declared_format="csv", content_type="text/html")
    assert getattr(raised.value, "code", None) == CODE_UNEXPECTED_CONTENT


def test_html_signature_is_rejected_without_a_content_type(tmp_path):
    path = write(tmp_path, "data.csv", b"<!-- proxy error -->\n<html><body>error</body></html>")
    with pytest.raises(Exception) as raised:
        inspect_download(path, declared_format="csv", content_type=None)
    assert getattr(raised.value, "code", None) == CODE_UNEXPECTED_CONTENT


def test_declared_format_must_match_the_bytes(tmp_path):
    path = write(tmp_path, "data.csv", arff_bytes())
    with pytest.raises(Exception) as raised:
        inspect_download(path, declared_format="csv", content_type="text/csv")
    assert getattr(raised.value, "code", None) == CODE_UNEXPECTED_CONTENT


def test_executable_content_is_rejected(tmp_path):
    path = write(tmp_path, "data.csv", b"MZ\x90\x00not a dataset")
    with pytest.raises(Exception) as raised:
        inspect_download(path, declared_format="csv", content_type="text/csv")
    assert getattr(raised.value, "code", None) == CODE_UNEXPECTED_CONTENT


def test_empty_artifact_is_rejected(tmp_path):
    path = write(tmp_path, "data.csv", b"")
    with pytest.raises(Exception) as raised:
        inspect_download(path, declared_format="csv", content_type=None)
    assert getattr(raised.value, "code", None) == CODE_UNEXPECTED_CONTENT


def test_single_member_archive_extracts_only_that_member(tmp_path):
    archive = write(tmp_path, "churn.zip", zip_bytes())
    info = extract_archive(archive, tmp_path / "out", max_members=10, max_extracted_bytes=10_000)

    assert info.selected_member == "churn.csv"
    assert info.extracted_path is not None
    assert info.extracted_path.read_bytes().startswith(b"customer_id")
    assert not (tmp_path / "out" / "evil.txt").exists()


def test_path_traversal_member_rejects_the_whole_archive(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(extra=[("../evil.csv", b"bad", 0o600 << 16)]),
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_backslash_member_path_rejects_the_whole_archive(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(extra=[("sub\\churn.csv", csv_bytes(), 0o600 << 16)]),
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_symlink_member_rejects_the_whole_archive(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(extra=[("link.csv", b"target", symlink_external_attr())]),
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_executable_member_rejects_the_whole_archive(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(extra=[("setup.exe", b"MZ", 0o600 << 16)]),
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_control_character_member_name_rejects_the_whole_archive(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(name="bad\x00.csv")
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_multiple_tabular_members_are_ambiguous(tmp_path):
    archive = write(
        tmp_path,
        "churn.zip",
        zip_bytes(extra=[("other.csv", csv_bytes(), 0o600 << 16)]),
    )
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_archive_without_tabular_member_is_rejected(tmp_path):
    archive = write(tmp_path, "churn.zip", zip_bytes(name="README.txt", content=b"hello"))
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_archive_member_count_limit(tmp_path):
    archive = write(tmp_path, "churn.zip", zip_bytes())
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=0, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_UNSAFE_ARCHIVE


def test_archive_expansion_limit(tmp_path):
    archive = write(tmp_path, "churn.zip", zip_bytes(content=csv_bytes(rows=100)))
    with pytest.raises(Exception) as raised:
        inspect_archive(archive, max_members=10, max_extracted_bytes=50)
    assert getattr(raised.value, "code", None) == CODE_SIZE_LIMIT


def test_corrupt_archive_is_rejected(tmp_path):
    path = write(tmp_path, "churn.zip", b"PK\x03\x04truncated")
    with pytest.raises(Exception) as raised:
        inspect_archive(path, max_members=10, max_extracted_bytes=10_000)
    assert getattr(raised.value, "code", None) == CODE_CORRUPT_ARCHIVE


def test_archive_is_detected_as_zip_when_declared(tmp_path):
    path = write(tmp_path, "churn.zip", zip_bytes())
    inspection = inspect_download(path, declared_format="zip", content_type="application/zip")
    assert inspection.is_archive is True
    assert inspection.format == "zip"
