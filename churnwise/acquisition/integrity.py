"""Artifact identity, content sniffing and safe archive handling.

A filename and an HTTP Content-Type are hints, never proof. The signature of
the downloaded bytes decides whether the artifact is accepted. Archives are
inspected before extraction; only one supported tabular member is extracted,
and unsafe paths, symlinks, encryption and executable entries are rejected.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from churnwise.acquisition.contracts import ACQUISITION_FORMATS
from churnwise.acquisition.errors import (
    CATEGORY_ARCHIVE,
    CATEGORY_FORMAT,
    CATEGORY_STORAGE,
    CODE_CORRUPT_ARCHIVE,
    CODE_SIZE_LIMIT,
    CODE_STORAGE,
    CODE_UNEXPECTED_CONTENT,
    CODE_UNSAFE_ARCHIVE,
    AcquisitionError,
    failure,
)

READ_HEAD = 8192
DANGEROUS_EXTENSIONS = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bat",
        ".cmd",
        ".ps1",
        ".sh",
        ".bash",
        ".py",
        ".rb",
        ".pl",
        ".js",
        ".jar",
        ".msi",
        ".vbs",
        ".scr",
        ".com",
        ".pif",
        ".lnk",
        ".html",
        ".htm",
        ".svg",
        ".php",
    }
)
TABULAR_EXTENSIONS = frozenset({".csv", ".tsv", ".arff", ".parquet", ".json"})
HTML_PREFIXES = ("<!doctype html", "<html", "<head", "<body")
EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"\xfe\xed\xfa", b"\xcf\xfa\xed", b"#!")


@dataclass
class Inspection:
    format: str
    content_type: str | None
    is_archive: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ArchiveInfo:
    members: list[dict[str, Any]]
    selected_member: str | None
    extracted_path: Path | None
    extracted_format: str | None
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_head(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(READ_HEAD)
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff \t\r\n")


def _html_like(text: str) -> bool:
    lowered = text.lower().lstrip()
    # A short leading comment is common on error pages; remove it before testing.
    lowered = re.sub(r"^<!--.*?-->", "", lowered, count=1, flags=re.DOTALL).lstrip()
    return lowered.startswith(HTML_PREFIXES)


def _looks_executable(raw: bytes) -> bool:
    return raw.startswith(EXECUTABLE_MAGIC)


def _csv_like(text: str) -> str | None:
    lines = [line for line in text.splitlines()[:8] if line.strip()]
    if len(lines) < 2:
        return None
    header = lines[0]
    if "\t" in header and "," not in header and ";" not in header:
        return "tsv"
    if "," in header or ";" in header:
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t")
        except csv.Error:
            delimiter = "," if "," in header else ";"
        else:
            delimiter = dialect.delimiter
        return "tsv" if delimiter == "\t" else "csv"
    return None


def inspect_download(
    path: os.PathLike[str] | str,
    *,
    declared_format: str,
    content_type: str | None,
) -> Inspection:
    path = Path(path)
    size = path.stat().st_size
    if size <= 0:
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            "downloaded artifact is empty",
            category=CATEGORY_FORMAT,
        )

    with path.open("rb") as handle:
        head = handle.read(READ_HEAD)
        handle.seek(max(0, size - 4))
        tail = handle.read(4)

    content_type = (content_type or "").lower().split(";", 1)[0].strip() or None
    if content_type in {"text/html", "application/xhtml+xml"}:
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            "source returned an HTML document instead of a dataset",
            category=CATEGORY_FORMAT,
        )
    if _looks_executable(head):
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            "downloaded bytes look executable; refusing the artifact",
            category=CATEGORY_FORMAT,
        )

    text = head.decode("utf-8", errors="replace").lstrip("\ufeff \t\r\n")
    if _html_like(text):
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            "source returned an HTML page instead of a dataset",
            category=CATEGORY_FORMAT,
        )

    actual: str | None = None
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        actual = "zip" if declared_format != "xlsx" else "xlsx"
    elif head.startswith(b"PAR1") and tail == b"PAR1":
        actual = "parquet"
    elif re.search(r"(?im)^\s*@relation\b", text) or re.search(r"(?im)^\s*@data\b", text):
        actual = "arff"
    elif text.lstrip().startswith(("{", "[")):
        actual = "json"
    else:
        csv_kind = _csv_like(text)
        if csv_kind:
            actual = csv_kind
        elif content_type in {"application/vnd.apache.parquet", "application/parquet"}:
            actual = "parquet"

    if declared_format not in ACQUISITION_FORMATS:
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            f"format {declared_format!r} is not supported for acquisition",
            category=CATEGORY_FORMAT,
        )
    if actual is None:
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            f"could not verify downloaded content as {declared_format}; content type {content_type!r}",
            category=CATEGORY_FORMAT,
        )
    if declared_format == "xlsx" and actual == "xlsx":
        return Inspection(actual, content_type, is_archive=False)
    if actual != declared_format:
        raise failure(
            CODE_UNEXPECTED_CONTENT,
            f"content looks like {actual!r} but the candidate declares {declared_format!r}",
            category=CATEGORY_FORMAT,
        )
    return Inspection(actual, content_type, is_archive=actual == "zip")


def inspect_archive(
    path: os.PathLike[str] | str,
    *,
    max_members: int,
    max_extracted_bytes: int,
) -> ArchiveInfo:
    members: list[dict[str, Any]] = []
    supported: list[zipfile.ZipInfo] = []
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > max_members:
                raise failure(
                    CODE_UNSAFE_ARCHIVE,
                    f"archive has {len(infos)} members; limit is {max_members}",
                    category=CATEGORY_ARCHIVE,
                )
            for info in infos:
                if "\\" in info.filename:
                    raise failure(
                        CODE_UNSAFE_ARCHIVE,
                        f"archive member {info.filename!r} contains a backslash path",
                        category=CATEGORY_ARCHIVE,
                    )
                name = info.filename
                reason: str | None = None
                member = {
                    "name": name,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "is_dir": info.is_dir(),
                }
                if info.flag_bits & 0x1:
                    raise failure(
                        CODE_UNSAFE_ARCHIVE,
                        f"archive member {name!r} is encrypted",
                        category=CATEGORY_ARCHIVE,
                    )
                if not _safe_member_name(name):
                    raise failure(
                        CODE_UNSAFE_ARCHIVE,
                        f"archive member {name!r} has an unsafe path",
                        category=CATEGORY_ARCHIVE,
                    )
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise failure(
                        CODE_UNSAFE_ARCHIVE,
                        f"archive member {name!r} is a symlink",
                        category=CATEGORY_ARCHIVE,
                    )
                if Path(name).suffix.lower() in DANGEROUS_EXTENSIONS:
                    raise failure(
                        CODE_UNSAFE_ARCHIVE,
                        f"archive member {name!r} is executable or a script",
                        category=CATEGORY_ARCHIVE,
                    )
                if info.is_dir():
                    reason = "directory"
                else:
                    total_uncompressed += info.file_size
                    if total_uncompressed > max_extracted_bytes:
                        raise failure(
                            CODE_SIZE_LIMIT,
                            f"archive expands beyond {max_extracted_bytes} bytes",
                            category=CATEGORY_ARCHIVE,
                        )
                    if Path(name).suffix.lower() in TABULAR_EXTENSIONS:
                        supported.append(info)
                    else:
                        reason = "not a supported tabular member"
                if reason:
                    member["ignored_reason"] = reason
                members.append(member)

            if len(supported) != 1:
                detail = (
                    "no supported tabular member"
                    if not supported
                    else f"{len(supported)} supported tabular members; archive is ambiguous"
                )
                raise failure(
                    CODE_UNSAFE_ARCHIVE,
                    detail,
                    category=CATEGORY_ARCHIVE,
                    causes=[info.filename for info in supported],
                )
    except zipfile.BadZipFile as exc:
        raise failure(
            CODE_CORRUPT_ARCHIVE,
            f"archive is corrupt or truncated: {exc}",
            category=CATEGORY_ARCHIVE,
        ) from exc
    except zipfile.LargeZipFile as exc:
        raise failure(
            CODE_UNSAFE_ARCHIVE,
            f"archive requires ZIP64 support: {exc}",
            category=CATEGORY_ARCHIVE,
        ) from exc
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        raise failure(
            CODE_UNSAFE_ARCHIVE,
            f"archive uses an unsupported or unsafe feature: {exc}",
            category=CATEGORY_ARCHIVE,
        ) from exc

    selected = supported[0]
    return ArchiveInfo(
        members=members,
        selected_member=selected.filename.replace("\\", "/"),
        extracted_path=None,
        extracted_format=Path(selected.filename).suffix.lower().lstrip("."),
    )


def extract_archive(
    path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    max_members: int,
    max_extracted_bytes: int,
) -> ArchiveInfo:
    info = inspect_archive(
        path, max_members=max_members, max_extracted_bytes=max_extracted_bytes
    )
    assert info.selected_member is not None
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    relative = PurePosixPath(info.selected_member)
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(info.selected_member) as source, target.open("wb") as handle:
                while True:
                    chunk = source.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_extracted_bytes:
                        raise failure(
                            CODE_SIZE_LIMIT,
                            f"extracted member exceeds {max_extracted_bytes} bytes",
                            category=CATEGORY_ARCHIVE,
                        )
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except zipfile.BadZipFile as exc:
        _remove_tree(destination)
        raise failure(
            CODE_CORRUPT_ARCHIVE,
            f"archive failed its integrity check during extraction: {exc}",
            category=CATEGORY_ARCHIVE,
        ) from exc
    except AcquisitionError:
        _remove_tree(destination)
        raise
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        _remove_tree(destination)
        raise failure(
            CODE_UNSAFE_ARCHIVE,
            f"archive member could not be extracted safely: {exc}",
            category=CATEGORY_ARCHIVE,
        ) from exc
    except OSError as exc:
        _remove_tree(destination)
        raise failure(
            CODE_STORAGE,
            f"could not extract archive: {exc}",
            category=CATEGORY_STORAGE,
        ) from exc

    info.extracted_path = target
    return info


_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "com0", "lpt0"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return False
    if "\\" in name:
        return False
    # NUL bytes truncate paths on some platforms and other control characters
    # make logs and extraction paths misleading. Reject them all.
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return False
    parts = PurePosixPath(name).parts
    if not parts:
        return False
    for part in parts:
        if part in ("", ".", "..") or ":" in part:
            return False
        # Windows strips trailing dots/spaces and resolves reserved device
        # names, which could make extraction write somewhere other than the
        # inspected member path.
        if part != part.rstrip(" .") or part.split(".", 1)[0].strip().lower() in _WINDOWS_RESERVED:
            return False
    return True


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass
