"""Dataset loading for Phase 4.

The loader consumes the verified artifact selected by Phase 3. It reads the
file once, returns a pandas frame and records structural warnings such as
duplicate header names. Parsing failures are explicit ``DatasetLoadError``
values with a stable code; they are never silently converted to an empty frame.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_FORMATS = frozenset({"csv", "tsv", "arff", "parquet", "xlsx", "json"})
NA_VALUES = ["", "?", "NA", "N/A", "NULL", "null", "None", "nan", "NaN"]
_ARFF_ATTRIBUTE = re.compile(
    r"""(?im)^\s*@attribute\s+(?:"([^"]+)"|'([^']+)'|(\S+))"""
)


class DatasetLoadError(Exception):
    """A parse or structural error that makes validation impossible."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    format: str
    source_path: str
    duplicate_headers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _sniff_delimiter(path: Path) -> str:
    text, _ = _read_text(path)
    first = next((line for line in text.splitlines() if line.strip()), "")
    if "\t" in first and "," not in first:
        return "\t"
    if ";" in first and "," not in first:
        return ";"
    return ","


def _header_names(path: Path, format_name: str) -> tuple[list[str], list[str]]:
    """Return header names and duplicated names without relying on pandas."""
    names: list[str] = []
    if format_name in {"csv", "tsv"}:
        text, _ = _read_text(path)
        delimiter = "\t" if format_name == "tsv" else _sniff_delimiter(path)
        try:
            first = next(csv.reader(text.splitlines(), delimiter=delimiter))
        except (csv.Error, StopIteration):
            first = []
        names = [str(name).strip() for name in first]
    elif format_name == "arff":
        text, _ = _read_text(path)
        names = [
            next(group for group in match.groups() if group is not None).strip()
            for match in _ARFF_ATTRIBUTE.finditer(text)
        ]
    elif format_name == "xlsx":
        try:
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                sheet = workbook.active
                row = next(sheet.iter_rows(values_only=True), ())
                names = [str(name).strip() for name in row if name is not None]
            finally:
                workbook.close()
        except Exception:
            names = []
    if not names:
        return [], []
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return names, duplicates


def _decode_arff_frame(frame: pd.DataFrame) -> pd.DataFrame:
    decoded = frame.copy()
    for name in decoded.columns:
        series = decoded[name]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            decoded[name] = series.map(
                lambda value: (
                    None
                    if isinstance(value, bytes) and value.strip() in {b"", b"?"}
                    else value.decode("utf-8", errors="replace").strip()
                    if isinstance(value, bytes)
                    else value
                )
            )
    return decoded


def load_dataset(
    path: Path | str,
    format: str | None = None,
    *,
    max_rows: int | None = None,
) -> LoadedDataset:
    """Load a supported tabular artifact without modifying it."""
    source = Path(path)
    format_name = (format or source.suffix.lstrip(".")).lower()
    if format_name == "zip":
        raise DatasetLoadError(
            "CW-VAL-001",
            "ZIP artifacts must be extracted by Phase 3; pass the extracted tabular member",
        )
    if format_name not in SUPPORTED_FORMATS:
        raise DatasetLoadError(
            "CW-VAL-001", f"unsupported tabular format {format_name!r}"
        )
    if not source.is_file():
        raise DatasetLoadError("CW-VAL-001", f"dataset artifact does not exist: {source}")
    if source.stat().st_size <= 0:
        raise DatasetLoadError("CW-VAL-002", "dataset artifact is empty")

    header_names, duplicate_headers = _header_names(source, format_name)
    warnings: list[str] = []
    try:
        if format_name == "arff":
            from scipy.io import arff

            records, _ = arff.loadarff(source)
            frame = _decode_arff_frame(pd.DataFrame(records))
        elif format_name in {"csv", "tsv"}:
            delimiter = "\t" if format_name == "tsv" else _sniff_delimiter(source)
            try:
                frame = pd.read_csv(
                    source,
                    sep=delimiter,
                    encoding="utf-8-sig",
                    na_values=NA_VALUES,
                    keep_default_na=True,
                    low_memory=False,
                )
            except UnicodeDecodeError:
                frame = pd.read_csv(
                    source,
                    sep=delimiter,
                    encoding="latin-1",
                    na_values=NA_VALUES,
                    keep_default_na=True,
                    low_memory=False,
                )
                warnings.append("file was decoded as latin-1 after UTF-8 failed")
        elif format_name == "parquet":
            frame = pd.read_parquet(source)
        elif format_name == "xlsx":
            frame = pd.read_excel(source, engine="openpyxl")
        elif format_name == "json":
            try:
                frame = pd.read_json(source)
            except ValueError:
                frame = pd.read_json(source, lines=True)
        else:  # pragma: no cover - guarded above
            raise DatasetLoadError("CW-VAL-001", f"unsupported format {format_name!r}")
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(
            "CW-VAL-002",
            f"dataset could not be parsed as {format_name}: {type(exc).__name__}: {exc}",
        ) from exc

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[:max_rows].copy()
        warnings.append(f"profile was limited to the first {max_rows} rows")
    if frame.shape[1] == 0:
        raise DatasetLoadError("CW-VAL-002", "dataset has no columns")
    frame.columns = [str(name).strip() for name in frame.columns]
    if duplicate_headers:
        warnings.append(
            "duplicate header names were present: " + ", ".join(duplicate_headers)
        )
    return LoadedDataset(
        frame=frame,
        format=format_name,
        source_path=str(source),
        duplicate_headers=duplicate_headers,
        warnings=warnings,
    )


def load_extracted_member(
    data_dir: Path | str,
    extracted_path: str,
    *,
    max_rows: int | None = None,
) -> LoadedDataset:
    """Load the tabular member recorded by a Phase 3 ZIP acquisition."""
    path = Path(data_dir) / extracted_path
    format_name = path.suffix.lstrip(".").lower()
    return load_dataset(path, format_name, max_rows=max_rows)
