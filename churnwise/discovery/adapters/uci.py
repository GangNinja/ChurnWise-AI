"""UCI Machine Learning Repository adapter.

The documented REST API exposes search, dataset metadata, declared target
column and per-variable roles. It does **not** expose a licence field, so UCI
candidates are always marked ``unverified`` until the landing page is checked.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlsplit

from churnwise.contracts import (
    DatasetCandidate,
    FieldInfo,
    LicenseStatus,
    PII_PATTERN,
    SearchQuery,
    utc_now_iso,
)
from churnwise.discovery.adapters.base import (
    SourceAdapter,
    as_int,
    as_text,
    canonical_format,
    clean_description,
    publisher_from,
)
from churnwise.net import CODE_BUDGET, CODE_PROTOCOL, CODE_RATE_LIMITED, CODE_TIMEOUT, DiscoveryHttpError

BASE = "https://archive.ics.uci.edu"
ALLOWED_HOST = "archive.ics.uci.edu"
_FATAL_METADATA_CODES = {CODE_BUDGET, CODE_RATE_LIMITED, CODE_TIMEOUT}

_NUMERIC_TYPES = {"integer", "continuous", "real"}
_CATEGORICAL_TYPES = {"binary", "categorical", "nominal", "ordinal"}


class UCIAdapter(SourceAdapter):
    def _discover(self, query: SearchQuery) -> tuple[list[DatasetCandidate], list[str]]:
        notes: list[str] = []
        hits: dict[str, str] = {}

        for term in query.terms:
            url = f"{BASE}/api/datasets/list?search={quote(term, safe='')}"
            payload = self.client.get_json(url, timeout=query.timeout_s)
            rows = payload.get("data")
            if rows is None:
                continue
            if not isinstance(rows, list):
                notes.append(f"UCI search term {term!r} returned an unexpected payload")
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                dataset_id = as_int(row.get("id"))
                if dataset_id is None:
                    continue
                hits.setdefault(str(dataset_id), as_text(row.get("name")) or "")

        if not hits:
            return [], notes

        candidates: list[DatasetCandidate] = []
        for dataset_id, fallback_name in list(hits.items())[: query.enrich_limit]:
            try:
                payload = self.client.get_json(
                    f"{BASE}/api/dataset?id={dataset_id}", timeout=query.timeout_s
                )
                metadata = payload.get("data")
                if not isinstance(metadata, dict):
                    raise DiscoveryHttpError(CODE_PROTOCOL, "UCI metadata missing data block")
            except DiscoveryHttpError as exc:
                if exc.code in _FATAL_METADATA_CODES:
                    raise
                notes.append(f"UCI dataset {dataset_id} skipped: {exc.message}")
                continue
            candidates.append(self._build(dataset_id, fallback_name, metadata))

        return candidates, notes

    def _build(
        self, dataset_id: str, fallback_name: str, metadata: dict[str, Any]
    ) -> DatasetCandidate:
        limitations: list[str] = [
            "UCI API does not expose a licence field; verify the dataset page before use"
        ]

        variables = metadata.get("variables")
        variables = [v for v in variables if isinstance(v, dict)] if isinstance(variables, list) else []

        schema: list[FieldInfo] = []
        target_column: str | None = None
        target_class_count: int | None = None
        for variable in variables:
            name = as_text(variable.get("name"))
            if not name:
                continue
            declared_role = (as_text(variable.get("role")) or "").lower()
            role = "target" if declared_role == "target" else ("id" if declared_role == "id" else "feature")
            data_type = as_text(variable.get("type"))
            schema.append(FieldInfo(name=name, data_type=data_type, role=role))
            if role == "target":
                target_column = name
                if (data_type or "").lower() == "binary":
                    target_class_count = 2

        declared_target = metadata.get("target_col")
        if isinstance(declared_target, list):
            target_column = target_column or (as_text(declared_target[0]) if declared_target else None)
        else:
            target_column = target_column or as_text(declared_target)

        if target_column and target_class_count is None:
            limitations.append("target cardinality is not declared; Phase 4 must confirm it")
        elif target_class_count == 2:
            limitations.append("binary target inferred from the declared variable type")

        row_count = as_int(metadata.get("num_instances"))
        feature_count = as_int(metadata.get("num_features")) or (len(schema) or None)

        data_url = as_text(metadata.get("data_url"))
        resource_url = None
        if data_url:
            parts = urlsplit(data_url)
            if parts.scheme == "https" and (parts.hostname or "").lower() == ALLOWED_HOST:
                resource_url = data_url
            else:
                limitations.append("source-declared data URL did not pass the host allowlist")

        declared_format = "unknown"
        if resource_url:
            path = urlsplit(resource_url).path.lower()
            for suffix in (".csv", ".zip", ".arff", ".parquet", ".xlsx"):
                if path.endswith(suffix):
                    declared_format = canonical_format(suffix.lstrip("."))
                    break

        numeric = any((f.data_type or "").lower() in _NUMERIC_TYPES for f in schema)
        categorical = any((f.data_type or "").lower() in _CATEGORICAL_TYPES for f in schema)

        pii = [
            field.name
            for field in schema
            if field.role != "target" and PII_PATTERN.search(field.name)
        ]
        if pii:
            limitations.append(f"possible PII column(s): {', '.join(sorted(pii))}")

        if not schema:
            limitations.append("column metadata unavailable from the source")
        limitations.append("class balance not reported by discovery")

        year = as_int(metadata.get("year_of_dataset_creation"))

        return DatasetCandidate(
            candidate_id=self._candidate_id(dataset_id),
            source_id=self.spec.source_id,
            source_name=self.spec.name,
            source_dataset_id=dataset_id,
            title=as_text(metadata.get("name")) or fallback_name or f"UCI dataset {dataset_id}",
            description=clean_description(metadata.get("abstract")),
            landing_url=f"{BASE}/dataset/{dataset_id}",
            resource_url=resource_url,
            format=declared_format,
            publisher=publisher_from(metadata.get("creators")),
            license_name=None,
            license_url=None,
            license_status=LicenseStatus.UNVERIFIED,
            published_at=str(year) if year else None,
            updated_at=as_text(metadata.get("last_updated")),
            size_bytes=None,
            row_count=row_count,
            feature_count=feature_count,
            target_column=target_column,
            target_class_count=target_class_count,
            class_balance=None,
            schema_fields=schema or None,
            has_numeric=numeric if schema else None,
            has_categorical=categorical if schema else None,
            access="open",
            limitations=limitations,
            retrieved_at=utc_now_iso(),
            evidence={
                "uci_id": int(dataset_id),
                "dataset_doi": as_text(metadata.get("dataset_doi")),
                "tasks": metadata.get("tasks"),
                "feature_types": metadata.get("feature_types"),
                "target_col": metadata.get("target_col"),
            },
        )
