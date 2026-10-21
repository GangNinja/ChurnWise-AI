"""OpenML adapter.

Uses the documented JSON API: dataset search by name, dataset description,
data qualities and feature metadata. No authentication is required. Only
metadata is fetched; dataset payloads are Phase 3 work.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

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

BASE = "https://www.openml.org"
_FATAL_METADATA_CODES = {CODE_BUDGET, CODE_RATE_LIMITED, CODE_TIMEOUT}

_AMBIGUOUS_LICENCES = {"", "unknown", "none", "unspecified", "?", "n/a", "no license"}

# OpenML currently emits a raw PHP notice inside the optional ``md5_checksum``
# string when a dataset has no checksum. The notice contains unescaped quotes,
# so the response is not valid JSON. Dropping only that broken value keeps the
# rest of the search usable; checksums are re-read from the dataset description
# during enrichment, where a missing value remains optional.
_PHP_NOTICE_IN_CHECKSUM = re.compile(
    r'"(?P<key>md5_checksum)"\s*:\s*"\s*<div style=.*?</div>"',
    re.DOTALL,
)


def _is_no_results(exc: DiscoveryHttpError) -> bool:
    """OpenML signals an empty name search with HTTP 412 and error code 372.

    That is a normal empty result, not a source failure; treating it as an
    error would abort a multi-term search as soon as one term has no match.
    """
    if exc.status != 412:
        return False
    try:
        payload = json.loads(exc.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return b"No results" in exc.body
    error = payload.get("error") if isinstance(payload, dict) else None
    return isinstance(error, dict) and str(error.get("code")) == "372"


class OpenMLAdapter(SourceAdapter):
    def _get_json(self, url: str, query: SearchQuery) -> dict[str, Any]:
        response = self.client.get(url, timeout=query.timeout_s)
        try:
            text = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DiscoveryHttpError(
                CODE_PROTOCOL,
                "response was not valid UTF-8 JSON",
                status=response.status,
                url=url,
            ) from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            repaired = _PHP_NOTICE_IN_CHECKSUM.sub('"md5_checksum":""', text)
            if repaired == text:
                raise DiscoveryHttpError(
                    CODE_PROTOCOL,
                    "response was not valid UTF-8 JSON",
                    status=response.status,
                    url=url,
                ) from None
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError as exc:
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

    def _discover(self, query: SearchQuery) -> tuple[list[DatasetCandidate], list[str]]:
        notes: list[str] = []
        stubs: dict[str, dict[str, Any]] = {}

        for term in query.terms:
            url = (
                f"{BASE}/api/v1/json/data/list/data_name/"
                f"{quote(term, safe='')}/limit/{query.max_results_per_source}"
            )
            try:
                payload = self._get_json(url, query)
            except DiscoveryHttpError as exc:
                if _is_no_results(exc):
                    continue
                raise
            block = payload.get("data") or {}
            if not isinstance(block, dict):
                raise DiscoveryHttpError(CODE_PROTOCOL, "OpenML search returned an unexpected shape")
            datasets = block.get("dataset") or []
            if not isinstance(datasets, list):
                notes.append(f"search term {term!r} returned a non-list dataset block")
                continue
            for stub in datasets:
                if not isinstance(stub, dict):
                    continue
                did = as_int(stub.get("did") or stub.get("id"))
                if did is None:
                    continue
                key = str(did)
                if key not in stubs:
                    stubs[key] = {**stub, "_search_term": term}

        if not stubs:
            return [], notes

        candidates: list[DatasetCandidate] = []
        for stub in list(stubs.values())[: query.enrich_limit]:
            did = str(as_int(stub.get("did") or stub.get("id")))
            try:
                payload = self._get_json(f"{BASE}/api/v1/json/data/{did}", query)
                description = payload.get("data_set_description")
                if not isinstance(description, dict):
                    raise DiscoveryHttpError(
                        CODE_PROTOCOL, "OpenML dataset description missing data_set_description"
                    )
            except DiscoveryHttpError as exc:
                if exc.code in _FATAL_METADATA_CODES:
                    raise
                notes.append(f"OpenML dataset {did} skipped: {exc.message}")
                continue
            qualities = self._qualities(did, query, notes)
            features = self._features(did, query, notes)
            candidates.append(self._build(did, stub, description, qualities, features))

        return candidates, notes

    # -- metadata helpers ---------------------------------------------------
    def _qualities(
        self, did: str, query: SearchQuery, notes: list[str]
    ) -> dict[str, str]:
        try:
            payload = self._get_json(f"{BASE}/api/v1/json/data/qualities/{did}", query)
        except DiscoveryHttpError as exc:
            if exc.code in _FATAL_METADATA_CODES:
                raise
            notes.append(f"OpenML qualities unavailable for {did}: {exc.message}")
            return {}
        block = payload.get("data_qualities") or {}
        items = block.get("quality") if isinstance(block, dict) else []
        if not isinstance(items, list):
            return {}
        return {
            str(item.get("name")): str(item.get("value"))
            for item in items
            if isinstance(item, dict) and item.get("name") is not None
        }

    def _features(
        self, did: str, query: SearchQuery, notes: list[str]
    ) -> list[dict[str, Any]]:
        try:
            payload = self._get_json(f"{BASE}/api/v1/json/data/features/{did}", query)
        except DiscoveryHttpError as exc:
            if exc.code in _FATAL_METADATA_CODES:
                raise
            notes.append(f"OpenML features unavailable for {did}: {exc.message}")
            return []
        block = payload.get("data_features") or {}
        items = block.get("feature") if isinstance(block, dict) else []
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    # -- mapping ------------------------------------------------------------
    def _build(
        self,
        did: str,
        stub: dict[str, Any],
        description: dict[str, Any],
        qualities: dict[str, str],
        features: list[dict[str, Any]],
    ) -> DatasetCandidate:
        limitations: list[str] = []
        schema: list[FieldInfo] = []
        target_column: str | None = None
        target_class_count: int | None = None

        for feature in features:
            name = as_text(feature.get("name"))
            if not name:
                continue
            is_target = str(feature.get("is_target")).lower() == "true"
            is_id = str(feature.get("is_row_identifier")).lower() == "true"
            role = "target" if is_target else ("id" if is_id else "feature")
            schema.append(
                FieldInfo(
                    name=name,
                    data_type=as_text(feature.get("data_type")),
                    role=role,
                    missing_count=as_int(feature.get("number_of_missing_values")),
                )
            )
            if is_target:
                target_column = name
                nominal = feature.get("nominal_value")
                if isinstance(nominal, list) and len(nominal) >= 2:
                    target_class_count = len(nominal)

        if target_column is None:
            target_column = as_text(description.get("default_target_attribute"))

        class_count_quality = as_int(qualities.get("NumberOfClasses"))
        if target_class_count is None and class_count_quality and class_count_quality >= 2:
            target_class_count = class_count_quality

        majority = as_int(qualities.get("MajorityClassSize"))
        minority = as_int(qualities.get("MinorityClassSize"))
        balance = (
            {"majority": majority, "minority": minority}
            if majority and minority and majority > 0 and minority > 0
            else None
        )

        row_count = as_int(qualities.get("NumberOfInstances"))
        feature_count = as_int(qualities.get("NumberOfFeatures")) or (len(schema) or None)

        licence_raw = as_text(description.get("licence"))
        if licence_raw and licence_raw.lower() not in _AMBIGUOUS_LICENCES:
            license_status = LicenseStatus.VERIFIED
            license_name = licence_raw
        else:
            license_status = LicenseStatus.UNVERIFIED
            license_name = None
            limitations.append("licence metadata unavailable or ambiguous at the source")

        if class_count_quality == 0:
            limitations.append(
                "source reports zero target classes; binary cardinality must be confirmed from data"
            )
        if not schema:
            limitations.append("column metadata unavailable from the source")
        if balance is None:
            limitations.append("class balance not reported by the source")
        if target_column is None:
            limitations.append("no target column declared by the source")

        pii = [
            field.name
            for field in schema
            if field.role != "target" and PII_PATTERN.search(field.name)
        ]
        if pii:
            limitations.append(f"possible PII column(s): {', '.join(sorted(pii))}")

        file_id = as_int(stub.get("file_id")) or as_int(description.get("file_id"))
        resource_url = (
            f"{BASE}/data/v1/download/{file_id}" if file_id is not None and file_id > 0 else None
        )
        if resource_url is None:
            limitations.append("direct resource URL not exposed; Phase 3 must resolve it")

        return DatasetCandidate(
            candidate_id=self._candidate_id(did),
            source_id=self.spec.source_id,
            source_name=self.spec.name,
            source_dataset_id=did,
            title=as_text(description.get("name")) or as_text(stub.get("name")) or f"OpenML {did}",
            description=clean_description(description.get("description")),
            landing_url=f"{BASE}/d/{did}",
            resource_url=resource_url,
            format=canonical_format(description.get("format") or stub.get("format")),
            publisher=publisher_from(description.get("creator")),
            license_name=license_name,
            license_url=None,
            license_status=license_status,
            published_at=as_text(description.get("upload_date")),
            updated_at=None,
            size_bytes=None,
            row_count=row_count,
            feature_count=feature_count,
            target_column=target_column,
            target_class_count=target_class_count,
            class_balance=balance,
            schema_fields=schema or None,
            has_numeric=(
                any((f.data_type or "").lower() == "numeric" for f in schema)
                if schema
                else None
            ),
            has_categorical=(
                any((f.data_type or "").lower() == "nominal" for f in schema)
                if schema
                else None
            ),
            access="open",
            limitations=limitations,
            retrieved_at=utc_now_iso(),
            evidence={
                "openml_did": int(did),
                "file_id": file_id,
                "version": as_text(description.get("version")),
                "md5_checksum": as_text(description.get("md5_checksum")),
                "search_term": stub.get("_search_term"),
                "default_target_attribute": as_text(
                    description.get("default_target_attribute")
                ),
                "quality": {
                    key: qualities.get(key)
                    for key in (
                        "NumberOfInstances",
                        "NumberOfFeatures",
                        "NumberOfClasses",
                        "MajorityClassSize",
                        "MinorityClassSize",
                    )
                },
            },
        )
