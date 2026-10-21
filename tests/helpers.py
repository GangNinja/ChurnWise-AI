"""Shared, network-free test helpers."""

from __future__ import annotations

import json
from typing import Any

from churnwise.contracts import (
    DatasetCandidate,
    FieldInfo,
    LicenseStatus,
    SourceResult,
    SourceStatus,
    utc_now_iso,
)
from churnwise.net import HttpClient, HttpResponse


def json_response(
    payload: Any, status: int = 200, headers: dict[str, str] | None = None
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={k.lower(): v for k, v in (headers or {}).items()},
        body=json.dumps(payload).encode("utf-8"),
        url="",
    )


class FakeTransport:
    """Returns queued responses per URL; raises queued exceptions; records calls."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self._counts: dict[str, int] = {}

    def __call__(self, url: str, timeout: float, max_bytes: int) -> HttpResponse:
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
            return value(url, timeout, max_bytes)
        return value


def make_client(
    routes: dict[str, Any],
    *,
    hosts: tuple[str, ...] = ("www.openml.org", "archive.ics.uci.edu", "api.github.com"),
    sleeps: list[float] | None = None,
    max_requests: int = 30,
    max_retries: int = 2,
    timeout_s: float = 12.0,
) -> tuple[HttpClient, FakeTransport, list[float]]:
    transport = FakeTransport(routes)
    slept = sleeps if sleeps is not None else []
    client = HttpClient(
        hosts,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_requests=max_requests,
        transport=transport,
        sleep=slept.append,
    )
    return client, transport, slept


def make_candidate(**overrides: Any) -> DatasetCandidate:
    defaults: dict[str, Any] = {
        "candidate_id": "openml:46915",
        "source_id": "openml",
        "source_name": "OpenML",
        "source_dataset_id": "46915",
        "title": "churn",
        "description": "Telephony customer churn dataset.",
        "landing_url": "https://www.openml.org/d/46915",
        "resource_url": "https://www.openml.org/data/v1/download/4965302",
        "format": "arff",
        "license_name": "MIT License",
        "license_status": LicenseStatus.VERIFIED,
        "retrieved_at": utc_now_iso(),
        "row_count": 5000,
        "feature_count": 21,
        "target_column": "CustomerChurned",
        "target_class_count": 2,
        "class_balance": {"majority": 4293, "minority": 707},
        "schema_fields": [
            FieldInfo("CustomerChurned", "nominal", "target"),
            FieldInfo("tenure", "numeric", "feature"),
            FieldInfo("contract", "nominal", "feature"),
        ],
        "has_numeric": True,
        "has_categorical": True,
        "evidence": {"openml_did": 46915},
    }
    defaults.update(overrides)
    return DatasetCandidate(**defaults)


def make_result(
    source_id: str,
    candidates: list[DatasetCandidate] | None = None,
    status: SourceStatus = SourceStatus.OK,
    **kwargs: Any,
) -> SourceResult:
    return SourceResult(
        source_id=source_id,
        status=status,
        candidates=candidates or [],
        **kwargs,
    )


class FakeAdapter:
    def __init__(self, result: SourceResult) -> None:
        self.source_id = result.source_id
        self.result = result
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return self.result


def openml_search_payload(*dids: int) -> dict[str, Any]:
    return {
        "data": {
            "dataset": [
                {
                    "did": did,
                    "name": f"churn {did}",
                    "format": "ARFF",
                    "file_id": 4_000_000 + did,
                }
                for did in dids
            ]
        }
    }


def openml_description_payload(did: int, **overrides: Any) -> dict[str, Any]:
    description = {
        "id": str(did),
        "name": f"churn {did}",
        "format": "ARFF",
        "licence": "Public",
        "default_target_attribute": "class",
        "description": "A telephony churn dataset. " * 3,
        "upload_date": "2023-01-27T11:40:40",
        "creator": "Test Creator",
    }
    description.update(overrides)
    return {"data_set_description": description}


def openml_qualities_payload(**values: str) -> dict[str, Any]:
    base = {
        "NumberOfInstances": "5000",
        "NumberOfFeatures": "21",
        "NumberOfClasses": "2",
        "MajorityClassSize": "4293",
        "MinorityClassSize": "707",
    }
    base.update(values)
    return {
        "data_qualities": {
            "quality": [{"name": name, "value": value} for name, value in base.items()]
        }
    }


def openml_features_payload(
    *,
    target: str = "class",
    classes: tuple[str, ...] = ("No", "Yes"),
    include_pii: bool = False,
) -> dict[str, Any]:
    features = [
        {
            "index": "0",
            "name": "tenure",
            "data_type": "numeric",
            "is_target": "false",
            "is_row_identifier": "false",
            "number_of_missing_values": "0",
        },
        {
            "index": "1",
            "name": "contract",
            "data_type": "nominal",
            "is_target": "false",
            "is_row_identifier": "false",
            "number_of_missing_values": "0",
            "nominal_value": ["monthly", "yearly"],
        },
        {
            "index": "2",
            "name": target,
            "data_type": "nominal",
            "is_target": "true",
            "is_row_identifier": "false",
            "number_of_missing_values": "0",
            "nominal_value": list(classes),
        },
    ]
    if include_pii:
        features.append(
            {
                "index": "3",
                "name": "customer_email",
                "data_type": "string",
                "is_target": "false",
                "is_row_identifier": "false",
                "number_of_missing_values": "0",
            }
        )
    return {"data_features": {"feature": features}}


def uci_search_payload(*rows: tuple[int, str]) -> dict[str, Any]:
    return {"data": [{"id": dataset_id, "name": name} for dataset_id, name in rows]}


def uci_metadata_payload(
    dataset_id: int = 563,
    *,
    name: str = "Iranian Churn",
    target_type: str = "Binary",
    **overrides: Any,
) -> dict[str, Any]:
    metadata = {
        "uci_id": dataset_id,
        "name": name,
        "num_instances": 3150,
        "num_features": 13,
        "tasks": ["Classification", "Regression"],
        "area": "Business",
        "abstract": "Customer churn data from an Iranian telecom company.",
        "target_col": ["Churn"],
        "data_url": f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv",
        "dataset_doi": "10.24432/C5JW3Z",
        "year_of_dataset_creation": 2020,
        "last_updated": "2023-01-01",
        "creators": [{"name": "Test Author"}],
        "variables": [
            {"name": "Call Failure", "role": "Feature", "type": "Integer"},
            {"name": "Complains", "role": "Feature", "type": "Binary"},
            {"name": "Churn", "role": "Target", "type": target_type},
        ],
    }
    metadata.update(overrides)
    return {"data": metadata}


def github_repo_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "full_name": "IBM/telco-customer-churn-on-icp4d",
        "license": {"spdx_id": "Apache-2.0", "name": "Apache License 2.0"},
    }
    payload.update(overrides)
    return payload
