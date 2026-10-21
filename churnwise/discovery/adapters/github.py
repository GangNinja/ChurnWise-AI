"""Pinned GitHub resource adapter.

GitHub is not used as a dataset search engine (code search needs a token).
Instead the registry holds explicit, reviewed resources. Only repository
metadata is fetched: the dataset file itself is Phase 3 work, so shape, target
and schema stay unknown and this candidate cannot pass discovery gates on its
own.
"""

from __future__ import annotations

from typing import Any

from churnwise.contracts import (
    DatasetCandidate,
    LicenseStatus,
    SearchQuery,
    utc_now_iso,
)
from churnwise.discovery.adapters.base import SourceAdapter, clean_description
from churnwise.net import DiscoveryHttpError

PINNED: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "ibm-telco-churn",
        "title": "IBM Telco Customer Churn",
        "description": (
            "Telco customer churn sample dataset published in IBM's public "
            "customer churn analytics repository."
        ),
        "repository": "IBM/telco-customer-churn-on-icp4d",
        "landing_url": "https://github.com/IBM/telco-customer-churn-on-icp4d",
        "resource_url": (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv"
        ),
        "format": "csv",
        "keywords": ("churn", "telco", "customer churn", "attrition", "subscription"),
        "publisher": "IBM",
    },
)


class GitHubPinnedAdapter(SourceAdapter):
    def _discover(self, query: SearchQuery) -> tuple[list[DatasetCandidate], list[str]]:
        notes: list[str] = []
        terms = [term.lower() for term in query.terms]
        candidates: list[DatasetCandidate] = []

        for entry in PINNED:
            keywords = tuple(str(k).lower() for k in entry["keywords"])
            if not any(term in keywords or any(k in term for k in keywords) for term in terms):
                continue
            repository_license, note = self._repository_license(
                str(entry["repository"]), query, notes
            )
            candidates.append(self._build(entry, repository_license, note))

        return candidates, notes

    def _repository_license(
        self, repository: str, query: SearchQuery, notes: list[str]
    ) -> tuple[str | None, str | None]:
        try:
            payload = self.client.get_json(
                f"https://api.github.com/repos/{repository}", timeout=query.timeout_s
            )
        except DiscoveryHttpError as exc:
            notes.append(
                f"GitHub repository metadata unavailable for {repository}: {exc.message}"
            )
            return None, f"repository metadata unavailable ({exc.code})"
        license_block = payload.get("license")
        if isinstance(license_block, dict):
            spdx = license_block.get("spdx_id")
            name = license_block.get("name")
            text = str(spdx or name or "").strip()
            if text and text.upper() != "NOASSERTION":
                return text, None
        return None, "repository declares no licence"

    def _build(
        self, entry: dict[str, Any], repository_license: str | None, license_note: str | None
    ) -> DatasetCandidate:
        limitations = [
            "dataset shape, target column and schema are not verified until Phase 3 inspects the file",
            "repository licence does not establish the dataset licence",
        ]
        if repository_license:
            limitations.append(f"repository licence: {repository_license}; dataset licence unverified")
        if license_note:
            limitations.append(license_note)

        return DatasetCandidate(
            candidate_id=self._candidate_id(str(entry["dataset_id"])),
            source_id=self.spec.source_id,
            source_name=self.spec.name,
            source_dataset_id=str(entry["dataset_id"]),
            title=str(entry["title"]),
            description=clean_description(entry.get("description")),
            landing_url=str(entry["landing_url"]),
            resource_url=str(entry["resource_url"]),
            format=str(entry["format"]),
            publisher=str(entry.get("publisher") or "") or None,
            license_name=None,
            license_url=None,
            license_status=LicenseStatus.UNVERIFIED,
            published_at=None,
            updated_at=None,
            size_bytes=None,
            row_count=None,
            feature_count=None,
            target_column=None,
            target_class_count=None,
            class_balance=None,
            schema_fields=None,
            has_numeric=None,
            has_categorical=None,
            access="open",
            limitations=limitations,
            retrieved_at=utc_now_iso(),
            evidence={
                "repository": entry["repository"],
                "repository_license": repository_license,
                "pinned": True,
            },
        )
