"""Static, reviewed source registry.

The registry is intentionally not a web crawl. Each entry declares where the
adapter may connect, how trustworthy its metadata is, and whether credentials
are required (none are, for the enabled sources).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    name: str
    adapter_key: str
    trust: float
    priority: int
    hosts: tuple[str, ...]
    requires_credentials: bool = False
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "trust": self.trust,
            "priority": self.priority,
            "hosts": list(self.hosts),
            "requires_credentials": self.requires_credentials,
            "enabled": self.enabled,
            "notes": self.notes,
        }


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="openml",
        name="OpenML",
        adapter_key="openml",
        trust=1.0,
        priority=1,
        hosts=("www.openml.org",),
        requires_credentials=False,
        notes="Documented REST API; per-dataset licence field and column metadata.",
    ),
    SourceSpec(
        source_id="uci",
        name="UCI Machine Learning Repository",
        adapter_key="uci",
        trust=0.95,
        priority=2,
        hosts=("archive.ics.uci.edu",),
        requires_credentials=False,
        notes="Documented REST API; target and variable roles exposed, licence is not.",
    ),
    SourceSpec(
        source_id="github-ibm",
        name="IBM public sample repository (GitHub)",
        adapter_key="github",
        trust=0.70,
        priority=3,
        hosts=("raw.githubusercontent.com", "api.github.com", "github.com"),
        requires_credentials=False,
        notes=(
            "Pinned public sample file, not a search engine. Dataset licence is not "
            "established by the repository licence; GitHub API is rate-limited without a token."
        ),
    ),
)

_SOURCE_BY_ID = {spec.source_id: spec for spec in SOURCES}


def get_source(source_id: str) -> SourceSpec:
    try:
        return _SOURCE_BY_ID[source_id]
    except KeyError as exc:
        raise KeyError(f"unknown source_id {source_id!r}") from exc


def enabled_sources() -> tuple[SourceSpec, ...]:
    return tuple(spec for spec in SOURCES if spec.enabled)


def allowed_hosts() -> frozenset[str]:
    hosts: set[str] = set()
    for spec in enabled_sources():
        hosts.update(spec.hosts)
    return frozenset(hosts)


def source_priority(source_id: str) -> int:
    spec = _SOURCE_BY_ID.get(source_id)
    return spec.priority if spec else 99


def source_trust(source_id: str) -> float:
    spec = _SOURCE_BY_ID.get(source_id)
    return spec.trust if spec else 0.0
