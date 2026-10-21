"""Dataset discovery subsystem (Phase 2).

Public entry points:

* :func:`discover` — run all adapters and return a :class:`DiscoveryReport`.
* :func:`build_adapters` — construct the enabled live adapters.
"""

from churnwise.discovery.adapters.github import GitHubPinnedAdapter
from churnwise.discovery.adapters.openml import OpenMLAdapter
from churnwise.discovery.adapters.uci import UCIAdapter
from churnwise.discovery.coordinator import discover
from churnwise.discovery.registry import SOURCES, SourceSpec
from churnwise.net import HttpClient, RequestsTransport

__all__ = [
    "GitHubPinnedAdapter",
    "HttpClient",
    "OpenMLAdapter",
    "RequestsTransport",
    "SOURCES",
    "SourceSpec",
    "UCIAdapter",
    "build_adapters",
    "discover",
]


def build_adapters(settings, query):
    """Create one adapter per enabled source, each with its own request budget."""
    from churnwise.config import Settings  # local import avoids a cycle at module load

    if not isinstance(settings, Settings):  # pragma: no cover - defensive
        raise TypeError("settings must be a Settings instance")

    adapters = []
    for spec in SOURCES:
        if not spec.enabled:
            continue
        client = HttpClient(
            spec.hosts,
            timeout_s=query.timeout_s,
            max_retries=query.max_retries,
            max_bytes=settings.max_response_bytes,
            max_requests=query.max_requests_per_source,
            transport=RequestsTransport(settings.user_agent),
        )
        if spec.adapter_key == "openml":
            adapters.append(OpenMLAdapter(spec, client))
        elif spec.adapter_key == "uci":
            adapters.append(UCIAdapter(spec, client))
        elif spec.adapter_key == "github":
            adapters.append(GitHubPinnedAdapter(spec, client))
    return adapters
