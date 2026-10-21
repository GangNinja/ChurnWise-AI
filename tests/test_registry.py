from __future__ import annotations

import pytest

from churnwise.discovery.registry import allowed_hosts, enabled_sources, get_source


def test_every_enabled_source_is_credential_free_and_uses_bare_https_hosts():
    specs = enabled_sources()
    assert specs
    for spec in specs:
        assert spec.requires_credentials is False, spec.source_id
        assert spec.hosts, spec.source_id
        assert all("://" not in host and "/" not in host for host in spec.hosts), spec.source_id
        assert spec.trust > 0.0
        assert spec.priority > 0


def test_allowed_hosts_is_the_union_of_enabled_source_hosts():
    hosts = allowed_hosts()
    assert "www.openml.org" in hosts
    assert "archive.ics.uci.edu" in hosts
    assert "api.github.com" in hosts


def test_unknown_sources_are_rejected_rather_than_defaulted():
    with pytest.raises(KeyError, match="unknown source_id"):
        get_source("not-a-source")
