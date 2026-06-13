"""Shared test fixtures and the hermetic network guard.

Connector tests must mock HTTP (respx) or use synthetic local payloads; none
should touch a real upstream, or the suite becomes slow and flaky. This autouse
guard blocks DNS resolution for any non-local host, so an unmocked call fails
fast. Opt out with @pytest.mark.network (deselected in CI via -m "not network").
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest

from cos.core.models import ReductionSpec

_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "", None}
_real_getaddrinfo = socket.getaddrinfo


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host not in _ALLOWED_HOSTS:
        raise RuntimeError(
            f"Blocked network access to {host!r} during tests. Mock with respx, "
            "or mark the test with @pytest.mark.network."
        )
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)


@pytest.fixture
def window() -> tuple[datetime, datetime]:
    return datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)


@pytest.fixture
def basin_spec() -> ReductionSpec:
    """A medium/large basin: bbox + centroid + area for the reduction path."""
    return ReductionSpec(
        domain_name="testbasin",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,
    )
