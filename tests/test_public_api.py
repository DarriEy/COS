from datetime import UTC, datetime

import pytest

import cos
from cos.core.models import ReductionSpec


@pytest.mark.asyncio
async def test_fetch_resolves_single_token_without_overriding_explicit_config(monkeypatch):
    seen = []

    class FakeConnector:
        auth = frozenset({"openet"})

        def __init__(self, config=None):
            seen.append(config)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def fetch_series(self, spec, start, end):
            return []

    monkeypatch.setattr(cos, "discover", lambda: None)
    monkeypatch.setattr(cos, "get_connector", lambda slug: FakeConnector)
    monkeypatch.setattr(cos, "resolve_credentials", lambda auth: {"openet": {"token": "from-env"}})
    window = (datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 2, 1, tzinfo=UTC))
    await cos.fetch_series("openet", ReductionSpec(), *window)
    await cos.fetch_series("openet", ReductionSpec(), *window, config={"token": "explicit"})
    assert seen[0]["token"] == "from-env"
    assert seen[0]["credentials"]["openet"]["token"] == "from-env"
    assert seen[1]["token"] == "explicit"


@pytest.mark.asyncio
async def test_discover_sites_reports_query_and_connector_provenance(monkeypatch):
    from cos.core.models import SiteRef

    class FakeConnector:
        auth = frozenset()

        def __init__(self, config=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def list_sites(self, spec):
            return [SiteRef(kind="station", site_id="fake:1")]

    monkeypatch.setattr(cos, "discover", lambda: None)
    monkeypatch.setattr(cos, "get_connector", lambda slug: FakeConnector)
    result = await cos.discover_sites("fake", ReductionSpec(bbox=(1, 2, 3, 4), options={"max_sites": 5}))
    assert result.query == "bbox"
    assert result.sites[0].site_id == "fake:1"
    assert result.source_info["connector"].endswith("FakeConnector")
