"""Connector lifecycle validation and runtime behavior."""

from datetime import datetime

import pytest

from cos.connectors.base import BaseObservationConnector, ConnectorLifecycle
from cos.core.exceptions import ConnectorError
from cos.core.models import ObservationKind, ReductionSpec, SiteRef


def test_invalid_lifecycle_is_rejected_at_class_definition() -> None:
    with pytest.raises(TypeError, match="lifecycle must be"):

        class InvalidConnector(BaseObservationConnector):
            lifecycle = "broken"


def test_retired_connector_requires_retirement_note() -> None:
    with pytest.raises(TypeError, match="retirement_note"):

        class MissingMetadataConnector(BaseObservationConnector):
            lifecycle = ConnectorLifecycle.RETIRED


class RetiredConnector(BaseObservationConnector):
    slug = "retired"
    display_name = "Retired"
    kind = ObservationKind.SWE
    structural_class = "point_network"
    lifecycle = ConnectorLifecycle.RETIRED
    replacement = "replacement"
    retirement_note = "removed upstream"
    base_url = "https://example.invalid"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        return []

    async def fetch_series(
        self, spec: ReductionSpec, start: datetime, end: datetime
    ) -> list:
        return []


@pytest.mark.asyncio
async def test_retired_connector_is_blocked_before_network_access() -> None:
    connector = RetiredConnector()
    with pytest.raises(ConnectorError, match="removed upstream.*replacement"):
        await connector.__aenter__()
    assert connector._client is None
