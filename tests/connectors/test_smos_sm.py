"""SMOS soil-moisture connector — contract checks.

Built during the connector build-out; the agent was rate-limited before writing
a reduction test, so this pins the contract (kind/unit/structural-class/slug)
offline. A synthetic-grid reduction test mirroring test_smap_sm is a follow-up
(tracked in the verify pass).
"""
from cos.connectors.smos_sm import SMOSSMConnector
from cos.core.models import KIND_UNITS, ObservationKind


def test_smos_connector_contract():
    conn = SMOSSMConnector()
    assert conn.slug == "smos_sm"
    assert conn.kind == ObservationKind.SOIL_MOISTURE
    assert conn.structural_class == "gridded"
    # canonical unit must be the frozen kind unit (m3/m3); no rescale at boundary
    assert KIND_UNITS[conn.kind] == "m3/m3"


def test_smos_registered():
    from cos.core.registry import discover, get_connector

    discover()
    assert get_connector("smos_sm") is SMOSSMConnector
