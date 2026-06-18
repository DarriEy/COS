"""ISMN soil-moisture connector — hermetic test of the point-network path.

Parses a small synthetic ISMN station CSV; no network, no credentials. Proves
the architecture-critical station → canonical-series path for volumetric soil
moisture: m³/m³ identity unit, the native percent→fraction rule, blank→MISSING,
and the half-open UTC window trim. Mirrors test_snotel.py / test_smap_sm.py.
"""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.ismn_sm import ISMNSoilMoistureConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction

# Already-volumetric ISMN station CSV (m³/m³): header, then DateTime,soil_moisture
# rows. One blank row -> MISSING; 2021-06-01 is outside the [start, end) window.
MOCK_CSV_VOLUMETRIC = """\
# ISMN station MAQU:CST05
DateTime,soil_moisture
2020-01-01,0.20
2020-01-02,0.35
2020-01-03,
2021-06-01,0.10
"""

# Percent-saturation variant (values > 1.5): the native rule divides by 100.
MOCK_CSV_PERCENT = """\
DateTime,soil_moisture
2020-01-01,20.0
2020-01-02,35.0
"""


def test_parse_volumetric_identity_and_window():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_VOLUMETRIC,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2021-06-01 outside half-open [start, end); the blank row is MISSING.
    assert "2021-06-01" not in by_date
    # Volumetric m³/m³ passes through unchanged (identity conversion).
    assert by_date["2020-01-01"].value == pytest.approx(0.20)
    assert by_date["2020-01-01"].quality == QualityFlag.GOOD
    assert by_date["2020-01-02"].value == pytest.approx(0.35)
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality == QualityFlag.MISSING


def test_parse_percent_divides_by_100():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_PERCENT,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # Series exceeds the 1.5 ceiling -> read as percent saturation, /100 -> m³/m³.
    assert by_date["2020-01-01"].value == pytest.approx(0.20)
    assert by_date["2020-01-02"].value == pytest.approx(0.35)


def test_window_trim_half_open():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_VOLUMETRIC,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in points}
    # Half-open [01-01, 01-03): includes 01-01 and 01-02, excludes 01-03.
    assert days == {1, 2}


def test_missing_sm_column_raises():
    with pytest.raises(ConnectorError):
        ISMNSoilMoistureConnector.parse_station_csv(
            "# c\nDateOnly\n2020-01-01\n",
            datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


def test_namespaced_and_bare_ids_resolve():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("ismn:MAQU:CST05", "OZNET:Y1"))
    ids = conn._station_ids(spec)
    # leading "ismn:" namespace stripped; embedded ":" in the station id kept.
    assert ids == ["MAQU:CST05", "OZNET:Y1"]


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("MAQU:CST05", "ismn:OZNET:Y1"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"ismn:MAQU:CST05", "ismn:OZNET:Y1"}
    assert all(s.kind == "station" for s in sites)


@pytest.mark.asyncio
async def test_fetch_series_without_station_ids_errors():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x")
    with pytest.raises(ConnectorError, match="station ids"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://ismn\.earth/.*").mock(
        return_value=httpx.Response(200, text=MOCK_CSV_VOLUMETRIC)
    )
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="maqu", station_ids=("ismn:MAQU:CST05",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.SOIL_MOISTURE
    assert s.unit == "m3/m3"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "ismn:MAQU:CST05"
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ismn():
    """LIVE smoke against the real ISMN dataviewer (needs registered creds).

    Run with: pytest -m network tests/connectors/test_ismn_sm.py -k live
    """
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="maqu", station_ids=("MAQU:CST05",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2015, 3, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m3/m3"
