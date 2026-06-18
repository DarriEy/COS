"""GGMN groundwater connector — hermetic test of the point-network / station path."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.ggmn_gw import GGMNGroundwaterConnector
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# GGMN WellLevelMeasurement/list payload: each row carries an HTML fragment with
# hidden inputs name="time" (ISO timestamp) and name="value_value" (metres).
def _row(time: str, value: str) -> dict:
    return {
        "html": (
            f'<form><input type="hidden" name="time" value="{time}">'
            f'<input type="hidden" name="value_value" value="{value}"></form>'
        )
    }


MOCK_PAYLOAD = {
    "data": [
        _row("2020-01-01T00:00:00", "12.5"),
        _row("2020-06-15T00:00:00", "13.0"),
        _row("2020-03-01T00:00:00", ""),       # empty -> MISSING
        _row("2020-04-01T00:00:00", "n/a"),    # non-numeric -> MISSING
        _row("2021-06-01T00:00:00", "9.0"),    # outside [start, end)
    ]
}


def test_parse_measurements_units_window_and_quality():
    points = GGMNGroundwaterConnector.parse_measurements(
        MOCK_PAYLOAD,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # window-trim: 2021-06-01 dropped (half-open [start, end))
    assert "2021-06-01" not in by_date
    # units: GGMN metres == canonical groundwater metres (identity, no scaling)
    assert by_date["2020-01-01"].value == pytest.approx(12.5)
    assert by_date["2020-01-01"].quality.value == "good"
    assert by_date["2020-06-15"].value == pytest.approx(13.0)
    # fill / non-numeric -> MISSING
    assert by_date["2020-03-01"].value is None
    assert by_date["2020-03-01"].quality.value == "missing"
    assert by_date["2020-04-01"].value is None
    assert by_date["2020-04-01"].quality.value == "missing"
    # sorted ascending
    assert [p.timestamp for p in points] == sorted(p.timestamp for p in points)


def test_parse_measurements_empty_data():
    assert GGMNGroundwaterConnector.parse_measurements(
        {"data": []}, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    ) == []


def test_parse_measurements_non_mapping_raises():
    with pytest.raises(DataFormatError):
        GGMNGroundwaterConnector.parse_measurements(
            [], datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_z_suffix_timestamp_parsed_as_utc():
    payload = {"data": [_row("2020-02-02T06:00:00Z", "5.0")]}
    points = GGMNGroundwaterConnector.parse_measurements(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert len(points) == 1
    assert points[0].timestamp == datetime(2020, 2, 2, 6, 0, tzinfo=UTC)
    assert points[0].value == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = GGMNGroundwaterConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("4242", "ggmn_gw:4343"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"ggmn_gw:4242", "ggmn_gw:4343"}
    assert all(s.kind == "station" for s in sites)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://ggis\.un-igrac\.org/groundwater/record/.*").mock(
        return_value=httpx.Response(200, json=MOCK_PAYLOAD)
    )
    conn = GGMNGroundwaterConnector()
    spec = ReductionSpec(domain_name="aquifer", station_ids=("4242",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.GROUNDWATER
    assert s.unit == "m"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "ggmn_gw:4242"
    # two GOOD points in-window (12.5, 13.0), two MISSING, one trimmed
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ggmn():
    """LIVE smoke against the real anonymous IGRAC GGMN endpoint.

    Run with: pytest -m network tests/connectors/test_ggmn_gw.py -k live
    """
    conn = GGMNGroundwaterConnector()
    # A small bbox somewhere with known GGMN coverage; discovery-driven.
    spec = ReductionSpec(domain_name="nl", bbox=(51.0, 4.0, 53.0, 7.0))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2010, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
