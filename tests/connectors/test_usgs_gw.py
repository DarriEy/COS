"""USGS NWIS groundwater connector — hermetic test of the point/station path."""

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.usgs_gw import FEET_TO_METERS, USGSGroundwaterConnector
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# Minimal NWIS waterservices JSON: one 72019 (feet) series with a good value,
# a blank, the -999999 fill, and an out-of-window row.
MOCK_NWIS = json.dumps(
    {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "parameterCode": "72019",
                        "variableName": "Depth to water level, feet below land surface",
                        "unit": {"unitCode": "ft"},
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2020-01-01T00:00:00.000-05:00", "value": "10.0"},
                                {"dateTime": "2020-01-02T00:00:00.000-05:00", "value": ""},
                                {"dateTime": "2020-01-03T00:00:00.000-05:00", "value": "-999999"},
                                {"dateTime": "2021-06-01T00:00:00.000-05:00", "value": "5.0"},
                            ]
                        }
                    ],
                }
            ]
        }
    }
)


def test_parse_feet_to_metres_and_window():
    points = USGSGroundwaterConnector.parse_series(
        MOCK_NWIS,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2021-06-01 is outside [start, end).
    assert "2021-06-01" not in by_date
    # 10.0 ft -> metres at the connector boundary.
    good = by_date["2020-01-01"]
    assert good.value == pytest.approx(10.0 * FEET_TO_METERS)
    assert good.quality.value == "good"
    # blank and -999999 fill both map to MISSING.
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality.value == "missing"


def test_window_is_half_open():
    # the start-of-day local timestamp 2020-01-01T00:00-05:00 is 05:00 UTC, so
    # an end exactly at 2020-01-01T05:00Z must exclude it (half-open).
    pts = USGSGroundwaterConnector.parse_series(
        MOCK_NWIS,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, 5, tzinfo=UTC),
    )
    assert pts == []


def test_unit_already_metres_passes_through():
    payload = json.dumps(
        {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "parameterCode": "72019",
                            "variableName": "Depth to water level",
                            "unit": {"unitCode": "m"},
                        },
                        "values": [
                            {"value": [{"dateTime": "2020-01-01T00:00:00Z", "value": "3.5"}]}
                        ],
                    }
                ]
            }
        }
    )
    pts = USGSGroundwaterConnector.parse_series(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert len(pts) == 1
    assert pts[0].value == pytest.approx(3.5)  # no feet conversion


def test_non_gw_variable_is_skipped():
    payload = json.dumps(
        {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "parameterCode": "00060",
                            "variableName": "Discharge, cubic feet per second",
                            "unit": {"unitCode": "ft3/s"},
                        },
                        "values": [
                            {"value": [{"dateTime": "2020-06-01T00:00:00Z", "value": "100.0"}]}
                        ],
                    }
                ]
            }
        }
    )
    pts = USGSGroundwaterConnector.parse_series(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert pts == []


def test_empty_timeseries_yields_no_points():
    pts = USGSGroundwaterConnector.parse_series(
        json.dumps({"value": {"timeSeries": []}}),
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert pts == []


def test_invalid_json_raises_data_format_error():
    with pytest.raises(DataFormatError):
        USGSGroundwaterConnector.parse_series(
            "{not json", datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_station_id_normalisation():
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(
        domain_name="x",
        station_ids=("385854121023801", "usgs:123456", "USGS-987654"),
    )
    assert conn._station_ids(spec) == ["385854121023801", "123456", "987654"]


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("385854121023801",))
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].site_id == "usgs:385854121023801"
    assert sites[0].kind == "station"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/.*").mock(
        return_value=httpx.Response(200, text=MOCK_NWIS)
    )
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(domain_name="aquifer", station_ids=("385854121023801",))
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
    assert s.site.site_id == "usgs:385854121023801"
    # one good value within the window; the two MISSING rows carry value=None.
    assert len([p for p in s.points if p.value is not None]) == 1


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_usgs_gw():
    """LIVE smoke against the real anonymous USGS NWIS endpoint.

    Run with: pytest -m network tests/connectors/test_usgs_gw.py -k live
    """
    conn = USGSGroundwaterConnector()
    # a long-running NWIS groundwater observation well.
    spec = ReductionSpec(domain_name="live", station_ids=("385854121023801",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
