"""SWOT WSE connector — hermetic test of the per-reach Hydrocron path.

SWOT WSE has NO SYMFLUENCE native, so this is *spec-validated*: the assertions
reproduce the published Hydrocron product spec on a synthetic inline fixture —
the metres unit (identity scale), the -999999999999.0 fill sentinel, the
``no_data`` time_str placeholder, the half-open UTC window, and the gridded
reduction — with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.swot_wse import (
    SOURCE_WSE_SCALE,
    SWOT_FILL_VALUE,
    SWOTWaterLevelConnector,
)
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# Hydrocron CSV: header, then valid rows, a fill-sentinel row, a 'no_data' row,
# and an out-of-window row. wse is in metres (canonical water_level unit).
MOCK_CSV = """\
reach_id,time_str,wse,wse_units
78340600051,2024-01-05T12:00:00Z,1024.50,m
78340600051,2024-01-12T12:00:00Z,1025.25,m
78340600051,2024-01-19T12:00:00Z,-999999999999.0,m
78340600051,no_data,-999999999999.0,m
78340600051,2025-06-01T12:00:00Z,1030.00,m
"""


def test_parse_metres_identity_scale_and_window():
    """Spec: wse is metres -> canonical 'm' via identity scale; window is half-open."""
    assert SOURCE_WSE_SCALE == 1.0  # documented spec contract: metres == canonical
    points = SWOTWaterLevelConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2025-06-01 is outside [start, end); 'no_data' row is dropped (no timestamp).
    assert "2025-06-01" not in by_date
    assert "no_data" not in by_date
    # Metres pass through unchanged (identity scale).
    assert by_date["2024-01-05"].value == pytest.approx(1024.50)
    assert by_date["2024-01-05"].quality.value == "good"
    assert by_date["2024-01-12"].value == pytest.approx(1025.25)


def test_fill_sentinel_maps_to_missing():
    """Spec: -999999999999.0 is the SWOT no-observation fill -> MISSING/None."""
    assert SWOT_FILL_VALUE == -999999999999.0
    points = SWOTWaterLevelConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2024-01-19"].value is None
    assert by_date["2024-01-19"].quality.value == "missing"


def test_no_data_time_str_is_dropped():
    """A 'no_data' time_str placeholder has no anchor timestamp and is skipped."""
    points = SWOTWaterLevelConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    # 2 valid + 1 fill (still inside window) = 3 anchored points; the 'no_data'
    # row and the out-of-window row are absent.
    assert len(points) == 3


def test_out_of_range_wse_masked_to_missing():
    """Spec: a finite wse outside the physical band is treated as fill -> MISSING."""
    csv_text = (
        "reach_id,time_str,wse\n"
        "1,2024-02-01T00:00:00Z,50000.0\n"   # absurdly high -> out of VALID_WSE_RANGE
        "1,2024-02-02T00:00:00Z,42.0\n"      # plausible -> good
    )
    points = SWOTWaterLevelConnector.parse_timeseries(
        csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2024-02-01"].value is None
    assert by_date["2024-02-01"].quality.value == "missing"
    assert by_date["2024-02-02"].value == pytest.approx(42.0)


def test_non_metre_units_rejected():
    """Spec contract: a wse_units other than metres must not be silently mis-scaled."""
    csv_text = (
        "reach_id,time_str,wse,wse_units\n"
        "1,2024-02-01T00:00:00Z,1024.5,ft\n"
    )
    with pytest.raises(DataFormatError):
        SWOTWaterLevelConnector.parse_timeseries(
            csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_missing_required_column_raises():
    with pytest.raises(DataFormatError):
        SWOTWaterLevelConnector.parse_timeseries(
            "reach_id,wse\n1,1024.5\n",  # no time_str column
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_json_envelope_is_unwrapped():
    """Hydrocron may wrap the CSV in {'results': {'csv': ...}}; parser unwraps it."""
    import json

    body = json.dumps({"status": "200 OK", "results": {"csv": MOCK_CSV}})
    points = SWOTWaterLevelConnector.parse_timeseries(
        body, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert len(points) == 3
    assert points[0].value == pytest.approx(1024.50)


def test_column_order_independent():
    """Columns are matched by header name, not position."""
    csv_text = (
        "time_str,wse,reach_id\n"
        "2024-03-01T00:00:00Z,7.5,1\n"
    )
    points = SWOTWaterLevelConnector.parse_timeseries(
        csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert len(points) == 1
    assert points[0].value == pytest.approx(7.5)


@pytest.mark.asyncio
async def test_fetch_series_builds_reach_series(monkeypatch):
    conn = SWOTWaterLevelConnector()

    async def _fake_fetch(self, feature, feature_id, start, end):  # noqa: ANN001
        assert feature == "Reach"
        assert feature_id == "78340600051"
        return MOCK_CSV

    monkeypatch.setattr(SWOTWaterLevelConnector, "_fetch_timeseries", _fake_fetch)
    spec = ReductionSpec(domain_name="amazon", station_ids=("swot:78340600051",))
    series_list = await conn.fetch_series(
        spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.WATER_LEVEL
    assert s.unit == "m"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "swot:78340600051"
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = SWOTWaterLevelConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("78340600051", "swot:99999999999"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"swot:78340600051", "swot:99999999999"}
    assert all(s.kind == "station" for s in sites)


# -- gridded path (synthetic WSE NetCDF) -------------------------------------


@pytest.fixture
def swot_grid_nc(tmp_path):
    """A synthetic SWOT-like WSE NetCDF (metres), with a fill cell to be masked."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2024-06-15", "2024-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.full((2, 3, 3), 1000.0)
    data[0, 0, 0] = SWOT_FILL_VALUE  # one fill cell -> masked to NaN before mean
    ds = xr.Dataset(
        {"wse": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "swot_wse_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_metres(swot_grid_nc):
    conn = SWOTWaterLevelConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        swot_grid_nc, spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.WATER_LEVEL
    assert series.unit == "m"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # All valid cells are 1000 m (the lone fill cell is masked out) -> mean 1000.
    for p in series.points:
        assert p.value == pytest.approx(1000.0, abs=1e-6)
        assert p.quality.value == "good"


def test_reduce_file_window_trim_half_open(swot_grid_nc):
    conn = SWOTWaterLevelConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2024-06-01, 2024-07-15): includes 06-15, excludes 07-15.
    series = conn.reduce_file(
        swot_grid_nc, spec,
        datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2024, 6) in months
    assert (2024, 7) not in months


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_swot():
    """LIVE smoke against the real anonymous Hydrocron endpoint.

    Run with: pytest -m network tests/connectors/test_swot_wse.py -k live
    """
    conn = SWOTWaterLevelConnector()
    spec = ReductionSpec(domain_name="amazon", station_ids=("78340600051",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
