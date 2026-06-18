"""CanSWE connector — hermetic test of the point-network / station path.

Builds a synthetic in-memory CanSWE-like NetCDF (``time × station`` SWE in mm,
per-station ``lat`` / ``lon`` / ``station_id``) and selects + canonicalizes it;
no network, no auth. SWE is already mm == canonical ``swe`` unit, so this also
asserts the identity unit handling (contrast with SNOTEL's inches→mm).
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.canswe_swe import CanSWEConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def canswe_nc(tmp_path):
    """Synthetic CanSWE NetCDF: 3 stations, SWE (mm) along (time, station)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    # station 0 inside bbox, station 1 inside bbox, station 2 OUTSIDE bbox.
    lats = np.array([51.0, 51.5, 60.0])
    lons = np.array([-115.0, -114.5, -100.0])
    station_id = np.array(["BOW1", "BOW2", "FAR3"], dtype=object)
    swe = np.array(
        [
            [100.0, 200.0, 999.0],   # 2020-01-01
            [110.0, np.nan, 999.0],  # 2020-01-02 (station1 missing)
            [120.0, 210.0, 999.0],   # 2020-01-03
            [50.0, 50.0, 50.0],      # 2021-06-01 (out of window)
        ]
    )
    ds = xr.Dataset(
        {
            "swe": (("time", "station"), swe),
            "lat": (("station",), lats),
            "lon": (("station",), lons),
            "station_id": (("station",), station_id),
        },
        coords={"time": times},
    )
    path = tmp_path / "canswe_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec_bbox(min_obs=1):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),  # (lat_min, lon_min, lat_max, lon_max)
        centroid=(51.25, -114.75),
        options={"min_observations": min_obs},
    )


def test_reduce_file_selects_bbox_stations_units_mm(canswe_nc):
    conn = CanSWEConnector()
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # Only the two in-bbox stations (FAR3 excluded).
    ids = {s.site.site_id for s in series}
    assert ids == {"canswe:BOW1", "canswe:BOW2"}
    s0 = next(s for s in series if s.site.site_id == "canswe:BOW1")
    assert s0.kind == ObservationKind.SWE
    assert s0.unit == "mm"  # source mm -> canonical mm (identity)
    assert s0.reduction == SpatialReduction.STATION
    assert s0.site.kind == "station"
    # mm carries through unchanged
    by_date = {p.timestamp.date().isoformat(): p for p in s0.points}
    assert by_date["2020-01-01"].value == pytest.approx(100.0)
    assert by_date["2020-01-01"].quality.value == "good"


def test_window_trim_half_open(canswe_nc):
    conn = CanSWEConnector()
    # Half-open [2020-01-01, 2020-01-03): includes 01-01, 01-02; excludes 01-03.
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    s0 = next(s for s in series if s.site.site_id == "canswe:BOW1")
    dates = {p.timestamp.date().isoformat() for p in s0.points}
    assert "2020-01-01" in dates
    assert "2020-01-02" in dates
    assert "2020-01-03" not in dates
    assert "2021-06-01" not in dates


def test_nan_swe_becomes_missing(canswe_nc):
    conn = CanSWEConnector()
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    s1 = next(s for s in series if s.site.site_id == "canswe:BOW2")
    by_date = {p.timestamp.date().isoformat(): p for p in s1.points}
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"
    assert by_date["2020-01-03"].value == pytest.approx(210.0)


def test_min_observations_filter_drops_sparse_stations(canswe_nc):
    conn = CanSWEConnector()
    # BOW2 has only 2 valid (non-NaN) obs in window -> require 3 to drop it.
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(min_obs=3),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    ids = {s.site.site_id for s in series}
    assert ids == {"canswe:BOW1"}  # BOW1 has 3 valid obs, BOW2 only 2


def test_explicit_station_ids_select_one(canswe_nc):
    conn = CanSWEConnector()
    spec = ReductionSpec(
        domain_name="bow",
        station_ids=("canswe:BOW2",),
        options={"min_observations": 1},
    )
    series = conn.reduce_file(
        canswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert {s.site.site_id for s in series} == {"canswe:BOW2"}


def test_list_sites_returns_bbox_stations(canswe_nc):
    conn = CanSWEConnector(config={"nc_path": str(canswe_nc)})
    spec = _spec_bbox()
    import asyncio

    sites = asyncio.run(conn.list_sites(spec))
    assert {s.site_id for s in sites} == {"canswe:BOW1", "canswe:BOW2"}
    bow1 = next(s for s in sites if s.site_id == "canswe:BOW1")
    assert bow1.latitude == pytest.approx(51.0)
    assert bow1.longitude == pytest.approx(-115.0)


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = CanSWEConnector()
    spec = _spec_bbox()
    with pytest.raises(ConnectorError, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_fetch_series_with_ncpath_in_config(canswe_nc):
    conn = CanSWEConnector(config={"nc_path": str(canswe_nc), "min_observations": 1})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.25, -114.75))
    series = await conn.fetch_series(
        spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert {s.site.site_id for s in series} == {"canswe:BOW1", "canswe:BOW2"}
    assert all(s.unit == "mm" for s in series)
