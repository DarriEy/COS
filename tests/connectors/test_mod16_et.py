"""MOD16 ET connector — hermetic test of the gridded basin-reduction path.

Builds synthetic in-memory MOD16-like NetCDFs and reduces them; no network, no
auth. Proves the gridded -> canonical-series path, the kg/m²/8day -> mm/day unit
boundary, fill-value masking, the nearest-cell small-basin default, and the
pre-reduced ET_basin_mean series path.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.mod16_et import MOD16ETConnector
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    QualityFlag,
    ReductionSpec,
    SpatialReduction,
)


@pytest.fixture
def mod16_daily_nc(tmp_path):
    """Gridded ET already in mm/day (the acquirer's default output)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-10", "2020-06-18", "2020-06-26", "2020-07-04"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3))
    data[0] = 2.0
    data[1] = 3.0
    data[2] = 4.0
    data[3] = 5.0
    ds = xr.Dataset(
        {"ET": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_daily.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mod16_composite_nc(tmp_path):
    """Gridded ET as an 8-day composite (kg/m2/8day) with a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    data = np.full((2, 2, 2), 8.0)  # 8 kg/m2/8day -> 1.0 mm/day
    # inject a fill/special pixel (>= 3276.1) that must be masked
    data[0, 0, 0] = 3276.7
    ds = xr.Dataset(
        {"ET_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET_500m"].attrs["units"] = "kg/m2/8day"
    path = tmp_path / "mod16_composite.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mod16_prereduced_nc(tmp_path):
    """Already basin-reduced ET_basin_mean(time) series in mm/day."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18", "2020-06-26"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"ET_basin_mean": (("time",), np.array([1.5, 2.5, np.nan]))},
        coords={"time": times},
    )
    ds["ET_basin_mean"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_prereduced.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2=8000.0):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_basin_mean_daily_units_canonical(mod16_daily_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_daily_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == KIND_UNITS[ObservationKind.ET] == "mm/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # each timestep is spatially uniform, so basin-mean equals the cell value
    by_month_day = {(p.timestamp.month, p.timestamp.day): p.value for p in series.points}
    assert by_month_day[(6, 10)] == pytest.approx(2.0)
    assert by_month_day[(7, 4)] == pytest.approx(5.0)
    assert all(p.quality == QualityFlag.GOOD for p in series.points)


def test_composite_units_divide_by_eight_and_fill_masked(mod16_composite_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_composite_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    assert series.source_info["source_units"] == "kg/m2/8day"
    # 8 kg/m2/8day / 8 = 1.0 mm/day; the fill pixel was masked, so the basin
    # mean of the remaining 1.0-valued cells is still 1.0.
    for p in series.points:
        assert p.value == pytest.approx(1.0)


def test_small_basin_defaults_to_nearest_cell(mod16_daily_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_daily_nc, _spec(area_km2=500.0),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("mod16_et:cell:")


def test_window_trim_half_open(mod16_daily_nc):
    conn = MOD16ETConnector()
    # [2020-06-18, 2020-07-04): includes 06-18 and 06-26, excludes 06-10 and 07-04.
    series = conn.reduce_file(
        mod16_daily_nc, _spec(),
        datetime(2020, 6, 18, tzinfo=UTC), datetime(2020, 7, 4, tzinfo=UTC),
    )
    days = {(p.timestamp.month, p.timestamp.day) for p in series.points}
    assert (6, 18) in days
    assert (6, 26) in days
    assert (6, 10) not in days
    assert (7, 4) not in days  # half-open excludes the end


def test_prereduced_series_path_and_missing(mod16_prereduced_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_prereduced_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.unit == "mm/day"
    vals = {(p.timestamp.day): (p.value, p.quality) for p in series.points}
    assert vals[10][0] == pytest.approx(1.5)
    assert vals[10][1] == QualityFlag.GOOD
    # NaN timestep -> MISSING with None value
    assert vals[26][0] is None
    assert vals[26][1] == QualityFlag.MISSING


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MOD16ETConnector()
    spec = _spec()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region(mod16_daily_nc):
    conn = MOD16ETConnector()
    sites = await conn.list_sites(_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "mod16_et:domain:bow"
