"""SSEBop ET connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory SSEBop-like NetCDF and reduces it; no network, no
auth. Proves the gridded -> canonical-series path for the ``et`` kind: mm/day
pass-through, basin_mean vs nearest_cell policy, half-open window trim, and
nodata/negative masking -> QualityFlag.MISSING.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ssebop_et import NODATA, SSEBopETConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ssebop_nc(tmp_path):
    """A synthetic SSEBop-like NetCDF: et (mm/day) over a small grid.

    Four monthly timesteps; one timestep is all-nodata (-> MISSING), the grid
    spans the Bow-at-Banff-ish bbox. Values chosen so basin_mean is exact.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-07-15", "2020-08-15", "2021-06-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 4.0          # 2020-06: uniform 4 mm/day -> mean 4.0
    data[1] = NODATA       # 2020-07: all nodata -> MISSING
    data[2] = 6.0          # 2020-08: uniform 6 mm/day -> mean 6.0
    data[3] = 3.0          # 2021-06: outside the default test window
    ds = xr.Dataset(
        {"et": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ssebop_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_mm_per_day_passthrough(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == "mm/day"  # KIND_UNITS[ET]
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "ssebop_et:domain:bow"

    by_month = {p.timestamp.month: p for p in series.points}
    # mm/day is canonical -> exact pass-through, no scaling.
    assert by_month[6].value == pytest.approx(4.0, abs=1e-9)
    assert by_month[6].quality == QualityFlag.GOOD
    assert by_month[8].value == pytest.approx(6.0, abs=1e-9)
    # All-nodata timestep masks to MISSING / None.
    assert by_month[7].value is None
    assert by_month[7].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=300.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ssebop_et:cell:")
    # nearest cell to (51,-115) is the center cell; same uniform values.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[6] == pytest.approx(4.0, abs=1e-9)


def test_window_trim_half_open(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    # Half-open [2020-06-01, 2020-08-15): includes 06-15 & 07-15, excludes 08-15.
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) in months
    assert (2020, 8) not in months  # 08-15 == end -> excluded (half-open)
    assert (2021, 6) not in months


def test_negatives_masked_to_missing(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    # All-negative layer -> every cell masked -> MISSING.
    data = np.full((1, 2, 2), -2.0, dtype="float64")
    ds = xr.Dataset(
        {"et": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "neg.nc"
    ds.to_netcdf(path)

    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
        centroid=(50.5, -115.5), area_km2=8000.0,
    )
    series = conn.reduce_file(
        path, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality == QualityFlag.MISSING


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0),
    )
    with pytest.raises(Exception, match="NetCDF path"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_list_sites_reduced_region(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "ssebop_et:domain:bow"
