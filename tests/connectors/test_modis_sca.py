"""MODIS SCA connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory MODIS-snow NetCDF (NDSI percent + flag bytes) and
reduces it; no network, no auth. Proves the percent→fraction canonicalization,
the byte-flag masking, the basin-mean / nearest-cell reductions, and half-open
UTC window-trim — the parts that mirror the native ``modis_snow`` handler.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_sca import MODISSCAConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def modis_nc(tmp_path):
    """Synthetic MODIS SCA NetCDF: NDSI_Snow_Cover (percent + flag bytes)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    # t0: uniform 50% NDSI -> fraction 0.5 everywhere.
    data[0] = 50.0
    # t1: uniform 100% NDSI -> 1.0.
    data[1] = 100.0
    # t2: mix of valid 80% and cloud flag (250) -> masked cells ignored,
    #     mean over valid = 0.8.
    data[2] = 80.0
    data[2, 0, 0] = 250.0  # cloud -> NaN
    data[2, 1, 1] = 255.0  # fill -> NaN
    # t3: fully cloud/fill -> all masked -> NaN -> MISSING.
    data[3] = 200.0
    ds = xr.Dataset(
        {"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_synth.nc"
    ds.to_netcdf(path)
    return path


def _by_month(series):
    return {p.timestamp.month: p for p in series.points}


def test_reduce_file_basin_mean_percent_to_fraction(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SNOW_COVER
    assert series.unit == "fraction"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"

    by_month = _by_month(series)
    # 50% -> 0.5
    assert by_month[1].value == pytest.approx(0.5, abs=1e-9)
    assert by_month[1].quality == QualityFlag.GOOD
    # 100% -> 1.0
    assert by_month[2].value == pytest.approx(1.0, abs=1e-9)
    # 80% valid cells (cloud/fill masked out) -> mean still 0.8
    assert by_month[3].value == pytest.approx(0.8, abs=1e-9)
    # all-flag timestep -> MISSING, value None
    assert by_month[4].value is None
    assert by_month[4].quality == QualityFlag.MISSING


def test_fraction_never_exceeds_one(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    vals = [p.value for p in series.points if p.value is not None]
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_small_basin_defaults_to_nearest_cell(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_sca:cell:")
    # nearest cell (51,-115) at t0 = 50% -> 0.5
    by_month = _by_month(series)
    assert by_month[1].value == pytest.approx(0.5, abs=1e-9)


def test_window_trim_half_open(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2020-02-01, 2020-04-15): includes 02-15 and 03-15, excludes 04-15.
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 2, 1, tzinfo=UTC), datetime(2020, 4, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert 2 in months
    assert 3 in months
    assert 1 not in months  # before window
    assert 4 not in months  # end is exclusive


def test_list_sites_one_region(modis_nc):
    import asyncio

    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    sites = asyncio.run(conn.list_sites(spec))
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "modis_sca:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC),
                                datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_earthdata_fetch_placeholder():
    pytest.skip("Live Earthdata MODIS fetch not wired; reduce path is the proven part.")
