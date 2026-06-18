"""SNODAS SWE connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory SNODAS-like NetCDF (SWE in metres) and reduces it;
no network, no auth. Proves m→mm canonicalization, half-open window trim,
negative clipping, and both reduction policies.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.snodas_swe import SNODASSWEConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def snodas_nc(tmp_path):
    """A synthetic SNODAS-like NetCDF: swe (metres) over a small daily grid.

    4 daily timesteps, 3x3 grid. Day 0 = 0.10 m, day 1 = 0.25 m, day 2 = 0.40 m,
    day 3 carries one slightly-negative cell (assimilation artifact) to test the
    non-negative clip. One cell is NaN on day 2 to exercise skipna in basin_mean.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 0.10
    data[1] = 0.25
    data[2] = 0.40
    data[2, 0, 0] = np.nan       # missing cell -> skipna in basin_mean
    data[3] = -0.001             # tiny negative everywhere -> clip to 0
    ds = xr.Dataset(
        {"swe": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "snodas_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_m_to_mm(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SWE
    assert series.unit == "mm"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    by_day = {p.timestamp.day: p for p in series.points}
    # 0.10 m -> 100 mm, 0.25 m -> 250 mm; both uniform so basin-mean is exact.
    assert by_day[1].value == pytest.approx(100.0, abs=1e-6)
    assert by_day[2].value == pytest.approx(250.0, abs=1e-6)
    # day 3 uniform 0.40 m except one NaN cell -> skipna mean still 400 mm.
    assert by_day[3].value == pytest.approx(400.0, abs=1e-6)


def test_negative_swe_clipped_to_zero(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[4].value == 0.0
    assert by_day[4].quality == QualityFlag.ESTIMATED


def test_small_basin_defaults_to_nearest_cell(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=10.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("snodas_swe:cell:")
    by_day = {p.timestamp.day: p for p in series.points}
    # nearest cell to centroid (51, -115) is the center cell = 0.25 m -> 250 mm.
    assert by_day[2].value == pytest.approx(250.0, abs=1e-6)


def test_window_trim_half_open(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2022-01-02, 2022-01-04): includes 01-02, 01-03; excludes 01-04.
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 2, tzinfo=UTC), datetime(2022, 1, 4, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {2, 3}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2022, 1, 1, tzinfo=UTC),
                                datetime(2022, 2, 1, tzinfo=UTC))
