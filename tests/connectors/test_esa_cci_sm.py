"""ESA CCI soil-moisture connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory ESA-CCI-like NetCDF (variable ``sm``) and reduces
it; no network, no auth. This proves the architecture-critical gridded ->
canonical-series path for the merged volumetric soil-moisture product: identity
unit (m3/m3), inclusive [0, 1] mask (the native handler's
``sm.where((sm >= 0) & (sm <= 1))``), basin-mean vs nearest-cell reduction, and
half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.esa_cci_sm import ESACCISMConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def esa_cci_nc(tmp_path):
    """A synthetic ESA-CCI-like NetCDF: ``sm`` (m3/m3) over a small grid.

    Four timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). The last
    timestep is entirely NaN (fill) so it must reduce to MISSING; one cell in an
    otherwise-valid layer is out of range (>1) to exercise the inclusive clip
    mask. A boundary cell at exactly 0.0 stays valid (inclusive lower bound).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = 0.20          # uniform valid layer -> mean 0.20
    data[1] = 0.40          # uniform valid layer
    data[1, 0, 0] = 2.0     # out of range (>1) -> masked, mean stays 0.40
    data[2] = 0.30
    data[3] = np.nan        # entirely fill -> MISSING
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # canonical, identity-converted from source m3/m3
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "esa_cci_sm:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 0.20 layer -> basin mean 0.20 (no scaling applied).
    assert by_day[15].value == pytest.approx(0.20, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Out-of-range cell masked; remaining cells are 0.40 -> mean unchanged.
    assert by_day[16].value == pytest.approx(0.40, abs=1e-9)


def test_fill_reduces_to_missing(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-NaN (fill) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_inclusive_zero_boundary_kept(tmp_path):
    """A cell at exactly sm == 0.0 is kept (inclusive lower bound), not masked."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    data = np.zeros((1, 2, 2))  # all 0.0 -> inclusive [0,1] keeps them
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_zero.nc"
    ds.to_netcdf(path)

    conn = ESACCISMConnector()
    series = conn.reduce_file(
        path,
        ReductionSpec(domain_name="b", bbox=(50.0, -116.0, 51.0, -115.0), area_km2=8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(0.0, abs=1e-9)
    assert series.points[0].quality == QualityFlag.GOOD


def test_small_basin_defaults_to_nearest_cell(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("esa_cci_sm:cell:")


def test_window_trim_half_open(esa_cci_nc):
    conn = ESACCISMConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = ESACCISMConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
