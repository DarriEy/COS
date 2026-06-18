"""MSWEP precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MSWEP-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded → canonical-series path for a
merged precipitation product: identity unit (mm), non-finite (fill) masking,
basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.mswep_precip import MSWEPPrecipConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def mswep_nc(tmp_path):
    """A synthetic MSWEP-like NetCDF: precipitation (mm) over a small grid.

    Four daily timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). The
    last timestep is entirely NaN (fill) so it must reduce to MISSING; one cell
    in an otherwise-uniform layer is NaN to exercise the finite-cell masking.
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
    data[0] = 5.0           # uniform valid layer -> mean 5.0 mm
    data[1] = 10.0          # uniform valid layer
    data[1, 0, 0] = np.nan  # one masked cell -> mean stays 10.0
    data[2] = 2.0
    data[3] = np.nan        # entirely fill -> MISSING
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "mswep_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical, identity-converted from source mm
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "mswep:domain:bow"
    assert series.provider == "mswep_precip"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5.0 mm layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Masked NaN cell skipped; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    assert by_day[16].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-NaN (fill) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("mswep:cell:")
    # Nearest cell to centroid (51, -115) -> uniform layers, exact values.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[17].value == pytest.approx(2.0, abs=1e-9)


def test_window_trim_half_open(mswep_nc):
    conn = MSWEPPrecipConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


def test_connector_metadata():
    conn = MSWEPPrecipConnector()
    assert conn.slug == "mswep_precip"
    assert conn.kind == ObservationKind.PRECIPITATION
    assert conn.structural_class == "gridded"
    assert conn.auth == frozenset({"gloh2o"})


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MSWEPPrecipConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
def test_live_placeholder():
    """Live GloH2O fetch is network/auth-gated; reduction is the proven path."""
    pytest.skip("live MSWEP fetch requires GloH2O credentials + rclone")
