"""CHIRPS precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory CHIRPS-like NetCDF and reduces it; no network, no
auth (CHIRPS is anonymous). This proves the architecture-critical gridded ->
canonical-series path for a rainfall product: identity unit (mm daily depth),
fill / negative masking, basin-mean vs nearest-cell reduction, and half-open UTC
window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.chirps_precip import FILL_VALUE, CHIRPSPrecipitationConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def chirps_nc(tmp_path):
    """A synthetic CHIRPS-like NetCDF: precip (mm/day) over a small grid.

    Four daily timesteps on a 3x3 grid. The last timestep is entirely fill
    (-9999) so it must reduce to MISSING; one cell in an otherwise-valid layer is
    negative (a partial no-data) to exercise the ``precip < 0`` mask.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3))
    data[0] = 5.0            # uniform valid layer -> mean 5.0 mm
    data[1] = 10.0           # uniform valid layer
    data[1, 0, 0] = -9999.0  # one fill cell -> masked, mean stays 10.0
    data[2] = 0.0            # dry day -> mean 0.0 (valid, non-negative)
    data[3] = FILL_VALUE     # entirely fill -> MISSING
    ds = xr.Dataset(
        {"precip": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "chirps_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical; CHIRPS mm/day daily depth -> identity
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "chirps:domain:bow"
    assert series.provider == "chirps_precip"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5.0 mm layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # The single fill (-9999) cell is masked; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    # A genuine dry day (0 mm) stays a valid GOOD zero, not masked.
    assert by_day[17].value == pytest.approx(0.0, abs=1e-9)
    assert by_day[17].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("chirps:cell:")
    # nearest cell to centroid (51, -115) is the grid center -> 5.0 on day 15.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)


def test_window_trim_half_open(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = CHIRPSPrecipitationConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_fetch_placeholder():
    """Live UCSB CHG fetch is not wired; reduction is the proven path."""
    pytest.skip("CHIRPS live UCSB download not wired; reduction path is hermetic")
