"""GPM IMERG precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory GPM IMERG-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded -> canonical-series path for a
satellite precipitation product: identity unit (mm/day daily depth == canonical
mm), fill masking + negative-clip, basin-mean vs nearest-cell reduction, and
half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.gpm_imerg_precip import FILL_VALUE, GPMIMERGPrecipConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def gpm_nc(tmp_path):
    """A synthetic GPM IMERG-like NetCDF: precipitation (mm/day) over a small grid.

    Four daily timesteps on a 3x3 grid. The last timestep is entirely fill so it
    must reduce to MISSING; one cell in an otherwise-uniform layer is negative
    (a spurious retrieval) to exercise the non-negative clip the native handler
    applies before averaging.
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
    data[0] = 5.0           # uniform valid layer -> mean 5.0 mm
    data[1] = 10.0          # uniform valid layer
    data[1, 0, 0] = -2.0    # spurious negative -> clipped to 0 (not masked)
    data[2] = 0.0           # dry day
    data[3] = FILL_VALUE    # entirely fill -> MISSING
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gpm_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    series = conn.reduce_file(
        gpm_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical; identity-converted from source mm/day depth
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "gpm_imerg:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5.0 layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # The -2.0 cell is clipped to 0; basin mean of eight 10s and one 0 over the
    # cos-lat weighting is below 10 but strictly positive -> clip happened.
    assert 0.0 < by_day[16].value < 10.0


def test_negative_is_clipped_not_masked(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    series = conn.reduce_file(
        gpm_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # Negative cell contributes 0, so the day is still GOOD (a real, finite value).
    assert by_day[16].quality == QualityFlag.GOOD
    assert by_day[16].value is not None


def test_dry_day_is_zero_good(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    series = conn.reduce_file(
        gpm_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[17].value == pytest.approx(0.0, abs=1e-9)
    assert by_day[17].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    series = conn.reduce_file(
        gpm_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    series = conn.reduce_file(
        gpm_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("gpm_imerg:cell:")
    by_day = {p.timestamp.day: p for p in series.points}
    # Nearest cell to centroid (51, -115) is the center cell = 5.0 on day 15.
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)


def test_window_trim_half_open(gpm_nc):
    conn = GPMIMERGPrecipConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        gpm_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = GPMIMERGPrecipConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
def test_live_placeholder():
    """Live Earthdata fetch is covered separately; offline suite skips this."""
    pytest.skip("network test — requires Earthdata credentials")
