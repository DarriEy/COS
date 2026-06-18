"""Daymet precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory Daymet-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded -> canonical-series path for a
daily precipitation product: identity unit (mm/day daily total -> canonical mm),
fill masking, basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.daymet_precip import FILL_VALUE, DaymetPrecipitationConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def daymet_nc(tmp_path):
    """A synthetic Daymet-like NetCDF: prcp (mm/day) over a small grid.

    Four daily timesteps on a 3x3 grid in North America. The last timestep is
    entirely the native missing value (-9999) so it must reduce to MISSING; one
    cell in an otherwise-uniform layer is also fill to confirm masked cells are
    skipped by the basin mean (the remaining cells set the mean).
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
    data[0] = 5.0            # uniform 5 mm/day -> basin mean 5.0
    data[1] = 10.0           # uniform 10 mm/day
    data[1, 0, 0] = FILL_VALUE  # one fill cell -> masked, mean stays 10.0
    data[2] = 0.0            # dry day -> 0.0 mm (a real zero, GOOD)
    data[3] = FILL_VALUE     # entirely fill -> MISSING
    ds = xr.Dataset(
        {"prcp": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "daymet_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical, identity-converted from source mm/day total
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "daymet:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5 mm/day layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Fill cell masked; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    # A dry day is a genuine zero, not missing.
    assert by_day[17].value == pytest.approx(0.0, abs=1e-9)
    assert by_day[17].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("daymet:cell:")
    # Nearest cell to centroid (51, -115) is the uniform-layer value.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)


def test_window_trim_half_open(daymet_nc):
    conn = DaymetPrecipitationConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = DaymetPrecipitationConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
