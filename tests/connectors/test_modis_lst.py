"""MODIS LST connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MODIS-LST-like NetCDF (packed DN) and reduces it;
no network, no auth. This proves the architecture-critical gridded ->
canonical-series path for land surface temperature: DN -> Kelvin scale
(``* 0.02``), valid-range / fill masking, basin-mean vs nearest-cell reduction,
band selection (day/night), and the half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_lst import LST_SCALE_FACTOR, MODISLSTConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def lst_nc(tmp_path):
    """A synthetic MODIS-LST-like NetCDF: packed DN day + night over a small grid.

    Four timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). DN 15000 ->
    300 K, DN 14000 -> 280 K. The last timestep is entirely fill (0) so it must
    reduce to MISSING; one cell in an otherwise-valid layer is out of range
    (below 7500) to exercise the valid-range mask.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)

    day = np.empty((4, 3, 3))
    day[0] = 15000.0          # uniform valid -> 15000 * 0.02 = 300 K
    day[1] = 14000.0          # uniform valid -> 280 K
    day[1, 0, 0] = 100.0      # below-range cell -> masked, mean stays 280 K
    day[2] = 16000.0          # 320 K
    day[3] = 0.0              # entirely fill -> MISSING

    night = np.full((4, 3, 3), 13500.0)   # uniform valid -> 270 K (all timesteps)

    ds = xr.Dataset(
        {
            "LST_Day_1km": (("time", "lat", "lon"), day),
            "LST_Night_1km": (("time", "lat", "lon"), night),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lst_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.LST
    assert series.unit == "K"  # canonical Kelvin, scaled from packed DN
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "modis_lst:domain:bow"
    assert series.source_info["band"] == "day"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform DN 15000 -> 15000 * 0.02 = 300 K basin mean.
    assert by_day[15].value == pytest.approx(15000.0 * LST_SCALE_FACTOR, abs=1e-9)
    assert by_day[15].value == pytest.approx(300.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Below-range cell masked; remaining DN 14000 -> 280 K, mean unchanged.
    assert by_day[16].value == pytest.approx(280.0, abs=1e-9)


def test_fill_value_reduces_to_missing(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (DN 0) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_night_band_selection(lst_nc):
    conn = MODISLSTConnector({"band": "night"})
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.source_info["band"] == "night"
    assert series.source_info["variable"] == "LST_Night_1km"
    by_day = {p.timestamp.day: p for p in series.points}
    # Night DN 13500 -> 13500 * 0.02 = 270 K everywhere.
    assert by_day[15].value == pytest.approx(270.0, abs=1e-9)


def test_small_basin_defaults_to_nearest_cell(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_lst:cell:")


def test_window_trim_half_open(lst_nc):
    conn = MODISLSTConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MODISLSTConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.live
async def test_live_fetch_placeholder():
    """Live AppEEARS/Earthdata fetch is not wired; reduction path is the proven part."""
    pytest.skip("live AppEEARS download not wired; see reduce_file tests")
