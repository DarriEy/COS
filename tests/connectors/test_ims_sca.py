"""IMS snow-cover connector — hermetic test of the gridded basin-reduction path.

Builds synthetic in-memory IMS-like NetCDFs (a value-code grid, and a pre-reduced
``snow_fraction`` series) and reduces them; no network, no auth. This proves the
code → fraction reduction (native parity), the unit (canonical ``fraction``,
no scalar conversion), window-trim, and MISSING handling.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ims_sca import (
    CODE_LAND,
    CODE_SEA_ICE,
    CODE_SNOW,
    CODE_WATER,
    IMSSnowCoverConnector,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ims_code_nc(tmp_path):
    """Synthetic IMS value-code grid NetCDF: (time, lat, lon) of surface codes.

    3x3 grid fully inside the bbox. Per timestep we lay out a known mix of
    land/snow/water so the expected SCA = snow_land / all_land is exact.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    codes = np.empty((3, 3, 3), dtype="int16")

    # t0: 9 land cells, 3 of them snow -> SCA = 3/9 = 1/3.
    codes[0] = CODE_LAND
    codes[0, 0, :] = CODE_SNOW  # one row snow (3 cells)

    # t1: water everywhere except a 2x... mix: 4 land cells, 2 snow, rest water.
    codes[1] = CODE_WATER
    codes[1, 0, 0] = CODE_SNOW
    codes[1, 0, 1] = CODE_SNOW
    codes[1, 1, 0] = CODE_LAND
    codes[1, 1, 1] = CODE_LAND  # land=4 (2 snow + 2 land) -> SCA = 2/4 = 0.5

    # t2: only water + sea ice -> no land pixels -> MISSING.
    codes[2] = CODE_WATER
    codes[2, 1, 1] = CODE_SEA_ICE

    ds = xr.Dataset(
        {"IMS_Surface_Values": (("time", "lat", "lon"), codes)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ims_codes_synth.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def ims_fraction_nc(tmp_path):
    """Synthetic pre-reduced IMS NetCDF: snow_fraction(time) already in 0-1."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-07-15", "2020-08-15"], dtype="datetime64[ns]"
    )
    # second value > 1 to prove clipping; third is NaN -> MISSING.
    frac = np.array([0.25, 1.4, np.nan])
    ds = xr.Dataset(
        {"snow_fraction": (("time",), frac)},
        coords={"time": times},
    )
    path = tmp_path / "ims_fraction_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec():
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,
    )


def test_reduce_codes_native_sca_ratio(ims_code_nc):
    conn = IMSSnowCoverConnector()
    series = conn.reduce_file(
        ims_code_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # canonical contract: SNOW_COVER unit is the dimensionless "fraction".
    assert series.kind == ObservationKind.SNOW_COVER
    assert series.unit == "fraction"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "ims_sca:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # t0: 3 snow / 9 land = 1/3 ; t1: 2 snow / 4 land = 0.5
    assert by_day[1].value == pytest.approx(1.0 / 3.0)
    assert by_day[1].quality == QualityFlag.GOOD
    assert by_day[2].value == pytest.approx(0.5)
    # t2: no land pixels -> MISSING / None.
    assert by_day[3].value is None
    assert by_day[3].quality == QualityFlag.MISSING


def test_fraction_passthrough_clips_and_masks(ims_fraction_nc):
    conn = IMSSnowCoverConnector()
    series = conn.reduce_file(
        ims_fraction_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "fraction"
    by_month = {p.timestamp.month: p for p in series.points}
    assert by_month[6].value == pytest.approx(0.25)
    # 1.4 clipped to 1.0 (native process clips to [0, 1]).
    assert by_month[7].value == pytest.approx(1.0)
    # NaN -> MISSING.
    assert by_month[8].value is None
    assert by_month[8].quality == QualityFlag.MISSING


def test_window_trim_half_open(ims_fraction_nc):
    conn = IMSSnowCoverConnector()
    # Half-open [2020-06-01, 2020-08-15): includes 06-15 & 07-15, excludes 08-15.
    series = conn.reduce_file(
        ims_fraction_nc, _spec(),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert 6 in months
    assert 7 in months
    assert 8 not in months


def test_all_values_in_unit_range(ims_code_nc):
    conn = IMSSnowCoverConnector()
    series = conn.reduce_file(
        ims_code_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    for p in series.points:
        if p.value is not None:
            assert 0.0 <= p.value <= 1.0


@pytest.mark.network
@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = IMSSnowCoverConnector()
    spec = _spec()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
