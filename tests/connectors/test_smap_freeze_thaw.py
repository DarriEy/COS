"""SMAP freeze/thaw connector — hermetic test of the gridded reduction path.

SMAP L3 Freeze/Thaw State (SPL3FTP) has NO SYMFLUENCE native, so this is
*spec-validated*: the assertions reproduce the published SPL3FTP product spec on a
synthetic inline fixture — the 0=thawed / 1=frozen categorical flag, the ``-9999``
fill sentinel, the reduction of the binary frozen field to a basin frozen-fraction
in ``[0, 1]`` (canonical ``freeze_thaw`` unit, identity scale), fill→MISSING, and
the half-open UTC window — with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.smap_freeze_thaw import (
    FILL_VALUE,
    FROZEN_CODE,
    SOURCE_FT_SCALE,
    THAWED_CODE,
    SMAPFreezeThawConnector,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ft_nc(tmp_path):
    """A synthetic SPL3FTP-like NetCDF: a categorical freeze/thaw flag.

    Four daily timesteps on a 3x3 grid (0-360 longitudes, = -116..-114), AM
    overpass variable ``freeze_thaw``:

    * day 15: all FROZEN (1) -> frozen fraction 1.0;
    * day 16: all THAWED (0) -> frozen fraction 0.0;
    * day 17: half frozen / half thawed, with one fill cell that must be excluded
      so the fraction is computed over VALID cells only;
    * day 18: entirely FILL (-9999) -> MISSING.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = FROZEN_CODE                       # all frozen -> 1.0
    data[1] = THAWED_CODE                       # all thawed -> 0.0
    # day 17: of the 9 cells, make 4 frozen and 4 thawed, 1 fill. Excluding the
    # fill cell leaves 8 valid -> 4/8 = 0.5 frozen fraction.
    layer = np.array([
        [FROZEN_CODE, FROZEN_CODE, FROZEN_CODE],
        [FROZEN_CODE, THAWED_CODE, THAWED_CODE],
        [THAWED_CODE, THAWED_CODE, FILL_VALUE],
    ])
    data[2] = layer
    data[3] = FILL_VALUE                        # entirely fill -> MISSING
    ds = xr.Dataset(
        {"freeze_thaw": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "spl3ftp_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_unit_and_fraction(ft_nc):
    """Spec: 0/1 flag reduced to a frozen FRACTION in [0,1]; canonical unit '1'."""
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        ft_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 12, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.FREEZE_THAW
    assert series.unit == "1"  # canonical freeze_thaw unit (dimensionless fraction)
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "smap_freeze_thaw:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform layers -> exact fractions regardless of cos-lat weighting.
    assert by_day[15].value == pytest.approx(1.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    assert by_day[16].value == pytest.approx(0.0, abs=1e-9)


def test_scale_is_identity():
    """Spec contract: the binary frozen field's mean IS the fraction (scale 1.0)."""
    assert SOURCE_FT_SCALE == 1.0


def test_fill_value_reduces_to_missing(ft_nc):
    """Spec: the -9999 fill sentinel -> no valid cell -> MISSING/None."""
    assert FILL_VALUE == -9999.0
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        ft_nc, _spec(8000.0),
        datetime(2020, 12, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_fill_cell_excluded_from_fraction(ft_nc):
    """Spec: a fill cell within an otherwise-valid layer is excluded from the
    frozen-fraction denominator (4 frozen / 8 valid = 0.5), not counted as thawed.

    With a single latitude row this would be exact; across the 3 rows here the
    cos-lat weighting nudges it slightly, so assert against the cos-lat-weighted
    fraction directly (the documented reduction), and pin that it is NOT 4/9.
    """
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        ft_nc, _spec(8000.0),
        datetime(2020, 12, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p.value for p in series.points}
    val = by_day[17]
    # cos-lat-weighted frozen fraction over the 8 VALID cells (fill excluded).
    lats = np.array([50.0, 51.0, 52.0])
    w = np.cos(np.deg2rad(lats))
    frozen = np.array([[1, 1, 1], [1, 0, 0], [0, 0, np.nan]], dtype="float64")
    w2d = np.broadcast_to(w[:, None], frozen.shape)
    fin = np.isfinite(frozen)
    expected = float(np.sum(frozen[fin] * w2d[fin]) / np.sum(w2d[fin]))
    assert val == pytest.approx(expected, abs=1e-9)
    # The unweighted-over-ALL-9-cells value (4/9) would be wrong: fill is excluded.
    assert abs(val - 4.0 / 9.0) > 1e-3


def test_nearest_cell_returns_raw_flag(ft_nc):
    """Spec: a single cell (small basin) yields the raw 0/1 flag, not a fraction.

    The centroid (51,-115) maps to grid cell (lat=51, lon=245) which is FROZEN(1)
    on day 15 and THAWED(0) on day 16.
    """
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        ft_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 12, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("smap_freeze_thaw:cell:")
    by_day = {p.timestamp.day: p.value for p in series.points}
    assert by_day[15] == pytest.approx(1.0)
    assert by_day[16] == pytest.approx(0.0)


def test_out_of_set_value_treated_as_fill(tmp_path):
    """Spec: a value outside the categorical {0,1} set is not-retrieved -> excluded.

    A stray code (e.g. 2 / 254) must be masked like fill, not coerced into the
    frozen/thawed tally.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-01-10"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    # One frozen, one thawed, rest a stray out-of-set code (2) -> only 2 valid,
    # fraction = 1 frozen / 2 valid = 0.5 (both survivors on the same lat row so
    # cos-lat weights cancel -> exact 0.5).
    layer = np.full((3, 3), 2.0)
    layer[1, 0] = FROZEN_CODE
    layer[1, 2] = THAWED_CODE
    ds = xr.Dataset(
        {"freeze_thaw": (("time", "lat", "lon"), layer[None, :, :])},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "spl3ftp_stray.nc"
    ds.to_netcdf(path)

    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 2, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(0.5, abs=1e-9)


def test_window_trim_half_open(ft_nc):
    """Half-open [12-15, 12-17): includes 12-15 and 12-16, excludes 12-17."""
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        ft_nc, _spec(8000.0),
        datetime(2020, 12, 15, tzinfo=UTC), datetime(2020, 12, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


def test_pm_overpass_variable_selected(tmp_path):
    """Spec: SPL3FTP carries AM and PM half-orbit flags; the configured overpass
    selects the corresponding variable (PM here)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-01-10"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    am = np.full((1, 3, 3), THAWED_CODE)   # AM all thawed -> 0.0
    pm = np.full((1, 3, 3), FROZEN_CODE)   # PM all frozen -> 1.0
    ds = xr.Dataset(
        {
            "freeze_thaw": (("time", "lat", "lon"), am),
            "freeze_thaw_pm": (("time", "lat", "lon"), pm),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "spl3ftp_ampm.nc"
    ds.to_netcdf(path)

    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
        options={"overpass": "pm"},
    )
    conn = SMAPFreezeThawConnector()
    series = conn.reduce_file(
        path, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 2, 1, tzinfo=UTC),
    )
    assert series.source_info["overpass"] == "pm"
    assert series.source_info["variable"] == "freeze_thaw_pm"
    assert series.points[0].value == pytest.approx(1.0, abs=1e-9)


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region(ft_nc):
    conn = SMAPFreezeThawConnector()
    sites = await conn.list_sites(_spec(8000.0))
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "smap_freeze_thaw:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = SMAPFreezeThawConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_smap_freeze_thaw():
    """LIVE smoke against a real SPL3FTP granule (requires Earthdata + a cached
    NetCDF supplied via config 'nc_path').

    Run with: pytest -m network tests/connectors/test_smap_freeze_thaw.py -k live
    """
    import os

    nc_path = os.environ.get("COS_SMAP_FT_NC")
    if not nc_path:
        pytest.skip("set COS_SMAP_FT_NC to a cached SPL3FTP NetCDF for the live smoke")
    conn = SMAPFreezeThawConnector({"nc_path": nc_path})
    spec = _spec(8000.0)
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "1"
