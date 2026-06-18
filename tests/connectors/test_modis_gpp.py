"""MODIS MOD17A2H GPP connector — hermetic test of the gridded reduction path.

Builds synthetic in-memory MOD17A2H-like NetCDFs and reduces them; no network,
no auth. Proves the gridded -> canonical-series path, the digital -> gC/m2/day
unit boundary (scale 0.0001, kgC -> gC, /interval_days), fill-value masking, the
nearest-cell small-basin default, the pre-reduced series path, and the half-open
window trim.

SPEC-VALIDATED: there is no SYMFLUENCE native MOD17 GPP handler to compare
against, so the parity section reproduces the PUBLISHED MOD17A2H product spec
(scale factor 0.0001, valid digital range 0..32760, fill floor 32761) on the
synthetic fixture and checks the connector against it.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_gpp import (
    DAYS_IN_COMPOSITE,
    KG_TO_G,
    SCALE_FACTOR,
    SPECIAL_VALUE_MIN_DIGITAL,
    MODISGPPConnector,
)
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    QualityFlag,
    ReductionSpec,
    SpatialReduction,
)


def _spec(area_km2=8000.0):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


@pytest.fixture
def gpp_digital_nc(tmp_path):
    """Gridded GPP as RAW MOD17A2H digital counts (8-day composite) + a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-10", "2020-06-18", "2020-06-26", "2020-07-04"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # digital 800 -> 800 * 0.0001 = 0.08 kgC/m2/8day -> *1000/8 = 10.0 gC/m2/day
    data = np.empty((4, 3, 3))
    data[0] = 800.0
    data[1] = 1600.0   # -> 0.16 kgC/m2/8day -> 20.0 gC/m2/day
    data[2] = 2400.0   # -> 0.24 kgC/m2/8day -> 30.0 gC/m2/day
    data[3] = 3200.0   # -> 0.32 kgC/m2/8day -> 40.0 gC/m2/day
    ds = xr.Dataset(
        {"Gpp_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["Gpp_500m"].attrs["units"] = "kgC/m2/8day"
    path = tmp_path / "gpp_digital.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def gpp_prereduced_nc(tmp_path):
    """Already basin-reduced GPP_basin_mean(time) series in canonical gC/m2/day."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18", "2020-06-26"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"GPP_basin_mean": (("time",), np.array([10.0, 20.0, np.nan]))},
        coords={"time": times},
    )
    ds["GPP_basin_mean"].attrs["units"] = "gC/m2/day"
    path = tmp_path / "gpp_prereduced.nc"
    ds.to_netcdf(path)
    return path


def test_basin_mean_digital_to_canonical_units(gpp_digital_nc):
    conn = MODISGPPConnector()
    series = conn.reduce_file(
        gpp_digital_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.GPP
    assert series.unit == KIND_UNITS[ObservationKind.GPP] == "gC/m2/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # each timestep is spatially uniform, so basin-mean equals the cell value.
    by_day = {p.timestamp.day: p.value for p in series.points}
    assert by_day[10] == pytest.approx(10.0)
    assert by_day[18] == pytest.approx(20.0)
    assert by_day[26] == pytest.approx(30.0)
    assert by_day[4] == pytest.approx(40.0)
    assert all(p.quality == QualityFlag.GOOD for p in series.points)
    assert series.source_info["scale_factor"] == "0.0001"


def test_fill_value_masked_to_missing(tmp_path):
    """Digital >= 32761 is a fill pixel -> NaN; an all-fill timestep -> MISSING."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    data = np.empty((2, 2, 2))
    data[0] = 800.0     # -> 10.0 gC/m2/day
    data[0, 0, 0] = 32767  # one fill pixel, masked; the rest still 10.0
    data[1] = 32761.0   # ALL fill -> NaN -> MISSING
    ds = xr.Dataset(
        {"Gpp_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["Gpp_500m"].attrs["units"] = "kgC/m2/8day"
    path = tmp_path / "gpp_fill.nc"
    ds.to_netcdf(path)

    conn = MODISGPPConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # fill pixel dropped, remaining cells all 10.0 -> basin mean still 10.0
    assert by_day[10].value == pytest.approx(10.0)
    assert by_day[10].quality == QualityFlag.GOOD
    # all-fill timestep -> None + MISSING
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_interval_days_configurable_for_short_composite(gpp_digital_nc):
    """A 5-day trailing composite scales by /5 instead of /8."""
    conn = MODISGPPConnector(config={"interval_days": 5})
    series = conn.reduce_file(
        gpp_digital_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p.value for p in series.points}
    # digital 800 * 0.0001 * 1000 / 5 = 16.0 gC/m2/day
    assert by_day[10] == pytest.approx(800.0 * SCALE_FACTOR * KG_TO_G / 5.0)
    assert by_day[10] == pytest.approx(16.0)


def test_daily_source_units_pass_through(tmp_path):
    """A source already in gC/m2/day is NOT rescaled — pass-through identity."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    data = np.full((1, 2, 2), 12.5)  # already gC/m2/day
    ds = xr.Dataset(
        {"GPP": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["GPP"].attrs["units"] = "gC/m2/day"
    path = tmp_path / "gpp_daily.nc"
    ds.to_netcdf(path)

    conn = MODISGPPConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.source_info["source_units"] == "gC/m2/day"
    assert series.points[0].value == pytest.approx(12.5)


def test_small_basin_defaults_to_nearest_cell(gpp_digital_nc):
    conn = MODISGPPConnector()
    series = conn.reduce_file(
        gpp_digital_nc, _spec(area_km2=500.0),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_gpp:cell:")


def test_window_trim_half_open(gpp_digital_nc):
    conn = MODISGPPConnector()
    # [2020-06-18, 2020-07-04): includes 06-18 and 06-26, excludes 06-10 and 07-04.
    series = conn.reduce_file(
        gpp_digital_nc, _spec(),
        datetime(2020, 6, 18, tzinfo=UTC), datetime(2020, 7, 4, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert 18 in days
    assert 26 in days
    assert 10 not in days
    assert 4 not in days  # half-open excludes the end


def test_prereduced_series_path_and_missing(gpp_prereduced_nc):
    conn = MODISGPPConnector()
    series = conn.reduce_file(
        gpp_prereduced_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.unit == "gC/m2/day"
    vals = {p.timestamp.day: (p.value, p.quality) for p in series.points}
    assert vals[10][0] == pytest.approx(10.0)
    assert vals[10][1] == QualityFlag.GOOD
    # NaN timestep -> MISSING with None value
    assert vals[26][0] is None
    assert vals[26][1] == QualityFlag.MISSING


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MODISGPPConnector()
    spec = _spec()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region():
    conn = MODISGPPConnector()
    sites = await conn.list_sites(_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "modis_gpp:domain:bow"


# ===========================================================================
# SPEC-VALIDATED reproduction of the PUBLISHED MOD17A2H product spec.
#
# There is NO SYMFLUENCE native MOD17 GPP handler, so instead of a native parity
# we reimplement the published product spec inline and assert the connector
# reproduces it exactly on the synthetic fixture:
#   * scale factor 0.0001 (kgC/m2/8day = digital * 0.0001)
#   * canonical conversion gC/m2/day = kgC/m2/8day * 1000 / interval_days
#   * valid digital range 0..32760; digital >= 32761 is fill -> NaN
#
# COS deliberately diverges from a plain unweighted mean in ONE benign way: its
# gridded basin_mean is a cos(latitude) AREA-WEIGHTED mean
# (cos.core.reduce.basin_mean), a documented approximation of polygon-weighted
# zonal stats. On a spatially-constant field the weighted and unweighted means
# are identical to float tolerance, which the constant-field test below pins.
# ===========================================================================

_SPEC_SCALE = 0.0001
_SPEC_FILL_MIN_DIGITAL = 32761
_SPEC_KG_TO_G = 1000.0


def _spec_convert(digital, *, interval_days):
    """Reproduce the published MOD17A2H digital -> gC/m2/day conversion.

    Mirrors the spec verbatim: mask digital >= 32761 to NaN, scale by 0.0001 to
    kgC/m2/8day, then * 1000 / interval_days to gC/m2/day.
    """
    arr = np.asarray(digital, dtype="float64")
    arr = np.where(arr >= _SPEC_FILL_MIN_DIGITAL, np.nan, arr)
    return arr * _SPEC_SCALE * _SPEC_KG_TO_G / interval_days


def _spec_basin_reduce(digital, *, interval_days):
    """Unweighted basin mean per timestep of the spec-converted gC/m2/day grid."""
    conv = _spec_convert(digital, interval_days=interval_days)
    out = np.full(conv.shape[0], np.nan, dtype="float64")
    for t in range(conv.shape[0]):
        layer = conv[t]
        if np.isfinite(layer).any():
            out[t] = float(np.nanmean(layer))
    return out


def test_spec_connector_constants_match_published_spec():
    """The connector's constants ARE the published MOD17A2H spec values."""
    assert SCALE_FACTOR == _SPEC_SCALE
    assert SPECIAL_VALUE_MIN_DIGITAL == _SPEC_FILL_MIN_DIGITAL
    assert KG_TO_G == _SPEC_KG_TO_G
    assert DAYS_IN_COMPOSITE == 8.0


def test_spec_constant_field_exact_factor_and_fill(gpp_digital_nc):
    """Constant field: COS cos-lat mean == spec unweighted mean to FLOAT tol.

    Pins the EXACT 0.0001 scale + *1000/8 unit factor and the 32761 fill floor.
    """
    conn = MODISGPPConnector()
    series = conn.reduce_file(
        gpp_digital_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    digital = np.array([
        np.full((3, 3), 800.0),
        np.full((3, 3), 1600.0),
        np.full((3, 3), 2400.0),
        np.full((3, 3), 3200.0),
    ])
    spec_vals = _spec_basin_reduce(digital, interval_days=DAYS_IN_COMPOSITE)
    cos = [p.value for p in series.points]
    assert len(cos) == len(spec_vals) == 4
    for c, n in zip(cos, spec_vals):
        assert c == pytest.approx(n, abs=1e-12)  # weighted==unweighted on constant
    # absolute canonical values from the published spec
    assert cos[0] == pytest.approx(10.0, abs=1e-12)
    assert cos[3] == pytest.approx(40.0, abs=1e-12)


def test_spec_fill_floor_boundary_is_inclusive():
    """Digital 32760 is the last VALID count; 32761 is the first fill count."""
    conn = MODISGPPConnector()
    _, converted = conn._canonicalize_units(
        "kgC/m2/8day", np.array([[32760.0, 32761.0]]),
    )
    # 32760 valid -> finite; 32761 fill -> NaN
    assert np.isfinite(converted[0, 0])
    assert converted[0, 0] == pytest.approx(32760.0 * _SPEC_SCALE * _SPEC_KG_TO_G / 8.0)
    assert np.isnan(converted[0, 1])


def test_spec_nearest_cell_is_identity(tmp_path):
    """Point reduction picks one cell — spec scale applies, no weighting drift."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((1, 3, 3))
    # gentle gradient by row; centroid (51,-115) selects the middle cell (1600).
    data[0] = np.array([[800.0, 800.0, 800.0],
                        [1600.0, 1600.0, 1600.0],
                        [2400.0, 2400.0, 2400.0]])
    ds = xr.Dataset(
        {"Gpp_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["Gpp_500m"].attrs["units"] = "kgC/m2/8day"
    path = tmp_path / "gpp_grad.nc"
    ds.to_netcdf(path)

    conn = MODISGPPConnector()
    series = conn.reduce_file(
        path, _spec(area_km2=500.0),  # small -> nearest_cell
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    # middle cell digital 1600 -> 20.0 gC/m2/day, exactly the spec conversion.
    assert series.points[0].value == pytest.approx(20.0, abs=1e-12)


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_earthdata_fetch_placeholder():
    """Live Earthdata MOD17A2H download is not yet wired; marked for network runs."""
    pytest.skip("Live Earthdata MOD17A2H download not yet wired")
