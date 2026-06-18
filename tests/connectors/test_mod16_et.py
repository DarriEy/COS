"""MOD16 ET connector — hermetic test of the gridded basin-reduction path.

Builds synthetic in-memory MOD16-like NetCDFs and reduces them; no network, no
auth. Proves the gridded -> canonical-series path, the kg/m²/8day -> mm/day unit
boundary, fill-value masking, the nearest-cell small-basin default, and the
pre-reduced ET_basin_mean series path.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.mod16_et import MOD16ETConnector
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    QualityFlag,
    ReductionSpec,
    SpatialReduction,
)


@pytest.fixture
def mod16_daily_nc(tmp_path):
    """Gridded ET already in mm/day (the acquirer's default output)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-10", "2020-06-18", "2020-06-26", "2020-07-04"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3))
    data[0] = 2.0
    data[1] = 3.0
    data[2] = 4.0
    data[3] = 5.0
    ds = xr.Dataset(
        {"ET": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_daily.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mod16_composite_nc(tmp_path):
    """Gridded ET as an 8-day composite (kg/m2/8day) with a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    data = np.full((2, 2, 2), 8.0)  # 8 kg/m2/8day -> 1.0 mm/day
    # inject a fill/special pixel (>= 3276.1) that must be masked
    data[0, 0, 0] = 3276.7
    ds = xr.Dataset(
        {"ET_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET_500m"].attrs["units"] = "kg/m2/8day"
    path = tmp_path / "mod16_composite.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mod16_prereduced_nc(tmp_path):
    """Already basin-reduced ET_basin_mean(time) series in mm/day."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18", "2020-06-26"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"ET_basin_mean": (("time",), np.array([1.5, 2.5, np.nan]))},
        coords={"time": times},
    )
    ds["ET_basin_mean"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_prereduced.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2=8000.0):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_basin_mean_daily_units_canonical(mod16_daily_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_daily_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == KIND_UNITS[ObservationKind.ET] == "mm/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # each timestep is spatially uniform, so basin-mean equals the cell value
    by_month_day = {(p.timestamp.month, p.timestamp.day): p.value for p in series.points}
    assert by_month_day[(6, 10)] == pytest.approx(2.0)
    assert by_month_day[(7, 4)] == pytest.approx(5.0)
    assert all(p.quality == QualityFlag.GOOD for p in series.points)


def test_composite_units_divide_by_eight_and_fill_masked(mod16_composite_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_composite_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    assert series.source_info["source_units"] == "kg/m2/8day"
    # 8 kg/m2/8day / 8 = 1.0 mm/day; the fill pixel was masked, so the basin
    # mean of the remaining 1.0-valued cells is still 1.0.
    for p in series.points:
        assert p.value == pytest.approx(1.0)


def test_small_basin_defaults_to_nearest_cell(mod16_daily_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_daily_nc, _spec(area_km2=500.0),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("mod16_et:cell:")


def test_window_trim_half_open(mod16_daily_nc):
    conn = MOD16ETConnector()
    # [2020-06-18, 2020-07-04): includes 06-18 and 06-26, excludes 06-10 and 07-04.
    series = conn.reduce_file(
        mod16_daily_nc, _spec(),
        datetime(2020, 6, 18, tzinfo=UTC), datetime(2020, 7, 4, tzinfo=UTC),
    )
    days = {(p.timestamp.month, p.timestamp.day) for p in series.points}
    assert (6, 18) in days
    assert (6, 26) in days
    assert (6, 10) not in days
    assert (7, 4) not in days  # half-open excludes the end


def test_prereduced_series_path_and_missing(mod16_prereduced_nc):
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_prereduced_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.unit == "mm/day"
    vals = {(p.timestamp.day): (p.value, p.quality) for p in series.points}
    assert vals[10][0] == pytest.approx(1.5)
    assert vals[10][1] == QualityFlag.GOOD
    # NaN timestep -> MISSING with None value
    assert vals[26][0] is None
    assert vals[26][1] == QualityFlag.MISSING


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MOD16ETConnector()
    spec = _spec()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region(mod16_daily_nc):
    conn = MOD16ETConnector()
    sites = await conn.list_sites(_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "mod16_et:domain:bow"


# ===========================================================================
# PARITY-BY-CONSTRUCTION against the SYMFLUENCE native MOD16 handler.
#
# Native reduction semantics (reimplemented inline below), taken verbatim from
#   symfluence/data/observation/handlers/modis_et.py :: _process_netcdf
#   symfluence/data/acquisition/handlers/modis_et.py :: _process_modis_hdf_files
#                                                     :: _process_appeears_products
# All three native spatial reductions are an UNWEIGHTED lat/lon mean over the
# valid (non-NaN) cells:
#     observation handler : da.mean(dim=spatial_dims, skipna=True)
#     acquirer earthaccess: float(np.nanmean(et_data))
#     acquirer appeears   : et_merged.mean(dim=[lat,lon], skipna=True)
# Unit handling: an 8-day composite (kg/m2/8day) is divided by DAYS_IN_COMPOSITE
# (=8) to mm/day; 1 kg/m2 of water == 1 mm. Fill / special pixels (digital
# 32761..32767, i.e. >= 32761*0.1 = 3276.1 after the 0.1 scale) are NaN.
#
# COS deliberately diverges in ONE benign way: its gridded basin_mean is a
# cos(latitude) AREA-WEIGHTED mean (cos.core.reduce.basin_mean), a documented
# approximation of polygon-weighted zonal stats, whereas native is unweighted.
# For ET (a basin-mean flux objective) this is benign: over a narrow-latitude
# bbox the cos-lat weights are nearly constant, so the two agree to ~1e-3. The
# unit factor and the fill rule are EXACT, and a spatially-constant field makes
# the weighted and unweighted means identical to float tolerance.
# ===========================================================================

# native constants (mirrored, not imported, so the test pins the contract)
_NATIVE_DAYS_IN_COMPOSITE = 8.0
_NATIVE_SCALE = 0.1
_NATIVE_SPECIAL_VALUE_MIN = 32761  # digital; >= this is fill/special -> NaN


def _native_basin_reduce(values_scaled, *, is_8day_composite):
    """Reimplement the native MOD16 reduction on a (time, lat, lon) array.

    *values_scaled* is already 0.1-scaled (kg/m2 units), exactly the array the
    native handlers operate on after `et_data * SCALE_FACTOR`. Returns a length-
    `time` vector of UNWEIGHTED basin means in mm/day (if composite) or in the
    source unit (if already daily).
    """
    arr = np.asarray(values_scaled, dtype="float64")
    # native fill rule: digital >= 32761  <=>  scaled >= 3276.1
    arr = np.where(arr >= _NATIVE_SPECIAL_VALUE_MIN * _NATIVE_SCALE, np.nan, arr)
    out = np.full(arr.shape[0], np.nan, dtype="float64")
    for t in range(arr.shape[0]):
        layer = arr[t]
        if np.isfinite(layer).any():
            out[t] = float(np.nanmean(layer))  # UNWEIGHTED, skipna
    if is_8day_composite:
        out = out / _NATIVE_DAYS_IN_COMPOSITE
    return out


def _cos_values(series):
    return [p.value for p in series.points]


@pytest.fixture
def mod16_constant_nc(tmp_path):
    """Spatially CONSTANT composite field: cos-lat-weighted == unweighted EXACTLY."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((2, 3, 3))
    data[0] = 16.0   # kg/m2/8day -> 2.0 mm/day
    data[1] = 24.0   # -> 3.0 mm/day
    ds = xr.Dataset(
        {"ET_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET_500m"].attrs["units"] = "kg/m2/8day"
    path = tmp_path / "mod16_const.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mod16_varying_nc(tmp_path):
    """Spatially VARYING daily field over a narrow-lat bbox + one fill pixel.

    A GENTLE latitude gradient — the realistic regime for a basin's ET field
    over a narrow latitude band — so the benign cos-lat-vs-unweighted discrepancy
    stays within relative 1e-3 (verified: ~5e-4 here). A fill pixel exercises the
    shared masking rule (both sides drop it before averaging). The steep-gradient
    worst case is covered separately by test_parity_steep_gradient_bounded.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((2, 3, 3))
    # t0: gentle latitude gradient (rows): 2.9, 3.0, 3.1 mm/day
    data[0] = np.array([[2.9, 2.9, 2.9], [3.0, 3.0, 3.0], [3.1, 3.1, 3.1]])
    # t1: gentle gradient + a single fill/special pixel both sides mask
    data[1] = np.array([[3.9, 3.9, 3.9], [4.0, 4.0, 4.0], [4.1, 4.1, 4.1]])
    data[1, 0, 0] = 3276.7  # >= 3276.1 -> fill -> NaN on both sides
    ds = xr.Dataset(
        {"ET": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_varying.nc"
    ds.to_netcdf(path)
    return path


def test_parity_constant_field_exact_unit_factor_and_fill(mod16_constant_nc):
    """Constant field: COS cos-lat mean == native unweighted mean to FLOAT tol.

    Also pins the EXACT /8 composite -> mm/day unit factor and the fill floor.
    """
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_constant_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # native reduction on the SAME scaled (kg/m2) array, composite -> /8.
    native = _native_basin_reduce(
        np.array([np.full((3, 3), 16.0), np.full((3, 3), 24.0)]),
        is_8day_composite=True,
    )
    cos = _cos_values(series)
    assert len(cos) == len(native) == 2
    for c, n in zip(cos, native):
        assert c == pytest.approx(n, abs=1e-12)  # weighted==unweighted on constant
    # and the absolute canonical values are the expected mm/day
    assert cos[0] == pytest.approx(2.0, abs=1e-12)
    assert cos[1] == pytest.approx(3.0, abs=1e-12)
    assert series.unit == "mm/day"
    assert series.source_info["source_units"] == "kg/m2/8day"


def test_parity_varying_field_cos_lat_vs_unweighted_within_tol(mod16_varying_nc):
    """Varying field over narrow-lat bbox: COS == native within relative 1e-3.

    This is the benign documented divergence (cos-lat area weighting vs the
    native unweighted mean). Over lat 50..52 deg the cos weights span only
    cos(50)=0.643..cos(52)=0.616, ~4% spread, and the resulting basin-mean
    discrepancy stays well under 1e-3 relative. The fill pixel is masked
    identically on both sides before averaging.
    """
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_varying_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # SAME input array as the fixture, already mm/day (no composite divide).
    raw = np.empty((2, 3, 3))
    raw[0] = np.array([[2.9, 2.9, 2.9], [3.0, 3.0, 3.0], [3.1, 3.1, 3.1]])
    raw[1] = np.array([[3.9, 3.9, 3.9], [4.0, 4.0, 4.0], [4.1, 4.1, 4.1]])
    raw[1, 0, 0] = 3276.7
    native = _native_basin_reduce(raw, is_8day_composite=False)
    cos = _cos_values(series)
    assert len(cos) == len(native) == 2
    for c, n in zip(cos, native):
        assert c is not None and np.isfinite(n)
        # benign cos-lat vs unweighted: agree to relative 1e-3 over narrow bbox
        assert c == pytest.approx(n, rel=1e-3)
    # the discrepancy is real but tiny: COS cos-weights upweight the southern
    # (larger-cos) rows, so it must NOT be bitwise-equal on a lat gradient.
    assert cos[0] != native[0]


def test_parity_steep_gradient_bounded(tmp_path):
    """Worst case: a STEEP lat gradient is where cos-lat diverges most.

    Documents the boundary of the benign divergence: a 1,3,5 mm/day gradient
    across lat 50..52 makes COS cos-lat (2.971) differ from native unweighted
    (3.0) by ~1%, ABOVE the 1e-3 graduation tolerance. This is why parity is
    value-bounded (cos-lat basin-mean), not bitwise, and why the suggested grade
    is scoped to realistic narrow-band ET fields. The two still track closely
    (well within 2%) and the divergence is purely the documented weighting choice.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((1, 3, 3))
    data[0] = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])
    ds = xr.Dataset(
        {"ET": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET"].attrs["units"] = "mm/day"
    path = tmp_path / "mod16_steep.nc"
    ds.to_netcdf(path)

    conn = MOD16ETConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    native = _native_basin_reduce(data.copy(), is_8day_composite=False)
    cos = _cos_values(series)
    # exceeds 1e-3 (the point of this test) but stays within 2% — the cos-lat
    # weighting upweights the southern (larger-cos-lat) rows.
    assert cos[0] == pytest.approx(native[0], rel=2e-2)
    assert abs(cos[0] - native[0]) / native[0] > 1e-3


def test_parity_nearest_cell_is_identity_to_native_pick(mod16_varying_nc):
    """Point reduction is EXACT: nearest cell value, no weighting, no unit drift.

    Native point sampling (small basin) selects a single pixel; COS nearest_cell
    does the same argmin pick, so they must agree bitwise (identity tolerance).
    """
    conn = MOD16ETConnector()
    series = conn.reduce_file(
        mod16_varying_nc, _spec(area_km2=500.0),  # small -> nearest_cell
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    # centroid (51, -115) -> middle cell: lat row idx 1, lon col idx 1.
    # t0 middle row = 3.0; t1 middle row = 4.0 (the fill is at [0,0], not here).
    cos = _cos_values(series)
    assert cos[0] == pytest.approx(3.0, abs=1e-12)
    assert cos[1] == pytest.approx(4.0, abs=1e-12)


def test_parity_fill_missing_maps_to_missing_quality(tmp_path):
    """An all-fill timestep -> native NaN -> COS value None + QualityFlag.MISSING."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-10", "2020-06-18"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    data = np.empty((2, 2, 2))
    data[0] = 8.0          # -> 1.0 mm/day
    data[1] = 3276.7       # ALL fill -> native NaN -> COS MISSING
    ds = xr.Dataset(
        {"ET_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["ET_500m"].attrs["units"] = "kg/m2/8day"
    path = tmp_path / "mod16_allfill.nc"
    ds.to_netcdf(path)

    conn = MOD16ETConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    native = _native_basin_reduce(
        np.array([np.full((2, 2), 8.0), np.full((2, 2), 3276.7)]),
        is_8day_composite=True,
    )
    assert np.isfinite(native[0]) and np.isnan(native[1])
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[10].value == pytest.approx(1.0, abs=1e-12)
    assert by_day[10].quality == QualityFlag.GOOD
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_parity_window_trim_half_open_matches_native_filter(mod16_varying_nc):
    """Half-open [start, end) UTC trim — COS keeps exactly the native-kept stamps."""
    conn = MOD16ETConnector()
    start = datetime(2020, 6, 10, tzinfo=UTC)
    end = datetime(2020, 6, 18, tzinfo=UTC)  # excludes 06-18
    series = conn.reduce_file(mod16_varying_nc, _spec(), start, end)
    kept = {p.timestamp.day for p in series.points}
    # native semantics: start <= t < end
    all_stamps = [datetime(2020, 6, 10, tzinfo=UTC), datetime(2020, 6, 18, tzinfo=UTC)]
    native_kept = {t.day for t in all_stamps if start <= t < end}
    assert kept == native_kept == {10}
