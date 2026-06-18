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


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION
#
# These tests reimplement the SYMFLUENCE native IMS semantics *inline* and
# assert the COS connector's pure reducers reproduce them bit-for-bit on the
# SAME synthetic input. The two native code paths are:
#
#   1. acquisition handler IMSSnowAcquirer._extract_domain_stats (acquire/handlers/ims_snow.py):
#         land  = count(code == VALUE_LAND) | (code == VALUE_SNOW)
#         snow  = count(code == VALUE_SNOW)
#         snow_fraction = snow / land   if land > 0 else NaN
#      -> UNWEIGHTED integer pixel-count ratio, no cos-lat weighting, no unit
#         factor (the ratio *is* the canonical "fraction" unit). The native bbox
#         is a *pixel-index* subset of the polar-stereographic grid; for a grid
#         whose cell centers all fall inside the request bbox this selects the
#         identical pixel set COS selects by lat/lon membership, so parity here
#         is IDENTITY (exact float equality), not a tolerance.
#
#   2. observation handler IMSSnowHandler._compute_sca_from_grid + .process:
#         same snow/land ratio (NaN when land == 0), then .process clips the
#         final series to [0, 1].
#
# COS's reduce_codes mirrors (1)+(2): same unweighted count ratio, clip to
# [0,1], None/MISSING when land == 0. Because the ratio is dimensionless and
# already in [0,1], the clip is a benign no-op on the grid path and the cos-lat
# weighting that COS.basin_mean would apply is *deliberately not used* here
# (reduce_codes is a pure pixel count), so there is NO area-weighting divergence
# from native. Parity is therefore EXACT on the grid path.
# --------------------------------------------------------------------------- #

# IMS native value codes (mirrors symfluence acquire/obs handlers).
_NATIVE_VALUE_LAND = 2
_NATIVE_VALUE_SNOW = 4


def _native_sca_grid(codes):
    """Inline reimplementation of the native code-grid reduction.

    Mirrors IMSSnowAcquirer._extract_domain_stats / IMSSnowHandler.
    _compute_sca_from_grid exactly: per (time) snapshot the UNWEIGHTED ratio
    count(==SNOW) / count(==LAND or ==SNOW), NaN when no land pixels.
    ``codes`` is (time, y, x) over the already-bbox-subset grid.
    """
    out = []
    for t in range(codes.shape[0]):
        snap = codes[t]
        land = int(np.sum((snap == _NATIVE_VALUE_LAND) | (snap == _NATIVE_VALUE_SNOW)))
        snow = int(np.sum(snap == _NATIVE_VALUE_SNOW))
        out.append(snow / land if land > 0 else float("nan"))
    return out


def _native_fraction_process(fractions):
    """Inline reimplementation of native IMSSnowHandler.process for a pre-reduced
    snow_fraction series: clip to [0, 1]; NaN stays NaN (-> MISSING in COS)."""
    out = []
    for v in np.asarray(fractions, dtype="float64"):
        if np.isfinite(v):
            out.append(min(max(float(v), 0.0), 1.0))
        else:
            out.append(float("nan"))
    return out


def test_parity_code_grid_exact_against_native(ims_code_nc):
    """COS reduce_codes == native _extract_domain_stats / _compute_sca_from_grid.

    Tolerance: IDENTITY. The reduction is an unweighted integer pixel-count
    ratio over the same pixel set (whole synthetic grid is inside the bbox), so
    the two implementations must agree to exact float equality, not a tolerance.
    """
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(ims_code_nc) as ds:
        codes = np.asarray(ds["IMS_Surface_Values"].values)

    # NATIVE expected (inline reimplementation on the SAME input).
    native = _native_sca_grid(codes)

    # COS pure reducer on the SAME input (no bbox subset needed: whole grid in box).
    cos_points = IMSSnowCoverConnector.reduce_codes(
        np.asarray([np.datetime64(f"2020-01-0{t+1}") for t in range(codes.shape[0])]),
        codes, None, None, None,
    )
    cos_vals = [p.value for p in cos_points]

    assert len(cos_vals) == len(native)
    for cos_v, nat_v in zip(cos_vals, native):
        if np.isnan(nat_v):
            # native NaN <-> COS None/MISSING
            assert cos_v is None
        else:
            # EXACT: identical unweighted count ratio (no float drift expected).
            assert cos_v == nat_v


def test_parity_code_grid_bbox_subset_matches_native_pixel_subset(tmp_path):
    """When a bbox excludes part of the grid, COS's lat/lon-membership subset must
    select the SAME pixels the native pixel-index bbox would, so the ratio matches.

    Build a 3x3 grid where only the southern 2 rows fall in the bbox; the excluded
    northern row is all SNOW and would inflate SCA if (wrongly) included.
    """
    lats = np.array([50.0, 51.0, 60.0])  # 60.0 row excluded by bbox lat_max=52
    lons = np.array([-116.0, -115.0, -114.0])
    codes = np.empty((1, 3, 3), dtype="int16")
    codes[0] = CODE_LAND
    codes[0, 2, :] = CODE_SNOW  # northern (excluded) row all snow
    codes[0, 0, 0] = CODE_SNOW  # one in-box snow pixel

    bbox = (50.0, -116.0, 52.0, -114.0)
    # NATIVE on the in-box pixel subset (rows 0,1 -> 6 land cells, 1 snow).
    in_box = codes[:, 0:2, :]
    native = _native_sca_grid(in_box)
    assert native[0] == pytest.approx(1.0 / 6.0)  # sanity: excluded snow row ignored

    cos_points = IMSSnowCoverConnector.reduce_codes(
        np.asarray([np.datetime64("2020-01-01")]), codes, lats, lons, bbox,
    )
    assert cos_points[0].value == pytest.approx(native[0])
    assert cos_points[0].value == pytest.approx(1.0 / 6.0)


def test_parity_fraction_passthrough_exact_against_native(ims_fraction_nc):
    """COS fraction_series == native process() clip rule, exact / identity.

    Covers: unit (no scalar conversion — fraction stays fraction), the [0,1]
    clip, and NaN -> MISSING.
    """
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(ims_fraction_nc) as ds:
        frac = np.asarray(ds["snow_fraction"].values, dtype="float64")
        times = np.asarray(ds["snow_fraction"]["time"].values)

    native = _native_fraction_process(frac)
    cos_points = IMSSnowCoverConnector.fraction_series(times, frac)
    cos_vals = [p.value for p in cos_points]

    assert len(cos_vals) == len(native)
    for cos_v, nat_v in zip(cos_vals, native):
        if np.isnan(nat_v):
            assert cos_v is None
        else:
            assert cos_v == nat_v  # EXACT


def test_parity_constant_field_agrees_to_float_tolerance():
    """A constant-snow field: COS and native must agree to float tolerance.

    All-SNOW grid -> SCA == 1.0 for every implementation regardless of weighting,
    the strongest invariant (cos-lat weighting, if it were applied, still yields
    1.0; this pins that COS did not silently introduce weighting that diverges).
    """
    lats = np.array([10.0, 45.0, 80.0])  # wide latitude spread -> weights differ a lot
    lons = np.array([-116.0, -115.0, -114.0])
    codes = np.full((1, 3, 3), CODE_SNOW, dtype="int16")
    bbox = (0.0, -120.0, 90.0, -110.0)

    native = _native_sca_grid(codes[:, :, :])  # whole grid in box -> 9/9 = 1.0
    cos_points = IMSSnowCoverConnector.reduce_codes(
        np.asarray([np.datetime64("2020-01-01")]), codes, lats, lons, bbox,
    )
    assert native[0] == pytest.approx(1.0)
    assert cos_points[0].value == pytest.approx(native[0], abs=1e-12)


def test_parity_window_trim_native_closed_vs_cos_half_open(ims_fraction_nc):
    """Document the one intentional convention difference: native filters with a
    CLOSED [start, end] window; COS uses the canonical half-open [start, end).

    For an endpoint that is not a sample boundary the two agree; the difference
    only manifests for an observation landing exactly on ``end``. Here we pin
    that COS excludes the 08-15 sample when end == 2020-08-15 (half-open), which
    native would *include* (closed). This divergence is benign for daily snow
    cover (the kind's objective is a daily KGE/correlation, and the standard COS
    contract is half-open everywhere), so it does not corrupt parity.
    """
    conn = IMSSnowCoverConnector()
    series = conn.reduce_file(
        ims_fraction_nc, _spec(),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert 8 not in months  # COS half-open excludes the endpoint sample.
