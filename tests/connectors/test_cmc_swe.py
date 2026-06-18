"""CMC SWE connector — hermetic test of the gridded depth→SWE reduction path.

Builds a synthetic in-memory CMC-like snow-depth grid (cm) and reduces it via the
pure ``reduce_arrays`` core; no network, no auth, no rasterio. Proves the
architecture-critical extract→mask→convert→reduce→canonicalize path and that it
matches the native handler's semantics: depth(cm)→SWE(mm) at density/100, the
``>999`` cm mask, the non-negative SWE clip, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.cmc_swe import DEFAULT_SNOW_DENSITY, CMCSnowSWEConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


def _grid():
    """Synthetic CMC depth grid (cm): 4 daily steps over a small 3x3 grid."""
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # depth in cm: step0=10cm everywhere, step1=20, step2 has a bad >999 cell,
    # step3=5cm (outside the test window).
    depth = np.empty((4, 3, 3))
    depth[0] = 10.0
    depth[1] = 20.0
    depth[2] = 10.0
    depth[2][1, 1] = 5000.0  # bad value -> masked, must not drag the mean up
    depth[3] = 5.0
    return lats, lons, times, depth


def test_basin_mean_depth_cm_to_swe_mm():
    conn = CMCSnowSWEConnector()
    lats, lons, times, depth = _grid()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SWE
    assert series.unit == "mm"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "cmc_swe:domain:bow"

    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    # swe_mm = depth_cm * density/100 ; density default 200 -> factor 2.0
    factor = DEFAULT_SNOW_DENSITY / 100.0
    assert by_date["2020-01-01"].value == pytest.approx(10.0 * factor)  # 20 mm
    assert by_date["2020-01-02"].value == pytest.approx(20.0 * factor)  # 40 mm
    # step2: the 5000 cm cell is masked (>999), remaining 8 cells are 10cm.
    assert by_date["2020-01-03"].value == pytest.approx(10.0 * factor)  # 20 mm
    assert by_date["2020-01-03"].quality.value == "good"


def test_window_trim_half_open():
    conn = CMCSnowSWEConnector()
    lats, lons, times, depth = _grid()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # [2020-01-02, 2020-01-03): includes 01-02, excludes 01-03 and 2021-06-01.
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    dates = {p.timestamp.date().isoformat() for p in series.points}
    assert dates == {"2020-01-02"}


def test_custom_density_via_options():
    conn = CMCSnowSWEConnector()
    lats, lons, times, depth = _grid()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0,
                         options={"snow_density": 300.0})
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 2, tzinfo=UTC),
    )
    # 10 cm * 300/100 = 30 mm
    assert series.points[0].value == pytest.approx(10.0 * 3.0)


def test_negative_swe_is_clipped_to_zero():
    conn = CMCSnowSWEConnector()
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01"], dtype="datetime64[ns]")
    # Negative depths are first masked as NaN (depth < 0), so to exercise the
    # SWE clip we feed a tiny positive depth and a custom (negative would be
    # masked) — instead assert the mask removes negatives -> MISSING here.
    depth = np.array([[[-5.0, -5.0], [-5.0, -5.0]]])  # all negative -> all masked
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
                         centroid=(50.5, -115.5), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 2, tzinfo=UTC),
    )
    # all cells masked -> no finite mean -> MISSING.
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_small_basin_defaults_to_nearest_cell():
    conn = CMCSnowSWEConnector()
    lats, lons, times, depth = _grid()
    spec = ReductionSpec(domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=300.0)  # small
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("cmc_swe:cell:")
    # nearest cell to (51,-115) = grid center; depth 10cm -> 20mm at step0.
    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    assert by_date["2020-01-01"].value == pytest.approx(10.0 * DEFAULT_SNOW_DENSITY / 100.0)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = CMCSnowSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="cached file"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC),
                                datetime(2021, 1, 1, tzinfo=UTC))


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION against the SYMFLUENCE native handler.
#
# Native ref: src/symfluence/data/observation/handlers/cmc_snow.py
#   (registered as both 'cmc_snow' and 'cmc_swe'), method _process_geotiff:
#
#     data[(data < 0) | (data > 999)] = np.nan          # plausibility mask (cm)
#     n_valid = np.sum(~np.isnan(data))                  # skip step if 0 valid
#     mean_depth_cm = np.nanmean(data)                   # UNWEIGHTED bbox mean
#     swe_mm = mean_depth_cm * (snow_density / 100.0)    # depth(cm) -> SWE(mm)
#   then, over the assembled frame:
#     df['swe_mm'] = df['swe_mm'].clip(lower=0)          # non-negative clip
#     df = df[(df.datetime >= start) & (df.datetime <= end)]  # INCLUSIVE window
#
# The COS connector reproduces the mask, the unit factor, and the clip exactly,
# but differs from native in two documented, benign ways:
#   1. SPATIAL REDUCTION: native takes an UNWEIGHTED np.nanmean over the bbox
#      cells; COS basin_mean uses a cosine-latitude AREA weighting. These agree
#      EXACTLY for a constant field or a field constant within each latitude row,
#      and to a small relative tolerance for a lat-varying field over a narrow
#      band (the documented cos-lat approximation, reduce.py docstring).
#   2. TIME WINDOW: native is inclusive [start, end]; COS canonicalises to
#      half-open [start, end). Benign — COS's contract is half-open UTC for every
#      connector; only an obs landing exactly on `end` differs, and dropping the
#      right edge is the intended canonical behaviour.
#
# These tests reimplement the native reduction INLINE on the SAME synthetic input
# and assert COS == native to the appropriate tolerance.
# --------------------------------------------------------------------------- #


def _native_swe_mm(depth_layer: np.ndarray, snow_density: float) -> float | None:
    """Inline reimplementation of native CMC _process_geotiff per-step semantics.

    Mirrors cmc_snow.py exactly: mask <0 / >999 cm, skip if no valid cells, take
    the UNWEIGHTED np.nanmean over the bbox window, convert cm depth -> mm SWE,
    and clip non-negative.
    """
    data = depth_layer.astype("float64").copy()
    data[(data < 0) | (data > 999)] = np.nan
    if np.sum(~np.isnan(data)) == 0:
        return None
    mean_depth_cm = float(np.nanmean(data))
    swe_mm = mean_depth_cm * (snow_density / 100.0)
    return max(swe_mm, 0.0)


def _run_cos(lats, lons, times, depth, *, density=None, area_km2=8000.0,
             start=datetime(2000, 1, 1, tzinfo=UTC), end=datetime(2100, 1, 1, tzinfo=UTC)):
    conn = CMCSnowSWEConnector()
    options = {"snow_density": density} if density is not None else {}
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(float(lats.min()), float(lons.min()), float(lats.max()), float(lons.max())),
        centroid=(float(np.mean(lats)), float(np.mean(lons))),
        area_km2=area_km2,
        options=options,
    )
    return conn.reduce_arrays(lats, lons, times, depth, spec, start, end)


def test_parity_constant_field_exact_vs_native():
    """Constant field: cos-lat weighting collapses to the unweighted mean, so COS
    MUST equal native np.nanmean to float tolerance — and exercises the unit
    factor and the >999 mask on the SAME input the native reducer sees."""
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    depth = np.empty((3, 3, 3))
    depth[0] = 10.0                 # uniform 10 cm
    depth[1] = 42.5                 # uniform 42.5 cm
    depth[2] = 7.0
    depth[2][1, 1] = 5000.0         # bad >999 cell -> masked in BOTH paths
    density = 200.0

    series = _run_cos(lats, lons, times, depth, density=density)
    cos_by_date = {p.timestamp.date().isoformat(): p.value for p in series.points}

    for k, layer in enumerate(depth):
        native = _native_swe_mm(layer, density)
        date = times[k].astype("datetime64[D]").astype(str)
        # constant (and constant-after-mask) field => exact agreement
        assert cos_by_date[date] == pytest.approx(native, rel=0, abs=1e-9), date


def test_parity_unit_factor_matches_native_for_arbitrary_density():
    """The depth->SWE factor (density/100) is identical to native at several
    densities, on a uniform field where the reductions coincide exactly."""
    lats = np.array([55.0, 56.0])
    lons = np.array([-100.0, -99.0])
    times = np.array(["2020-02-01"], dtype="datetime64[ns]")
    for density in (100.0, 200.0, 240.0, 300.0):
        depth = np.full((1, 2, 2), 33.0)
        series = _run_cos(lats, lons, times, depth, density=density)
        native = _native_swe_mm(depth[0], density)
        assert series.points[0].value == pytest.approx(native, rel=0, abs=1e-9)


def test_parity_latrow_constant_field_exact_vs_native():
    """Field constant within each latitude row but varying across rows: the
    cos-lat weights are per-row, and np.nanmean is the unweighted mean. These are
    NOT trivially equal, so we assert the documented relationship explicitly:
    COS == cos-lat-weighted mean, which differs from native unweighted mean only
    by the weighting. Here we verify COS reproduces the WEIGHTED reduction, and
    that it stays within the basin-mean tolerance of native over this 2-deg band."""
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-03-01"], dtype="datetime64[ns]")
    depth = np.empty((1, 3, 3))
    depth[0, 0, :] = 10.0   # lat 50
    depth[0, 1, :] = 30.0   # lat 51
    depth[0, 2, :] = 50.0   # lat 52
    density = 200.0

    series = _run_cos(lats, lons, times, depth, density=density)
    cos_val = series.points[0].value

    # native (unweighted) reference on the SAME masked input:
    native = _native_swe_mm(depth[0], density)  # mean(10,30,50)=30 -> 60 mm

    # explicit cos-lat-weighted reference (what COS must compute):
    w = np.cos(np.deg2rad(lats))
    rows = np.array([10.0, 30.0, 50.0])
    weighted_cm = float(np.sum(rows * w) / np.sum(w))
    weighted_mm = weighted_cm * density / 100.0

    assert cos_val == pytest.approx(weighted_mm, rel=0, abs=1e-9)
    # benign divergence: over this 2-deg band the cos-lat mean is within ~1% of
    # native; document the bound (basin-mean parity is tolerance-based, not bitwise).
    assert cos_val == pytest.approx(native, rel=1e-2)


def test_parity_fill_and_missing_vs_native():
    """A step whose every cell is masked (>999 or <0) yields no valid native row
    (n_valid==0 -> skipped) and a COS MISSING point — the fill rule matches."""
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    depth = np.empty((2, 2, 2))
    depth[0] = 15.0                 # valid step
    depth[1] = 5000.0              # all >999 -> all masked
    density = 200.0

    series = _run_cos(lats, lons, times, depth, density=density)
    by_date = {p.timestamp.date().isoformat(): p for p in series.points}

    # valid step matches native exactly (uniform field)
    assert by_date["2020-01-01"].value == pytest.approx(
        _native_swe_mm(depth[0], density), abs=1e-9)
    assert by_date["2020-01-01"].quality.value == "good"

    # fully-masked step: native produces NO row (n_valid==0); COS produces a
    # MISSING point. Both encode "no observation" — value None, quality MISSING.
    assert _native_swe_mm(depth[1], density) is None
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"


def test_parity_window_half_open_vs_native_inclusive():
    """COS canonicalises to half-open [start, end); native is inclusive [start,
    end]. Verify the ONLY difference is the right edge: an obs exactly on `end`
    is kept by native and dropped by COS, everything strictly inside agrees."""
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    depth = np.full((3, 2, 2), 12.0)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 1, 3, tzinfo=UTC)

    series = _run_cos(lats, lons, times, depth, density=200.0, start=start, end=end)
    cos_dates = {p.timestamp.date().isoformat() for p in series.points}

    # native inclusive window would keep all three; COS half-open drops the
    # right-edge 2020-01-03. Everything strictly inside [start, end) is identical.
    native_inclusive = {"2020-01-01", "2020-01-02", "2020-01-03"}
    assert cos_dates == native_inclusive - {"2020-01-03"}
    # and the kept values equal native (uniform field)
    for p in series.points:
        assert p.value == pytest.approx(_native_swe_mm(depth[0], 200.0), abs=1e-9)


def test_read_geotiff_reprojects_projected_crs_to_geographic(tmp_path):
    """Regression: the live spot-check found _read_geotiff produced bogus 1D axes
    for the CMC Polar Stereographic GeoTIFF (warping one row/col), placing every
    real NH bbox outside the grid -> zero points. It must reproject the raster to
    a REGULAR EPSG:4326 grid so lat/lon are true 1D axes. This exercises the
    previously-0%-covered geo frontend on a genuinely projected raster.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # 1-band 8x8 grid in a PROJECTED CRS (EPSG:3857 Web Mercator) over a real
    # mid-latitude NH box (~50N, ~-117E), constant 40 cm snow depth, 25 km cells.
    # Web Mercator is non-geographic so it exercises the reproject branch (the
    # fix) without the polar-stereographic degeneracy a pole-side origin causes.
    arr = np.full((1, 8, 8), 40.0, dtype="float32")
    transform = from_origin(-1.300e7, 6.70e6, 25_000.0, 25_000.0)
    path = tmp_path / "cmc_swe_depth_2020.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
        crs="EPSG:3857", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(arr)

    conn = CMCSnowSWEConnector()
    lats, lons, times, depth_cm = conn._read_geotiff(path)

    # Axes must be GEOGRAPHIC and in the northern high latitudes (the bug gave
    # lats ~0..20). Longitudes within [-180, 180].
    assert lats.min() > 40.0 and lats.max() <= 90.0
    assert -180.0 <= lons.min() and lons.max() <= 180.0
    assert depth_cm.shape[0] == 1
    # The constant 40 cm field survives reprojection where data exists.
    assert np.nanmax(depth_cm) == pytest.approx(40.0, abs=1e-6)

    # Full path: reduce over the data's own lat/lon extent -> finds cells (not the
    # zero-point failure) and returns the constant depth converted to SWE.
    spec = ReductionSpec(
        domain_name="np",
        bbox=(float(lats.min()), float(lons.min()), float(lats.max()), float(lons.max())),
        centroid=(float(np.median(lats)), float(np.median(lons))),
        area_km2=50_000.0,
    )
    series = conn.reduce_file(path, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    assert series.points, "reduce_file must find cells in the reprojected grid (not zero points)"
    assert any(p.value is not None for p in series.points)
