"""CMC snow-depth connector — hermetic test of the gridded cm→m reduction path.

Builds a synthetic in-memory CMC-like snow-depth grid (cm) and reduces it via the
pure ``reduce_arrays`` core; no network, no auth, no rasterio. Proves the
architecture-critical extract→mask→convert→reduce→canonicalize path and that it
matches the native handler's semantics: the ``>999`` / ``<0`` cm mask, the
depth(cm)→depth(m) cm/100 scale, the canonical unit ``m`` for
``ObservationKind.SNOW_DEPTH``, half-open UTC window trim, and fill→MISSING.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.cmc_snow_depth import CM_TO_M, CMCSnowDepthConnector
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


def test_basin_mean_depth_cm_to_m():
    conn = CMCSnowDepthConnector()
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
    assert series.kind == ObservationKind.SNOW_DEPTH
    assert series.unit == "m"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "cmc_snow_depth:domain:bow"

    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    # depth_m = depth_cm / 100
    assert by_date["2020-01-01"].value == pytest.approx(10.0 * CM_TO_M)  # 0.10 m
    assert by_date["2020-01-02"].value == pytest.approx(20.0 * CM_TO_M)  # 0.20 m
    # step2: the 5000 cm cell is masked (>999), remaining 8 cells are 10cm.
    assert by_date["2020-01-03"].value == pytest.approx(10.0 * CM_TO_M)  # 0.10 m
    assert by_date["2020-01-03"].quality.value == "good"


def test_scale_factor_is_one_hundredth():
    """The cm→m boundary conversion is exactly /100 (canonical unit metres)."""
    assert pytest.approx(0.01) == CM_TO_M
    conn = CMCSnowDepthConnector()
    lats = np.array([55.0, 56.0])
    lons = np.array([-100.0, -99.0])
    times = np.array(["2020-02-01"], dtype="datetime64[ns]")
    depth = np.full((1, 2, 2), 250.0)  # 250 cm
    spec = ReductionSpec(domain_name="x", bbox=(55.0, -100.0, 56.0, -99.0),
                         centroid=(55.5, -99.5), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(2.5)  # 250 cm -> 2.5 m


def test_window_trim_half_open():
    conn = CMCSnowDepthConnector()
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


def test_fill_mask_yields_missing():
    conn = CMCSnowDepthConnector()
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01"], dtype="datetime64[ns]")
    depth = np.array([[[-5.0, -5.0], [-5.0, -5.0]]])  # all negative -> all masked
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
                         centroid=(50.5, -115.5), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 2, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_small_basin_defaults_to_nearest_cell():
    conn = CMCSnowDepthConnector()
    lats, lons, times, depth = _grid()
    spec = ReductionSpec(domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=300.0)  # small
    series = conn.reduce_arrays(
        lats, lons, times, depth, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("cmc_snow_depth:cell:")
    # nearest cell to (51,-115) = grid center; depth 10cm -> 0.10 m at step0.
    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    assert by_date["2020-01-01"].value == pytest.approx(10.0 * CM_TO_M)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = CMCSnowDepthConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="cached file"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC),
                                datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_fetch_placeholder():
    """Live NSIDC/Earthdata fetch is not yet wired; placeholder for the network
    gate so the live path is explicitly marked and deselected in CI."""
    pytest.skip("CMC live NSIDC/Earthdata download not yet wired")


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION against the SYMFLUENCE native handler.
#
# Native ref: src/symfluence/data/observation/handlers/cmc_snow.py (registered as
# 'cmc_snow' AND 'cmc_swe'), method _process_geotiff:
#
#     data[(data < 0) | (data > 999)] = np.nan          # plausibility mask (cm)
#     n_valid = np.sum(~np.isnan(data))                  # skip step if 0 valid
#     mean_depth_cm = np.nanmean(data)                   # UNWEIGHTED bbox mean
#     # native then -> SWE; the underlying physical observable is mean_depth_cm.
#   then, over the assembled frame, an INCLUSIVE [start, end] window.
#
# The native 'cmc_snow' observable IS the basin-mean snow DEPTH (the SWE step is a
# density multiply applied afterward). This connector emits that depth in metres
# (mean_depth_cm / 100). So the native depth observable, in canonical metres, is:
#
#     native_depth_m = mean_depth_cm / 100
#
# COS reproduces the mask and the cm→m scale exactly, differing from native in two
# documented, benign ways:
#   1. SPATIAL REDUCTION: native takes an UNWEIGHTED np.nanmean over the bbox
#      cells; COS basin_mean uses a cosine-latitude AREA weighting. These agree
#      EXACTLY for a constant field (or a field constant within each latitude row
#      they do not, but collapse for a fully-constant field), and to a small
#      relative tolerance for a lat-varying field over a narrow band (the
#      documented cos-lat approximation, reduce.py docstring).
#   2. TIME WINDOW: native is inclusive [start, end]; COS canonicalises to
#      half-open [start, end). Benign — only an obs landing exactly on `end`
#      differs, and dropping the right edge is the intended canonical behaviour.
# --------------------------------------------------------------------------- #


def _native_depth_m(depth_layer: np.ndarray) -> float | None:
    """Inline reimplementation of the native CMC depth observable per step.

    Mirrors cmc_snow.py exactly: mask <0 / >999 cm, skip if no valid cells, take
    the UNWEIGHTED np.nanmean over the bbox window. The canonical SNOW_DEPTH unit
    is metres, so the native cm mean is reported as cm/100 (the density step that
    turns it into SWE is NOT part of the depth observable).
    """
    data = depth_layer.astype("float64").copy()
    data[(data < 0) | (data > 999)] = np.nan
    if np.sum(~np.isnan(data)) == 0:
        return None
    mean_depth_cm = float(np.nanmean(data))
    return mean_depth_cm / 100.0


def _run_cos(lats, lons, times, depth, *, area_km2=8000.0,
             start=datetime(2000, 1, 1, tzinfo=UTC), end=datetime(2100, 1, 1, tzinfo=UTC)):
    conn = CMCSnowDepthConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(float(lats.min()), float(lons.min()), float(lats.max()), float(lons.max())),
        centroid=(float(np.mean(lats)), float(np.mean(lons))),
        area_km2=area_km2,
    )
    return conn.reduce_arrays(lats, lons, times, depth, spec, start, end)


def test_parity_constant_field_exact_vs_native():
    """Constant field: cos-lat weighting collapses to the unweighted mean, so COS
    MUST equal native depth (cm/100) to float tolerance — and exercises the cm→m
    scale and the >999 mask on the SAME input the native reducer sees."""
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    depth = np.empty((3, 3, 3))
    depth[0] = 10.0                 # uniform 10 cm
    depth[1] = 42.5                 # uniform 42.5 cm
    depth[2] = 7.0
    depth[2][1, 1] = 5000.0         # bad >999 cell -> masked in BOTH paths

    series = _run_cos(lats, lons, times, depth)
    cos_by_date = {p.timestamp.date().isoformat(): p.value for p in series.points}

    for k, layer in enumerate(depth):
        native = _native_depth_m(layer)
        date = times[k].astype("datetime64[D]").astype(str)
        # constant (and constant-after-mask) field => exact agreement
        assert cos_by_date[date] == pytest.approx(native, rel=0, abs=1e-12), date


def test_parity_scale_matches_native_for_arbitrary_depth():
    """The cm→m scale (/100) is identical to native at several depths, on a
    uniform field where the reductions coincide exactly."""
    lats = np.array([55.0, 56.0])
    lons = np.array([-100.0, -99.0])
    times = np.array(["2020-02-01"], dtype="datetime64[ns]")
    for depth_cm in (1.0, 33.0, 100.0, 875.0):
        depth = np.full((1, 2, 2), depth_cm)
        series = _run_cos(lats, lons, times, depth)
        native = _native_depth_m(depth[0])
        assert series.points[0].value == pytest.approx(native, rel=0, abs=1e-12)


def test_parity_latrow_field_cos_lat_vs_native_unweighted():
    """Field constant within each latitude row but varying across rows: COS uses
    per-row cos-lat weights; native uses the unweighted np.nanmean. Assert COS ==
    the explicit cos-lat-weighted reference, and stays within the documented
    basin-mean tolerance of native over this 2-deg band."""
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-03-01"], dtype="datetime64[ns]")
    depth = np.empty((1, 3, 3))
    depth[0, 0, :] = 10.0   # lat 50
    depth[0, 1, :] = 30.0   # lat 51
    depth[0, 2, :] = 50.0   # lat 52

    series = _run_cos(lats, lons, times, depth)
    cos_val = series.points[0].value

    native = _native_depth_m(depth[0])  # mean(10,30,50)/100 = 0.30 m

    w = np.cos(np.deg2rad(lats))
    rows = np.array([10.0, 30.0, 50.0])
    weighted_m = float(np.sum(rows * w) / np.sum(w)) / 100.0

    assert cos_val == pytest.approx(weighted_m, rel=0, abs=1e-12)
    # benign divergence: over this 2-deg band cos-lat mean within ~1% of native.
    assert cos_val == pytest.approx(native, rel=1e-2)


def test_parity_fill_and_missing_vs_native():
    """A step whose every cell is masked yields no valid native row (n_valid==0 ->
    skipped) and a COS MISSING point — the fill rule matches."""
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    depth = np.empty((2, 2, 2))
    depth[0] = 15.0                 # valid step
    depth[1] = 5000.0               # all >999 -> all masked

    series = _run_cos(lats, lons, times, depth)
    by_date = {p.timestamp.date().isoformat(): p for p in series.points}

    assert by_date["2020-01-01"].value == pytest.approx(_native_depth_m(depth[0]), abs=1e-12)
    assert by_date["2020-01-01"].quality.value == "good"

    assert _native_depth_m(depth[1]) is None
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"


def test_parity_window_half_open_vs_native_inclusive():
    """COS canonicalises to half-open [start, end); native is inclusive. Verify the
    ONLY difference is the right edge: an obs exactly on `end` is kept by native
    and dropped by COS, everything strictly inside agrees."""
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    depth = np.full((3, 2, 2), 12.0)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 1, 3, tzinfo=UTC)

    series = _run_cos(lats, lons, times, depth, start=start, end=end)
    cos_dates = {p.timestamp.date().isoformat() for p in series.points}

    native_inclusive = {"2020-01-01", "2020-01-02", "2020-01-03"}
    assert cos_dates == native_inclusive - {"2020-01-03"}
    for p in series.points:
        assert p.value == pytest.approx(_native_depth_m(depth[0]), abs=1e-12)
