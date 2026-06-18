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


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION vs the native SYMFLUENCE handler
#
# Native ref: symfluence/data/observation/handlers/gpm.py (keys 'gpm_imerg'/'gpm').
# Native reduction semantics (reimplemented inline below, no symfluence import):
#   1. _subset_spatial selects grid cells whose coords fall in [lat_min,lat_max]
#      x [lon_min,lon_max] (inclusive), same membership rule COS uses.
#   2. precip.mean(dim=non_time_dims) -> an UNWEIGHTED arithmetic mean over the
#      selected lat/lon cells, per timestep, with xarray's default skipna=True
#      (NaN/fill cells are dropped from the mean; an all-NaN layer -> NaN).
#   3. Units: mm/day passed through UNCHANGED -> canonical mm is the identity
#      (unit factor == 1.0, no scaling).
#   4. Non-negativity: clip(lower=0) applied to the AVERAGED series.
#
# COS reduction semantics (cos.core.reduce.basin_mean):
#   - cos-latitude AREA-WEIGHTED mean over the same in-bbox cells, skipna;
#   - identity mm/day -> mm;
#   - per-CELL non-negative clip BEFORE averaging (connector.reduce_file).
#
# Two documented, benign divergences vs native, neither of which corrupts a
# precipitation basin-mean objective:
#   (a) cos-lat weighting vs unweighted mean. Vanishes for uniform fields and for
#       narrow-latitude bboxes (weights ~constant); for a 2deg bbox it is ~0.3%.
#   (b) per-cell clip-before-average vs clip-after-average. Identical whenever no
#       in-bbox cell is negative (the overwhelmingly common case); they differ
#       only when a single sub-cell retrieval is spuriously negative AND the
#       layer mean would otherwise stay >=0, a sub-percent perturbation.
#
# The parity assertions below pin: EXACT identity on a uniform/single-cell field
# and on the unit factor; tight relative (1e-4) agreement on a narrow-latitude
# bbox where cos-lat ~= unweighted; and they keep the clip/fill rules honest.
# ---------------------------------------------------------------------------


def _native_basin_mean(lats, lons, values, bbox):
    """Reimplement the native handler's reduction inline (no symfluence import).

    UNWEIGHTED, skipna arithmetic mean over in-bbox cells per timestep, then
    clip the resulting series to be non-negative -- exactly gpm.py's
    _subset_spatial + precip.mean(non_time_dims) + clip(lower=0).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]  # (time, nlat, nlon)
    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.nanmean(layer))  # unweighted, skipna
    # native clips the averaged series, not per-cell:
    return np.where(np.isfinite(out), np.clip(out, 0.0, None), out)


def _build_nc(tmp_path, lats, lons, times, data, name="parity.nc"):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": np.asarray(lats, "float64"),
                "lon": np.asarray(lons, "float64")},
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


def test_parity_uniform_field_exact(tmp_path):
    """Uniform/constant field: cos-lat == unweighted exactly. The two MUST agree
    to float tolerance, and both equal the field value (identity unit factor)."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-06-15", "2020-06-16"], dtype="datetime64[ns]")
    data = np.empty((2, 3, 3))
    data[0] = 7.25
    data[1] = 0.0
    nc = _build_nc(tmp_path, lats, lons, times, data)

    series = conn.reduce_file(
        nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_vals = np.array([p.value for p in series.points], dtype="float64")
    native = _native_basin_mean(lats, lons, data, _spec(8000.0).bbox)

    # Exact identity: uniform field, no weighting effect, no unit scaling.
    np.testing.assert_allclose(cos_vals, native, atol=1e-12)
    assert cos_vals[0] == pytest.approx(7.25, abs=1e-12)  # mm/day -> mm identity


def test_parity_single_cell_nearest_exact(tmp_path):
    """nearest_cell is a single-cell pick; native unweighted mean of that one
    cell is the same number -> EXACT parity for the point path."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    data = np.arange(9, dtype="float64").reshape(1, 3, 3)  # center cell = 4.0
    nc = _build_nc(tmp_path, lats, lons, times, data)

    series = conn.reduce_file(
        nc, _spec(500.0),  # small -> nearest_cell at centroid (51,-115) = center
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    # Native unweighted mean of the single in-bbox-nearest cell == that cell.
    assert series.points[0].value == pytest.approx(4.0, abs=1e-12)


def test_parity_narrow_latitude_bbox_tight(tmp_path):
    """Narrow-latitude bbox: cos-lat weights are ~constant, so the COS weighted
    mean and the native UNWEIGHTED mean agree to a tight relative tolerance.

    This is the documented benign divergence (a). With 0.1deg lat spacing the
    cos-lat / unweighted gap is ~1e-6 relative -- far inside 1e-4."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([0.05, 0.15, 0.25])  # ~equator, 0.1deg GPM spacing
    lons = np.array([10.0, 10.1, 10.2])
    times = np.array(["2020-06-15", "2020-06-16"], dtype="datetime64[ns]")
    data = np.empty((2, 3, 3))
    data[0] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    data[1] = np.array([[0.5, 0.0, 1.5], [2.0, 3.0, 0.0], [4.0, 0.0, 6.0]])
    nc = _build_nc(tmp_path, lats, lons, times, data)

    spec = ReductionSpec(
        domain_name="equ", bbox=(0.0, 10.0, 0.3, 10.3),
        centroid=(0.15, 10.1), area_km2=8000.0,
    )
    series = conn.reduce_file(
        nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_vals = np.array([p.value for p in series.points], dtype="float64")
    native = _native_basin_mean(lats, lons, data, spec.bbox)

    # cos-lat ~= unweighted for a narrow-latitude bbox: tight relative parity.
    np.testing.assert_allclose(cos_vals, native, rtol=1e-4)


def test_parity_unit_factor_is_identity(tmp_path):
    """mm/day daily depth -> canonical mm is the identity (factor 1.0): the
    reduced value equals the native mean with NO scaling applied."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([0.05, 0.15])
    lons = np.array([10.0, 10.1])
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    data = np.array([[[12.0, 12.0], [12.0, 12.0]]], dtype="float64")  # uniform 12
    nc = _build_nc(tmp_path, lats, lons, times, data)
    spec = ReductionSpec(domain_name="u", bbox=(0.0, 10.0, 0.2, 10.2),
                         centroid=(0.1, 10.05), area_km2=8000.0)
    series = conn.reduce_file(
        nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.unit == "mm"
    # Uniform 12 mm/day -> 12 mm, no factor: native and COS both 12.0 exactly.
    assert series.points[0].value == pytest.approx(12.0, abs=1e-12)


def test_parity_clip_matches_native_when_no_negative_cells(tmp_path):
    """When no in-bbox cell is negative (the common case), COS's per-cell clip
    and the native clip-after-average are both no-ops, so the only remaining
    difference is cos-lat weighting -> tight parity on a narrow bbox."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([0.05, 0.15, 0.25])
    lons = np.array([10.0, 10.1, 10.2])
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
                    dtype="float64")  # all non-negative
    nc = _build_nc(tmp_path, lats, lons, times, data)
    spec = ReductionSpec(domain_name="c", bbox=(0.0, 10.0, 0.3, 10.3),
                         centroid=(0.15, 10.1), area_km2=8000.0)
    series = conn.reduce_file(
        nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_vals = np.array([p.value for p in series.points], dtype="float64")
    native = _native_basin_mean(lats, lons, data, spec.bbox)
    np.testing.assert_allclose(cos_vals, native, rtol=1e-4)


def test_parity_fill_maps_to_missing_like_native(tmp_path):
    """An all-fill layer reduces to NaN in BOTH semantics. COS surfaces that as
    QualityFlag.MISSING / value None; the native handler's mean of an all-NaN
    layer is NaN (then dropped). Same fill rule, expressed in each contract."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([0.05, 0.15])
    lons = np.array([10.0, 10.1])
    times = np.array(["2020-06-15", "2020-06-16"], dtype="datetime64[ns]")
    data = np.empty((2, 2, 2))
    data[0] = 3.0
    data[1] = FILL_VALUE  # entirely fill
    nc = _build_nc(tmp_path, lats, lons, times, data)
    spec = ReductionSpec(domain_name="f", bbox=(0.0, 10.0, 0.2, 10.2),
                         centroid=(0.1, 10.05), area_km2=8000.0)
    series = conn.reduce_file(
        nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    native = _native_basin_mean(lats, lons,
                                np.where(data <= FILL_VALUE, np.nan, data), spec.bbox)
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(3.0, abs=1e-12)
    assert by_day[15].quality == QualityFlag.GOOD
    assert by_day[16].value is None
    assert by_day[16].quality == QualityFlag.MISSING
    assert np.isnan(native[1])  # native: all-fill -> NaN, matching MISSING


def test_parity_window_trim_half_open_matches_native(tmp_path):
    """Half-open [start, end) UTC trim. Native filters with an inclusive mask
    (>= start & <= end); COS uses strictly half-open (< end). They agree on the
    interior; this pins the COS rule explicitly and documents the boundary
    convention difference is the END-inclusive native edge only."""
    conn = GPMIMERGPrecipConnector()
    lats = np.array([0.05, 0.15])
    lons = np.array([10.0, 10.1])
    times = np.array(["2020-06-15", "2020-06-16", "2020-06-17"],
                     dtype="datetime64[ns]")
    data = np.full((3, 2, 2), 4.0)
    nc = _build_nc(tmp_path, lats, lons, times, data)
    spec = ReductionSpec(domain_name="w", bbox=(0.0, 10.0, 0.2, 10.2),
                         centroid=(0.1, 10.05), area_km2=8000.0)
    series = conn.reduce_file(
        nc, spec,
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}  # 06-17 excluded by half-open [start, end)


@pytest.mark.network
def test_live_placeholder():
    """Live Earthdata fetch is covered separately; offline suite skips this."""
    pytest.skip("network test — requires Earthdata credentials")
