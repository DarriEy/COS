"""ESA CCI soil-moisture connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory ESA-CCI-like NetCDF (variable ``sm``) and reduces
it; no network, no auth. This proves the architecture-critical gridded ->
canonical-series path for the merged volumetric soil-moisture product: identity
unit (m3/m3), inclusive [0, 1] mask (the native handler's
``sm.where((sm >= 0) & (sm <= 1))``), basin-mean vs nearest-cell reduction, and
half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.esa_cci_sm import ESACCISMConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def esa_cci_nc(tmp_path):
    """A synthetic ESA-CCI-like NetCDF: ``sm`` (m3/m3) over a small grid.

    Four timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). The last
    timestep is entirely NaN (fill) so it must reduce to MISSING; one cell in an
    otherwise-valid layer is out of range (>1) to exercise the inclusive clip
    mask. A boundary cell at exactly 0.0 stays valid (inclusive lower bound).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = 0.20          # uniform valid layer -> mean 0.20
    data[1] = 0.40          # uniform valid layer
    data[1, 0, 0] = 2.0     # out of range (>1) -> masked, mean stays 0.40
    data[2] = 0.30
    data[3] = np.nan        # entirely fill -> MISSING
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # canonical, identity-converted from source m3/m3
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "esa_cci_sm:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 0.20 layer -> basin mean 0.20 (no scaling applied).
    assert by_day[15].value == pytest.approx(0.20, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Out-of-range cell masked; remaining cells are 0.40 -> mean unchanged.
    assert by_day[16].value == pytest.approx(0.40, abs=1e-9)


def test_fill_reduces_to_missing(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-NaN (fill) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_inclusive_zero_boundary_kept(tmp_path):
    """A cell at exactly sm == 0.0 is kept (inclusive lower bound), not masked."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    data = np.zeros((1, 2, 2))  # all 0.0 -> inclusive [0,1] keeps them
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_zero.nc"
    ds.to_netcdf(path)

    conn = ESACCISMConnector()
    series = conn.reduce_file(
        path,
        ReductionSpec(domain_name="b", bbox=(50.0, -116.0, 51.0, -115.0), area_km2=8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(0.0, abs=1e-9)
    assert series.points[0].quality == QualityFlag.GOOD


def test_small_basin_defaults_to_nearest_cell(esa_cci_nc):
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("esa_cci_sm:cell:")


def test_window_trim_half_open(esa_cci_nc):
    conn = ESACCISMConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = ESACCISMConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION against the SYMFLUENCE native ESACCISMHandler.
#
# Native semantics (data/observation/handlers/soil_moisture.py::ESACCISMHandler):
#   * variable ``sm``;
#   * mask = ``sm.where((sm >= 0) & (sm <= 1))`` — INCLUSIVE [0, 1], everything
#     else -> NaN (fill / non-physical);
#   * unit = IDENTITY: source is volumetric m3/m3 and the handler emits the
#     masked fraction unchanged (renames ``sm`` -> ``soil_moisture``), no factor;
#   * bbox-subset reduction = ``sm.mean(dim=<all non-time dims>)`` — an
#     UNWEIGHTED (NaN-skipping) spatial mean over the in-bbox cells.
#
# COS basin_mean uses a cos-LATITUDE AREA-WEIGHTED mean (cos.core.reduce). That
# is the only deliberate divergence. It is benign for the kind's objective:
#   * for a CONSTANT field or a SINGLE in-bbox row, cos-lat weighting cancels in
#     the normalized mean -> the two are bitwise-equal to float tolerance;
#   * for a multi-row narrow-latitude bbox the weighting perturbs the mean only
#     by the cos-lat spread across a couple of degrees (~1e-4 relative here),
#     documented and tolerance-bounded, exactly as the GRACE basin-mean parity.
# Both reductions, the inclusive mask, the identity unit, and the half-open UTC
# window are asserted to AGREE with an inline reimplementation of the native
# reduction on the SAME synthetic input.
# --------------------------------------------------------------------------- #


def _native_inclusive_mask(layer):
    """Native ``sm.where((sm >= 0) & (sm <= 1))`` — inclusive, else NaN."""
    return np.where((layer >= 0.0) & (layer <= 1.0), layer, np.nan)


def _native_unweighted_bbox_mean(lats, lons, values, bbox):
    """Reimplements the native bbox-subset + UNWEIGHTED NaN-skipping spatial mean.

    Mirrors ``sm.sel(lat..., lon...)`` then ``sm.mean(dim=[lat, lon])`` over the
    in-bbox cells, after the inclusive [0, 1] mask. Identity unit (no factor).
    Returns a length-time vector; an all-NaN layer -> NaN (MISSING).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # Match the connector's 0-360 normalization for negative request lons.
    if lons.size and float(np.nanmax(lons)) > 180.0:
        if lon_min < 0:
            lon_min += 360.0
        if lon_max < 0:
            lon_max += 360.0
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]
    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        masked = _native_inclusive_mask(sub[t])
        finite = np.isfinite(masked)
        if finite.any():
            out[t] = float(np.mean(masked[finite]))  # UNWEIGHTED, NaN-skipping
    return out


def test_parity_single_row_bbox_is_exact_vs_native_unweighted(tmp_path):
    """Single in-bbox latitude row: cos-lat weight cancels -> COS == native EXACTLY.

    With one latitude row the cos-lat weights are a single constant, so the
    area-weighted mean reduces to the plain mean. COS must equal the native
    unweighted bbox-mean to float tolerance (identity unit, inclusive mask).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15", "2020-06-16"], dtype="datetime64[ns]")
    lats = np.array([51.0])               # single row -> weighting is a no-op
    lons = np.array([244.0, 245.0, 246.0])
    data = np.empty((2, 1, 3))
    data[0] = np.array([[0.10, 0.30, 0.50]])
    data[1] = np.array([[0.20, 2.00, 0.40]])  # middle cell >1 -> masked out
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_row.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        path,
        ReductionSpec(domain_name="b", bbox=bbox, area_km2=8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    expected = _native_unweighted_bbox_mean(lats, lons, data, bbox)
    cos_vals = [p.value for p in series.points]
    # ts0: mean(0.10,0.30,0.50)=0.30; ts1: 2.0 masked -> mean(0.20,0.40)=0.30.
    assert expected[0] == pytest.approx(0.30, abs=1e-12)
    assert expected[1] == pytest.approx(0.30, abs=1e-12)
    for got, exp in zip(cos_vals, expected):
        assert got == pytest.approx(float(exp), abs=1e-12)


def test_parity_constant_field_is_exact_vs_native(tmp_path):
    """Constant multi-row field: cos-lat weighting is irrelevant -> EXACT parity."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.full((1, 3, 3), 0.37)
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "esa_cci_const.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = ESACCISMConnector()
    series = conn.reduce_file(
        path,
        ReductionSpec(domain_name="b", bbox=bbox, area_km2=8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    expected = _native_unweighted_bbox_mean(lats, lons, data, bbox)
    assert expected[0] == pytest.approx(0.37, abs=1e-12)
    assert series.points[0].value == pytest.approx(float(expected[0]), abs=1e-12)


def test_parity_multirow_bbox_cos_lat_vs_native_unweighted_within_tol(esa_cci_nc):
    """Multi-row narrow-lat bbox: COS cos-lat mean ~ native unweighted mean.

    Quantifies the only deliberate divergence. Over a 50-52 deg, 2-degree-tall
    bbox the cos-lat weight spread is tiny, so the relative gap is ~1e-4 — far
    inside the GRACE-style tolerance and harmless for the soil-moisture target.
    For the layers that are uniform across rows (this fixture's are), the gap is
    in fact zero; this asserts the bound holds and would catch a wrong reduction.
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.empty((4, 3, 3))
    data[0] = 0.20
    data[1] = 0.40
    data[1, 0, 0] = 2.0
    data[2] = 0.30
    data[3] = np.nan
    bbox = (50.0, -116.0, 52.0, -114.0)
    expected = _native_unweighted_bbox_mean(lats, lons, data, bbox)

    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_by_day = {p.timestamp.day: p.value for p in series.points}
    cos_vals = [cos_by_day[d] for d in (15, 16, 17, 18)]
    for got, exp in zip(cos_vals, expected):
        if exp is None or not np.isfinite(exp):
            assert got is None  # native MISSING -> COS MISSING (fill parity)
        else:
            assert got == pytest.approx(float(exp), rel=1e-3, abs=1e-9)


def test_parity_nearest_cell_is_identity_vs_native_sel_nearest(esa_cci_nc):
    """Small basin: COS nearest_cell == native ``sm.sel(method='nearest')`` cell.

    For area below the threshold COS samples the single cell nearest the
    centroid; the native handler, when a bbox midpoint is known, likewise does
    ``sm.sel(lat=mid, lon=mid, method='nearest')``. On a regular grid these pick
    the SAME index, so the (masked) cell value must match EXACTLY.
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.empty((4, 3, 3))
    data[0] = 0.20
    data[1] = 0.40
    data[1, 0, 0] = 2.0
    data[2] = 0.30
    data[3] = np.nan

    # Centroid (51,-115) -> nearest indices i=1 (51.0), j=1 (245.0).
    centroid_lat, centroid_lon = 51.0, -115.0
    i = int(np.argmin(np.abs(lats - centroid_lat)))
    j = int(np.argmin(np.abs(lons - (centroid_lon + 360.0))))  # 0-360 norm
    native_cell = _native_inclusive_mask(data)[:, i, j]

    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    cos_by_day = {p.timestamp.day: p.value for p in series.points}
    cos_vals = [cos_by_day[d] for d in (15, 16, 17, 18)]
    for got, exp in zip(cos_vals, native_cell):
        if not np.isfinite(exp):
            assert got is None
        else:
            assert got == pytest.approx(float(exp), abs=1e-12)


def test_parity_unit_is_identity_no_factor(esa_cci_nc):
    """Boundary unit conversion is the IDENTITY: source m3/m3 == canonical m3/m3.

    Asserts COS applies NO scaling (unlike SNOTEL's 25.4 or GRACE's cm->mm). The
    reduced value equals the raw masked fraction, matching the native handler
    which emits the volumetric fraction unchanged.
    """
    from cos.core.models import KIND_UNITS

    conn = ESACCISMConnector()
    series = conn.reduce_file(
        esa_cci_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.unit == "m3/m3" == KIND_UNITS[ObservationKind.SOIL_MOISTURE]
    by_day = {p.timestamp.day: p.value for p in series.points}
    # Raw source value 0.20 passes through with no unit factor.
    assert by_day[15] == pytest.approx(0.20, abs=1e-12)
