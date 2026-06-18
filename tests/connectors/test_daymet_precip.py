"""Daymet precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory Daymet-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded -> canonical-series path for a
daily precipitation product: identity unit (mm/day daily total -> canonical mm),
fill masking, basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.daymet_precip import FILL_VALUE, DaymetPrecipitationConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def daymet_nc(tmp_path):
    """A synthetic Daymet-like NetCDF: prcp (mm/day) over a small grid.

    Four daily timesteps on a 3x3 grid in North America. The last timestep is
    entirely the native missing value (-9999) so it must reduce to MISSING; one
    cell in an otherwise-uniform layer is also fill to confirm masked cells are
    skipped by the basin mean (the remaining cells set the mean).
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
    data[0] = 5.0            # uniform 5 mm/day -> basin mean 5.0
    data[1] = 10.0           # uniform 10 mm/day
    data[1, 0, 0] = FILL_VALUE  # one fill cell -> masked, mean stays 10.0
    data[2] = 0.0            # dry day -> 0.0 mm (a real zero, GOOD)
    data[3] = FILL_VALUE     # entirely fill -> MISSING
    ds = xr.Dataset(
        {"prcp": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "daymet_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical, identity-converted from source mm/day total
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "daymet:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5 mm/day layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Fill cell masked; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    # A dry day is a genuine zero, not missing.
    assert by_day[17].value == pytest.approx(0.0, abs=1e-9)
    assert by_day[17].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(daymet_nc):
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("daymet:cell:")
    # Nearest cell to centroid (51, -115) is the uniform-layer value.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)


def test_window_trim_half_open(daymet_nc):
    conn = DaymetPrecipitationConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = DaymetPrecipitationConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION against the native SYMFLUENCE Daymet handler.
#
# Native ref: symfluence/data/observation/handlers/daymet.py
#   DaymetHandler._process_netcdf:
#     * subset the variable to the basin/bbox via da.sel(lat=slice, lon=slice)
#       — selects cells whose centers fall inside the (inclusive) bbox;
#     * spatial mean: da.mean(dim=spatial_dims, skipna=True) — an UNWEIGHTED
#       arithmetic mean over the in-box cells, skipping missing (fill->NaN) cells;
#     * unit: VARIABLE_MAP maps prcp -> precip_mm with NO rescaling (CSV path maps
#       'prcp (mm/day)' -> precip_mm identically). The conversion is the IDENTITY.
#     * fill: xarray decodes Daymet's _FillValue/-9999 to NaN at open; skipna=True
#       drops them; an all-missing layer yields NaN.
#
# COS daymet_precip.reduce_file -> cos.core.reduce.basin_mean is the SAME selection
# and the SAME identity unit, but uses a cos-LATITUDE AREA-WEIGHTED mean. The only
# semantic divergence from native is unweighted-vs-cos-weighted; that is the
# documented benign basin-mean approximation. The reimplementations below encode
# the native semantics inline and assert:
#   * EXACT equality for uniform / single-cell / narrow fields where the two means
#     coincide;
#   * tolerance-bounded equality for a genuinely varying field over the (1-degree-
#     tall) test bbox, where cos-lat weights differ from uniform weights by < 1%.
# --------------------------------------------------------------------------


def _native_unweighted_basin_mean(lats, lons, values, bbox):
    """Reimplement DaymetHandler._process_netcdf's reduction inline.

    Inclusive bbox cell selection + UNWEIGHTED, NaN-skipping spatial mean,
    identity unit. Returns a length-time vector (NaN where no finite in-box cell).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]
    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.mean(layer[finite]))  # UNWEIGHTED, skipna
    return out


def test_parity_uniform_field_cos_equals_native_exactly(daymet_nc):
    """On the synthetic fixture (uniform/constant per-layer fields), the COS cos-lat
    weighted mean and the native unweighted mean must agree to float tolerance:
    a weighted mean of a constant equals that constant regardless of weights."""
    import xarray as xr

    conn = DaymetPrecipitationConnector()
    bbox = (50.0, -116.0, 52.0, -114.0)
    with xr.open_dataset(daymet_nc) as ds:
        lats = np.asarray(ds["lat"].values, dtype="float64")
        lons = np.asarray(ds["lon"].values, dtype="float64")
        vals = np.asarray(ds["prcp"].values, dtype="float64")
    vals = np.where(vals == FILL_VALUE, np.nan, vals)

    native = _native_unweighted_basin_mean(lats, lons, vals, bbox)

    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos = [p.value for p in series.points]

    # Identity unit + constant layers => bitwise-tight parity, and the all-fill
    # layer is NaN (native) / None+MISSING (COS) on both sides.
    for native_v, cos_v in zip(native, cos):
        if np.isnan(native_v):
            assert cos_v is None
        else:
            assert cos_v == pytest.approx(float(native_v), abs=1e-12)


def _native_coslat_weighted_basin_mean(lats, lons, values, bbox):
    """Reimplement cos.core.reduce.basin_mean's cos-lat weighting inline, so the
    COS result can be pinned EXACTLY (not just bounded) on the shared input."""
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]
    weights = np.cos(np.deg2rad(lats[lat_sel]))
    w2d = np.broadcast_to(weights[:, None], sub.shape[1:])
    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.sum(layer[finite] * w2d[finite]) / np.sum(w2d[finite]))
    return out


def test_parity_varying_field_coslat_vs_unweighted(tmp_path):
    """A genuinely lat-varying field: COS cos-lat mean vs native unweighted mean.

    The ONLY divergence is the cos-latitude weighting (selection, identity unit, and
    fill rule are identical to native). Over this deliberately adversarial fixture --
    a 2-degree-tall bbox (50-52 N) with a strong south->north gradient -- the cos-lat
    vs unweighted gap is ~1.2%. That is the documented benign basin-mean
    approximation (papers/cos_design.md: basin-mean parity is tolerance-based, not
    bitwise), NOT a unit/fill/reduction bug. We therefore:
      * pin COS EXACTLY against an inline cos-lat reimplementation (bitwise), and
      * bound the COS-vs-native gap by a quantified, justified tolerance.
    For realistic basins (narrower bbox and/or smoother fields) the gap shrinks well
    below 1e-3; this test uses the worst case to keep the tolerance load-bearing."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    times = np.array(["2021-03-01"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # Strong south->north gradient so weighting actually matters.
    layer = np.array([
        [1.0, 2.0, 3.0],     # 50 N
        [10.0, 11.0, 12.0],  # 51 N
        [20.0, 21.0, 22.0],  # 52 N
    ])
    data = layer[None, :, :]
    ds = xr.Dataset(
        {"prcp": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "daymet_gradient.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)[0]
    expected_cos = _native_coslat_weighted_basin_mean(lats, lons, data, bbox)[0]

    conn = DaymetPrecipitationConnector()
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC),
    )
    cos = series.points[0].value

    # COS is EXACTLY the cos-lat weighted mean of the shared input.
    assert cos == pytest.approx(float(expected_cos), abs=1e-12)
    # COS-vs-native gap is the cos-lat weighting only, bounded < 1.5% on this
    # worst-case 2-degree bbox with a near-10x N-S gradient.
    assert cos == pytest.approx(float(native), rel=1.5e-2)
    # ...and demonstrably NOT identical (the weighting is real), so the tolerance
    # is load-bearing rather than vacuous.
    assert abs(cos - float(native)) > 1e-6


def test_parity_single_cell_identity(tmp_path):
    """Single-cell bbox: weighted and unweighted means are both just that cell.
    Guarantees the identity-unit + reduction agree to float tolerance with no
    weighting freedom at all."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-03-01"], dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    data = np.array([[[7.25]]])  # single cell, 7.25 mm/day
    ds = xr.Dataset(
        {"prcp": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "daymet_single.nc"
    ds.to_netcdf(path)

    bbox = (50.5, -115.5, 51.5, -114.5)
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)[0]

    conn = DaymetPrecipitationConnector()
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(float(native), abs=1e-12)
    assert series.points[0].value == pytest.approx(7.25, abs=1e-12)


def test_parity_unit_factor_is_identity(daymet_nc):
    """Native maps prcp (mm/day) -> precip_mm with NO scaling; COS emits canonical
    'mm' with the same identity. Assert the unit factor is exactly 1.0 by checking
    a known input value passes through unscaled (5 mm/day -> 5 mm)."""
    conn = DaymetPrecipitationConnector()
    series = conn.reduce_file(
        daymet_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 16, tzinfo=UTC),
    )
    assert series.unit == "mm"
    # Source value 5.0 mm/day -> 5.0 canonical mm: factor == 1.0.
    assert series.points[0].value == pytest.approx(5.0, abs=1e-12)
