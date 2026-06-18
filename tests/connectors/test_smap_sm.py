"""SMAP soil-moisture connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory SMAP-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded → canonical-series path for a
volumetric soil-moisture product: identity unit (m³/m³), fill/out-of-range
masking, basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.smap_sm import FILL_VALUE, SMAPSoilMoistureConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def smap_nc(tmp_path):
    """A synthetic SMAP-like NetCDF: soil_moisture (m³/m³) over a small grid.

    Four timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). The last
    timestep is entirely fill (-9999) so it must reduce to MISSING; one cell in
    an otherwise-valid layer is out of range (>1) to exercise the clip mask.
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
    data[1, 0, 0] = 2.0     # one out-of-range cell -> masked, mean stays 0.40
    data[2] = 0.30
    data[3] = FILL_VALUE    # entirely fill -> MISSING
    ds = xr.Dataset(
        {"soil_moisture": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "smap_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(smap_nc):
    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # canonical, identity-converted from source m³/m³
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "smap:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 0.20 layer -> basin mean 0.20 (no scaling applied).
    assert by_day[15].value == pytest.approx(0.20, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Out-of-range cell masked; remaining cells are 0.40 -> mean unchanged.
    assert by_day[16].value == pytest.approx(0.40, abs=1e-9)


def test_fill_value_reduces_to_missing(smap_nc):
    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(smap_nc):
    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("smap:cell:")


def test_window_trim_half_open(smap_nc):
    conn = SMAPSoilMoistureConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = SMAPSoilMoistureConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION against the native SYMFLUENCE SMAP semantics.
#
# Native reference (SYMFLUENCE):
#   * data/acquisition/handlers/smap.py  (earthaccess streaming path, the one
#     that carries the fill/clip mask): for each granule it builds
#         mask = (lat in bbox) & (lon in bbox) & (sm != -9999) & (sm > 0) & (sm < 1)
#     then takes float(np.mean(sm[mask])) — an UNWEIGHTED arithmetic mean over the
#     surviving cells, in the native unit m³/m³ (identity, no scaling).
#   * data/observation/handlers/soil_moisture.py SMAPHandler.process():
#         ds[var].mean(dim=<all non-time dims>)  — also an UNWEIGHTED mean,
#     identity unit.
#
# COS (smap_sm.py + core/reduce.basin_mean) applies the SAME fill/clip mask
# (== -9999 | !finite | <=0 | >=1 -> NaN) and the SAME identity unit, but
# reduces with a COS-LATITUDE AREA-WEIGHTED mean instead of an unweighted one.
#
# Therefore the only semantic difference is the weighting. The reductions are:
#   * EXACTLY equal when the surviving field is uniform, single-cell, or
#     single-latitude-row (the cos-lat weights cancel or there is one weight);
#   * within a tiny relative tolerance over a narrow-latitude bbox, where
#     cos(lat) varies negligibly across rows. This is the documented,
#     benign approximation (papers/cos_design.md §2; same posture as GRACE).
#
# The fixtures below reimplement the native unweighted reduction inline and
# assert COS == native at the appropriate tolerance.
# --------------------------------------------------------------------------


def _native_unweighted_basin_mean(lats, lons, values, bbox):
    """Reimplement the NATIVE SMAP reduction: masked UNWEIGHTED mean per step.

    Mirrors the earthaccess-streaming path of the native handler: fill/clip
    mask identical to the connector's, then a plain arithmetic mean (no cos-lat
    weighting) over surviving in-bbox cells. Returns a length-time vector with
    NaN where no cell survives (-> the native handler would emit nothing / the
    canonical contract emits MISSING).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # Match reduce._normalize_lons: shift request lon into 0-360 grid convention.
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
        layer = sub[t]
        # Native fill/clip mask (identical to the connector's invalid mask).
        valid = np.isfinite(layer) & (layer != FILL_VALUE) & (layer > 0.0) & (layer < 1.0)
        if valid.any():
            out[t] = float(np.mean(layer[valid]))  # UNWEIGHTED native mean
    return out


def test_parity_uniform_field_exact(smap_nc):
    """Uniform / out-of-range-masked layers: COS cos-lat mean == native unweighted
    mean to float tolerance (weights cancel when survivors share a value)."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(smap_nc) as ds:
        lats = np.asarray(ds["lat"].values, dtype="float64")
        lons = np.asarray(ds["lon"].values, dtype="float64")
        values = np.asarray(ds["soil_moisture"].values, dtype="float64")

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_unweighted_basin_mean(lats, lons, values, bbox)

    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 6, 30, tzinfo=UTC),
    )
    cos_by_day = {p.timestamp.day: p.value for p in series.points}
    # days 15,16,17 are uniform-survivor layers -> exact agreement.
    for day, idx in ((15, 0), (16, 1), (17, 2)):
        assert cos_by_day[day] == pytest.approx(native[idx], abs=1e-12)


def test_parity_unit_factor_is_identity(smap_nc):
    """The boundary unit conversion is the identity: source m³/m³ == canonical
    m³/m³, so COS values equal the native (unscaled) values, not a scaled copy."""
    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 6, 30, tzinfo=UTC),
    )
    assert series.unit == "m3/m3"
    by_day = {p.timestamp.day: p.value for p in series.points}
    # Source layer was literally 0.30 m³/m³ -> canonical 0.30 (factor 1.0).
    assert by_day[17] == pytest.approx(0.30, abs=1e-12)


def test_parity_narrow_bbox_relative_tolerance(tmp_path):
    """Over a multi-latitude bbox with a NON-uniform survivor field, COS's
    cos-lat-weighted mean differs from the native unweighted mean only by the
    weighting. Across a narrow latitude span the difference is < 1e-3 relative.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    # ~0.18 deg latitude span: cos(lat) varies negligibly across rows, so the
    # cos-lat-weighted (COS) and unweighted (native) means agree to < 1e-3
    # relative even for a steep meridional gradient in the survivor field. A
    # wider span (e.g. 1 deg with the same 6x gradient) pushes the benign
    # divergence to ~4e-3 — still tolerance-bounded, just above 1e-3.
    lats = np.array([50.0, 50.09, 50.18])
    lons = np.array([244.0, 245.0])
    # Non-uniform survivors so the weighting actually bites.
    data = np.array([[[0.10, 0.20],
                      [0.30, 0.40],
                      [0.50, 0.60]]], dtype="float64")
    ds = xr.Dataset(
        {"soil_moisture": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "smap_narrow.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)[0]

    conn = SMAPSoilMoistureConnector()
    spec = ReductionSpec(domain_name="narrow", bbox=bbox, centroid=(50.5, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 6, 30, tzinfo=UTC),
    )
    cos_val = series.points[0].value
    # Benign cos-lat-vs-unweighted divergence over a narrow bbox.
    assert cos_val == pytest.approx(native, rel=1e-3)
    # ...and it is genuinely NOT bitwise-equal (weighting is real, not a no-op).
    assert abs(cos_val - native) > 0


def test_parity_single_latitude_row_exact(tmp_path):
    """If all survivors lie on ONE latitude row, the single cos-lat weight is a
    common factor that cancels -> COS == native EXACTLY even for a non-uniform
    field. Pins the claim that the divergence is purely the across-row weighting.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.full((1, 3, 3), FILL_VALUE, dtype="float64")
    # Only the middle latitude row (lat=51) carries valid, non-uniform values.
    data[0, 1, :] = [0.10, 0.20, 0.45]
    ds = xr.Dataset(
        {"soil_moisture": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "smap_onerow.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)[0]
    assert native == pytest.approx((0.10 + 0.20 + 0.45) / 3.0, abs=1e-12)

    conn = SMAPSoilMoistureConnector()
    spec = ReductionSpec(domain_name="onerow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 6, 30, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(native, abs=1e-12)


def test_parity_fill_rule_matches_native_missing(smap_nc):
    """Native fill/clip rule (== -9999 | <=0 | >=1) -> the all-fill layer yields
    NO surviving cells; the canonical contract surfaces that as MISSING/None,
    matching the native handler emitting nothing for that step."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(smap_nc) as ds:
        lats = np.asarray(ds["lat"].values, dtype="float64")
        lons = np.asarray(ds["lon"].values, dtype="float64")
        values = np.asarray(ds["soil_moisture"].values, dtype="float64")
    native = _native_unweighted_basin_mean(lats, lons, values, (50.0, -116.0, 52.0, -114.0))
    assert np.isnan(native[3])  # native: no survivors on the all-fill layer

    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 6, 30, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_parity_window_trim_half_open_matches_native(smap_nc):
    """Half-open [start, end) UTC trim: a step exactly at `end` is excluded,
    exactly as a native loc[index >= start & index < end] window would drop it."""
    conn = SMAPSoilMoistureConnector()
    series = conn.reduce_file(
        smap_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 18, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    # 06-18 == end is excluded (half-open); 15,16,17 retained.
    assert days == {15, 16, 17}
