# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GLEAM ET connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory GLEAM-like NetCDF (variable ``E`` in mm/day) and
reduces it; no network, no GLEAM credentials. Proves the gridded -> canonical
ET-series path, the canonical mm/day unit, and half-open window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.gleam_et import GLEAMETConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def gleam_nc(tmp_path):
    """A synthetic GLEAM-like NetCDF: E (mm/day) over a small 0-360 grid."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2015-01-15", "2015-02-15", "2015-03-15", "2015-04-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    # mm/day: constant per timestep so basin-mean == that constant.
    data = np.empty((4, 3, 3))
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    ds = xr.Dataset(
        {"E": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gleam_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_units_mm_per_day(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == "mm/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "gleam_et:domain:bow"
    # Constant-per-layer field -> basin-mean equals the layer constant.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[1] == pytest.approx(1.0, abs=1e-9)
    assert by_month[2] == pytest.approx(2.0, abs=1e-9)
    assert by_month[4] == pytest.approx(4.0, abs=1e-9)


def test_unit_conversion_multiplier(gleam_nc):
    # Mirrors native ET_UNIT_CONVERSION: apply a boundary multiplier.
    conn = GLEAMETConnector(config={"unit_conversion": 0.5})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[4] == pytest.approx(2.0, abs=1e-9)  # 4.0 * 0.5


def test_small_basin_defaults_to_nearest_cell(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("gleam_et:cell:")
    # Constant layers -> nearest-cell value equals the layer constant.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[3] == pytest.approx(3.0, abs=1e-9)


def test_window_trim_half_open(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2015-02-01, 2015-04-15): includes 02-15 and 03-15, excludes 04-15.
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 2, 1, tzinfo=UTC), datetime(2015, 4, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert months == {2, 3}
    assert 4 not in months  # half-open excludes the exact end timestamp


def test_variable_autodetect_evaporation_name(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2015-01-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    data = np.full((1, 2, 2), 2.5)
    ds = xr.Dataset(
        {"evaporation": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    path = tmp_path / "gleam_alt.nc"
    ds.to_netcdf(path)

    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
                         centroid=(50.5, -115.5), area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.source_info["variable"] == "evaporation"
    assert series.points[0].value == pytest.approx(2.5, abs=1e-9)


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION vs the SYMFLUENCE native handler.
#
# Native ref: symfluence/data/observation/handlers/gleam.py::GLEAMETHandler.process
#   line 92:  et_mean = et_data.mean(dim=[d for d in et_data.dims if d != 'time'])
#
# The native basin reduction is an UNWEIGHTED arithmetic mean over the bbox-
# subset lat/lon cells (xarray ``.mean()`` is plain mean, ``skipna=True`` so
# NaN/fill cells drop out). Unit handling is the identity, mm/day -> mm/day,
# with an OPTIONAL ``ET_UNIT_CONVERSION`` boundary multiplier (gleam.py:103-108).
#
# COS's basin_mean (cos.core.reduce.basin_mean) is a COSINE-LATITUDE AREA-
# WEIGHTED mean (also NaN-skipping). This is the one deliberate, documented
# divergence (see reduce.py module docstring: "a documented approximation of
# full polygon-weighted zonal stats — basin-mean parity is tolerance-based").
#
# These tests reimplement the NATIVE semantics inline on the SAME synthetic
# grid and assert COS == native:
#   * EXACT (float tol) for a constant field and for a single-latitude-row
#     bbox (cos-lat weights cancel),
#   * relative ~1e-3 over a small/narrow-latitude bbox (weighted vs unweighted),
#   * EXACT for the unit-conversion factor and for NaN/fill -> MISSING.
# ---------------------------------------------------------------------------


def _native_unweighted_basin_mean(lats, lons, values, bbox, conversion=None):
    """Reimplements gleam.py::process basin reduction inline.

    bbox = (lat_min, lon_min, lat_max, lon_max) to match ReductionSpec.bbox.
    Selects cells with centers inside the (sorted) bbox, takes a plain NaN-
    skipping arithmetic mean over lat+lon per timestep, then applies the
    optional ET_UNIT_CONVERSION multiplier (after the mean — the native code
    multiplies the reduced DataFrame, but for a linear factor order is
    irrelevant). Returns a length-time vector (NaN where no finite cell).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # native shifts negative request lons into 0-360 when the grid is 0-360
    if lons.size and np.nanmax(lons) > 180 and (lon_min < 0 or lon_max < 0):
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    lat_min, lat_max = sorted([lat_min, lat_max])
    lon_min, lon_max = sorted([lon_min, lon_max])
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]
    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.mean(layer[finite]))  # UNWEIGHTED, NaN-skipping
    if conversion is not None:
        out = out * float(conversion)
    return out


def _read_grid(nc_path, conn):
    """Pull (lats, lons, values) the way the connector does, for the inline native calc."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(nc_path) as ds:
        var = conn._select_et_variable(ds)
        lat_name = next(n for n in ds.coords if str(n).lower() in {"lat", "latitude"})
        lon_name = next(n for n in ds.coords if str(n).lower() in {"lon", "longitude"})
        da = ds[var].transpose("time", lat_name, lon_name)
        return (
            np.asarray(ds[lat_name].values, dtype="float64"),
            np.asarray(ds[lon_name].values, dtype="float64"),
            np.asarray(da.values, dtype="float64"),
        )


def test_parity_constant_field_exact(gleam_nc):
    """Constant-per-layer field: cos-lat weighted (COS) == unweighted (native), exactly."""
    conn = GLEAMETConnector()
    bbox = (50.0, -116.0, 52.0, -114.0)
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    lats, lons, values = _read_grid(gleam_nc, conn)
    native = _native_unweighted_basin_mean(lats, lons, values, bbox)
    cos = [p.value for p in series.points]
    assert len(cos) == len(native)
    for c, n in zip(cos, native):
        assert c == pytest.approx(n, abs=1e-12)


def test_parity_narrow_latitude_bbox_within_relative_tol(tmp_path):
    """Non-constant field over a narrow-latitude bbox: cos-lat weighted ~ unweighted.

    The grid spans lat 50-52 (~2 deg). Over this band the cos-lat weights
    differ from uniform by only ~0.5% top-to-bottom, so for a *physically
    realistic* (gently lat-varying) ET field the area-weighted mean (COS) and
    the unweighted mean (native) agree to ~1e-3 relative. We assert that
    documented tolerance and ALSO that the weighted result is NOT bitwise-equal
    (proving the weighting is real).

    Note: the divergence scales with the field's latitude gradient, not the
    bbox width alone. A sharply lat-varying field would exceed 1e-3 — which is
    exactly why COS documents basin-mean parity as tolerance-based, and why a
    real basin objective (a smooth ET field) stays inside it.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2015-01-15", "2015-02-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    # gently latitude-varying field (realistic ET gradient): the cos-lat
    # weighting bites, but stays inside the documented ~1e-3 tolerance.
    data = np.empty((2, 3, 3))
    data[0] = np.array([[2.98, 2.98, 2.98], [3.00, 3.00, 3.00], [3.02, 3.02, 3.02]])
    data[1] = data[0] + 1.0
    ds = xr.Dataset(
        {"E": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gleam_latvary.nc"
    ds.to_netcdf(path)

    conn = GLEAMETConnector()
    bbox = (50.0, -116.0, 52.0, -114.0)
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(
        path, spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)
    cos = np.array([p.value for p in series.points])
    # within documented cos-lat tolerance...
    assert np.allclose(cos, native, rtol=1e-3, atol=0.0)
    # ...but the weighting is genuinely applied (not identical to unweighted).
    assert not np.allclose(cos, native, rtol=0.0, atol=1e-9)


def test_parity_single_latitude_row_exact(tmp_path):
    """A single-lat-row bbox: only one cos-lat weight, so weighted == unweighted exactly."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2015-01-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.empty((1, 3, 3))
    data[0] = np.array([[2.0, 4.0, 6.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]])
    ds = xr.Dataset(
        {"E": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gleam_onerow.nc"
    ds.to_netcdf(path)

    conn = GLEAMETConnector()
    # bbox selects only the lat=50.0 row (lat_max just below 51).
    bbox = (49.5, -116.0, 50.5, -114.0)
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(50.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(
        path, spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)
    cos = series.points[0].value
    # native = mean(2,4,6) = 4.0; single lat row -> COS weight cancels exactly.
    assert native == pytest.approx(4.0, abs=1e-12)
    assert cos == pytest.approx(native, abs=1e-12)


def test_parity_unit_conversion_factor_exact(gleam_nc):
    """ET_UNIT_CONVERSION boundary multiplier: COS factor == native factor, exactly."""
    bbox = (50.0, -116.0, 52.0, -114.0)
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    # COS path with the native ET_UNIT_CONVERSION key name.
    conn = GLEAMETConnector(config={"ET_UNIT_CONVERSION": 0.5})
    series = conn.reduce_file(
        gleam_nc, spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    lats, lons, values = _read_grid(gleam_nc, conn)
    native = _native_unweighted_basin_mean(lats, lons, values, bbox, conversion=0.5)
    cos = [p.value for p in series.points]
    for c, n in zip(cos, native):
        assert c == pytest.approx(n, abs=1e-12)


def test_parity_fill_missing_maps_to_missing_quality(tmp_path):
    """NaN/fill: native .mean(skipna=True) drops it; an all-NaN layer -> MISSING.

    Mirrors native: a finite cell is averaged (skipna), an all-fill timestep
    yields NaN -> COS emits QualityFlag.MISSING with value None.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    from cos.core.models import QualityFlag

    times = np.array(["2015-01-15", "2015-02-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    data = np.empty((2, 2, 2))
    # t0: one finite cell + NaNs -> native skipna mean = the finite value.
    data[0] = np.array([[5.0, np.nan], [np.nan, np.nan]])
    # t1: all NaN -> native mean = NaN -> MISSING.
    data[1] = np.full((2, 2), np.nan)
    ds = xr.Dataset(
        {"E": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gleam_fill.nc"
    ds.to_netcdf(path)

    conn = GLEAMETConnector()
    bbox = (49.0, -117.0, 52.0, -114.0)
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(50.5, -115.5),
                         area_km2=8000.0)
    series = conn.reduce_file(
        path, spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    native = _native_unweighted_basin_mean(lats, lons, data, bbox)
    pts = {p.timestamp.month: p for p in series.points}
    # t0: finite cell present -> matches native (skipna) value, GOOD.
    assert pts[1].value == pytest.approx(native[0], abs=1e-12) == pytest.approx(5.0)
    assert pts[1].quality == QualityFlag.GOOD
    # t1: all-fill -> native NaN -> COS MISSING / None.
    assert np.isnan(native[1])
    assert pts[2].value is None
    assert pts[2].quality == QualityFlag.MISSING
