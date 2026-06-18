"""ASCAT soil-moisture connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory ASCAT-like NetCDF (degree of saturation) and
reduces it; no network, no auth. This proves the architecture-critical gridded
-> canonical-series path for a C-band active-microwave product: the native
saturation -> volumetric conversion (percentage detection, porosity scaling),
physical-range masking -> MISSING, basin-mean vs nearest-cell reduction, and the
half-open UTC window trim. Canonical unit is m³/m³.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ascat_sm import (
    DEFAULT_POROSITY,
    ASCATSoilMoistureConnector,
    _canonicalize_saturation,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ascat_nc(tmp_path):
    """Synthetic ASCAT-like NetCDF: surface_soil_moisture_saturation as percent.

    Four timesteps on a 3x3 grid (0-360 longitudes = -116..-114). Saturation is
    stored as a percentage (0-100), the common CDS representation, so the
    connector must divide by 100 then multiply by porosity. The last timestep is
    entirely NaN (no retrieval) so it must reduce to MISSING; one cell in an
    otherwise-valid layer is out of range to exercise the physical-range mask.
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
    data[0] = 40.0          # 40% saturation -> 0.40 frac -> *0.45 = 0.18 m³/m³
    data[1] = 80.0          # 80% saturation -> 0.80 frac -> *0.45 = 0.36 m³/m³
    data[1, 0, 0] = 500.0   # absurd value: /100 -> 5.0 -> *0.45 = 2.25 -> masked
    data[2] = 60.0          # 60% -> 0.60 -> *0.45 = 0.27
    data[3] = np.nan        # entirely missing -> MISSING
    ds = xr.Dataset(
        {"surface_soil_moisture_saturation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ascat_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_canonicalize_saturation_percent_to_volumetric():
    """Pure helper: 40% saturation -> 0.40 frac -> *0.45 porosity = 0.18 m³/m³."""
    arr = np.full((1, 2, 2), 40.0)
    out = _canonicalize_saturation(arr, "surface_soil_moisture_saturation", DEFAULT_POROSITY)
    assert np.allclose(out, 0.18)


def test_canonicalize_saturation_already_fraction():
    """A saturation fraction (0-1) skips the /100 step but still scales by porosity."""
    arr = np.full((1, 2, 2), 0.40)
    out = _canonicalize_saturation(arr, "surface_soil_moisture_saturation", DEFAULT_POROSITY)
    assert np.allclose(out, 0.18)


def test_canonicalize_masks_out_of_range():
    """Values that fall outside 0 < sm < 1 after conversion become NaN (MISSING)."""
    arr = np.array([[[500.0]]])  # /100 -> 5.0 -> *0.45 = 2.25 -> out of range
    out = _canonicalize_saturation(arr, "saturation", DEFAULT_POROSITY)
    assert np.isnan(out).all()


def test_reduce_file_basin_mean_units_and_values(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # canonical volumetric, converted from saturation
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "ascat:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # 40% saturation -> 0.40 -> *0.45 porosity = 0.18 m³/m³.
    assert by_day[15].value == pytest.approx(0.18, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # 80% layer with one absurd (masked) cell -> remaining cells 0.36 m³/m³.
    assert by_day[16].value == pytest.approx(0.36, abs=1e-9)
    # 60% -> 0.27 m³/m³.
    assert by_day[17].value == pytest.approx(0.27, abs=1e-9)


def test_all_missing_reduces_to_missing(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_porosity_override(ascat_nc):
    """A config porosity overrides the default in the saturation conversion."""
    conn = ASCATSoilMoistureConnector(config={"porosity": 0.50})
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # 40% -> 0.40 -> *0.50 = 0.20 m³/m³.
    assert by_day[15].value == pytest.approx(0.20, abs=1e-9)


def test_small_basin_defaults_to_nearest_cell(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ascat:cell:")


def test_window_trim_half_open(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = ASCATSoilMoistureConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION                                                       #
#                                                                             #
# The native reference is symfluence's ASCATSMHandler.process (observation    #
# handler: data/observation/handlers/soil_moisture.py). Per timestep it:      #
#   1. valid = ~isnan(layer)                                                   #
#   2. if nanmean(valid) > 1.0:  valid /= 100.0          (percentage -> frac)  #
#   3. if 'saturation' in var or nanmean(valid) > 0.5:                         #
#          valid *= porosity                              (sat -> volumetric)  #
#   4. phys = (valid > 0) & (valid < 1)                   (strict physical mask)#
#   5. value = float(np.mean(valid[phys]))               (UNWEIGHTED mean)     #
#                                                                             #
# COS reproduces 1-4 bit-for-bit in `_canonicalize_saturation` (it masks the  #
# same strict 0<sm<1 range to NaN). The ONE deliberate divergence is step 5:  #
# COS's `basin_mean` takes a cos(latitude) AREA-WEIGHTED mean over the in-box  #
# finite cells, whereas native takes an UNWEIGHTED arithmetic mean. This is    #
# the same documented gridded-reduction approximation GRACE uses (see         #
# cos/core/reduce.py module docstring); it is benign for the soil-moisture     #
# objective because (a) for a spatially constant layer the two are bitwise     #
# identical, and (b) over a narrow-latitude basin the cos-lat weights are      #
# nearly uniform so the means agree to ~1e-3. The native percentage/porosity/  #
# fill handling — the parts that would corrupt the objective if wrong — match  #
# EXACTLY.                                                                      #
# --------------------------------------------------------------------------- #


def _native_reduce_timestep(layer, var_name, porosity):
    """Reimplements ASCATSMHandler.process's per-timestep reduction (native).

    Returns the native float value for one (lat, lon) layer, or None if the
    layer reduces to MISSING (no finite, or no in-physical-range cells). This
    is a faithful inline copy of the native semantics — UNWEIGHTED np.mean.
    """
    sm_slice = np.asarray(layer, dtype="float64")
    valid_mask = ~np.isnan(sm_slice)
    if not np.any(valid_mask):
        return None
    sm_valid = sm_slice[valid_mask]
    if np.nanmean(sm_valid) > 1.0:
        sm_valid = sm_valid / 100.0
    if "saturation" in var_name.lower() or np.nanmean(sm_valid) > 0.5:
        sm_valid = sm_valid * porosity
    phys_mask = (sm_valid > 0) & (sm_valid < 1)
    if not np.any(phys_mask):
        return None
    return float(np.mean(sm_valid[phys_mask]))


def _make_nc(tmp_path, times, lats, lons, data, var="surface_soil_moisture_saturation"):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), np.asarray(data, dtype="float64"))},
        coords={"time": np.asarray(times, dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    path = tmp_path / "ascat_parity.nc"
    ds.to_netcdf(path)
    return path


def test_parity_constant_field_is_bitwise_identical_to_native(tmp_path):
    """Constant layers: cos-lat weighting == unweighted mean -> COS == native exactly.

    A spatially constant layer makes the ONLY divergence (weighting) vanish, so
    the COS basin_mean must equal the native unweighted mean to float tolerance.
    This also exercises percentage->fraction, saturation->porosity, and the
    strict physical-range mask through the full connector path.
    """
    times = ["2020-06-15", "2020-06-16", "2020-06-17"]
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 == -116..-114
    var = "surface_soil_moisture_saturation"
    data = np.empty((3, 3, 3))
    data[0] = 40.0   # -> 0.40 -> *0.45 = 0.18
    data[1] = 80.0   # -> 0.80 -> *0.45 = 0.36
    data[2] = 60.0   # -> 0.60 -> *0.45 = 0.27
    path = _make_nc(tmp_path, times, lats, lons, data, var=var)

    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}

    # Native: canonicalize the SAME bbox cells, unweighted mean.
    porosity = DEFAULT_POROSITY
    expected = {
        15: _native_reduce_timestep(data[0], var, porosity),
        16: _native_reduce_timestep(data[1], var, porosity),
        17: _native_reduce_timestep(data[2], var, porosity),
    }
    for day, exp in expected.items():
        # Constant layer: cos-lat weighting collapses to the unweighted mean, so
        # COS and native agree to machine epsilon.
        assert by_day[day].value == pytest.approx(exp, abs=1e-12, rel=0.0), f"day {day}"
        assert by_day[day].quality == QualityFlag.GOOD
    # And the canonical unit is the native objective's unit.
    assert series.unit == "m3/m3"


def test_parity_varying_field_cos_lat_vs_native_unweighted_within_tol(tmp_path):
    """Spatially varying layer: bound the cos-lat vs native-unweighted divergence.

    With a non-constant layer the deliberate weighting divergence is exposed.
    Over this narrow 2-degree-latitude bbox the cos-lat weights are nearly
    uniform, so COS (area-weighted) must still match native (unweighted) to a
    tight relative tolerance. We assert both that they are CLOSE (parity holds)
    and that they are NOT bitwise identical (the divergence is real, documented).
    """
    times = ["2020-06-15"]
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    var = "surface_soil_moisture_saturation"
    # A gradient in saturation-percent across the grid (all stay in-range post
    # conversion: 0.45*[0.30..0.70] = 0.135..0.315).
    layer = np.array([
        [30.0, 40.0, 50.0],
        [45.0, 55.0, 65.0],
        [50.0, 60.0, 70.0],
    ])
    data = layer[np.newaxis, ...]
    path = _make_nc(tmp_path, times, lats, lons, data, var=var)

    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_value = series.points[0].value
    native_value = _native_reduce_timestep(layer, var, DEFAULT_POROSITY)

    # Parity bound: this is a deliberately adversarial fixture — the saturation
    # gradient is aligned with latitude, which MAXIMISES the cos-lat vs
    # unweighted divergence. Even so, over this 2-degree-latitude bbox the
    # cos-weight spread (cos50..cos52 ~= 0.643..0.616, ~4%) caps the divergence
    # at ~2.8e-3 relative. A real basin's field is not perfectly lat-correlated,
    # so the operational divergence is smaller; 3e-3 is the worst-case envelope.
    assert cos_value == pytest.approx(native_value, rel=3e-3)
    # Sanity: the divergence is genuine (weights differ across the 3 latitudes),
    # so they are close but not bitwise identical.
    assert cos_value != native_value
    # Document the measured worst-case so a regression in either direction trips.
    assert abs(cos_value - native_value) / native_value < 3e-3


def test_parity_unit_and_porosity_factor_match_native(tmp_path):
    """The porosity conversion factor is identical to native for a constant field."""
    times = ["2020-06-15"]
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    var = "surface_soil_moisture_saturation"
    data = np.full((1, 2, 2), 40.0)  # 40% saturation everywhere
    path = _make_nc(tmp_path, times, lats, lons, data, var=var)

    for porosity in (0.40, 0.45, 0.50):
        conn = ASCATSoilMoistureConnector(config={"porosity": porosity})
        series = conn.reduce_file(
            path, _spec(8000.0),
            datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
        )
        native = _native_reduce_timestep(data[0], var, porosity)
        # Constant field -> cos-lat == unweighted; identical float ops both
        # paths, so they agree to machine epsilon (not bitwise only because of
        # associativity in the weighted sum).
        assert series.points[0].value == pytest.approx(native, abs=1e-12, rel=0.0)
        # 0.40 frac * porosity is the exact volumetric value.
        assert series.points[0].value == pytest.approx(0.40 * porosity, abs=1e-12)


def test_parity_fill_and_out_of_range_to_missing_match_native(tmp_path):
    """Fill (NaN) and out-of-physical-range cells reduce to MISSING exactly as native.

    Two timesteps: one entirely NaN (native: no valid -> dropped/MISSING), one
    where every cell is out of range after conversion (native: phys_mask empty
    -> dropped/MISSING). COS must emit a MISSING point (value None) for both.
    """
    times = ["2020-06-15", "2020-06-16"]
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    var = "surface_soil_moisture_saturation"
    data = np.empty((2, 2, 2))
    data[0] = np.nan          # entirely missing
    data[1] = 500.0           # /100 -> 5.0 -> *0.45 = 2.25 -> all out of range
    path = _make_nc(tmp_path, times, lats, lons, data, var=var)

    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}

    # Native reduces both layers to None (MISSING).
    assert _native_reduce_timestep(data[0], var, DEFAULT_POROSITY) is None
    assert _native_reduce_timestep(data[1], var, DEFAULT_POROSITY) is None
    for day in (15, 16):
        assert by_day[day].value is None
        assert by_day[day].quality == QualityFlag.MISSING


def test_parity_half_open_window_matches_native_slice(tmp_path):
    """COS's half-open [start, end) UTC trim matches a native pandas [start, end) slice.

    Native applies `df.loc[(idx >= start) & (idx <= end)]` (closed), but COS's
    canonical contract is half-open [start, end). We verify COS keeps exactly the
    timestamps in [start, end) and drops the boundary `end` timestamp — the
    canonical-contract behaviour the kind's downstream join relies on.
    """
    times = ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"]
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    var = "surface_soil_moisture_saturation"
    data = np.full((4, 2, 2), 40.0)
    path = _make_nc(tmp_path, times, lats, lons, data, var=var)

    conn = ASCATSoilMoistureConnector()
    start = datetime(2020, 6, 15, tzinfo=UTC)
    end = datetime(2020, 6, 17, tzinfo=UTC)
    series = conn.reduce_file(path, _spec(8000.0), start, end)
    kept = sorted(p.timestamp.day for p in series.points)
    # Half-open: 15, 16 in; 17 (== end) and 18 excluded.
    assert kept == [15, 16]
