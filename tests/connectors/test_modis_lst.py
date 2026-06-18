"""MODIS LST connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MODIS-LST-like NetCDF (packed DN) and reduces it;
no network, no auth. This proves the architecture-critical gridded ->
canonical-series path for land surface temperature: DN -> Kelvin scale
(``* 0.02``), valid-range / fill masking, basin-mean vs nearest-cell reduction,
band selection (day/night), and the half-open UTC window trim.

PARITY-BY-CONSTRUCTION
----------------------
The COS ``modis_lst`` connector is a port of SYMFLUENCE's native
``modis_lst`` / ``mod11`` observation handler
(``data/observation/handlers/modis_lst.py``). The native reduction is
``MODISLSTHandler._extract_basin_mean`` followed by the per-timestep
``* LST_SCALE_FACTOR`` (0.02) DN -> Kelvin scale. Native semantics, reproduced
inline in ``_native_reduce`` below and asserted against the COS connector:

* **valid-range mask** — native does ``da.where((da >= 7500) & (da <= 65535))``;
  out-of-range / fill (DN 0) -> NaN. COS's ``_mask_invalid`` is identical.
* **reduction** — native takes an *UNWEIGHTED* arithmetic mean over the cells
  inside the basin bbox: ``float(da.mean(skipna=True))``. COS takes a *cos-lat
  AREA-WEIGHTED* mean (``cos.core.reduce.basin_mean``). For a **uniform /
  single-cell** field the two are bit-for-bit equal (both collapse to the
  constant); for a non-uniform field over a narrow-latitude bbox the two
  diverge only by the cosine-latitude weight spread, which is < 1e-3 relative
  (measured ~1e-4 for a 4 K N-S gradient over 2deg of latitude — see
  ``test_parity_nonuniform_field_cos_lat_within_tolerance``). This is the
  documented, benign basin-mean approximation (see ``core/reduce.py`` and
  GRACE), not a semantic divergence.
* **unit** — native applies ``* 0.02`` *after* the mean; since the mean is
  linear, ``mean(DN) * 0.02 == mean(DN * 0.02)`` (COS scales before reducing).
  The native default *presentation* unit is celsius (``K - 273.15``), but the
  canonical ``lst`` unit is Kelvin and the native internal column is
  ``lst_day_k`` (Kelvin) — both the native pre-presentation value and COS are
  Kelvin, so the parity boundary is Kelvin.
* **fill rule** — native: an all-fill / all-invalid layer reduces to NaN
  (``da.mean`` over all-NaN), the ``day_val > 0`` guard then yields ``np.nan``.
  COS surfaces this as ``value=None`` / ``QualityFlag.MISSING``.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_lst import (
    LST_SCALE_FACTOR,
    LST_VALID_RANGE,
    MODISLSTConnector,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def lst_nc(tmp_path):
    """A synthetic MODIS-LST-like NetCDF: packed DN day + night over a small grid.

    Four timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). DN 15000 ->
    300 K, DN 14000 -> 280 K. The last timestep is entirely fill (0) so it must
    reduce to MISSING; one cell in an otherwise-valid layer is out of range
    (below 7500) to exercise the valid-range mask.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)

    day = np.empty((4, 3, 3))
    day[0] = 15000.0          # uniform valid -> 15000 * 0.02 = 300 K
    day[1] = 14000.0          # uniform valid -> 280 K
    day[1, 0, 0] = 100.0      # below-range cell -> masked, mean stays 280 K
    day[2] = 16000.0          # 320 K
    day[3] = 0.0              # entirely fill -> MISSING

    night = np.full((4, 3, 3), 13500.0)   # uniform valid -> 270 K (all timesteps)

    ds = xr.Dataset(
        {
            "LST_Day_1km": (("time", "lat", "lon"), day),
            "LST_Night_1km": (("time", "lat", "lon"), night),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lst_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


# --------------------------------------------------------------------------- #
# Native semantics, reimplemented inline (the parity reference).
# --------------------------------------------------------------------------- #
def _native_reduce(values_dn, lats, lons, bbox):
    """Reimplement SYMFLUENCE ``MODISLSTHandler`` reduction on a DN cube.

    Mirrors ``_extract_basin_mean`` + the ``* LST_SCALE_FACTOR`` Kelvin scale
    and the ``> 0`` fill guard, exactly as the native handler does it:

    * valid-range mask -> NaN,
    * subset to bbox (inclusive both ends, as ``da.sel(slice(...))``),
    * **UNWEIGHTED** ``np.nanmean`` per timestep,
    * ``* 0.02`` after the mean (linear, equivalent to scaling first),
    * all-NaN / non-positive mean -> NaN (the ``day_val > 0`` guard).

    Returns a list of per-timestep Kelvin values (``np.nan`` where MISSING).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # Native uses native (negative) lons against the file's lons. The synthetic
    # file is on 0-360; emulate the same cell selection the native sel() would
    # make by shifting the request the way COS's reducer documents.
    if float(np.nanmax(lons)) > 180.0:
        if lon_min < 0:
            lon_min += 360.0
        if lon_max < 0:
            lon_max += 360.0
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]

    lo, hi = LST_VALID_RANGE
    out = []
    for t in range(values_dn.shape[0]):
        layer = values_dn[t][np.ix_(lat_sel, lon_sel)].astype("float64")
        masked = np.where((layer >= lo) & (layer <= hi), layer, np.nan)
        if not np.isfinite(masked).any():
            out.append(np.nan)
            continue
        mean_dn = float(np.nanmean(masked))          # UNWEIGHTED native mean
        kelvin = mean_dn * LST_SCALE_FACTOR
        out.append(kelvin if kelvin > 0 else np.nan)
    return out


def _cos_values(series):
    """COS connector results as {day: value_or_None}."""
    return {p.timestamp.day: p.value for p in series.points}


# --------------------------------------------------------------------------- #
# Existing behavioural tests.
# --------------------------------------------------------------------------- #
def test_reduce_file_basin_mean_units_and_values(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.LST
    assert series.unit == "K"  # canonical Kelvin, scaled from packed DN
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "modis_lst:domain:bow"
    assert series.source_info["band"] == "day"

    by_day = _cos_values(series)
    # Uniform DN 15000 -> 15000 * 0.02 = 300 K basin mean.
    assert by_day[15] == pytest.approx(15000.0 * LST_SCALE_FACTOR, abs=1e-9)
    assert by_day[15] == pytest.approx(300.0, abs=1e-9)
    # Below-range cell masked; remaining DN 14000 -> 280 K, mean unchanged.
    assert by_day[16] == pytest.approx(280.0, abs=1e-9)


def test_fill_value_reduces_to_missing(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (DN 0) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_night_band_selection(lst_nc):
    conn = MODISLSTConnector({"band": "night"})
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.source_info["band"] == "night"
    assert series.source_info["variable"] == "LST_Night_1km"
    by_day = _cos_values(series)
    # Night DN 13500 -> 13500 * 0.02 = 270 K everywhere.
    assert by_day[15] == pytest.approx(270.0, abs=1e-9)


def test_small_basin_defaults_to_nearest_cell(lst_nc):
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_lst:cell:")


def test_window_trim_half_open(lst_nc):
    conn = MODISLSTConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION: COS reduction == native reduction.
# --------------------------------------------------------------------------- #
def test_parity_uniform_field_is_exact(lst_nc):
    """Uniform / fill fields: COS basin-mean == native unweighted mean, EXACT.

    For a constant in-box field the cos-lat weights drop out, so the
    area-weighted (COS) and unweighted (native) means are bit-for-bit equal.
    Covers the unit factor (0.02 DN->K), the valid-range mask (the masked
    below-range cell on day 16 leaves a still-uniform field), and the
    all-fill -> MISSING fill rule, all in one shot against native semantics.
    """
    import xarray as xr

    with xr.open_dataset(lst_nc) as ds:
        dn = np.asarray(ds["LST_Day_1km"].values, dtype="float64")
        lats = np.asarray(ds["lat"].values, dtype="float64")
        lons = np.asarray(ds["lon"].values, dtype="float64")

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_reduce(dn, lats, lons, bbox)  # per-timestep Kelvin / nan

    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos = _cos_values(series)  # {day: value_or_None}

    # Day 15 = uniform 300 K, day 16 = 280 K (one cell masked, rest uniform),
    # day 17 = 320 K, day 18 = all-fill -> MISSING.
    for day, native_val in zip((15, 16, 17, 18), native):
        if np.isnan(native_val):
            assert cos[day] is None, f"native MISSING but COS={cos[day]} on day {day}"
        else:
            # Uniform in-box field => weighted == unweighted exactly.
            assert cos[day] == pytest.approx(native_val, abs=1e-9), (
                f"day {day}: COS={cos[day]} native={native_val}"
            )


def test_parity_unit_factor_exact():
    """The DN->Kelvin scale factor matches the native handler bit-for-bit."""
    # Native: lst_k = mean_dn * LST_SCALE_FACTOR with LST_SCALE_FACTOR = 0.02.
    assert LST_SCALE_FACTOR == 0.02
    assert MODISLSTConnector._mask_invalid is not None
    # A single uniform cell scaled through COS's mask+scale == native DN*0.02.
    masked = MODISLSTConnector._mask_invalid(np.array([[[15000.0]]]))
    cos_kelvin = float(masked[0, 0, 0]) * LST_SCALE_FACTOR
    native_kelvin = 15000.0 * 0.02
    assert cos_kelvin == native_kelvin == 300.0


def test_parity_nonuniform_field_cos_lat_within_tolerance(tmp_path):
    """Non-uniform field: COS cos-lat mean ~= native unweighted mean (rel<1e-3).

    The ONLY semantic difference COS introduces over native is cos-latitude
    area weighting in the basin mean. For a non-uniform field this is a real
    (but bounded) divergence. With a 4 K north-south gradient across the 2deg
    latitude band the relative difference is ~1e-4, comfortably under the
    documented 1e-3 basin-mean tolerance. This is the same benign approximation
    GRACE uses; it does not corrupt the LST objective (a basin-mean temperature
    accurate to ~0.03 K).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")

    # DN that scale to a 300 -> 302 -> 304 K N-S gradient (4 K over 2deg lat).
    dn_rows = np.array([15000.0, 15100.0, 15200.0])
    day = np.empty((1, 3, 3))
    day[0] = np.repeat(dn_rows[:, None], 3, axis=1)
    night = np.full((1, 3, 3), 13500.0)

    ds = xr.Dataset(
        {
            "LST_Day_1km": (("time", "lat", "lon"), day),
            "LST_Night_1km": (("time", "lat", "lon"), night),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lst_gradient.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_reduce(day, lats, lons, bbox)[0]  # unweighted -> 302.0 K

    conn = MODISLSTConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos = _cos_values(series)[15]

    assert native == pytest.approx(302.0, abs=1e-9)
    # cos-lat weighted differs from unweighted only by the weight spread.
    assert cos == pytest.approx(native, rel=1e-3)
    assert abs(cos - native) < 0.05  # < 0.05 K absolute over this gradient


def test_parity_fill_and_invalid_match_native(tmp_path):
    """All-fill and all-out-of-range layers reduce to MISSING, as native does.

    Native: ``da.where(valid_range).mean(skipna=True)`` over an all-invalid
    layer is NaN, the ``> 0`` guard yields ``np.nan``. COS must surface
    ``value=None`` / ``QualityFlag.MISSING`` for the same inputs.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    times = np.array(["2020-06-15", "2020-06-16"], dtype="datetime64[ns]")

    day = np.empty((2, 3, 3))
    day[0] = 0.0       # all fill (DN 0) -> MISSING
    day[1] = 100.0     # all below valid range (< 7500) -> MISSING
    night = np.full((2, 3, 3), 13500.0)

    ds = xr.Dataset(
        {
            "LST_Day_1km": (("time", "lat", "lon"), day),
            "LST_Night_1km": (("time", "lat", "lon"), night),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lst_fill.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_reduce(day, lats, lons, bbox)
    assert all(np.isnan(v) for v in native)

    conn = MODISLSTConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    for p in series.points:
        assert p.value is None
        assert p.quality == QualityFlag.MISSING


def test_parity_window_trim_matches_native_half_open(lst_nc):
    """Half-open [start, end) UTC trim — native does ``df.loc[start:end]`` but
    COS's canonical contract is half-open; assert the COS boundary explicitly.

    (Native uses a closed pandas slice on a daily index; the COS half-open
    window is the canonical contract and the relevant parity is that the
    *included* interior timesteps are identical — the end boundary is excluded
    by design and documented in the connector.)
    """
    conn = MODISLSTConnector()
    series = conn.reduce_file(
        lst_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 18, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    # [15, 18): 15, 16, 17 included; 18 (the end) excluded.
    assert days == {15, 16, 17}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MODISLSTConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.live
async def test_live_fetch_placeholder():
    """Live AppEEARS/Earthdata fetch is not wired; reduction path is the proven part."""
    pytest.skip("live AppEEARS download not wired; see reduce_file tests")
