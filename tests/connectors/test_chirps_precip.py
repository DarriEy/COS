"""CHIRPS precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory CHIRPS-like NetCDF and reduces it; no network, no
auth (CHIRPS is anonymous). This proves the architecture-critical gridded ->
canonical-series path for a rainfall product: identity unit (mm daily depth),
fill / negative masking, basin-mean vs nearest-cell reduction, and half-open UTC
window trim.

The ``test_parity_*`` block is a PARITY-BY-CONSTRUCTION check against the native
SYMFLUENCE CHIRPS observation handler
(``symfluence/data/observation/handlers/chirps.py``). Rather than import
SYMFLUENCE, it reimplements the native reduction semantics inline on the SAME
synthetic input and asserts how COS relates to them:

  * unit handling: IDENTITY (native carries CHIRPS mm/day through as
    ``precipitation_mm`` with no scaling; COS emits ``mm`` unscaled) — exact;
  * uniform field / single cell: COS basin-mean == native unweighted mean to
    float tolerance (cos-lat weights cancel when the field is constant);
  * spatial gradient over the bbox: COS uses a cos-latitude AREA-WEIGHTED mean,
    native uses an UNWEIGHTED ``DataArray.mean`` — they DIVERGE by ~the cos-lat
    weight spread (~0.9% over a 2 deg lat band here), which is why basin-mean
    parity is tolerance-based and this kind stays ungated unless restricted to
    the uniform/point regime;
  * fill timing: COS masks -9999 / negatives PER CELL BEFORE averaging; native
    masks only the POST-average value (``where(>= 0)``), so a partial-fill layer
    drops to MISSING in native but survives (correctly) in COS — a benign
    divergence in COS's favour, documented here so it is not mistaken for a bug.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.chirps_precip import FILL_VALUE, CHIRPSPrecipitationConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def chirps_nc(tmp_path):
    """A synthetic CHIRPS-like NetCDF: precip (mm/day) over a small grid.

    Four daily timesteps on a 3x3 grid. The last timestep is entirely fill
    (-9999) so it must reduce to MISSING; one cell in an otherwise-valid layer is
    negative (a partial no-data) to exercise the ``precip < 0`` mask.
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
    data[0] = 5.0            # uniform valid layer -> mean 5.0 mm
    data[1] = 10.0           # uniform valid layer
    data[1, 0, 0] = -9999.0  # one fill cell -> masked, mean stays 10.0
    data[2] = 0.0            # dry day -> mean 0.0 (valid, non-negative)
    data[3] = FILL_VALUE     # entirely fill -> MISSING
    ds = xr.Dataset(
        {"precip": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "chirps_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical; CHIRPS mm/day daily depth -> identity
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "chirps:domain:bow"
    assert series.provider == "chirps_precip"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5.0 mm layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # The single fill (-9999) cell is masked; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    # A genuine dry day (0 mm) stays a valid GOOD zero, not masked.
    assert by_day[17].value == pytest.approx(0.0, abs=1e-9)
    assert by_day[17].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    series = conn.reduce_file(
        chirps_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("chirps:cell:")
    # nearest cell to centroid (51, -115) is the grid center -> 5.0 on day 15.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)


def test_window_trim_half_open(chirps_nc):
    conn = CHIRPSPrecipitationConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        chirps_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = CHIRPSPrecipitationConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_fetch_placeholder():
    """Live UCSB CHG fetch is not wired; reduction is the proven path."""
    pytest.skip("CHIRPS live UCSB download not wired; reduction path is hermetic")


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION against the native SYMFLUENCE CHIRPS handler.
#
# native_reduce_layer() reimplements, inline, exactly what
# symfluence/data/observation/handlers/chirps.py:process does to one timestep:
#   1. select cells inside the bbox (label slice, inclusive on both ends),
#   2. UNWEIGHTED arithmetic mean over the spatial dims (xr.DataArray.mean,
#      skipna=True), with NO pre-masking of -9999 / negatives,
#   3. mask the post-average value: ``where(value >= 0, NaN)`` then drop NaN,
#   4. identity unit (mm/day carried through as precipitation_mm, no scaling).
# --------------------------------------------------------------------------- #


def native_reduce_layer(lats, lons, layer, bbox):
    """The native handler's reduction of ONE (lat, lon) layer -> scalar or None.

    Returns the native ``precipitation_mm`` for the timestep, or ``None`` if the
    native post-average ``where(>= 0)`` + ``dropna`` would drop it (its MISSING).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_sel = (lats >= lat_min) & (lats <= lat_max)
    lon_sel = (lons >= lon_min) & (lons <= lon_max)
    sub = layer[np.ix_(lat_sel, lon_sel)]
    # native does NOT pre-mask -9999 / negatives; skipna only drops NaN/inf.
    finite = np.isfinite(sub)
    if not finite.any():
        return None  # all-NaN slice -> NaN mean -> dropped
    mean = float(sub[finite].mean())  # UNWEIGHTED, no cos-lat weighting
    # native masks the post-average value: where(>= 0) then dropna.
    if not (mean >= 0):
        return None
    return mean  # identity unit (mm)


def _cos_basin(conn, nc, area_km2, start, end):
    """Run the COS pure reduction and index its points by day."""
    series = conn.reduce_file(nc, _spec(area_km2), start, end)
    return {p.timestamp.day: p for p in series.points}, series


def test_parity_uniform_field_exact(chirps_nc):
    """Uniform / dry layers: COS basin-mean == native unweighted mean, EXACTLY.

    When the field is constant over the bbox the cos-lat weights cancel, so the
    area-weighted mean and the native unweighted mean coincide to float tol. This
    is the regime in which parity is bitwise, and it covers the identity-unit
    claim (5.0 mm in == 5.0 mm out, 0.0 stays 0.0 GOOD).
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = CHIRPSPrecipitationConnector()
    cos, _ = _cos_basin(
        conn, chirps_nc, 8000.0,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )

    # day 15: uniform 5.0 ; day 17: uniform 0.0 (dry, valid).
    native_15 = native_reduce_layer(lats, lons, np.full((3, 3), 5.0), bbox)
    native_17 = native_reduce_layer(lats, lons, np.full((3, 3), 0.0), bbox)
    assert cos[15].value == pytest.approx(native_15, abs=1e-12)
    assert cos[17].value == pytest.approx(native_17, abs=1e-12)
    assert cos[17].quality == QualityFlag.GOOD  # dry zero is GOOD, not MISSING


def test_parity_single_cell_identity(tmp_path):
    """Single in-bbox cell: COS basin-mean == native mean == the cell value.

    With one cell the weighting is irrelevant and the two reductions and the raw
    value must all coincide exactly — the tightest possible parity anchor, and a
    second confirmation of the identity unit conversion.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    data = np.array([[[7.3]]])  # (time=1, lat=1, lon=1)
    ds = xr.Dataset(
        {"precip": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "chirps_one_cell.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = CHIRPSPrecipitationConnector()
    cos, series = _cos_basin(
        conn, path, 8000.0,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    native = native_reduce_layer(lats, lons, data[0], bbox)
    assert series.unit == "mm"
    assert native == pytest.approx(7.3, abs=1e-12)  # identity unit
    assert cos[15].value == pytest.approx(native, abs=1e-12)


def test_parity_gradient_divergence_is_cos_lat_only(tmp_path):
    """Spatial gradient: COS (cos-lat weighted) and native (unweighted) DIVERGE.

    Over the 50-52 deg N band the cos-lat weight spread makes the two means
    differ by ~0.9% on a 1..9 gradient — well outside any 1e-3 tolerance. This is
    the documented reason CHIRPS basin-mean cannot graduate to a tight value
    grade; the divergence is purely the weighting choice (we reproduce native's
    unweighted number AND COS's weighted number from the same array), not a bug.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    ds = xr.Dataset(
        {"precip": (("time", "lat", "lon"), grad[None, :, :])},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "chirps_gradient.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = CHIRPSPrecipitationConnector()
    cos, _ = _cos_basin(
        conn, path, 8000.0,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )

    native = native_reduce_layer(lats, lons, grad, bbox)  # unweighted = 5.0
    w = np.cos(np.deg2rad(lats))
    w2d = np.broadcast_to(w[:, None], grad.shape)
    cos_lat = float((grad * w2d).sum() / w2d.sum())  # weighted

    assert native == pytest.approx(5.0, abs=1e-12)
    # COS reproduces the cos-lat weighted number, NOT the native unweighted one.
    assert cos[15].value == pytest.approx(cos_lat, abs=1e-9)
    rel = abs(cos[15].value - native) / native
    assert rel > 1e-3  # the divergence exceeds a tight value tolerance...
    assert rel < 2e-2  # ...but is bounded by the cos-lat weight spread alone.


def test_parity_fill_timing_diverges_cos_is_stricter(chirps_nc):
    """Partial-fill layer: native drops it to MISSING; COS keeps the valid mean.

    Native masks only the POST-average value, so day-16's nine cells
    (eight 10.0 + one -9999) average to ~-1102, fail ``where(>= 0)`` and are
    DROPPED. COS pre-masks the -9999 cell and averages the surviving eight to
    10.0 GOOD. This divergence is in COS's favour (native's value is a -9999
    contamination artifact), so it does not corrupt the precipitation objective,
    but it is a genuine semantic difference -> CHIRPS stays ungated.
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    bbox = (50.0, -116.0, 52.0, -114.0)
    layer16 = np.full((3, 3), 10.0)
    layer16[0, 0] = FILL_VALUE  # one fill cell

    native_16 = native_reduce_layer(lats, lons, layer16, bbox)
    assert native_16 is None  # native drops the contaminated layer -> MISSING

    conn = CHIRPSPrecipitationConnector()
    cos, _ = _cos_basin(
        conn, chirps_nc, 8000.0,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    # COS keeps it: pre-masked fill, mean of remaining 10.0 cells.
    assert cos[16].value == pytest.approx(10.0, abs=1e-9)
    assert cos[16].quality == QualityFlag.GOOD


def test_parity_all_fill_both_missing(chirps_nc):
    """All-fill layer -> MISSING in BOTH COS and native (the agreeing fill case).

    Day 18 is entirely -9999. Native: all cells averaged -> -9999 -> where(>= 0)
    -> dropped. COS: all cells pre-masked to NaN -> no finite cell -> MISSING.
    Both reach MISSING, so for the common (whole-layer) no-data case parity holds.
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    bbox = (50.0, -116.0, 52.0, -114.0)
    native_18 = native_reduce_layer(lats, lons, np.full((3, 3), FILL_VALUE), bbox)
    assert native_18 is None

    conn = CHIRPSPrecipitationConnector()
    cos, _ = _cos_basin(
        conn, chirps_nc, 8000.0,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert cos[18].value is None
    assert cos[18].quality == QualityFlag.MISSING
