"""SMOS soil-moisture connector — contract + PARITY-BY-CONSTRUCTION tests.

Parity target: the SYMFLUENCE native handler
``data/observation/handlers/soil_moisture.py::SMOSSMHandler.process``. The native
reduction, reimplemented inline below from the source, is:

* variable picked from the same candidate list, same order;
* per-timestep validity mask ``(sm > 0) & (sm < 1) & ~isnan`` (physical
  volumetric range), identical to the COS connector's mask;
* spatial reduction = ``np.nanmean`` over the valid in-bbox pixels — an
  *UNWEIGHTED* mean per timestep;
* unit = identity (volumetric m3/m3 emitted unchanged), the canonical
  ``KIND_UNITS[SOIL_MOISTURE]``;
* window = closed ``[start, end]`` on the file's timestamps.

COS's connector differs in two *documented, benign* ways:

1. its ``basin_mean`` is a cosine-latitude AREA-WEIGHTED mean (``cos.core.reduce``),
   not the native unweighted mean. Over a single cell, a constant field, or a
   narrow-latitude bbox the two agree to float / ~1e-3 tolerance (the cos(lat)
   weights are then near-equal). This is the same approximation GRACE basin-mean
   makes and is harmless for the soil-moisture objective (a basin-average SM
   anomaly correlation), so it does not corrupt the kind.
2. its window is half-open ``[start, end)`` (the COS canonical convention shared
   by every connector). This is a deliberate framework choice, not a port bug.

These tests construct synthetic inputs inline, run the COS pure reducer, compute
the native expected result inline on the SAME input, and assert equality at the
right tolerance.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.smos_sm import SM_VARIABLES, SMOSSMConnector
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction


# --------------------------------------------------------------------------- #
# Contract (kept from the original file).
# --------------------------------------------------------------------------- #
def test_smos_connector_contract():
    conn = SMOSSMConnector()
    assert conn.slug == "smos_sm"
    assert conn.kind == ObservationKind.SOIL_MOISTURE
    assert conn.structural_class == "gridded"
    # canonical unit must be the frozen kind unit (m3/m3); no rescale at boundary
    assert KIND_UNITS[conn.kind] == "m3/m3"


def test_smos_registered():
    from cos.core.registry import discover, get_connector

    discover()
    assert get_connector("smos_sm") is SMOSSMConnector


# --------------------------------------------------------------------------- #
# Inline reimplementation of the native SMOSSMHandler.process reduction.
# Mirrors soil_moisture.py:558-566 exactly: per-timestep nanmean over the
# physical-range valid pixels. UNWEIGHTED. Unit identity.
# --------------------------------------------------------------------------- #
def _native_smos_series(values: np.ndarray) -> list[float | None]:
    """Native per-timestep reduction over the *already-subset* (time,lat,lon) cube.

    Returns one value per timestep (None where no valid pixel), unweighted mean
    of cells with 0 < sm < 1 and finite, exactly as the native handler emits one
    row per time with ``float(np.nanmean(sm_slice[valid_mask]))``.
    """
    out: list[float | None] = []
    for t in range(values.shape[0]):
        sl = values[t]
        valid = (sl > 0) & (sl < 1) & (~np.isnan(sl))
        out.append(float(np.nanmean(sl[valid])) if np.any(valid) else None)
    return out


def _make_nc(tmp_path, *, times, lats, lons, data, var="sm"):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "smos_synth.nc"
    ds.to_netcdf(path)
    return path


# --------------------------------------------------------------------------- #
# PARITY 1: constant field -> COS cos-lat weighted mean == native unweighted
#           mean to float tolerance (weights cancel for a constant layer).
#           Also exercises the variable-name selection + identity unit.
# --------------------------------------------------------------------------- #
def test_parity_constant_field_exact(tmp_path):
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([10.0, 11.0, 12.0])
    data = np.full((2, 3, 3), 0.3)  # constant volumetric SM
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data)

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="const",
        bbox=(50.0, 10.0, 52.0, 12.0),
        centroid=(51.0, 11.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        nc, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 12, 31, tzinfo=UTC)
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # identity boundary conversion
    assert series.reduction == SpatialReduction.BASIN_MEAN

    native = _native_smos_series(data)
    cos_vals = [p.value for p in series.points]
    assert len(cos_vals) == len(native) == 2
    for c, n in zip(cos_vals, native):
        assert n is not None and c == pytest.approx(0.3, abs=1e-12)
        assert c == pytest.approx(n, abs=1e-12)  # cos-lat == unweighted for constant


# --------------------------------------------------------------------------- #
# PARITY 2: varying field over a narrow-latitude bbox -> cos-lat weighted mean
#           agrees with native unweighted mean to relative ~1e-3. We assert both
#           the documented benign drift bound AND that COS exactly reproduces its
#           own cos-lat formula computed inline (proving the kernel, not luck).
# --------------------------------------------------------------------------- #
def test_parity_narrow_bbox_coslat_vs_unweighted_within_1e_3(tmp_path):
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    # ~1 degree of latitude span at ~51N: cos(50)=0.643, cos(51)=0.629 -> the
    # max weight ratio is ~1.02, so weighted vs unweighted means differ < 1e-3
    # for a field whose spatial variance is modest.
    lats = np.array([50.0, 51.0])
    lons = np.array([10.0, 11.0])
    data = np.array([[[0.20, 0.22], [0.24, 0.26]]])  # (1,2,2)
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data)

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="narrow",
        bbox=(50.0, 10.0, 51.0, 11.0),
        centroid=(50.5, 10.5),
        area_km2=8000.0,
    )
    series = conn.reduce_file(
        nc, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    (cos_val,) = [p.value for p in series.points]

    # Native unweighted mean over the same valid cells.
    (native_val,) = _native_smos_series(data)
    assert native_val == pytest.approx(0.23, abs=1e-12)

    # Documented benign divergence: cos-lat vs unweighted within relative 1e-3.
    assert cos_val == pytest.approx(native_val, rel=1e-3)

    # And COS exactly reproduces its OWN cos-lat formula (kernel correctness).
    w = np.cos(np.deg2rad(lats))
    w2d = np.broadcast_to(w[:, None], data[0].shape)
    expected_coslat = float(np.sum(data[0] * w2d) / np.sum(w2d))
    assert cos_val == pytest.approx(expected_coslat, abs=1e-12)


# --------------------------------------------------------------------------- #
# PARITY 3: single-cell (nearest_cell) -> identity with native nanmean of the
#           one selected pixel. No weighting ambiguity: MUST match to float tol.
# --------------------------------------------------------------------------- #
def test_parity_nearest_cell_single_pixel_identity(tmp_path):
    times = np.array(["2020-03-01", "2020-03-02"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([10.0, 11.0, 12.0])
    data = np.arange(2 * 3 * 3, dtype="float64").reshape(2, 3, 3) / 100.0  # all in (0,1)
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data)

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, 10.0, 52.0, 12.0),
        centroid=(51.0, 11.0),  # -> grid cell index (1,1)
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        nc, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    cos_vals = [p.value for p in series.points]
    # Native nanmean over a single in-box pixel == that pixel's value.
    expected = [float(data[t, 1, 1]) for t in range(2)]
    assert cos_vals == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# PARITY 4: fill / out-of-range -> QualityFlag.MISSING, identical mask to native.
#           Cell with sm=0 (fill) and sm>=1 (saturation overflow) drop out; a
#           timestep with NO valid pixel becomes MISSING/None in both.
# --------------------------------------------------------------------------- #
def test_parity_fill_and_outofrange_to_missing(tmp_path):
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([10.0, 11.0])
    data = np.empty((2, 2, 2))
    # t0: one valid (0.4), rest are fill(0)/overflow(1.5)/nan -> mean = 0.4
    data[0] = [[0.4, 0.0], [1.5, np.nan]]
    # t1: entirely non-physical -> MISSING in both
    data[1] = [[0.0, -9999.0], [np.nan, 2.0]]
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data)

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="fill",
        bbox=(50.0, 10.0, 51.0, 11.0),
        centroid=(50.5, 10.5),
        area_km2=8000.0,
    )
    series = conn.reduce_file(
        nc, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    pts = series.points
    native = _native_smos_series(data)

    # t0: single surviving cell -> 0.4 in both (weighting irrelevant, one cell).
    assert native[0] == pytest.approx(0.4, abs=1e-12)
    assert pts[0].value == pytest.approx(0.4, abs=1e-12)
    assert pts[0].quality.value == "good"
    # t1: no valid pixel -> MISSING/None in both.
    assert native[1] is None
    assert pts[1].value is None
    assert pts[1].quality.value == "missing"


# --------------------------------------------------------------------------- #
# WINDOW: COS half-open [start, end). Documented divergence from native closed
#         [start, end]; the boundary timestamp at `end` is excluded by COS.
# --------------------------------------------------------------------------- #
def test_window_trim_half_open(tmp_path):
    times = np.array(
        ["2020-06-15", "2020-07-15", "2020-08-15"], dtype="datetime64[ns]"
    )
    lats = np.array([50.0, 51.0])
    lons = np.array([10.0, 11.0])
    data = np.full((3, 2, 2), 0.3)
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data)

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="win", bbox=(50.0, 10.0, 51.0, 11.0),
        centroid=(50.5, 10.5), area_km2=8000.0,
    )
    # [2020-06-15, 2020-08-15): includes 06-15, EXCLUDES the 08-15 boundary obs.
    series = conn.reduce_file(
        nc, spec, datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 8, 15, tzinfo=UTC)
    )
    days = {(p.timestamp.month, p.timestamp.day) for p in series.points}
    assert (6, 15) in days
    assert (7, 15) in days
    assert (8, 15) not in days  # half-open upper bound


def test_variable_name_selection_matches_native_order(tmp_path):
    # Native and COS share the same candidate list and order; pick a non-first
    # name and confirm COS still finds it.
    assert SM_VARIABLES[0] == "sm"
    times = np.array(["2020-01-01"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([10.0, 11.0])
    data = np.full((1, 2, 2), 0.42)
    nc = _make_nc(tmp_path, times=times, lats=lats, lons=lons, data=data, var="Soil_Moisture")

    conn = SMOSSMConnector()
    spec = ReductionSpec(
        domain_name="vn", bbox=(50.0, 10.0, 51.0, 11.0),
        centroid=(50.5, 10.5), area_km2=8000.0,
    )
    series = conn.reduce_file(
        nc, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert series.source_info["variable"] == "Soil_Moisture"
    assert series.points[0].value == pytest.approx(0.42, abs=1e-12)


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = SMOSSMConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, 10.0, 52.0, 12.0), centroid=(51.0, 11.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
