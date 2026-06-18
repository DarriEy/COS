"""GLDAS-2.1 TWS connector — hermetic test of the gridded basin-reduction path
plus a PARITY-BY-CONSTRUCTION check against the native SYMFLUENCE GLDAS handler.

Builds a synthetic in-memory GLDAS-like NetCDF (the component variables the
native ``GLDASAcquirer`` sums) and reduces it; no network, no auth. This proves
the architecture-critical gridded -> canonical-series path: component summing,
the mm-is-canonical identity boundary, cos-lat basin-mean, half-open window
trim, and the anomaly baseline.

The parity block reimplements the native reduction inline (see
``symfluence/data/acquisition/handlers/gldas_tws.py`` ``GLDASAcquirer`` and
``symfluence/data/observation/handlers/gldas_tws.py`` ``GLDASHandler.process``):

    native, per granule (bbox sub-grid):
      total_sm = sum over SM_VARS of mean(v over lat,lon, skipna=True)
                 INCLUDING a component only if its spatial mean is finite
      swe_mm   = mean(SWE_inst over lat,lon, skipna=True)   (NO finite-guard)
      canopy   = mean(CanopInt_inst over lat,lon, skipna=True)
      tws_mm   = total_sm + swe_mm + canopy
    then process():
      tws_cm        = tws_mm / 10.0                # <-- native unit is cm
      anomaly_cm    = tws_cm - mean(tws_cm over baseline window, inclusive)

The native spatial mean is an UNWEIGHTED arithmetic mean; COS uses a cos-lat
AREA-WEIGHTED mean (a documented refinement). COS keeps the canonical ``mm``
unit; native re-expresses in ``cm``. The relationship is therefore:

    cos_anomaly_mm  ==  native_anomaly_cm * 10        (unit-only factor)

and the two spatial means agree EXACTLY on a constant / uniform field and to
~1e-3 relative on a narrow-latitude bbox. These are the parity invariants
asserted below.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.gldas_tws import COMPONENT_VARS, GLDASTWSConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

SOIL = (
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
)
SWE = "SWE_inst"
CANOPY = "CanopInt_inst"


# ---------------------------------------------------------------------------
# Inline reimplementation of the NATIVE GLDAS reduction (the parity oracle).
# Mirrors GLDASAcquirer.download() per-granule arithmetic + GLDASHandler.process()
# unit conversion + anomaly. Operates on the same numpy arrays COS sees so the
# two paths consume an identical input.
# ---------------------------------------------------------------------------
def native_gldas_anomaly_cm(
    comp_fields: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    bbox: tuple[float, float, float, float],
    baseline: tuple[str, str],
) -> tuple[dict[int, float], list[float]]:
    """Return ({year: native anomaly cm}, [per-timestep anomaly cm]) for a
    per-component (time,lat,lon) set.

    Uses an UNWEIGHTED, skipna spatial mean per component, the native finite
    guard on the soil-moisture components only, mm->cm, then a baseline-window
    (inclusive) anomaly. ``bbox`` is (lat_min, lon_min, lat_max, lon_max).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # match the grid's 0-360 convention as the connector / reduce.py does.
    if float(np.nanmax(lons)) > 180.0:
        if lon_min < 0:
            lon_min += 360.0
        if lon_max < 0:
            lon_max += 360.0
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]

    def spatial_mean(field_t: np.ndarray) -> float:
        sub = field_t[lat_sel[:, None], lon_sel[None, :]]
        finite = np.isfinite(sub)
        if not finite.any():
            return float("nan")
        return float(np.mean(sub[finite]))  # UNWEIGHTED arithmetic mean, skipna

    n_t = times.shape[0]
    tws_mm = np.full(n_t, np.nan)
    for t in range(n_t):
        total_sm = 0.0
        for v in SOIL:
            m = spatial_mean(comp_fields[v][t])
            if not np.isnan(m):  # native finite-guard: include only finite SM
                total_sm += m
        swe_mm = spatial_mean(comp_fields[SWE][t])      # no guard -> NaN propagates
        canopy = spatial_mean(comp_fields[CANOPY][t])
        tws_mm[t] = total_sm + swe_mm + canopy

    tws_cm = tws_mm / 10.0  # native re-expresses TWS in cm
    years = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in times])
    b0 = int(baseline[0][:4])
    b1 = int(baseline[1][:4])
    in_base = (years >= b0) & (years <= b1) & np.isfinite(tws_cm)
    base_mean = float(np.mean(tws_cm[in_base])) if in_base.any() else float(np.nanmean(tws_cm))
    # keyed by year for the uniform-field fixtures (one value per year there);
    # the full per-timestep anomaly vector is also returned for NaN-pattern checks.
    by_year = {int(y): (float(c) - base_mean) for y, c in zip(years, tws_cm)}
    per_step = [float(c) - base_mean for c in tws_cm]
    return by_year, per_step


def _build_nc(tmp_path, comp_fields, lats, lons, times, name="gldas_synth.nc"):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    data_vars = {v: (("time", "lat", "lon"), comp_fields[v]) for v in comp_fields}
    ds = xr.Dataset(data_vars, coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


def _flat_components(times, lats, lons):
    """Spatially-uniform component fields. Per-time component sum: 100 (baseline
    years 2004/2005), 150 (2020). SWE=Canopy=5; soil sums to 90 / 140."""
    shape = (times.shape[0], lats.shape[0], lons.shape[0])
    soil_per_time = {
        "SoilMoi0_10cm_inst": np.array([20.0, 20.0, 35.0, 35.0]),
        "SoilMoi10_40cm_inst": np.array([25.0, 25.0, 35.0, 35.0]),
        "SoilMoi40_100cm_inst": np.array([25.0, 25.0, 35.0, 35.0]),
        "SoilMoi100_200cm_inst": np.array([20.0, 20.0, 35.0, 35.0]),
    }
    swe = np.array([5.0, 5.0, 5.0, 5.0])
    canopy = np.array([5.0, 5.0, 5.0, 5.0])

    def field(per_time):
        arr = np.empty(shape)
        for t in range(shape[0]):
            arr[t] = per_time[t]
        return arr

    comp = {v: field(vals) for v, vals in soil_per_time.items()}
    comp[SWE] = field(swe)
    comp[CANOPY] = field(canopy)
    return comp


@pytest.fixture
def synth_grid():
    times = np.array(
        ["2004-06-15", "2005-06-15", "2020-06-15", "2020-07-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    return times, lats, lons


@pytest.fixture
def gldas_nc(tmp_path, synth_grid):
    times, lats, lons = synth_grid
    comp = _flat_components(times, lats, lons)
    return _build_nc(tmp_path, comp, lats, lons, times)


# ---------------------------------------------------------------------------
# Behavioural tests (unchanged contract coverage).
# ---------------------------------------------------------------------------
def test_reduce_file_basin_mean_components_mm_anomaly(gldas_nc):
    conn = GLDASTWSConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        gldas_nc, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.TWS
    assert series.unit == "mm"  # canonical; mm-in == mm-out identity boundary
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "gldas_tws:domain:bow"
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2004] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2005] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2020] == pytest.approx(50.0, abs=1e-6)


def test_small_basin_defaults_to_nearest_cell(gldas_nc):
    conn = GLDASTWSConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        gldas_nc, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("gldas_tws:cell:")
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2020] == pytest.approx(50.0, abs=1e-6)


def test_window_trim_half_open(gldas_nc):
    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2020-06-01, 2020-07-15): includes the 06-15 obs, excludes 07-15.
    series = conn.reduce_file(
        gldas_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
    )
    years_months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in years_months
    assert (2020, 7) not in years_months


def test_missing_components_masked(gldas_nc, tmp_path):
    """A cell-time with all components NaN -> MISSING; partial NaN sums finitely."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.open_dataset(gldas_nc)
    for v in COMPONENT_VARS:
        ds[v].values[3] = np.nan
    masked = tmp_path / "gldas_masked.nc"
    ds.to_netcdf(masked)
    ds.close()

    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        masked, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    july = [p for p in series.points if p.timestamp.month == 7 and p.timestamp.year == 2020]
    assert len(july) == 1
    assert july[0].value is None
    assert july[0].quality.value == "missing"


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC),
                                datetime(2021, 1, 1, tzinfo=UTC))


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION: COS reduce_file vs inline native reduction.
# ---------------------------------------------------------------------------
def _cos_anomaly_mm(nc_path, bbox, baseline=("2004-01-01", "2009-12-31")):
    conn = GLDASTWSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=bbox, centroid=(51.0, -115.0), area_km2=8000.0,
        options={"baseline": baseline},
    )
    series = conn.reduce_file(
        nc_path, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    return {p.timestamp.year: p.value for p in series.points}


def test_parity_constant_field_exact(gldas_nc, synth_grid):
    """Constant/uniform field: cos-lat and unweighted means are identical, so
    COS anomaly (mm) MUST equal native anomaly (cm) * 10 to float tolerance.
    This pins the unit factor AND the reduction equivalence with zero slack."""
    times, lats, lons = synth_grid
    comp = _flat_components(times, lats, lons)
    bbox = (50.0, -116.0, 52.0, -114.0)

    cos = _cos_anomaly_mm(gldas_nc, bbox)
    native_cm, _ = native_gldas_anomaly_cm(
        comp, lats, lons, times, bbox, ("2004-01-01", "2009-12-31")
    )
    for year in (2004, 2005, 2020):
        # native is cm; COS canonical is mm -> compare with the *10 unit factor.
        assert cos[year] == pytest.approx(native_cm[year] * 10.0, abs=1e-9), year
    # Sanity: the unit factor is real and non-trivial (2020 native +5 cm = +50 mm).
    assert native_cm[2020] == pytest.approx(5.0, abs=1e-9)
    assert cos[2020] == pytest.approx(50.0, abs=1e-9)


def test_parity_narrow_bbox_coslat_within_tolerance(tmp_path):
    """Non-constant latitude gradient over a narrow (2-deg) bbox: cos-lat
    weighting (COS) vs unweighted mean (native) agree to ~1e-3 relative.

    The field varies in latitude so the two weightings genuinely differ; the
    assertion is the documented tolerance-based parity for cos-lat basin-mean."""
    times = np.array(
        ["2004-06-15", "2005-06-15", "2020-06-15", "2020-07-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    shape = (4, 3, 3)

    # Soil layer with a latitude gradient (rows differ); SWE/Canopy uniform.
    # row offsets per latitude: +0,+2,+4 mm added to a base that is 90/140.
    lat_offset = np.array([0.0, 2.0, 4.0])  # along lat axis
    base_per_time = np.array([90.0, 90.0, 140.0, 140.0])  # total soil
    soil0 = np.empty(shape)
    for t in range(4):
        for i in range(3):
            soil0[t, i, :] = base_per_time[t] + lat_offset[i]
    # split soil into the four vars trivially: put it all in one, zeros elsewhere.
    comp = {
        "SoilMoi0_10cm_inst": soil0,
        "SoilMoi10_40cm_inst": np.zeros(shape),
        "SoilMoi40_100cm_inst": np.zeros(shape),
        "SoilMoi100_200cm_inst": np.zeros(shape),
        SWE: np.full(shape, 5.0),
        CANOPY: np.full(shape, 5.0),
    }
    nc = _build_nc(tmp_path, comp, lats, lons, times, name="gldas_grad.nc")
    bbox = (50.0, -116.0, 52.0, -114.0)

    cos = _cos_anomaly_mm(nc, bbox)
    native_cm, _ = native_gldas_anomaly_cm(comp, lats, lons, times, bbox, ("2004-01-01", "2009-12-31"))
    for year in (2004, 2005, 2020):
        native_mm = native_cm[year] * 10.0
        # narrow-latitude bbox: cos-lat vs unweighted differ <0.1% relative.
        denom = max(abs(native_mm), 1.0)
        assert abs(cos[year] - native_mm) / denom < 1e-3, (year, cos[year], native_mm)


def test_parity_window_trim_matches_native(gldas_nc, synth_grid):
    """Half-open [start, end) UTC trim is COS-specific (native keeps all granules
    it downloaded for the temporal range). Within an all-inclusive window the
    surviving timestamps are exactly native's, and values match the *10 factor."""
    times, lats, lons = synth_grid
    comp = _flat_components(times, lats, lons)
    bbox = (50.0, -116.0, 52.0, -114.0)
    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    # window covering everything -> all four native timestamps survive.
    series = conn.reduce_file(
        gldas_nc, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    cos_years = sorted(p.timestamp.year for p in series.points)
    native, _ = native_gldas_anomaly_cm(comp, lats, lons, times, bbox, ("2004-01-01", "2009-12-31"))
    assert cos_years == [2004, 2005, 2020, 2020]
    assert set(cos_years) == set(native)


def test_parity_all_nan_timestep_is_missing_like_native(gldas_nc, tmp_path):
    """When EVERY component is NaN at a timestep, native's tws_mm is NaN (dropped/
    missing) and COS flags MISSING -> they agree on the missing verdict."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.open_dataset(gldas_nc)
    for v in COMPONENT_VARS:
        ds[v].values[3] = np.nan  # 2020-07 fully NaN
    masked = tmp_path / "gldas_allnan.nc"
    ds.to_netcdf(masked)
    ds.close()

    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        masked, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    july = [p for p in series.points if p.timestamp.month == 7 and p.timestamp.year == 2020][0]
    assert july.value is None and july.quality.value == "missing"

    # native oracle: every component spatial-mean NaN -> tws_mm NaN at that step.
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    times = np.array(["2004-06-15", "2005-06-15", "2020-06-15", "2020-07-15"],
                     dtype="datetime64[ns]")
    comp = _flat_components(times, lats, lons)
    for v in COMPONENT_VARS:
        comp[v][3] = np.nan
    bbox = (50.0, -116.0, 52.0, -114.0)
    _, native_steps = native_gldas_anomaly_cm(comp, lats, lons, times, bbox, ("2004-01-01", "2009-12-31"))
    # timesteps: [2004-06, 2005-06, 2020-06, 2020-07]; the 07 step is fully NaN.
    assert np.isnan(native_steps[3])


def test_documented_divergence_swe_nan_timestep(gldas_nc, tmp_path):
    """DOCUMENTED, BENIGN-FOR-OBJECTIVE divergence (the reason parity is
    tolerance/regime-scoped, not bitwise across all NaN patterns):

    If SWE alone is entirely NaN at a timestep, the NATIVE handler propagates
    that NaN into tws_mm (it guards only the soil-moisture components), so the
    whole timestep becomes MISSING. COS sums components per-cell treating an
    all-NaN-SWE cell's SWE as 0 (because soil/canopy are finite there), so COS
    reports a *value* for that timestep.

    This is the precise behaviour the prior xfail test mislabelled (it assumed
    native NaN-skips SWE -> +45; native does NOT). We assert the ACTUAL native
    vs COS verdicts here so the divergence is pinned, not hidden behind xfail."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.open_dataset(gldas_nc)
    ds[SWE].values[2] = np.nan  # SWE all-NaN at 2020-06 only
    part = tmp_path / "gldas_swe_nan.nc"
    ds.to_netcdf(part)
    ds.close()

    conn = GLDASTWSConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        part, spec,
        datetime(2004, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    cos_2020_06 = [p for p in series.points
                   if p.timestamp.year == 2020 and p.timestamp.month == 6][0]
    # COS: soil(140)+canopy(5) finite -> value present (anomaly +45 mm).
    assert cos_2020_06.value is not None
    assert cos_2020_06.value == pytest.approx(45.0, abs=1e-6)

    # native oracle: SWE spatial-mean NaN -> tws_mm NaN -> MISSING (no value).
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    times = np.array(["2004-06-15", "2005-06-15", "2020-06-15", "2020-07-15"],
                     dtype="datetime64[ns]")
    comp = _flat_components(times, lats, lons)
    comp[SWE][2] = np.nan
    bbox = (50.0, -116.0, 52.0, -114.0)
    _, native_steps = native_gldas_anomaly_cm(comp, lats, lons, times, bbox, ("2004-01-01", "2009-12-31"))
    # timestep index 2 == 2020-06, where SWE is all-NaN -> native tws_mm NaN.
    assert np.isnan(native_steps[2])  # native drops the SWE-NaN timestep

    # The divergence is confined to NaN-pattern handling and does NOT affect the
    # TWS-anomaly objective on complete granules (GLDAS monthly fields are spatially
    # complete over land basins); on all-finite data the two paths agree exactly
    # (see test_parity_constant_field_exact / _narrow_bbox).
