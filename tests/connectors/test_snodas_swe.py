"""SNODAS SWE connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory SNODAS-like NetCDF (SWE in metres) and reduces it;
no network, no auth. Proves m→mm canonicalization, half-open window trim,
negative clipping, and both reduction policies.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.snodas_swe import SNODASSWEConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def snodas_nc(tmp_path):
    """A synthetic SNODAS-like NetCDF: swe (metres) over a small daily grid.

    4 daily timesteps, 3x3 grid. Day 0 = 0.10 m, day 1 = 0.25 m, day 2 = 0.40 m,
    day 3 carries one slightly-negative cell (assimilation artifact) to test the
    non-negative clip. One cell is NaN on day 2 to exercise skipna in basin_mean.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 0.10
    data[1] = 0.25
    data[2] = 0.40
    data[2, 0, 0] = np.nan       # missing cell -> skipna in basin_mean
    data[3] = -0.001             # tiny negative everywhere -> clip to 0
    ds = xr.Dataset(
        {"swe": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "snodas_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_m_to_mm(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SWE
    assert series.unit == "mm"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    by_day = {p.timestamp.day: p for p in series.points}
    # 0.10 m -> 100 mm, 0.25 m -> 250 mm; both uniform so basin-mean is exact.
    assert by_day[1].value == pytest.approx(100.0, abs=1e-6)
    assert by_day[2].value == pytest.approx(250.0, abs=1e-6)
    # day 3 uniform 0.40 m except one NaN cell -> skipna mean still 400 mm.
    assert by_day[3].value == pytest.approx(400.0, abs=1e-6)


def test_negative_swe_clipped_to_zero(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[4].value == 0.0
    assert by_day[4].quality == QualityFlag.ESTIMATED


def test_small_basin_defaults_to_nearest_cell(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=10.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 5, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("snodas_swe:cell:")
    by_day = {p.timestamp.day: p for p in series.points}
    # nearest cell to centroid (51, -115) is the center cell = 0.25 m -> 250 mm.
    assert by_day[2].value == pytest.approx(250.0, abs=1e-6)


def test_window_trim_half_open(snodas_nc):
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2022-01-02, 2022-01-04): includes 01-02, 01-03; excludes 01-04.
    series = conn.reduce_file(
        snodas_nc, spec,
        datetime(2022, 1, 2, tzinfo=UTC), datetime(2022, 1, 4, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {2, 3}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2022, 1, 1, tzinfo=UTC),
                                datetime(2022, 2, 1, tzinfo=UTC))


# ----------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION vs the native SYMFLUENCE handler
#
# Native ref: symfluence/data/observation/handlers/snodas.py (registry keys
# ``snodas`` / ``snodas_swe``). Its reduction is (lines 162-211):
#
#   mean_snow = snow.mean(dim=non_time_dims, skipna=True)   # UNWEIGHTED bbox mean
#   df[swe_m] = mean_snow                                    # stays in metres
#   df[swe_m] = df[swe_m].clip(lower=0)                      # clip in METRES, >= 0
#   df[swe_mm] = df[swe_m] * 1000                            # m -> mm AFTER clip
#
# Spatial subset is xarray ``.sel(slice(lat_min, lat_max))`` — inclusive on both
# ends. Temporal subset is ``.sel(time=slice(start, end))`` — ALSO inclusive on
# both ends (note: closed, not half-open; COS trims half-open [start, end), a
# deliberate canonical-window tightening, exercised separately below).
#
# The native canonical comparison field is ``swe_mm``. We reimplement that exact
# pipeline inline and assert the COS connector reproduces it.
#
# Two documented semantic differences from native, both proven benign here:
#   1. COS basin_mean is cos-latitude AREA-WEIGHTED; native is UNWEIGHTED. For a
#      uniform field (and a single cell) the two are identical to float epsilon;
#      for a varying field over a narrow latitude band the difference is bounded
#      by the spread of cos(lat) across the band (~5e-4 relative over 50-52 N) and
#      does not corrupt the SWE-tracking objective.
#   2. COS scales (m->mm) then clips >= 0; native clips (in m) then scales. Since
#      clip(x, 0) * 1000 == clip(x * 1000, 0) for a positive factor, these are
#      algebraically identical.
# ----------------------------------------------------------------------------


def _native_swe_mm(values_m, lats, bbox, start, end, times):
    """Reimplement the native snodas.py reduction on the SAME (time, lat, lon) m grid.

    UNWEIGHTED skipna spatial mean over the bbox, clip(>=0) in metres, then *1000.
    Returns {day -> swe_mm} restricted to the native CLOSED [start, end] window.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # 3x3 fixture grid lons are -116..-114, bbox spans the full grid; select by
    # lat only here (lon fully inside) to mirror the inclusive .sel slice.
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    out = {}
    for k, t in enumerate(times):
        day = t.astype("datetime64[D]").astype(datetime).day
        ts = t.astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        if not (start <= ts <= end):  # native CLOSED window
            continue
        layer = values_m[k][lat_sel, :]
        finite = np.isfinite(layer)
        mean_m = float(np.mean(layer[finite])) if finite.any() else np.nan
        if np.isfinite(mean_m):
            mean_m = max(mean_m, 0.0)          # clip in metres
            out[day] = mean_m * 1000.0          # then -> mm
        else:
            out[day] = None
    return out


def test_parity_uniform_field_exact_vs_native(snodas_nc):
    """Uniform fields: cos-lat weighting collapses to the native unweighted mean.

    Days 1 (0.10 m) and 2 (0.25 m) are spatially uniform, so the area-weighted
    and unweighted means are identical -> COS must equal native to float epsilon.
    Day 3 is uniform 0.40 m with one NaN cell; both reductions skip it the same
    way (the weighted mean of a constant over any cell subset is that constant).
    """
    lats = np.array([50.0, 51.0, 52.0])
    times = np.array(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
                     dtype="datetime64[ns]")
    values_m = np.empty((4, 3, 3), dtype="float64")
    values_m[0] = 0.10
    values_m[1] = 0.25
    values_m[2] = 0.40
    values_m[2, 0, 0] = np.nan
    values_m[3] = -0.001
    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2022, 1, 5, tzinfo=UTC)

    native = _native_swe_mm(values_m, lats, bbox, start, end, times)

    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(snodas_nc, spec, start, end)
    cos = {p.timestamp.day: p.value for p in series.points}

    for day in (1, 2, 3):
        assert cos[day] == pytest.approx(native[day], abs=1e-9), f"day {day}"


def test_parity_varying_field_cos_lat_vs_native_unweighted(snodas_nc, tmp_path):
    """Latitude-varying field: bound the cos-lat vs unweighted divergence.

    Build a field that varies with latitude so the area-weighting actually bites,
    then assert COS (cos-lat) tracks native (unweighted) within the relative bound
    set by the spread of cos(lat) across the 50-52 N band. This is the documented,
    benign basin-mean approximation (tolerance-based parity, like CAS attributes).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2022-01-10"], dtype="datetime64[ns]")
    # SWE increasing northward: a real latitudinal gradient.
    values_m = np.empty((1, 3, 3), dtype="float64")
    values_m[0, 0, :] = 0.20   # lat 50
    values_m[0, 1, :] = 0.40   # lat 51
    values_m[0, 2, :] = 0.60   # lat 52
    ds = xr.Dataset({"swe": (("time", "lat", "lon"), values_m)},
                    coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / "snodas_grad.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2022, 1, 31, tzinfo=UTC)
    native = _native_swe_mm(values_m, lats, bbox, start, end, times)

    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(path, spec, start, end)
    cos = {p.timestamp.day: p.value for p in series.points}

    # cos(lat) over 50-52 N spans cos(50 deg)..cos(52 deg); the weighted mean of a
    # monotone field differs from the unweighted by < the band's weight spread.
    w = np.cos(np.deg2rad(lats))
    rel_bound = float((w.max() - w.min()) / w.mean())  # ~0.046 here
    assert cos[10] == pytest.approx(native[10], rel=rel_bound)
    # And the cos-lat mean leans toward the (lower-weighted northern) high values
    # only slightly: it must stay well within a tight 1% of the unweighted mean
    # for this 2-degree band, documenting the divergence is small.
    assert cos[10] == pytest.approx(native[10], rel=1e-2)


def test_parity_single_cell_exact_vs_native(tmp_path):
    """Single-cell grid: weighted == unweighted == that cell. Must match to eps."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    times = np.array(["2022-02-01", "2022-02-02"], dtype="datetime64[ns]")
    values_m = np.array([[[0.137]], [[0.642]]], dtype="float64")  # (2,1,1)
    ds = xr.Dataset({"swe": (("time", "lat", "lon"), values_m)},
                    coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / "snodas_single.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2022, 2, 1, tzinfo=UTC)
    end = datetime(2022, 2, 3, tzinfo=UTC)
    native = _native_swe_mm(values_m, lats, bbox, start, end, times)

    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="pt", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(path, spec, start, end)
    cos = {p.timestamp.day: p.value for p in series.points}
    assert cos[1] == pytest.approx(native[1], abs=1e-9) == pytest.approx(137.0, abs=1e-9)
    assert cos[2] == pytest.approx(native[2], abs=1e-9) == pytest.approx(642.0, abs=1e-9)


def test_parity_unit_factor_is_exactly_1000(snodas_nc):
    """The m->mm canonicalization factor must be exactly 1000, matching native."""
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(snodas_nc, spec, datetime(2022, 1, 1, tzinfo=UTC),
                              datetime(2022, 1, 5, tzinfo=UTC))
    by_day = {p.timestamp.day: p.value for p in series.points}
    assert by_day[1] / 0.10 == pytest.approx(1000.0)   # 0.10 m -> 100 mm
    assert by_day[2] / 0.25 == pytest.approx(1000.0)   # 0.25 m -> 250 mm
    assert series.unit == "mm"


def test_parity_clip_order_equivalent_to_native(tmp_path):
    """COS scales-then-clips; native clips-then-scales. Prove equivalence.

    A cell at exactly -0.001 m: native clips to 0 m -> 0 mm; COS scales to -1 mm
    then clips to 0 mm. Both yield 0.0, and COS flags the clipped point ESTIMATED.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    times = np.array(["2022-03-01"], dtype="datetime64[ns]")
    values_m = np.array([[[-0.001]]], dtype="float64")
    ds = xr.Dataset({"swe": (("time", "lat", "lon"), values_m)},
                    coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / "snodas_neg.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2022, 3, 1, tzinfo=UTC)
    end = datetime(2022, 3, 2, tzinfo=UTC)
    native = _native_swe_mm(values_m, lats, bbox, start, end, times)  # clip->0

    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(path, spec, start, end)
    p = series.points[0]
    assert native[1] == 0.0
    assert p.value == 0.0
    assert p.quality == QualityFlag.ESTIMATED


def test_parity_fill_nan_maps_to_missing(tmp_path):
    """An all-NaN timestep -> native NaN/None; COS QualityFlag.MISSING, value None."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2022-04-01", "2022-04-02"], dtype="datetime64[ns]")
    values_m = np.empty((2, 3, 3), dtype="float64")
    values_m[0] = 0.30
    values_m[1] = np.nan          # entire layer missing
    ds = xr.Dataset({"swe": (("time", "lat", "lon"), values_m)},
                    coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / "snodas_allnan.nc"
    ds.to_netcdf(path)

    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2022, 4, 1, tzinfo=UTC)
    end = datetime(2022, 4, 3, tzinfo=UTC)
    native = _native_swe_mm(values_m, lats, bbox, start, end, times)

    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=bbox, centroid=(51.0, -115.0),
                         area_km2=8000.0)
    series = conn.reduce_file(path, spec, start, end)
    by_day = {p.timestamp.day: p for p in series.points}
    assert native[1] == pytest.approx(300.0) and by_day[1].value == pytest.approx(300.0)
    assert native[2] is None
    assert by_day[2].value is None
    assert by_day[2].quality == QualityFlag.MISSING


def test_parity_window_is_half_open_vs_native_closed(snodas_nc):
    """Document the one intentional divergence: COS window is half-open [start,end).

    Native uses a CLOSED [start, end] .sel slice. COS tightens the right edge to
    half-open. With end=2022-01-03 the 01-03 obs is INCLUDED by native but EXCLUDED
    by COS. This is a deliberate canonical-window rule, not a porting bug.
    """
    conn = SNODASSWEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(snodas_nc, spec, datetime(2022, 1, 1, tzinfo=UTC),
                              datetime(2022, 1, 3, tzinfo=UTC))
    days = {p.timestamp.day for p in series.points}
    assert days == {1, 2}          # COS half-open excludes 01-03
    # native CLOSED would have included day 3:
    lats = np.array([50.0, 51.0, 52.0])
    times = np.array(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
                     dtype="datetime64[ns]")
    values_m = np.empty((4, 3, 3), dtype="float64")
    values_m[0] = 0.10; values_m[1] = 0.25; values_m[2] = 0.40
    values_m[2, 0, 0] = np.nan; values_m[3] = -0.001
    native = _native_swe_mm(values_m, lats, (50.0, -116.0, 52.0, -114.0),
                            datetime(2022, 1, 1, tzinfo=UTC),
                            datetime(2022, 1, 3, tzinfo=UTC), times)
    assert 3 in native               # native closed includes the boundary day
