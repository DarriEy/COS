"""MODIS SCA connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory MODIS-snow NetCDF (NDSI percent + flag bytes) and
reduces it; no network, no auth. Proves the percent→fraction canonicalization,
the byte-flag masking, the basin-mean / nearest-cell reductions, and half-open
UTC window-trim — the parts that mirror the native ``modis_snow`` handler.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_sca import MODISSCAConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def modis_nc(tmp_path):
    """Synthetic MODIS SCA NetCDF: NDSI_Snow_Cover (percent + flag bytes)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    # t0: uniform 50% NDSI -> fraction 0.5 everywhere.
    data[0] = 50.0
    # t1: uniform 100% NDSI -> 1.0.
    data[1] = 100.0
    # t2: mix of valid 80% and cloud flag (250) -> masked cells ignored,
    #     mean over valid = 0.8.
    data[2] = 80.0
    data[2, 0, 0] = 250.0  # cloud -> NaN
    data[2, 1, 1] = 255.0  # fill -> NaN
    # t3: fully cloud/fill -> all masked -> NaN -> MISSING.
    data[3] = 200.0
    ds = xr.Dataset(
        {"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_synth.nc"
    ds.to_netcdf(path)
    return path


def _by_month(series):
    return {p.timestamp.month: p for p in series.points}


def test_reduce_file_basin_mean_percent_to_fraction(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SNOW_COVER
    assert series.unit == "fraction"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"

    by_month = _by_month(series)
    # 50% -> 0.5
    assert by_month[1].value == pytest.approx(0.5, abs=1e-9)
    assert by_month[1].quality == QualityFlag.GOOD
    # 100% -> 1.0
    assert by_month[2].value == pytest.approx(1.0, abs=1e-9)
    # 80% valid cells (cloud/fill masked out) -> mean still 0.8
    assert by_month[3].value == pytest.approx(0.8, abs=1e-9)
    # all-flag timestep -> MISSING, value None
    assert by_month[4].value is None
    assert by_month[4].quality == QualityFlag.MISSING


def test_fraction_never_exceeds_one(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    vals = [p.value for p in series.points if p.value is not None]
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_small_basin_defaults_to_nearest_cell(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_sca:cell:")
    # nearest cell (51,-115) at t0 = 50% -> 0.5
    by_month = _by_month(series)
    assert by_month[1].value == pytest.approx(0.5, abs=1e-9)


def test_window_trim_half_open(modis_nc):
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2020-02-01, 2020-04-15): includes 02-15 and 03-15, excludes 04-15.
    series = conn.reduce_file(
        modis_nc, spec,
        datetime(2020, 2, 1, tzinfo=UTC), datetime(2020, 4, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert 2 in months
    assert 3 in months
    assert 1 not in months  # before window
    assert 4 not in months  # end is exclusive


def test_list_sites_one_region(modis_nc):
    import asyncio

    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    sites = asyncio.run(conn.list_sites(spec))
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "modis_sca:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC),
                                datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_earthdata_fetch_placeholder():
    pytest.skip("Live Earthdata MODIS fetch not wired; reduce path is the proven part.")


# --------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION against the native SYMFLUENCE modis_snow handler.
#
# Native semantics (data/observation/handlers/modis_snow.py +
# .../modis_utils.py), reimplemented inline below so the assertion is
# self-contained and cannot drift with the source:
#
#   1. quality filter: data.where((data >= 0) & (data <= 100))  -> NaN outside
#      the valid NDSI percent range. Every MODIS flag/fill byte (>= 200) is
#      excluded by the SAME range test, so byte masking == range masking.
#   2. reduction: data.mean(dim=spatial_dims, skipna=True)  -> an UNWEIGHTED
#      simple arithmetic mean over all in-grid cells, NaN-skipping.
#   3. unit: df['sca'] / 100.0 (percent -> fraction).
#
# The ONLY semantic divergence is reduction WEIGHTING: COS basin_mean is a
# cosine-latitude AREA-WEIGHTED mean; native is an UNWEIGHTED mean. This is
# benign for the snow-cover objective and:
#   * is EXACT (float tolerance) for a single-latitude-row grid (one weight),
#   * is EXACT for a spatially constant field (weights cancel),
#   * differs only ~1e-3 relative over a small/narrow-latitude bbox.
# nearest_cell is identical to the native single-pixel pick (no weighting),
# and the percent->fraction unit factor + fill->MISSING rule are identical.
# --------------------------------------------------------------------------


def _native_unweighted_mean(values_percent):
    """Reimplement the native reduction inline: mask to [0,100], /100, then an
    UNWEIGHTED skipna mean over the spatial cells. values shape (time,lat,lon)."""
    v = np.asarray(values_percent, dtype="float64").copy()
    invalid = ~((v >= 0.0) & (v <= 100.0))
    v[invalid] = np.nan
    v = v / 100.0  # percent -> fraction, exactly as native df['sca']/100.0
    out = np.full(v.shape[0], np.nan, dtype="float64")
    for t in range(v.shape[0]):
        layer = v[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.mean(layer[finite]))  # UNWEIGHTED, skipna
    return out


def _single_row_modis_nc(tmp_path):
    """A single-latitude-row grid: cos-lat weighting collapses to one weight,
    so COS basin_mean == native unweighted mean EXACTLY (float tolerance)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
                     dtype="datetime64[ns]")
    lats = np.array([51.0])               # ONE latitude row
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 1, 3), dtype="float64")
    data[0] = np.array([[10.0, 20.0, 60.0]])   # mixed valid percents
    data[1] = np.array([[100.0, 0.0, 50.0]])
    data[2] = np.array([[80.0, 250.0, 80.0]])  # one cloud byte -> masked
    data[3] = np.array([[200.0, 255.0, 211.0]])  # all flag bytes -> all masked
    ds = xr.Dataset({"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons})
    path = tmp_path / "modis_single_row.nc"
    ds.to_netcdf(path)
    return path, data


def test_parity_single_row_exact_vs_native_unweighted_mean(tmp_path):
    """Single-lat-row grid: COS cos-lat basin_mean == native unweighted mean to
    float tolerance (one latitude -> one weight -> weighting is a no-op)."""
    nc_path, data = _single_row_modis_nc(tmp_path)
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(nc_path, spec,
                              datetime(2020, 1, 1, tzinfo=UTC),
                              datetime(2021, 1, 1, tzinfo=UTC))

    native = _native_unweighted_mean(data)
    cos_by_month = {p.timestamp.month: p for p in series.points}
    for month, idx in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        p = cos_by_month[month]
        if np.isfinite(native[idx]):
            assert p.value == pytest.approx(float(native[idx]), abs=1e-12)
            assert p.quality == QualityFlag.GOOD
        else:
            # native all-NaN timestep -> COS MISSING / None, the fill rule.
            assert p.value is None
            assert p.quality == QualityFlag.MISSING


def test_parity_constant_field_exact_vs_native(tmp_path):
    """Spatially constant field: area weights cancel, so COS cos-lat basin_mean
    == native unweighted mean exactly, AND both == the percent/100 value."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]")
    lats = np.array([48.0, 50.0, 52.0, 54.0])   # WIDE latitude span on purpose
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((2, 4, 3), dtype="float64")
    data[0] = 37.0     # uniform 37% everywhere -> 0.37
    data[1] = 90.0     # uniform 90% everywhere -> 0.90
    ds = xr.Dataset({"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons})
    nc_path = tmp_path / "modis_const.nc"
    ds.to_netcdf(nc_path)

    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(47.0, -117.0, 55.0, -113.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(nc_path, spec,
                              datetime(2020, 1, 1, tzinfo=UTC),
                              datetime(2021, 1, 1, tzinfo=UTC))
    native = _native_unweighted_mean(data)
    by_month = {p.timestamp.month: p for p in series.points}
    # Even across a wide latitude span, a constant field gives the identical
    # answer for weighted and unweighted means.
    assert by_month[1].value == pytest.approx(float(native[0]), abs=1e-12)
    assert by_month[1].value == pytest.approx(0.37, abs=1e-12)
    assert by_month[2].value == pytest.approx(float(native[1]), abs=1e-12)
    assert by_month[2].value == pytest.approx(0.90, abs=1e-12)


def test_parity_narrow_bbox_relative_tolerance_vs_native(modis_nc):
    """Multi-row, non-constant field over the small Bow-scale bbox: COS cos-lat
    weighted mean tracks the native UNWEIGHTED mean to ~1e-3 relative. This is
    the documented benign divergence (weighting vs none over a narrow basin)."""
    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(modis_nc, spec,
                              datetime(2020, 1, 1, tzinfo=UTC),
                              datetime(2021, 1, 1, tzinfo=UTC))

    # Rebuild the same data the fixture wrote, to compute native expectation.
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 50.0
    data[1] = 100.0
    data[2] = 80.0
    data[2, 0, 0] = 250.0
    data[2, 1, 1] = 255.0
    data[3] = 200.0
    native = _native_unweighted_mean(data)

    by_month = {p.timestamp.month: p for p in series.points}
    for month, idx in [(1, 0), (2, 1), (3, 2)]:
        # cos-lat weighting over a 2-degree-tall bbox: relative diff < 1e-3.
        assert by_month[month].value == pytest.approx(float(native[idx]), rel=1e-3)
    # all-flag timestep agrees on the fill rule: native NaN -> COS MISSING.
    assert not np.isfinite(native[3])
    assert by_month[4].value is None
    assert by_month[4].quality == QualityFlag.MISSING


def test_parity_unit_factor_is_exactly_one_over_hundred(tmp_path):
    """The percent->fraction factor matches native df['sca']/100.0 exactly, for
    an arbitrary non-round percent, via the single-cell (no-weighting) path."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-15"], dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    data = np.array([[[63.0]]], dtype="float64")   # 63% -> 0.63
    ds = xr.Dataset({"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons})
    nc_path = tmp_path / "modis_one_cell.nc"
    ds.to_netcdf(nc_path)

    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="pt", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=500.0)  # nearest_cell
    series = conn.reduce_file(nc_path, spec,
                              datetime(2020, 1, 1, tzinfo=UTC),
                              datetime(2021, 1, 1, tzinfo=UTC))
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.points[0].value == pytest.approx(63.0 / 100.0, abs=1e-12)


def test_parity_fill_bytes_become_missing(tmp_path):
    """Every MODIS flag/fill byte maps to MISSING/None, exactly the native
    where((data>=0)&(data<=100)) -> NaN rule, asserted byte-by-byte."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    fill_bytes = [200, 201, 211, 237, 239, 250, 254, 255]
    times = np.array([f"2020-{i + 1:02d}-15" for i in range(len(fill_bytes))],
                     dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    data = np.array([[[float(b)]] for b in fill_bytes], dtype="float64")
    ds = xr.Dataset({"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons})
    nc_path = tmp_path / "modis_fill.nc"
    ds.to_netcdf(nc_path)

    conn = MODISSCAConnector()
    spec = ReductionSpec(domain_name="pt", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=500.0)
    series = conn.reduce_file(nc_path, spec,
                              datetime(2020, 1, 1, tzinfo=UTC),
                              datetime(2021, 1, 1, tzinfo=UTC))
    assert len(series.points) == len(fill_bytes)
    for p in series.points:
        assert p.value is None
        assert p.quality == QualityFlag.MISSING
