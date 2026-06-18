"""SSEBop ET connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory SSEBop-like NetCDF and reduces it; no network, no
auth. Proves the gridded -> canonical-series path for the ``et`` kind: mm/day
pass-through, basin_mean vs nearest_cell policy, half-open window trim, and
nodata/negative masking -> QualityFlag.MISSING.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ssebop_et import NODATA, SSEBopETConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ssebop_nc(tmp_path):
    """A synthetic SSEBop-like NetCDF: et (mm/day) over a small grid.

    Four monthly timesteps; one timestep is all-nodata (-> MISSING), the grid
    spans the Bow-at-Banff-ish bbox. Values chosen so basin_mean is exact.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-07-15", "2020-08-15", "2021-06-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 4.0          # 2020-06: uniform 4 mm/day -> mean 4.0
    data[1] = NODATA       # 2020-07: all nodata -> MISSING
    data[2] = 6.0          # 2020-08: uniform 6 mm/day -> mean 6.0
    data[3] = 3.0          # 2021-06: outside the default test window
    ds = xr.Dataset(
        {"et": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ssebop_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_mm_per_day_passthrough(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == "mm/day"  # KIND_UNITS[ET]
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "ssebop_et:domain:bow"

    by_month = {p.timestamp.month: p for p in series.points}
    # mm/day is canonical -> exact pass-through, no scaling.
    assert by_month[6].value == pytest.approx(4.0, abs=1e-9)
    assert by_month[6].quality == QualityFlag.GOOD
    assert by_month[8].value == pytest.approx(6.0, abs=1e-9)
    # All-nodata timestep masks to MISSING / None.
    assert by_month[7].value is None
    assert by_month[7].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=300.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ssebop_et:cell:")
    # nearest cell to (51,-115) is the center cell; same uniform values.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[6] == pytest.approx(4.0, abs=1e-9)


def test_window_trim_half_open(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    # Half-open [2020-06-01, 2020-08-15): includes 06-15 & 07-15, excludes 08-15.
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) in months
    assert (2020, 8) not in months  # 08-15 == end -> excluded (half-open)
    assert (2021, 6) not in months


def test_negatives_masked_to_missing(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([-116.0, -115.0])
    # All-negative layer -> every cell masked -> MISSING.
    data = np.full((1, 2, 2), -2.0, dtype="float64")
    ds = xr.Dataset(
        {"et": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "neg.nc"
    ds.to_netcdf(path)

    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
        centroid=(50.5, -115.5), area_km2=8000.0,
    )
    series = conn.reduce_file(
        path, spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality == QualityFlag.MISSING


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0),
    )
    with pytest.raises(Exception, match="NetCDF path"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


def _native_netcdf_basin_mean(values, start, end, times):
    """Reimplement the native SYMFLUENCE ``ssebop`` NetCDF reduction inline.

    Mirrors ``SSEBopHandler._process_netcdf`` exactly for the in-bbox subgrid:

    * ``et.mean(dim=non_time_dims, skipna=True)`` -> an *UNWEIGHTED* lat/lon mean
      (numpy ``np.nanmean`` over the spatial axes), NOT cos-lat weighted;
    * units are a mm/day pass-through (no scale for the NetCDF path);
    * the final ``df['et_mm_day'].clip(lower=0)`` non-negativity clip.

    Returns ``{datetime: et_mm_day}`` for timesteps inside the half-open window,
    matching how the native CSV is later windowed/used.
    """
    out = {}
    for k, t in enumerate(times):
        ts = t.astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        if not (start <= ts < end):
            continue
        layer = values[k]
        finite = np.isfinite(layer)
        # all-skipna -> NaN (native drops/leaves missing); else UNWEIGHTED mean
        mean = np.nan if not finite.any() else float(np.nanmean(layer))
        if not np.isnan(mean):
            mean = max(mean, 0.0)  # df['et_mm_day'].clip(lower=0)
        out[ts] = mean
    return out


def test_parity_uniform_field_exact_vs_native_unweighted_mean(ssebop_nc):
    """PARITY (exact): on a CONSTANT field the COS cos-lat weighted basin_mean
    is bitwise-identical to the native UNWEIGHTED ``et.mean(skipna=True)``.

    For a uniform layer the cos-lat weights cancel in numerator/denominator, so
    the two reductions MUST agree to float tolerance — this is the strongest
    possible parity assertion and isolates unit handling from the weighting
    approximation.
    """
    xr = pytest.importorskip("xarray")
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(ssebop_nc, spec, start, end)
    cos_by_ts = {p.timestamp: p.value for p in series.points}

    # Native semantics on the SAME input (full grid is the bbox here).
    with xr.open_dataset(ssebop_nc) as ds:
        values = np.asarray(ds["et"].values, dtype="float64")
        times = np.asarray(ds["time"].values)
    # Native NetCDF path does not mask nodata; emulate by treating the SSEBop
    # nodata sentinel as the missing value the same way COS does, so the parity
    # compares like-for-like on the canonical missing rule.
    native_vals = np.where(values == NODATA, np.nan, values)
    native = _native_netcdf_basin_mean(native_vals, start, end, times)

    assert set(cos_by_ts) == set(native)
    for ts, cos_v in cos_by_ts.items():
        nat_v = native[ts]
        if np.isnan(nat_v):
            assert cos_v is None, f"{ts}: COS should be MISSING when native is NaN"
        else:
            assert cos_v is not None
            # CONSTANT field -> cos-lat weighting cancels -> EXACT agreement.
            assert cos_v == pytest.approx(nat_v, abs=1e-12), ts


def test_parity_nonuniform_field_cos_lat_vs_unweighted_within_tol(tmp_path):
    """PARITY (tolerance): on a NON-uniform field over a narrow latitude band,
    COS's cos-lat AREA-WEIGHTED basin_mean diverges from the native UNWEIGHTED
    mean only by the documented cos-lat approximation (reduce.py docstring §2).

    Over a ~2-degree lat band the cos weights vary by < ~2%, and with a bounded
    value spread the weighted/unweighted means agree to ~1e-3 relative. This
    documents the ONLY semantic difference between COS and native for ``et`` and
    shows it is benign for the basin-mean objective.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # Non-uniform but bounded ET field (mm/day). A mild latitude gradient is the
    # worst case for cos-lat vs unweighted divergence; a realistic basin-scale
    # ET spread of a few percent across a ~2-degree band keeps the cos-lat
    # weighting effect at the ~1e-3 relative level (see assertion below).
    layer = np.array([
        [3.90, 3.95, 4.00],
        [4.00, 4.05, 4.10],
        [4.10, 4.15, 4.20],
    ])
    data = layer[None, :, :].astype("float64")
    ds = xr.Dataset(
        {"et": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ssebop_nonuniform.nc"
    ds.to_netcdf(path)

    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(path, spec, start, end)
    cos_v = series.points[0].value

    # Native UNWEIGHTED mean on the identical input.
    native = _native_netcdf_basin_mean(data.astype("float64"), start, end, times)
    nat_v = next(iter(native.values()))

    # Cross-check the native value by hand: plain np.nanmean of the layer.
    assert nat_v == pytest.approx(float(np.nanmean(layer)), abs=1e-12)

    # cos-lat weighted (COS) vs unweighted (native): benign, ~1e-3 relative.
    assert cos_v == pytest.approx(nat_v, rel=1e-3)
    # And NOT bitwise-equal (the weighting genuinely does something) — guards
    # against the connector silently dropping to an unweighted mean.
    assert abs(cos_v - nat_v) > 0.0


def test_parity_unit_factor_is_identity_mm_per_day(ssebop_nc):
    """PARITY (unit): both COS and native treat the NetCDF product as mm/day
    with NO scale factor (the /10 applies only to the CONUS GeoTIFF). The COS
    canonical unit equals KIND_UNITS[ET] == native column 'et_mm_day' semantics.
    """
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        ssebop_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    # Input 4.0 mm/day -> output 4.0 mm/day, i.e. unit factor == 1.0 exactly.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[6] == pytest.approx(4.0, abs=1e-12)


@pytest.mark.asyncio
async def test_list_sites_reduced_region(ssebop_nc):
    conn = SSEBopETConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "ssebop_et:domain:bow"
