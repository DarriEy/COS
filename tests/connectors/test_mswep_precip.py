"""MSWEP precipitation connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MSWEP-like NetCDF and reduces it; no network, no
auth. This proves the architecture-critical gridded → canonical-series path for a
merged precipitation product: identity unit (mm), non-finite (fill) masking,
basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.mswep_precip import MSWEPPrecipConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def mswep_nc(tmp_path):
    """A synthetic MSWEP-like NetCDF: precipitation (mm) over a small grid.

    Four daily timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). The
    last timestep is entirely NaN (fill) so it must reduce to MISSING; one cell
    in an otherwise-uniform layer is NaN to exercise the finite-cell masking.
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
    data[0] = 5.0           # uniform valid layer -> mean 5.0 mm
    data[1] = 10.0          # uniform valid layer
    data[1, 0, 0] = np.nan  # one masked cell -> mean stays 10.0
    data[2] = 2.0
    data[3] = np.nan        # entirely fill -> MISSING
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "mswep_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_values(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.PRECIPITATION
    assert series.unit == "mm"  # canonical, identity-converted from source mm
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "mswep:domain:bow"
    assert series.provider == "mswep_precip"

    by_day = {p.timestamp.day: p for p in series.points}
    # Uniform 5.0 mm layer -> basin mean 5.0 (no scaling applied).
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Masked NaN cell skipped; remaining cells are 10.0 -> mean unchanged.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-9)
    assert by_day[16].quality == QualityFlag.GOOD


def test_fill_value_reduces_to_missing(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-NaN (fill) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(mswep_nc):
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("mswep:cell:")
    # Nearest cell to centroid (51, -115) -> uniform layers, exact values.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[15].value == pytest.approx(5.0, abs=1e-9)
    assert by_day[17].value == pytest.approx(2.0, abs=1e-9)


def test_window_trim_half_open(mswep_nc):
    conn = MSWEPPrecipConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


def test_connector_metadata():
    conn = MSWEPPrecipConnector()
    assert conn.slug == "mswep_precip"
    assert conn.kind == ObservationKind.PRECIPITATION
    assert conn.structural_class == "gridded"
    assert conn.auth == frozenset({"gloh2o"})


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MSWEPPrecipConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION against the native SYMFLUENCE MSWEP handler.
#
# Native ref: symfluence/data/observation/handlers/mswep.py
#   MSWEPHandler._extract_basin_precip -> the spatial reduction is
#       spatial_dims = [d for d in da.dims if d != time_name]
#       precip_mean = da.mean(dim=spatial_dims, skipna=True)
#   i.e. an UNWEIGHTED arithmetic mean over the in-bbox lat/lon cells, NaN
#   cells skipped (skipna=True). Units pass through unchanged (stored as
#   'precip_mm', no scale factor) -> identity mm conversion. Temporal
#   .resample().sum() is a SEPARATE downstream step and is not part of the
#   per-file spatial reduction we port here.
#
# COS reduce_grid basin_mean is a cos-LATITUDE AREA-WEIGHTED mean (reduce.py
# basin_mean). The two means coincide:
#   * EXACTLY for a uniform layer (any weights average a constant to itself);
#   * EXACTLY for the nearest_cell path on a uniform layer (single value);
#   * to a small, BOUNDED relative gap for a non-uniform field over a
#     narrow-latitude bbox (cos weights vary little across a few tenths of a
#     degree). We assert that gap is < 1e-3 over this fixture and document it
#     as the parity tolerance for the basin-mean path.
# ---------------------------------------------------------------------------


def _native_unweighted_basin_mean(values, lats, lons, bbox):
    """Reimplement the native handler's spatial reduction inline.

    Mirrors MSWEPHandler._extract_basin_precip: subset to the bbox (cell
    centers inside the bounds), then an unweighted nan-skipping mean over the
    spatial dims, per timestep. No unit scaling (identity mm).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    # 0-360 longitude normalization, matching native _subset_to_bounds.
    if lons.max() > 180 and lon_min < 0:
        lon_min %= 360
        lon_max %= 360
    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    sub = values[:, lat_sel[:, None], lon_sel[None, :]]
    out = np.full(sub.shape[0], np.nan)
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if finite.any():
            out[t] = float(np.nanmean(layer))  # unweighted, skipna
    return out


def test_parity_uniform_layers_basin_mean_exact(mswep_nc):
    """On uniform layers COS cos-lat mean == native unweighted mean EXACTLY.

    Uniform field: any weighting averages a constant to itself, so the cos-lat
    weighted COS result must equal the native unweighted result to float
    tolerance, and the unit must pass through unchanged (identity mm).
    """
    conn = MSWEPPrecipConnector()
    spec = _spec(8000.0)  # large -> basin_mean
    series = conn.reduce_file(
        mswep_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )

    # Native expectation, computed inline on the SAME synthetic grid.
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.empty((4, 3, 3))
    data[0] = 5.0
    data[1] = 10.0
    data[1, 0, 0] = np.nan
    data[2] = 2.0
    data[3] = np.nan
    bbox = (50.0, -116.0, 52.0, -114.0)
    native = _native_unweighted_basin_mean(data, lats, lons, bbox)

    by_day = {p.timestamp.day: p for p in series.points}
    # 06-15 -> uniform 5.0; 06-16 -> uniform 10.0 (one NaN skipped both ways);
    # 06-17 -> uniform 2.0. All uniform => exact agreement.
    assert by_day[15].value == pytest.approx(float(native[0]), abs=1e-12)
    assert by_day[16].value == pytest.approx(float(native[1]), abs=1e-12)
    assert by_day[17].value == pytest.approx(float(native[2]), abs=1e-12)
    # And the absolute native values (no unit rescale applied anywhere).
    assert float(native[0]) == 5.0
    assert float(native[1]) == 10.0
    assert float(native[2]) == 2.0


def test_parity_nonuniform_cos_lat_vs_native_within_tol(tmp_path):
    """Non-uniform field over a narrow-lat bbox: cos-lat ~ unweighted (<1e-3).

    Documents the ONLY benign semantic divergence: COS area-weights by
    cos(lat), native does not. Over this fixture's <=2 deg latitude span the
    cos weights are nearly equal, so the relative gap is bounded well under the
    basin-mean parity tolerance (1e-3). A pathological wide-latitude bbox would
    diverge more — that is why the basin-mean grade is tolerance-based, not
    identity.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    # MSWEP is a 0.1 deg product; a 3-cell bbox therefore spans ~0.2 deg of
    # latitude. We model that real resolution so the cos-lat vs unweighted gap
    # reflects the actual operating regime, not an exaggerated wide bbox.
    lats = np.array([50.0, 50.1, 50.2])
    lons = np.array([245.0, 245.1, 245.2])
    # A non-uniform, lat-varying layer so weighting actually matters.
    data = np.array([[[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]])  # (1, 3, 3); unweighted mean = 5.0
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "mswep_nonuniform.nc"
    ds.to_netcdf(path)

    bbox = (49.95, -115.0, 50.25, -114.7)  # encloses all three 0.1 deg cells
    conn = MSWEPPrecipConnector()
    spec = ReductionSpec(
        domain_name="narrow", bbox=bbox,
        centroid=(50.1, -114.85), area_km2=8000.0,
    )
    series = conn.reduce_file(
        path, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_val = series.points[0].value

    native = _native_unweighted_basin_mean(data, lats, lons, bbox)[0]

    # Native unweighted mean is exactly 5.0; cos-lat weighting pulls it a few
    # parts in 1e4 toward the equator-side rows. Gap bounded well under 1e-3.
    assert native == pytest.approx(5.0, abs=1e-12)
    assert cos_val == pytest.approx(native, rel=1e-3)
    # ...and they are genuinely close but NOT identical (weighting is active),
    # so this fixture actually exercises the divergence rather than masking it.
    gap = abs(cos_val - native) / abs(native)
    assert 0.0 < gap < 1e-3


def test_parity_nearest_cell_is_native_grid_value_identity(mswep_nc):
    """nearest_cell returns the raw grid cell value, identity unit (mm).

    The native handler always spatial-means, but on a uniform layer the mean
    equals every cell, so the nearest_cell value must match the native mean
    exactly here (and prove no unit rescaling on the point path either).
    """
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # Centroid (51, -115) -> exact grid cell; uniform layers => == native mean.
    assert by_day[15].value == pytest.approx(5.0, abs=1e-12)
    assert by_day[16].value == pytest.approx(10.0, abs=1e-12)
    assert by_day[17].value == pytest.approx(2.0, abs=1e-12)


def test_parity_fill_and_nan_skip_match_native(mswep_nc):
    """Fill handling matches native: skipna mean; all-NaN layer -> no value.

    Native uses skipna=True, so a partially-masked layer averages over the
    finite cells (same value here, uniform), and an all-NaN layer yields NaN
    which the native pipeline drops. COS surfaces the all-NaN layer as
    QualityFlag.MISSING with value None — the canonical equivalent of native's
    dropped/NaN cell.
    """
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # Partial NaN (06-16) -> finite-cell mean, GOOD.
    assert by_day[16].value == pytest.approx(10.0, abs=1e-12)
    assert by_day[16].quality == QualityFlag.GOOD
    # All-NaN (06-18) -> MISSING / None (native: NaN -> dropped).
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_parity_half_open_window_matches_native_loc_semantics(mswep_nc):
    """COS half-open [start, end) trim vs native inclusive .loc[start:end].

    Native filters with df.loc[start:end] (inclusive on both ends). COS uses a
    half-open [start, end) UTC window. The semantics differ only at the exact
    end timestamp; for an end strictly after the last retained obs (the normal
    case) the retained sets are identical. We assert the half-open behaviour
    explicitly so any drift is caught: [06-15, 06-17) keeps 15,16 and drops 17.
    """
    conn = MSWEPPrecipConnector()
    series = conn.reduce_file(
        mswep_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = sorted(p.timestamp.day for p in series.points)
    assert days == [15, 16]


@pytest.mark.network
def test_live_placeholder():
    """Live GloH2O fetch is network/auth-gated; reduction is the proven path."""
    pytest.skip("live MSWEP fetch requires GloH2O credentials + rclone")
