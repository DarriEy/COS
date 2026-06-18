"""MODIS FAPAR connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MODIS-FAPAR-like NetCDF (digital numbers) and
reduces it; no network, no auth. Proves the architecture-critical gridded →
canonical path for the Fraction of Absorbed PAR: the 0.01 DN scale factor to
the canonical dimensionless fraction unit ("1", 0..1), fill (255) /
out-of-range masking, the QC algorithm-path filter, basin-mean vs nearest-cell
reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_fapar import FAPAR_FILL_VALUE, MODISFAPARConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def fapar_nc(tmp_path):
    """A synthetic MODIS-FAPAR NetCDF: Fpar_500m digital numbers over a small grid.

    Four 8-day timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). Values
    are raw DN (pre-scale): layer 0 uniform DN 30 (-> FAPAR 0.30), layer 1 DN 80
    with one out-of-range cell (DN 200 > 100, masked), layer 2 DN 50, layer 3
    entirely fill (255) so it must reduce to MISSING.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-09", "2020-06-17", "2020-06-25", "2020-07-03"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = 30.0             # uniform DN 30 -> FAPAR 0.30
    data[1] = 80.0             # uniform DN 80 -> FAPAR 0.80
    data[1, 0, 0] = 200.0      # out-of-range DN -> masked, mean stays 0.80
    data[2] = 50.0             # uniform DN 50 -> FAPAR 0.50
    data[3] = FAPAR_FILL_VALUE  # entirely 255 fill -> MISSING
    ds = xr.Dataset(
        {"Fpar_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_fapar_synth.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def fapar_qc_nc(tmp_path):
    """A 1-timestep FAPAR NetCDF with a QC layer to exercise the algorithm filter.

    DN 40 everywhere (-> FAPAR 0.40). QC bits 5-7: one cell main (0 -> keep),
    one saturation (2 -> keep), the rest backup (1 -> drop). The two kept cells
    both hold DN 40, so the basin mean is FAPAR 0.40.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-09"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.full((1, 3, 3), 40.0)
    qc = np.full((1, 3, 3), 1 << 5, dtype="int16")  # algorithm path 1 (backup) -> drop
    qc[0, 0, 0] = 0 << 5      # main (0) -> keep
    qc[0, 1, 1] = 2 << 5      # saturation (2) -> keep
    ds = xr.Dataset(
        {
            "Fpar_500m": (("time", "lat", "lon"), data),
            "FparLai_QC": (("time", "lat", "lon"), qc),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_fapar_qc_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_scale(fapar_nc):
    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        fapar_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.FAPAR
    assert series.unit == "1"  # canonical dimensionless FAPAR fraction
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "modis_fapar:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # DN 30 * 0.01 = FAPAR 0.30.
    assert by_day[9].value == pytest.approx(0.30, abs=1e-9)
    assert by_day[9].quality == QualityFlag.GOOD
    # Out-of-range DN 200 masked; remaining DN 80 * 0.01 = 0.80.
    assert by_day[17].value == pytest.approx(0.80, abs=1e-9)
    # DN 50 * 0.01 = 0.50.
    assert by_day[25].value == pytest.approx(0.50, abs=1e-9)
    # All canonical FAPAR values are in [0, 1].
    for p in series.points:
        if p.value is not None:
            assert 0.0 <= p.value <= 1.0


def test_fill_value_reduces_to_missing(fapar_nc):
    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        fapar_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (255) layer must surface as MISSING with no value.
    assert by_day[3].value is None
    assert by_day[3].quality == QualityFlag.MISSING


def test_qc_algorithm_filter(fapar_qc_nc):
    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        fapar_qc_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    # Only the main (0) + saturation (2) cells survive; both DN 40 -> FAPAR 0.40.
    assert series.points[0].value == pytest.approx(0.40, abs=1e-9)
    assert series.points[0].quality == QualityFlag.GOOD


def test_small_basin_defaults_to_nearest_cell(fapar_nc):
    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        fapar_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_fapar:cell:")
    # Nearest cell to centroid is uniform within each layer: DN 30 * 0.01 = 0.30.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[9].value == pytest.approx(0.30, abs=1e-9)


def test_window_trim_half_open(fapar_nc):
    conn = MODISFAPARConnector()
    # Half-open [06-09, 06-25): includes 06-09 and 06-17, excludes 06-25.
    series = conn.reduce_file(
        fapar_nc, _spec(8000.0),
        datetime(2020, 6, 9, tzinfo=UTC), datetime(2020, 6, 25, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {9, 17}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MODISFAPARConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_earthdata_fetch_placeholder():
    """Live Earthdata fetch is not wired (parity is the reduce path); skip."""
    pytest.skip("MODIS FAPAR live Earthdata download is not wired in this connector")


# --------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION
#
# The native SYMFLUENCE handler
# (src/symfluence/data/observation/handlers/modis_lai.py) reduces a MODIS
# FAPAR grid to a basin-mean FAPAR series as follows, in MODISLAIHandler:
#
#   1. valid-range mask: da.where((da >= 0) & (da <= 100))          -> NaN
#      (masks the 255 fill byte and every out-of-range DN), via FPAR_VALID_RANGE;
#   2. QC algorithm-path filter, _apply_qc_filter:
#        algorithm_bits = (qc >> 5) & 0b111; keep where bits in {0, 2};
#   3. bbox subset via inclusive da.sel(lat=slice, lon=slice);
#   4. *** UNWEIGHTED *** spatial mean: float(da.mean(skipna=True)) over
#      all (lat, lon) cells -- NO cosine-latitude weighting;
#   5. scale: mean_val * FPAR_SCALE_FACTOR (0.01) -> FAPAR fraction [0, 1];
#   6. an all-NaN layer -> None (the canonical MISSING).
#
# COS's reduce.basin_mean instead takes a COS-LATITUDE AREA-WEIGHTED mean
# (reduce.py docstring: a documented, tolerance-based approximation of
# polygon-weighted zonal stats). That is the ONLY semantic divergence; it is
# benign for the FAPAR objective and vanishes for constant-within-layer fields
# (the weights factor out). The tests below pin BOTH facts: exact agreement on
# constant fields, and a bounded (small) divergence on a latitude-varying field.
#
# These tests reimplement the native semantics inline on the SAME synthetic
# input and compare against the COS connector's pure reduce_file helper.
# --------------------------------------------------------------------------

# Match the synthetic grid / bbox used by the fixtures above so the inline
# native reimplementation and the COS connector see exactly the same cells.
_LATS = np.array([50.0, 51.0, 52.0])
_LONS = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
_BBOX = (50.0, -116.0, 52.0, -114.0)


def _native_basin_mean_fapar(dn_layer):
    """Reimplement the native MODISLAIHandler FAPAR reduction on one DN layer.

    Mirrors _extract_basin_mean + the 0.01 scale exactly: valid-range mask,
    inclusive bbox (the whole synthetic grid lies in the bbox), then an
    UNWEIGHTED skipna mean, scaled by FPAR_SCALE_FACTOR (0.01). Returns None for
    an all-invalid layer (the native None -> canonical MISSING).
    """
    lo, hi = 0.0, 100.0  # native FPAR_VALID_RANGE
    arr = np.asarray(dn_layer, dtype="float64")
    arr = np.where((arr >= lo) & (arr <= hi), arr, np.nan)
    if not np.isfinite(arr).any():
        return None
    mean_dn = float(np.nanmean(arr))  # native da.mean(skipna=True), UNWEIGHTED
    return mean_dn * 0.01  # native FPAR_SCALE_FACTOR


def _cos_value_for(series, day):
    by_day = {p.timestamp.day: p for p in series.points}
    return by_day[day].value


def _make_fapar_nc(tmp_path, layers, times):
    """Write a synthetic Fpar_500m NetCDF from a list of (3x3) DN layers."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    data = np.stack([np.asarray(layer, dtype="float64") for layer in layers])
    ds = xr.Dataset(
        {"Fpar_500m": (("time", "lat", "lon"), data)},
        coords={"time": np.asarray(times, dtype="datetime64[ns]"),
                "lat": _LATS, "lon": _LONS},
    )
    path = tmp_path / "modis_fapar_parity.nc"
    ds.to_netcdf(path)
    return path


def test_parity_constant_field_exact(tmp_path):
    """On constant-within-layer fields COS == native to float tolerance.

    cos-lat weighting and the native unweighted mean are identical when every
    in-box cell holds the same value (the weights factor out). This is the
    strongest parity statement: a bitwise-tight agreement that also pins the
    unit factor (DN * 0.01) and the fill-byte -> MISSING rule.
    """
    layers = [
        np.full((3, 3), 30.0),                   # DN 30 -> FAPAR 0.30
        np.full((3, 3), 80.0),                   # DN 80 -> FAPAR 0.80
        np.full((3, 3), float(FAPAR_FILL_VALUE)),  # all fill -> native None / MISSING
    ]
    times = ["2020-06-09", "2020-06-17", "2020-06-25"]
    path = _make_fapar_nc(tmp_path, layers, times)

    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    assert series.unit == "1"

    # DN 30 / DN 80 layers: COS basin-mean must equal the native unweighted
    # mean * 0.01 exactly.
    assert _cos_value_for(series, 9) == pytest.approx(_native_basin_mean_fapar(layers[0]), abs=1e-12)
    assert _cos_value_for(series, 17) == pytest.approx(_native_basin_mean_fapar(layers[1]), abs=1e-12)

    # All-fill layer: native returns None; COS surfaces MISSING with no value.
    assert _native_basin_mean_fapar(layers[2]) is None
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[25].value is None
    assert by_day[25].quality == QualityFlag.MISSING


def test_parity_out_of_range_fill_rule(tmp_path):
    """The valid-range mask matches native: out-of-range DN dropped, not clipped.

    Native masks DN>100 (and the 255 fill) to NaN before meaning. A layer of
    DN 80 with one DN 200 cell must reduce, in BOTH paths, to the mean of the
    surviving DN-80 cells (= FAPAR 0.80), never a value pulled up by the 200.
    """
    layer = np.full((3, 3), 80.0)
    layer[0, 0] = 200.0  # out of range -> masked
    path = _make_fapar_nc(tmp_path, [layer], ["2020-06-09"])

    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    native = _native_basin_mean_fapar(layer)
    assert native == pytest.approx(0.80, abs=1e-12)
    # Surviving cells are all DN 80 (constant) so cos-lat == unweighted exactly.
    assert series.points[0].value == pytest.approx(native, abs=1e-12)


def test_parity_latitude_varying_divergence_is_bounded(tmp_path):
    """On a latitude-varying field COS (cos-lat) diverges from native, bounded.

    This is the documented, benign divergence: COS area-weights by cos(lat),
    native takes a plain unweighted mean. Over the synthetic 50-52 deg bbox the
    relative gap is < 1e-2 (here ~7e-3), well within the FAPAR objective's
    tolerance. The test pins the divergence so a regression to (e.g.) the wrong
    reduction or a silent unit change would break it.
    """
    # Field varies only by latitude row so weighting is what differs.
    layer = np.empty((3, 3))
    layer[0, :] = 20.0   # lat 50 -> DN 20
    layer[1, :] = 40.0   # lat 51 -> DN 40
    layer[2, :] = 60.0   # lat 52 -> DN 60
    path = _make_fapar_nc(tmp_path, [layer], ["2020-06-09"])

    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    cos_val = series.points[0].value

    native_val = _native_basin_mean_fapar(layer)  # unweighted: (20+40+60)/3 * 0.01 = 0.40
    assert native_val == pytest.approx(0.40, abs=1e-12)

    # Reproduce the COS cos-lat reduction inline to confirm it is exactly what
    # the connector applies (and to bound the divergence from native).
    w = np.cos(np.deg2rad(_LATS))
    rows = np.array([0.20, 0.40, 0.60])  # per-lat-row scaled FAPAR (DN*0.01)
    cos_lat_expected = float(np.sum(rows * w) / np.sum(w))
    assert cos_val == pytest.approx(cos_lat_expected, abs=1e-9)

    rel = abs(cos_val - native_val) / native_val
    assert rel < 1e-2, f"cos-lat vs native divergence {rel:.4f} exceeds bound"


def test_parity_window_trim_half_open(tmp_path):
    """Half-open [start, end) UTC trim matches the native experiment-period slice.

    Native trims with df.loc[start:end]; COS uses start <= ts < end. The
    half-open boundary is the canonical contract, exercised here on the parity
    fixture: a timestamp exactly at end is excluded.
    """
    layers = [np.full((3, 3), 30.0), np.full((3, 3), 80.0), np.full((3, 3), 50.0)]
    times = ["2020-06-09", "2020-06-17", "2020-06-25"]
    path = _make_fapar_nc(tmp_path, layers, times)

    conn = MODISFAPARConnector()
    series = conn.reduce_file(
        path, _spec(8000.0),
        datetime(2020, 6, 9, tzinfo=UTC), datetime(2020, 6, 25, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {9, 17}  # 06-25 == end is excluded (half-open)
