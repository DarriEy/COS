"""VODCA VOD connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory VODCA-like NetCDF (VOD stored *packed*: a scaled
integer with a ``_FillValue`` sentinel and ``scale_factor``/``add_offset``) and
reduces it; no network, no auth. This proves the architecture-critical
gridded -> canonical-series path for a dimensionless product:

* the packed scale/offset is unpacked at the boundary (``vod = raw*scale+offset``);
* the ``_FillValue`` sentinel and out-of-range cells surface as MISSING;
* basin-mean vs nearest-cell reduction and half-open UTC window trim;
* SPEC-VALIDATED: the connector reproduces the published VODCA product spec
  (packed scale/offset, fill sentinel, valid range) on the fixture — there is no
  native handler, so the spec is the parity reference.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.vodca_vod import VOD_MAX, VODCAVODConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction

# Published VODCA packing used by the fixture (a representative scaled-integer
# encoding): physical VOD = stored * SCALE + OFFSET, with FILL the sentinel.
SCALE = 0.001
OFFSET = 0.0
FILL = -9999


@pytest.fixture
def vodca_nc(tmp_path):
    """A synthetic VODCA-like NetCDF: packed daily VOD over a small grid.

    Four daily steps on a 3x3 grid (0-360 longitudes, = -116..-114). VOD is
    stored on disk as a scaled integer (scale 0.001) with a -9999 fill sentinel
    (xarray packs the physical field below on write):

      * step0: physical VOD 0.30 everywhere -> basin mean 0.30
      * step1: 0.50; one cell 9000 > VOD_MAX so it is masked and the mean stays 0.50
      * step2: 0.40
      * step3: entirely fill -> MISSING
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    # Build the PHYSICAL VOD field, then let xarray pack it on write via the
    # encoding (scale/offset/fill). Reading back with mask_and_scale=False yields the
    # raw packed integers + the packing attrs — exactly the VODCA on-disk spec.
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 0.30
    data[1] = 0.50
    data[1, 0, 0] = 9000.0      # > VOD_MAX after decode -> masked
    data[2] = 0.40
    data[3] = np.nan            # entirely fill -> packed to FILL -> MISSING
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons}, name="vod",
    )
    ds = xr.Dataset({"vod": da})
    path = tmp_path / "vodca_synth.nc"
    ds.to_netcdf(
        path,
        encoding={"vod": {
            "scale_factor": SCALE, "add_offset": OFFSET,
            "_FillValue": FILL, "dtype": "int32",
        }},
    )
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_units_and_scale_conversion(vodca_nc):
    conn = VODCAVODConnector()
    series = conn.reduce_file(
        vodca_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.VOD
    assert series.unit == "1"  # canonical dimensionless
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "vodca_vod:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # Packed 300 * 0.001 = 0.30 (scale conversion applied at the boundary).
    assert by_day[15].value == pytest.approx(0.30, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # Out-of-range cell (9000 > VOD_MAX) masked; remaining cells 0.50 -> mean 0.50.
    assert by_day[16].value == pytest.approx(0.50, abs=1e-9)
    assert by_day[17].value == pytest.approx(0.40, abs=1e-9)


def test_fill_value_reduces_to_missing(vodca_nc):
    conn = VODCAVODConnector()
    series = conn.reduce_file(
        vodca_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (-9999) layer must surface as MISSING with no value.
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(vodca_nc):
    conn = VODCAVODConnector()
    series = conn.reduce_file(
        vodca_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("vodca_vod:cell:")
    # Nearest cell to centroid (51, -115) is the uniform interior -> 0.30 at step0.
    by_day = {p.timestamp.day: p.value for p in series.points}
    assert by_day[15] == pytest.approx(0.30, abs=1e-9)


def test_window_trim_half_open(vodca_nc):
    conn = VODCAVODConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        vodca_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = VODCAVODConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="nc_path|path|NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


# --------------------------------------------------------------------------
# SPEC-VALIDATED parity (no native handler exists for VODCA).
#
# The parity reference is the published VODCA product spec: VOD is a packed
# scaled-integer field decoded as  vod = stored*scale_factor + add_offset,
# with the _FillValue sentinel masked and a physical valid range applied. The
# fixtures below reimplement that spec inline and assert the connector's pure
# decode/mask helper reproduces it bit-for-bit, then that the reduced series
# matches a spec-faithful (cos-lat weighted) reduction.
# --------------------------------------------------------------------------


def _spec_decode(raw, scale, offset, fill, vod_max):
    """Reimplement the published VODCA decode/mask spec, independently."""
    out = np.array(raw, dtype="float64", copy=True)
    out = np.where(out == float(fill), np.nan, out)
    out = out * scale + offset
    out = np.where((out < 0.0) | (out > vod_max), np.nan, out)
    return out


def test_decode_and_mask_matches_published_spec(vodca_nc):
    """The connector's pure decode/mask helper == the published packing spec."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(vodca_nc, mask_and_scale=False) as ds:
        raw = np.asarray(ds["vod"].values, dtype="float64")
        attrs = dict(ds["vod"].attrs)

    conn = VODCAVODConnector()
    decoded = conn._decode_and_mask(raw, attrs)
    expected = _spec_decode(raw, SCALE, OFFSET, FILL, VOD_MAX)

    # NaN positions agree (fill + out-of-range) and finite values agree exactly.
    assert np.array_equal(np.isnan(decoded), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert np.allclose(decoded[finite], expected[finite], atol=1e-12)


def test_reduction_matches_spec_cos_lat_mean(vodca_nc):
    """Reduced series == spec-faithful cos-lat weighted mean of decoded VOD."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(vodca_nc, mask_and_scale=False) as ds:
        lats = np.asarray(ds["lat"].values, dtype="float64")
        raw = np.asarray(ds["vod"].values, dtype="float64")

    decoded = _spec_decode(raw, SCALE, OFFSET, FILL, VOD_MAX)
    # cos-lat weighted mean over the whole (single-bbox) grid, per step.
    w = np.cos(np.deg2rad(lats))
    expected = np.full(decoded.shape[0], np.nan)
    for t in range(decoded.shape[0]):
        layer = decoded[t]
        fin = np.isfinite(layer)
        if fin.any():
            w2d = np.broadcast_to(w[:, None], layer.shape)
            expected[t] = float(np.sum(layer[fin] * w2d[fin]) / np.sum(w2d[fin]))

    conn = VODCAVODConnector()
    series = conn.reduce_file(
        vodca_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    cos_by_day = {p.timestamp.day: p.value for p in series.points}
    for day, idx in ((15, 0), (16, 1), (17, 2)):
        assert cos_by_day[day] == pytest.approx(expected[idx], abs=1e-12)


@pytest.mark.network
@pytest.mark.skip(reason="VODCA Zenodo archive download is not wired; supply nc_path")
def test_live_fetch_placeholder():
    """Live VODCA fetch would require a supplied Zenodo NetCDF (config nc_path)."""
