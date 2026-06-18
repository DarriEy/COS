# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CSR (UTexas) GRACE mascon connector — hermetic test of the gridded path.

Builds a synthetic CSR-mascon-like grid (``lwe_thickness`` in cm, lon on 0-360)
and reduces it; no network, no auth. Proves the architecture-critical
cm->mm + cos-lat reduction + window-trim + anomaly-baseline path and asserts
**parity-by-construction** against the native SYMFLUENCE ``grace`` handler's CSR
reduction (which processes the CSR mascon identically to the JPL mascon — same
``lwe_thickness`` cm variable, same basin-size strategy, same 2003-2008 baseline),
mirrored in COS by :class:`cos.connectors.grace.GRACEConnector`.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.csr_grace import (
    CM_TO_MM,
    DEFAULT_BASELINE,
    CSRGRACEConnector,
)
from cos.connectors.grace import GRACEConnector
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    QualityFlag,
    ReductionSpec,
    SpatialReduction,
)

# A 3x3 grid on 0-360 lons (244..246 == -116..-114), lats 50..52 — a Bow-like box.
_LATS = np.array([50.0, 51.0, 52.0])
_LONS = np.array([244.0, 245.0, 246.0])
# Baseline-window months ~2 cm, 2020 months ~5 cm.
_TIMES = np.array(
    ["2003-06-15", "2004-06-15", "2020-06-15", "2020-07-15"],
    dtype="datetime64[ns]",
)


def _cube(value_by_index):
    """A (time, lat, lon) cm cube; *value_by_index* maps time-index -> uniform cm."""
    data = np.empty((len(_TIMES), _LATS.size, _LONS.size), dtype="float64")
    for t, v in value_by_index.items():
        data[t] = v
    return data


def _uniform_cube():
    return _cube({0: 2.0, 1: 2.0, 2: 5.0, 3: 5.0})


def _full_window():
    return datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)


def _basin_spec(area_km2: float = 8000.0) -> ReductionSpec:
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


# --- unit / scale conversion ----------------------------------------------


def test_kind_and_canonical_unit():
    conn = CSRGRACEConnector()
    assert conn.kind == ObservationKind.TWS
    assert KIND_UNITS[conn.kind] == "mm"
    assert CM_TO_MM == 10.0


def test_cm_to_mm_scale_and_anomaly_basin_mean():
    conn = CSRGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(_LATS, _LONS, _TIMES, _uniform_cube(), _basin_spec(), start, end)

    assert series.kind == ObservationKind.TWS
    assert series.unit == "mm"
    assert series.provider == "csr_grace"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "csr_grace:domain:bow"
    # cm->mm: baseline (2003-2008) = 2 cm = 20 mm; 2020 = 5 cm = 50 mm.
    # Anomaly re-references to baseline mean (20 mm) -> 2003 ~ 0, 2020 ~ +30 mm.
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2003] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2020] == pytest.approx(30.0, abs=1e-6)


def test_source_info_records_csr_center():
    conn = CSRGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(_LATS, _LONS, _TIMES, _uniform_cube(), _basin_spec(), start, end)
    assert "CSR" in series.source_info["source"]
    assert series.source_info["baseline"] == "-".join(DEFAULT_BASELINE)


# --- reduction policy (basin-size) -----------------------------------------


def test_small_basin_defaults_to_nearest_cell():
    conn = CSRGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(
        _LATS, _LONS, _TIMES, _uniform_cube(), _basin_spec(area_km2=500.0), start, end
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("csr_grace:cell:")


# --- window trim (half-open [start, end)) ----------------------------------


def test_window_trim_half_open():
    conn = CSRGRACEConnector()
    # [2020-06-01, 2020-07-15): includes 2020-06-15, excludes 2020-07-15 exactly.
    start = datetime(2020, 6, 1, tzinfo=UTC)
    end = datetime(2020, 7, 15, tzinfo=UTC)
    series = conn.reduce_arrays(_LATS, _LONS, _TIMES, _uniform_cube(), _basin_spec(), start, end)
    years_months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in years_months
    assert (2020, 7) not in years_months  # end is exclusive
    assert (2003, 6) not in years_months  # before start


# --- fill -> MISSING -------------------------------------------------------


def test_nan_fill_becomes_missing():
    conn = CSRGRACEConnector()
    start, end = _full_window()
    cube = _uniform_cube()
    cube[2] = np.nan  # the 2020-06 layer is entirely fill -> MISSING
    series = conn.reduce_arrays(_LATS, _LONS, _TIMES, cube, _basin_spec(), start, end)
    by_time = {(p.timestamp.year, p.timestamp.month): p for p in series.points}
    missing = by_time[(2020, 6)]
    assert missing.value is None
    assert missing.quality == QualityFlag.MISSING
    # A finite neighbour stays GOOD.
    assert by_time[(2020, 7)].quality == QualityFlag.GOOD


def test_nearest_cell_nan_is_missing():
    conn = CSRGRACEConnector()
    start, end = _full_window()
    cube = _uniform_cube()
    cube[2] = np.nan
    series = conn.reduce_arrays(
        _LATS, _LONS, _TIMES, cube, _basin_spec(area_km2=500.0), start, end
    )
    by_time = {(p.timestamp.year, p.timestamp.month): p for p in series.points}
    assert by_time[(2020, 6)].value is None
    assert by_time[(2020, 6)].quality == QualityFlag.MISSING


# --- (lat, lon, time) dim-order pitfall ------------------------------------


def test_lat_lon_time_ordering_via_file(tmp_path):
    """A (lat, lon, time)-ordered NetCDF reduces identically to (time, lat, lon)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    # Build a (time, lat, lon) reference cube, then write it transposed to
    # (lat, lon, time) so reduce_file must reorder it before reducing.
    cube = _uniform_cube()
    cube_llt = np.transpose(cube, (1, 2, 0))  # (lat, lon, time)
    ds = xr.Dataset(
        {"lwe_thickness": (("lat", "lon", "time"), cube_llt)},
        coords={"time": _TIMES, "lat": _LATS, "lon": _LONS},
    )
    path = tmp_path / "csr_llt.nc"
    ds.to_netcdf(path)

    conn = CSRGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_file(path, _basin_spec(), start, end)
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2003] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2020] == pytest.approx(30.0, abs=1e-6)


class _FakeTimeDA:
    """Minimal DataArray stand-in: a numeric ``.values`` with a ``units`` attr."""

    def __init__(self, values, units):
        self.values = values
        self.attrs = {"units": units}


def test_days_since_time_axis_decoded():
    """An undecoded ``days since`` numeric time axis is decoded to real timestamps.

    The CSR mascon's CDF time axis is ``days since <origin>``; when a reader hands
    xarray a file it leaves undecoded, :func:`_decode_times` must turn the raw day
    offsets into real UTC timestamps (the native handler's ``_get_time_index``
    tolerance), not pass through raw offsets.
    """
    from cos.connectors.csr_grace import _decode_times

    origin = np.datetime64("2002-01-01")
    offsets = ((_TIMES - origin) / np.timedelta64(1, "D")).astype("float64")
    decoded = _decode_times(_FakeTimeDA(offsets, "days since 2002-01-01T00:00:00Z"))
    assert np.issubdtype(decoded.dtype, np.datetime64)
    assert list(decoded.astype("datetime64[D]").astype(str)) == [
        "2003-06-15", "2004-06-15", "2020-06-15", "2020-07-15",
    ]


def test_decode_times_passthrough_datetime64():
    """An already-decoded datetime64 axis is returned unchanged."""
    from cos.connectors.csr_grace import _decode_times

    out = _decode_times(_FakeTimeDA(_TIMES, ""))
    assert np.array_equal(out, _TIMES)


# --- parity-by-construction vs native (CSR == JPL-mascon reduction) --------


def _native_mascon_nc(tmp_path):
    """A (time, lat, lon) ``lwe_thickness`` (cm) NetCDF the native COS GRACE
    connector (JPL mascon) reduces — same product layout as the CSR mascon."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {"lwe_thickness": (("time", "lat", "lon"), _uniform_cube())},
        coords={"time": _TIMES, "lat": _LATS, "lon": _LONS},
    )
    path = tmp_path / "mascon.nc"
    ds.to_netcdf(path)
    return path


def test_parity_with_native_jpl_mascon_reduction(tmp_path):
    """CSR mascon reduces identically to the JPL mascon (same native reduction).

    The native SYMFLUENCE ``grace`` handler runs JPL/CSR/GSFC through one
    ``_extract_for_basin`` path (same ``lwe_thickness`` cm, same basin-size
    strategy, same baseline). COS's :class:`GRACEConnector` mirrors that for the
    JPL mascon; this asserts the CSR connector produces a bit-identical series on
    the same grid, proving parity-by-construction.
    """
    start, end = _full_window()
    spec = _basin_spec()

    csr = CSRGRACEConnector().reduce_arrays(_LATS, _LONS, _TIMES, _uniform_cube(), spec, start, end)
    jpl = GRACEConnector().reduce_file(_native_mascon_nc(tmp_path), spec, start, end)

    csr_vals = [p.value for p in csr.points]
    jpl_vals = [p.value for p in jpl.points]
    assert csr.unit == jpl.unit == "mm"
    assert csr.reduction == jpl.reduction
    assert csr_vals == pytest.approx(jpl_vals, abs=1e-9)


def test_parity_nearest_cell_with_native(tmp_path):
    start, end = _full_window()
    spec = _basin_spec(area_km2=500.0)  # small -> nearest_cell
    csr = CSRGRACEConnector().reduce_arrays(_LATS, _LONS, _TIMES, _uniform_cube(), spec, start, end)
    jpl = GRACEConnector().reduce_file(_native_mascon_nc(tmp_path), spec, start, end)
    assert csr.reduction == SpatialReduction.NEAREST_CELL == jpl.reduction
    assert [p.value for p in csr.points] == pytest.approx([p.value for p in jpl.points], abs=1e-9)


# --- list_sites ------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sites_one_reduced_region():
    conn = CSRGRACEConnector()
    sites = await conn.list_sites(_basin_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "csr_grace:domain:bow"


# --- live (network) --------------------------------------------------------


@pytest.mark.network
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_csr_download_not_wired():
    """Live fetch without a cached file raises a clear ConnectorError (download unwired)."""
    from cos.core.exceptions import ConnectorError

    conn = CSRGRACEConnector()
    start, end = _full_window()
    with pytest.raises(ConnectorError):
        await conn.fetch_series(_basin_spec(), start, end)
