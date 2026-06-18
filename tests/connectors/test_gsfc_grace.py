# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GSFC GRACE mascon TWS connector — hermetic test of the gridded basin-reduction path.

Builds synthetic in-memory GSFC-mascon-like arrays and reduces them; no network,
no auth, no NetCDF dependency (the pure :meth:`GSFCGRACEConnector.reduce_arrays`
core is exercised directly). Asserts:

* cm → mm (canonical ``tws`` unit) boundary conversion + scale;
* 2003-2008 anomaly baseline subtraction;
* half-open ``[start, end)`` UTC window trim;
* fill / non-finite cells → :class:`QualityFlag.MISSING`;
* basin_mean vs nearest_cell reduction selection;
* **parity-by-construction** vs the native SYMFLUENCE ``grace`` handler's GSFC
  branch (cm→mm via ×10, cos-lat-weighted basin mean, 2003-2008 anomaly);
* a ``(lat, lon, time)``-ordered granule is transposed correctly before reducing.

Live PO.DAAC/Earthdata fetch is marked ``@pytest.mark.network``.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.gsfc_grace import (
    CM_TO_MM,
    GSFCGRACEConnector,
)
from cos.core.models import (
    ObservationKind,
    QualityFlag,
    ReductionSpec,
    SpatialReduction,
)

# Four monthly mascon timesteps: two baseline years (2003, 2004) at 2.0 cm, two
# 2020 months at 5.0 cm. cm→mm makes baseline 20 mm, 2020 50 mm → anomaly +30 mm.
TIMES = np.array(
    ["2003-06-15", "2004-06-15", "2020-06-15", "2020-07-15"],
    dtype="datetime64[ns]",
)
LATS = np.array([50.0, 51.0, 52.0])
LONS = np.array([244.0, 245.0, 246.0])  # 0-360 convention (= -116..-114)
CM_VALUES = np.array([2.0, 2.0, 5.0, 5.0])  # per-timestep, uniform over the grid


def _grid_cm():
    """(time, lat, lon) cm field, uniform per timestep."""
    data = np.empty((4, 3, 3), dtype="float64")
    for t in range(4):
        data[t] = CM_VALUES[t]
    return data


def _spec(area_km2: float = 8000.0) -> ReductionSpec:
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def _full_window():
    return datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)


def test_reduce_arrays_cm_to_mm_anomaly_basin_mean():
    conn = GSFCGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(LATS, LONS, TIMES, _grid_cm(), _spec(), start, end)

    assert series.kind == ObservationKind.TWS
    assert series.unit == "mm"
    assert series.provider == "gsfc_grace"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "gsfc_grace:domain:bow"

    by_year = {p.timestamp.year: p.value for p in series.points}
    # Baseline (2003-2008) mean = 20 mm; 2020 = 50 mm → anomaly +30 mm.
    assert by_year[2003] == pytest.approx(0.0, abs=1e-9)
    assert by_year[2004] == pytest.approx(0.0, abs=1e-9)
    assert by_year[2020] == pytest.approx(30.0, abs=1e-9)
    assert all(p.quality == QualityFlag.GOOD for p in series.points)


def test_small_basin_defaults_to_nearest_cell():
    conn = GSFCGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(LATS, LONS, TIMES, _grid_cm(), _spec(area_km2=500.0), start, end)
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("gsfc_grace:cell:")
    # Anomaly is still +30 mm at the nearest cell (uniform grid).
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2020] == pytest.approx(30.0, abs=1e-9)


def test_window_trim_half_open():
    conn = GSFCGRACEConnector()
    # Half-open [2020-06-01, 2020-07-15): includes 06-15, excludes 07-15.
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, _grid_cm(), _spec(),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) not in months


def test_fill_and_nonfinite_become_missing():
    conn = GSFCGRACEConnector()
    data = _grid_cm()
    # Make the 2020-06 layer entirely land-mask fill -> NaN -> MISSING (no in-box
    # finite cell, so basin_mean yields NaN for that timestep).
    from cos.connectors.gsfc_grace import FILL_VALUE
    data[2] = FILL_VALUE
    # Make the 2020-07 layer non-finite (off-Earth inf).
    data[3] = np.inf
    start, end = _full_window()
    series = conn.reduce_arrays(LATS, LONS, TIMES, data, _spec(), start, end)
    by_month = {(p.timestamp.year, p.timestamp.month): p for p in series.points}
    assert by_month[(2020, 6)].quality == QualityFlag.MISSING
    assert by_month[(2020, 6)].value is None
    assert by_month[(2020, 7)].quality == QualityFlag.MISSING
    assert by_month[(2020, 7)].value is None
    # Baseline timesteps remain GOOD.
    assert by_month[(2003, 6)].quality == QualityFlag.GOOD


def test_lat_lon_time_ordered_granule_is_transposed():
    """A (lat, lon, time)-ordered file must reduce identically to (time, lat, lon).

    This guards the dim-order pitfall fixed in the gridded LST/SIF connectors:
    reduce_grid indexes (time, lat, lon), so the connector must transpose.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    import tempfile
    from pathlib import Path

    # (lat, lon, time) cm field, same uniform-per-timestep values.
    data_llt = np.empty((3, 3, 4), dtype="float64")
    for t in range(4):
        data_llt[:, :, t] = CM_VALUES[t]
    ds = xr.Dataset(
        {"lwe_thickness": (("lat", "lon", "time"), data_llt)},
        coords={"lat": LATS, "lon": LONS, "time": TIMES},
    )
    conn = GSFCGRACEConnector()
    start, end = _full_window()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gsfc_llt.nc"
        ds.to_netcdf(path)
        series = conn.reduce_file(path, _spec(), start, end)
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2020] == pytest.approx(30.0, abs=1e-9)
    assert by_year[2003] == pytest.approx(0.0, abs=1e-9)


def test_native_parity_by_construction():
    """Mirror the native ``grace`` handler's GSFC math on the same synthetic field.

    Native handler (data/observation/handlers/grace.py): basin > 1000 km² →
    cos-lat-weighted bbox mean of ``lwe_thickness``; values carried in cm, the
    canonical boundary scales cm→mm (×10); anomaly = series minus its 2003-2008
    baseline mean. We recompute that independently and assert equality.
    """
    conn = GSFCGRACEConnector()
    start, end = _full_window()
    series = conn.reduce_arrays(LATS, LONS, TIMES, _grid_cm(), _spec(), start, end)

    # Independent native-equivalent computation on the same arrays.
    lat_min, lon_min, lat_max, lon_max = 50.0, -116.0, 52.0, -114.0
    # 0-360 grid vs negative request lon: shift, as the native handler does.
    lo_min, lo_max = lon_min + 360.0, lon_max + 360.0
    lat_sel = np.where((LATS - lat_min >= 0) & (lat_max - LATS >= 0))[0]
    lon_sel = np.where((LONS - lo_min >= 0) & (lo_max - LONS >= 0))[0]
    weights = np.cos(np.deg2rad(LATS[lat_sel]))
    native_mm = []
    grid = _grid_cm()
    for t in range(4):
        sub = grid[t][np.ix_(lat_sel, lon_sel)] * CM_TO_MM  # cm → mm
        w2d = np.broadcast_to(weights[:, None], sub.shape)
        native_mm.append(float(np.sum(sub * w2d) / np.sum(w2d)))
    native_mm = np.array(native_mm)
    baseline_mean = native_mm[:2].mean()  # 2003 + 2004
    native_anom = native_mm - baseline_mean

    cos_anom = np.array([p.value for p in series.points])
    np.testing.assert_allclose(cos_anom, native_anom, atol=1e-9)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = GSFCGRACEConnector()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            _spec(), datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_gsfc_fetch_smoke():
    """Live smoke test: requires a real GSFC mascon NetCDF (Earthdata/PO.DAAC).

    Skipped by default; the live download path is not wired, so this only
    documents the live entry point.
    """
    pytest.skip("Live GSFC mascon fetch requires Earthdata credentials + wired download")
