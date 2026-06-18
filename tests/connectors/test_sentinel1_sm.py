"""Sentinel-1 SM connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory Sentinel-1-like grid and reduces it; no network, no
auth. Parity basis is the SYMFLUENCE native ``sentinel1_sm`` handler, whose basin
reduction (subset to the basin bbox, then a ``skipna`` spatial mean of the valid
cells) is reimplemented INLINE on the SAME fixture and asserted equivalent to the
COS reduction within tolerance — parity-by-construction.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.sentinel1_sm import (
    FILL_VALUE,
    SOURCE_SCALE,
    Sentinel1SoilMoistureConnector,
)
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# A small lat/lon grid over the Bow basin, 3 timesteps. Values are Copernicus
# SSM % of saturation (0–100). One cell carries the fill sentinel; one timestep
# is wholly fill (-> MISSING); one is fully out of the physical band.
LATS = np.array([50.0, 51.0, 52.0])
LONS = np.array([-116.0, -115.0, -114.0])
TIMES = np.array(["2024-01-15", "2024-02-15", "2024-03-15"], dtype="datetime64[ns]")
BBOX = (50.0, -116.0, 52.0, -114.0)
CENTROID = (51.0, -115.0)


def _percent_cube() -> np.ndarray:
    """(time, lat, lon) of SSM % saturation with a fill cell and a fill layer."""
    cube = np.empty((3, 3, 3), dtype="float64")
    cube[0] = 40.0                 # uniform 40% -> 0.40 m3/m3
    cube[0, 0, 0] = FILL_VALUE     # one fill cell, skipped by the mean
    cube[1] = FILL_VALUE           # whole layer is fill -> MISSING
    cube[2] = 60.0                 # uniform 60% -> 0.60 m3/m3
    return cube


def _native_basin_mean_percent(cube: np.ndarray) -> list[float | None]:
    """Reimplement the native handler's basin reduction on the SAME cube.

    Native ``Sentinel1SMHandler`` subsets the soil-moisture DataArray to the
    basin bbox then takes ``da.mean(skipna=True)`` per timestep. The grid here is
    the whole basin, so the bbox subset is the identity; we replicate the skipna
    mean over the percent values with the fill sentinel masked to NaN (xarray
    skips NaN). The series stays in % — the COS side scales by 0.01 at the
    boundary, so we apply the same scale here for the equivalence assertion.
    """
    masked = np.where(cube == FILL_VALUE, np.nan, cube)
    out: list[float | None] = []
    for t in range(masked.shape[0]):
        layer = masked[t]
        if not np.isfinite(layer).any():
            out.append(None)
        else:
            out.append(float(np.nanmean(layer)) * SOURCE_SCALE["SSM"])
    return out


def test_percent_saturation_scaled_to_m3m3():
    """Copernicus SSM % saturation is scaled by 0.01 to the canonical m3/m3."""
    assert SOURCE_SCALE["SSM"] == 0.01
    assert SOURCE_SCALE["soil_moisture"] == 1.0  # volumetric -> identity (native pass-through)
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, _percent_cube(), "SSM", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    by_month = {p.timestamp.month: p for p in series.points}
    # 40% -> 0.40 m3/m3 (the single fill cell is skipped, not pulled in).
    assert by_month[1].value == pytest.approx(0.40, abs=1e-9)
    assert by_month[1].quality.value == "good"
    assert by_month[3].value == pytest.approx(0.60, abs=1e-9)


def test_volumetric_variable_identity_scale():
    """A volumetric ``soil_moisture`` variable (already m3/m3) passes through."""
    cube = np.full((1, 3, 3), 0.25, dtype="float64")
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_arrays(
        LATS, LONS, np.array(["2024-01-15"], dtype="datetime64[ns]"), cube,
        "soil_moisture", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(0.25, abs=1e-9)


def test_fill_layer_maps_to_missing():
    """A wholly-fill timestep reduces to None / MISSING."""
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, _percent_cube(), "SSM", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p for p in series.points}
    assert by_month[2].value is None
    assert by_month[2].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """A scaled value outside [0, 1] m3/m3 is masked to MISSING."""
    cube = np.full((1, 3, 3), 250.0, dtype="float64")  # 250% -> 2.5 m3/m3 (out of band)
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_arrays(
        LATS, LONS, np.array(["2024-01-15"], dtype="datetime64[ns]"), cube,
        "SSM", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_window_trim_half_open():
    """Half-open [start, end): the boundary timestep at end is excluded."""
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    # [2024-01-15, 2024-03-15): includes 01-15 and 02-15, excludes the 03-15 end.
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, _percent_cube(), "SSM", spec,
        datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 3, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert months == {1, 2}


def test_small_basin_defaults_to_nearest_cell():
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="tiny", bbox=BBOX, centroid=CENTROID, area_km2=500.0)
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, _percent_cube(), "SSM", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("sentinel1_sm:cell:")


def test_parity_with_native_basin_mean():
    """PARITY-BY-CONSTRUCTION: COS basin-mean == native skipna basin-mean (scaled)."""
    cube = _percent_cube()
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_arrays(
        LATS, LONS, TIMES, cube, "SSM", spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    cos_vals = [p.value for p in series.points]
    native_vals = _native_basin_mean_percent(cube)
    assert len(cos_vals) == len(native_vals)
    for cos_v, nat_v in zip(cos_vals, native_vals):
        if nat_v is None:
            assert cos_v is None
        else:
            # cos_lat weighting over a near-equal-latitude basin matches the native
            # unweighted skipna mean within tolerance (basin-mean parity is
            # tolerance-based, exactly as documented for the gridded path).
            assert cos_v == pytest.approx(nat_v, rel=1e-3, abs=1e-3)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="x", bbox=BBOX, centroid=CENTROID)
    with pytest.raises(Exception, match="cached file"):
        await conn.fetch_series(
            spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_reduce_file_from_netcdf(tmp_path):
    """End-to-end file read path on a synthetic on-disk NetCDF (no real network)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {"SSM": (("time", "lat", "lon"), _percent_cube())},
        coords={"time": TIMES, "lat": LATS, "lon": LONS},
    )
    path = tmp_path / "s1_synth.nc"
    ds.to_netcdf(path)
    conn = Sentinel1SoilMoistureConnector()
    spec = ReductionSpec(domain_name="bow", bbox=BBOX, centroid=CENTROID, area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p for p in series.points}
    assert by_month[1].value == pytest.approx(0.40, abs=1e-9)
    assert by_month[2].value is None
