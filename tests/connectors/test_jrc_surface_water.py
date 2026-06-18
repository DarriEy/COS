"""JRC Surface Water connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory JRC occurrence grid (percent + fill bytes) and
reduces it; no network, no auth. Proves the percent->fraction canonicalization,
the fill / out-of-range masking, the basin-mean / nearest-cell reductions, and
the half-open UTC window-trim around the static 1984-2021 epoch — the parts that
mirror the native ``jrc_water`` handler's ``occurrence_mean`` statistic.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.jrc_surface_water import (
    JRC_EPOCH_START,
    JRCSurfaceWaterConnector,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


def _occurrence_grid():
    """Synthetic JRC occurrence grid: percent in [0,100] + a 255 fill byte.

    3x3 cells. Uniform 40% occurrence except one fill (255) cell which must be
    masked out of the mean. Valid mean = 40% -> fraction 0.40.
    """
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((3, 3), 40.0, dtype="float64")
    values[0, 0] = 255.0  # fill -> masked, must not drag the mean down
    return lats, lons, values


@pytest.fixture
def jrc_nc(tmp_path):
    """Synthetic JRC occurrence NetCDF (percent + fill byte)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats, lons, values = _occurrence_grid()
    ds = xr.Dataset(
        {"occurrence": (("lat", "lon"), values)},
        coords={"lat": lats, "lon": lons},
    )
    path = tmp_path / "jrc_occurrence_synth.nc"
    ds.to_netcdf(path)
    return path


# ---- pure reduce_arrays (no file IO at all) --------------------------------


def test_reduce_arrays_basin_mean_percent_to_fraction():
    lats, lons, values = _occurrence_grid()
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SURFACE_WATER
    assert series.unit == "fraction"
    assert series.provider == "jrc_surface_water"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"

    # One epoch point; 40% valid (fill masked) -> 0.40 fraction.
    assert len(series.points) == 1
    pt = series.points[0]
    assert pt.value == pytest.approx(0.40, abs=1e-9)
    assert pt.quality == QualityFlag.GOOD
    # Stamped at the JRC epoch start.
    assert pt.timestamp == datetime.fromisoformat(JRC_EPOCH_START).replace(tzinfo=UTC)


def test_reduce_arrays_fraction_bounded_0_1():
    lats, lons, values = _occurrence_grid()
    values[:] = 100.0  # full occurrence everywhere
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(1.0, abs=1e-9)
    vals = [p.value for p in series.points if p.value is not None]
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_all_fill_is_missing():
    lats, lons, values = _occurrence_grid()
    values[:] = 255.0  # all fill -> all masked -> NaN -> MISSING
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert len(series.points) == 1
    assert series.points[0].value is None
    assert series.points[0].quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell():
    lats, lons, values = _occurrence_grid()
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("jrc_surface_water:cell:")
    # nearest cell (51,-115) is a 40% valid cell -> 0.40 fraction.
    assert series.points[0].value == pytest.approx(0.40, abs=1e-9)


def test_window_trim_excludes_epoch_when_after_start():
    """Half-open [start, end): epoch point dropped if start is after the epoch."""
    lats, lons, values = _occurrence_grid()
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Window starts AFTER the JRC epoch (1984) -> epoch point excluded.
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(2000, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert series.points == []


def test_window_trim_end_exclusive():
    """end == epoch stamp must exclude the point (half-open)."""
    lats, lons, values = _occurrence_grid()
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    epoch = datetime.fromisoformat(JRC_EPOCH_START).replace(tzinfo=UTC)
    series = conn.reduce_arrays(
        lats, lons, values, spec,
        datetime(1980, 1, 1, tzinfo=UTC), epoch,  # end exclusive == epoch
    )
    assert series.points == []


# ---- reduce_file via NetCDF (still offline, no network) --------------------


def test_reduce_file_netcdf_path(jrc_nc):
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        jrc_nc, spec,
        datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "fraction"
    assert series.points[0].value == pytest.approx(0.40, abs=1e-9)


def test_basin_mean_requires_bbox():
    lats, lons, values = _occurrence_grid()
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="x", reduction=SpatialReduction.BASIN_MEAN,
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="bbox"):
        conn.reduce_arrays(
            lats, lons, values, spec,
            datetime(1980, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC),
        )


# ---- contract metadata -----------------------------------------------------


def test_connector_metadata():
    conn = JRCSurfaceWaterConnector()
    assert conn.slug == "jrc_surface_water"
    assert conn.kind == ObservationKind.SURFACE_WATER
    assert conn.structural_class == "gridded"
    assert conn.auth == frozenset()  # JRC = no auth


def test_list_sites_one_region():
    import asyncio

    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    sites = asyncio.run(conn.list_sites(spec))
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "jrc_surface_water:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = JRCSurfaceWaterConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="path"):
        await conn.fetch_series(spec, datetime(1980, 1, 1, tzinfo=UTC),
                                datetime(2030, 1, 1, tzinfo=UTC))


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_gcs_fetch_placeholder():
    pytest.skip("Live JRC GCS download not wired; reduce path is the proven part.")
