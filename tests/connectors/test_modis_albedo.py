"""MODIS MCD43A3 albedo connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MCD43A3-like grid of *stored 16-bit integers* and
reduces it via the pure ``reduce_arrays`` core; no network, no auth.

There is no SYMFLUENCE native albedo handler, so these tests are
**spec-validated**: they assert the connector reproduces the published MCD43A3
product specification on the synthetic fixture — the documented scale factor
(0.001), the valid stored range (0..1000 -> reflectance 0..1), the fill value
(32767 -> NaN -> MISSING), half-open UTC window trim, and the basin reduction.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_albedo import (
    FILL_VALUE,
    SCALE_FACTOR,
    VALID_MAX,
    MODISAlbedoConnector,
)
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


def _grid():
    """Synthetic MCD43A3 stored-integer grid: 4 daily steps over a 3x3 grid.

    Values are stored 16-bit integers (reflectance * 1000). Step 2 carries a
    fill cell and an out-of-range cell that must be masked out of the mean.
    """
    times = np.array(
        ["2020-06-01", "2020-06-02", "2020-06-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    raw = np.empty((4, 3, 3))
    raw[0] = 150.0   # stored 150 -> reflectance 0.150
    raw[1] = 300.0   # stored 300 -> reflectance 0.300
    raw[2] = 150.0   # stored 150 -> 0.150, but two cells corrupted below
    raw[2][0, 0] = FILL_VALUE   # 32767 fill -> masked
    raw[2][1, 1] = 5000.0       # > VALID_MAX (1000) -> masked
    raw[3] = 200.0   # outside the test window
    return lats, lons, times, raw


def test_basin_mean_scale_factor_to_reflectance():
    conn = MODISAlbedoConnector()
    lats, lons, times, raw = _grid()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_arrays(
        lats, lons, times, raw, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ALBEDO
    assert series.unit == "1"  # canonical dimensionless albedo unit
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "modis_albedo:domain:bow"
    assert series.source_info["scale_factor"] == "0.001"
    assert series.source_info["albedo_type"] == "white_sky"

    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    # Documented scale factor: stored * 0.001 -> reflectance (0..1).
    assert by_date["2020-06-01"].value == pytest.approx(150.0 * SCALE_FACTOR)  # 0.150
    assert by_date["2020-06-02"].value == pytest.approx(300.0 * SCALE_FACTOR)  # 0.300
    # Step2: fill (32767) and out-of-range (5000) cells masked; the remaining 7
    # cells are all stored 150 -> mean reflectance 0.150 exactly.
    assert by_date["2020-06-03"].value == pytest.approx(150.0 * SCALE_FACTOR)  # 0.150
    assert by_date["2020-06-03"].quality.value == "good"


def test_fill_and_out_of_range_become_missing_nearest_cell():
    """A centroid cell that is fill / out-of-range reduces to MISSING (None)."""
    conn = MODISAlbedoConnector()
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-06-01", "2020-06-02"], dtype="datetime64[ns]")
    raw = np.empty((2, 3, 3))
    raw[0] = 200.0
    raw[0][1, 1] = FILL_VALUE     # centroid (51,-115) is the fill cell
    raw[1] = 200.0
    raw[1][1, 1] = VALID_MAX + 1  # centroid out of valid range
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=300.0,  # small -> nearest_cell at the centroid
    )
    series = conn.reduce_arrays(
        lats, lons, times, raw, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_albedo:cell:")
    for p in series.points:
        assert p.value is None
        assert p.quality.value == "missing"


def test_valid_range_bounds_inclusive():
    """Stored values exactly at 0 and VALID_MAX are valid (-> 0.0 and 1.0)."""
    conn = MODISAlbedoConnector()
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    times = np.array(["2020-06-01", "2020-06-02"], dtype="datetime64[ns]")
    raw = np.empty((2, 3, 3))
    raw[0] = 0.0          # stored 0 -> reflectance 0.0 (valid, inclusive)
    raw[1] = VALID_MAX    # stored 1000 -> reflectance 1.0 (valid, inclusive)
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_arrays(
        lats, lons, times, raw, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in series.points}
    assert by_date["2020-06-01"].value == pytest.approx(0.0)
    assert by_date["2020-06-01"].quality.value == "good"
    assert by_date["2020-06-02"].value == pytest.approx(1.0)
    assert by_date["2020-06-02"].quality.value == "good"


def test_window_trim_half_open():
    conn = MODISAlbedoConnector()
    lats, lons, times, raw = _grid()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # [2020-06-02, 2020-06-03): includes 06-02, excludes 06-03 and 2021-06-01.
    series = conn.reduce_arrays(
        lats, lons, times, raw, spec,
        datetime(2020, 6, 2, tzinfo=UTC), datetime(2020, 6, 3, tzinfo=UTC),
    )
    dates = {p.timestamp.date().isoformat() for p in series.points}
    assert dates == {"2020-06-02"}


def test_black_sky_albedo_type_selected():
    """The albedo_type option flips the source_info band label."""
    conn = MODISAlbedoConnector()
    lats, lons, times, raw = _grid()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
        options={"albedo_type": "black_sky"},
    )
    series = conn.reduce_arrays(
        lats, lons, times, raw, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        albedo_type=conn._albedo_type(spec),
    )
    assert series.source_info["albedo_type"] == "black_sky"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MODISAlbedoConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="path"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_lpdaac_fetch_smoke():  # pragma: no cover - network gated
    """Placeholder for a live LP DAAC fetch (Earthdata creds required)."""
    pytest.skip("live LP DAAC MCD43A3 download not wired in this connector")
