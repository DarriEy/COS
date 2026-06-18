# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CNES/GRGS TWS connector — hermetic test of the gridded basin-reduction path.

Builds synthetic CNES/GRGS-style ASCII EWH grids (``GWH-2_*.txt``, ``lon lat
value`` rows, cm, lon on 0-360) and reduces them; no network, no auth. Proves the
ASCII-grid parse + cos-lat reduction + cm->mm + anomaly-baseline path, mirroring
the native ``cnes_grgs`` handler's semantics.
"""

from datetime import UTC, datetime

import pytest

from cos.connectors.cnes_grgs_tws import (
    CNESGRGSConnector,
    date_from_filename,
    parse_grids,
    parse_one_grid,
)
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# A 3x3 1°-ish grid on 0-360 lons (244..246 == -116..-114), lats 50..52.
_LATS = [50.0, 51.0, 52.0]
_LONS = [244.0, 245.0, 246.0]


def _grid_text(value_cm: float) -> str:
    """One synthetic ASCII grid (uniform value, cm), with comment header lines."""
    lines = ["# CNES/GRGS RL05 EWH grid (synthetic)", "# lon lat ewh_cm"]
    for la in _LATS:
        for lo in _LONS:
            lines.append(f"{lo:.4f} {la:.4f} {value_cm:.4f}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def grgs_dir(tmp_path):
    """Directory of monthly grids: 2005 & 2006 (baseline) ~2 cm, 2020 ~5 cm.

    Filenames carry a GWH-2_YYYYDDD-YYYYDDD_ span that resolves to a month start.
    Day-of-year 152..181 (non-leap) maps to mid ~ Jun 16 -> 2020-06-01;
    182..212 -> mid ~ Jul 12 -> 2020-07-01.
    """
    specs = [
        ("GWH-2_2005152-2005181_RL05.txt", 2.0),  # 2005-06
        ("GWH-2_2006152-2006181_RL05.txt", 2.0),  # 2006-06
        ("GWH-2_2020152-2020181_RL05.txt", 5.0),  # 2020-06
        ("GWH-2_2020182-2020212_RL05.txt", 5.0),  # 2020-07
    ]
    for name, val in specs:
        (tmp_path / name).write_text(_grid_text(val))
    return tmp_path


def test_parse_one_grid_units_and_shape():
    lats, lons, grid = parse_one_grid(_grid_text(3.5))
    assert list(lats) == _LATS
    assert list(lons) == _LONS
    assert grid.shape == (3, 3)
    # Parser preserves source cm values; conversion happens at the connector boundary.
    assert grid[0, 0] == pytest.approx(3.5)


def test_date_from_filename_month_start():
    ts = date_from_filename("GWH-2_2020152-2020181_RL05.txt")
    assert ts == datetime(2020, 6, 1, tzinfo=UTC)


def test_reduce_file_basin_mean_cm_to_mm_anomaly(grgs_dir):
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
        options={"baseline": ("2004-01-01", "2009-12-31")},
    )
    series = conn.reduce_file(
        grgs_dir, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.TWS
    assert series.unit == "mm"  # KIND_UNITS[TWS]
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "cnes_grgs:domain:bow"
    # Baseline (2005-2006) mean = 20 mm (2 cm). 2020 = 50 mm -> anomaly +30 mm.
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2005] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2020] == pytest.approx(30.0, abs=1e-6)


def test_small_basin_defaults_to_nearest_cell(grgs_dir):
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        grgs_dir, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("cnes_grgs:cell:")


def test_window_trim_half_open(grgs_dir):
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    # Half-open [2020-06-01, 2020-07-01): includes the 2020-06 obs, excludes 2020-07.
    series = conn.reduce_file(
        grgs_dir, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) not in months


def test_parse_grids_sorts_time(grgs_dir):
    _lats, _lons, times, values = parse_grids(grgs_dir)
    assert times == sorted(times)
    assert values.shape == (4, 3, 3)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = CNESGRGSConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="path"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
