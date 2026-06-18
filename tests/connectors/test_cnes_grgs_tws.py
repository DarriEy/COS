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


# --- PARITY-BY-CONSTRUCTION vs the native SYMFLUENCE cnes_grgs handler --------
#
# Native semantics (symfluence acquisition handler cnes_grgs_tws.py, lines
# 95-122) reduce one ASCII grid to a single monthly value by:
#
#   mask = (lats >= lat_min) & (lats <= lat_max) &     # INCLUSIVE on all four
#          (lons >= lon_min) & (lons <= lon_max)         # bounds, lon on 0-360
#   value_cm = float(vals[mask].mean())                # UNWEIGHTED arithmetic mean
#
# then writes that cm value and (handler process(), lines 78-84) subtracts the
# baseline-window mean to make a cm anomaly. So the native series is:
#   anomaly_cm(t) = unweighted_bbox_mean_cm(t) - mean_over_baseline(unweighted...)
#
# COS instead takes a cos-latitude AREA-WEIGHTED bbox mean, converts cm->mm at
# the boundary, and subtracts the baseline-window mean (mm). The only semantic
# difference is unweighted (native) vs cos-lat-weighted (COS) averaging; unit is
# a pure x10 factor; the bbox selection and inclusive bounds are identical; the
# anomaly re-referencing is the same subtraction. These tests reimplement the
# native reduction inline on the SAME synthetic grids and assert COS == native:
#   * identically (float tol) for a uniform field (weighting is irrelevant), and
#   * within ~1e-3 relative for a latitude-varying field over a narrow bbox.


def _native_bbox_mean_cm(lats_axis, lons_axis, grid_cm, bbox):
    """Reimplement the native acquirer's UNWEIGHTED inclusive-bbox mean (cm).

    Mirrors symfluence cnes_grgs acquisition handler lines 109-122 exactly: build
    a flat boolean mask with inclusive bounds, then a plain arithmetic mean of the
    selected cell values. ``bbox`` is COS's (lat_min, lon_min, lat_max, lon_max),
    lons given on 0-360 to match the synthetic grids (native runs on 0-360 too).
    """
    import numpy as np

    lat_min, lon_min, lat_max, lon_max = bbox
    flat_lats, flat_lons, flat_vals = [], [], []
    for i, la in enumerate(lats_axis):
        for j, lo in enumerate(lons_axis):
            v = grid_cm[i, j]
            if np.isfinite(v):
                flat_lats.append(la)
                flat_lons.append(lo)
                flat_vals.append(v)
    flat_lats = np.array(flat_lats)
    flat_lons = np.array(flat_lons)
    flat_vals = np.array(flat_vals)
    mask = (
        (flat_lats >= lat_min) & (flat_lats <= lat_max)
        & (flat_lons >= lon_min) & (flat_lons <= lon_max)
    )
    return float(flat_vals[mask].mean())


def test_parity_uniform_field_exact(grgs_dir):
    """Uniform field: cos-lat weighting is irrelevant, so COS == native exactly.

    Native bbox-mean of a uniform grid equals the cell value; cm->mm is x10; the
    anomaly subtracts the baseline-window mean. With baseline (2005,2006)=2cm and
    2020=5cm the native anomaly is (5-2)*10 = 30 mm, and COS must agree to float
    tolerance because every cell carries identical weight.
    """
    import numpy as np

    # COS series (mm anomaly), large basin -> basin_mean.
    bbox = (50.0, 244.0, 52.0, 246.0)  # 0-360 lons to match the grid directly
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=bbox, centroid=(51.0, 245.0),
        area_km2=8000.0, options={"baseline": ("2004-01-01", "2009-12-31")},
    )
    series = conn.reduce_file(
        grgs_dir, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    cos_by_year = {p.timestamp.year: p.value for p in series.points}

    # Native: reimplement inline on the SAME grids, in cm, then cm->mm + anomaly.
    lats, lons, times, vals_cm = parse_grids(grgs_dir)
    native_cm = {
        t.year: _native_bbox_mean_cm(lats, lons, vals_cm[k], bbox)
        for k, t in enumerate(times)
    }
    baseline = [v for y, v in native_cm.items() if 2004 <= y <= 2009]
    offset = sum(baseline) / len(baseline)
    native_mm_anom = {y: (v - offset) * CM_TO_MM_IMPORT for y, v in native_cm.items()}

    for y in (2005, 2006, 2020):
        assert cos_by_year[y] == pytest.approx(native_mm_anom[y], abs=1e-9)
    np.testing.assert_allclose(cos_by_year[2020], 30.0, atol=1e-9)


def test_parity_lat_varying_field_within_tolerance(tmp_path):
    """Latitude-varying field over a narrow bbox: COS (cos-lat) ~ native (unweighted).

    The documented benign divergence. Over a small/narrow-latitude bbox the cos-lat
    weights vary by < a few x 1e-3 across the band, so the weighted and unweighted
    means agree to ~1e-3 relative. We make the field vary with latitude so that the
    two reductions are genuinely different numbers (not trivially equal), then bound
    their disagreement.
    """

    lats = [50.0, 51.0, 52.0]
    lons = [244.0, 245.0, 246.0]

    def grid_text(per_lat_cm):
        lines = ["# synthetic lat-varying EWH grid"]
        for la, val in zip(lats, per_lat_cm):
            for lo in lons:
                lines.append(f"{lo:.4f} {la:.4f} {val:.4f}")
        return "\n".join(lines) + "\n"

    # Distinct value per latitude row so unweighted != cos-lat-weighted in general.
    # A physically realistic ~few-percent gradient across the 2-degree band (TWS
    # does not swing 50% over 2 deg); steeper gradients would push the cos-lat vs
    # unweighted gap above 1e-3 (that gap is exactly the documented benign
    # divergence and scales with the field's latitudinal contrast).
    (tmp_path / "GWH-2_2005152-2005181_RL05.txt").write_text(grid_text([1.0, 1.0, 1.0]))
    (tmp_path / "GWH-2_2020152-2020181_RL05.txt").write_text(grid_text([4.9, 5.0, 5.1]))

    bbox = (50.0, 244.0, 52.0, 246.0)
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=bbox, centroid=(51.0, 245.0),
        area_km2=8000.0, options={"baseline": ("2004-01-01", "2009-12-31")},
    )
    series = conn.reduce_file(
        tmp_path, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    cos_2020 = next(p.value for p in series.points if p.timestamp.year == 2020)

    lats_ax, lons_ax, times, vals_cm = parse_grids(tmp_path)
    native_cm = {
        t.year: _native_bbox_mean_cm(lats_ax, lons_ax, vals_cm[k], bbox)
        for k, t in enumerate(times)
    }
    offset = native_cm[2005]
    native_2020 = (native_cm[2020] - offset) * CM_TO_MM_IMPORT

    # The two reductions are genuinely different (lat-varying field), but agree to
    # better than 1e-3 relative over this narrow (2-degree) latitude band.
    assert cos_2020 != pytest.approx(native_2020, abs=0.0)  # not trivially identical
    assert cos_2020 == pytest.approx(native_2020, rel=1e-3)


def test_parity_unit_factor_is_exactly_cm_to_mm(grgs_dir):
    """The COS unit conversion is a pure x10 (cm->mm), matching native cm * 10.

    Pin the raw (pre-anomaly) magnitude: with a baseline window that excludes all
    data the connector falls back to the global mean, so a uniform 2 cm / 5 cm grid
    must reduce to raw 20 / 50 mm before anomaly. We verify via the difference,
    which is unit-factor-only and weighting-independent for a uniform field.
    """
    bbox = (50.0, 244.0, 52.0, 246.0)
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=bbox, centroid=(51.0, 245.0),
        area_km2=8000.0, options={"baseline": ("2005-01-01", "2006-12-31")},
    )
    series = conn.reduce_file(
        grgs_dir, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_year = {p.timestamp.year: p.value for p in series.points}
    # native cm diff (5-2)=3 cm; COS mm anomaly diff must be 3 * 10 = 30 mm.
    assert by_year[2020] - by_year[2005] == pytest.approx(30.0, abs=1e-9)


def test_parity_fill_missing_maps_to_quality_missing(tmp_path):
    """A timestep whose bbox cells are all NaN -> value None / QualityFlag.MISSING.

    Native would propagate NaN (mean of an empty/NaN selection); COS represents the
    no-finite-cell case explicitly as a MISSING-quality point with value None. We
    construct a month whose only in-grid cells are absent (so the folded grid is all
    NaN inside the bbox) and assert the canonical MISSING contract.
    """

    from cos.core.models import QualityFlag

    # Shared axes across both grids (parse_grids requires identical lat/lon axes):
    # an in-bbox block (lats 50-52) plus one out-of-bbox row (lat 10) so a grid can
    # carry data ONLY outside the bbox, folding the in-bbox cells to NaN.
    lats = [10.0, 50.0, 51.0, 52.0]
    lons = [244.0, 245.0, 246.0]

    def grid_text(in_bbox_value_cm):
        """Every grid lists the SAME cells (shared axes); in_bbox_value_cm=None
        writes NaN for in-bbox cells so they fold to a NaN block."""
        lines = ["# synthetic"]
        for la in lats:
            for lo in lons:
                if la == 10.0:
                    field = "99.0000"  # out-of-bbox sentinel, present in every grid
                elif in_bbox_value_cm is None:
                    field = "nan"  # in-bbox NaN -> MISSING after reduction
                else:
                    field = f"{in_bbox_value_cm:.4f}"
                lines.append(f"{lo:.4f} {la:.4f} {field}")
        return "\n".join(lines) + "\n"

    (tmp_path / "GWH-2_2005152-2005181_RL05.txt").write_text(grid_text(2.0))
    (tmp_path / "GWH-2_2020152-2020181_RL05.txt").write_text(grid_text(None))

    bbox = (50.0, 244.0, 52.0, 246.0)
    conn = CNESGRGSConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=bbox, centroid=(51.0, 245.0),
        area_km2=8000.0, options={"baseline": ("2004-01-01", "2009-12-31")},
    )
    series = conn.reduce_file(
        tmp_path, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_year = {p.timestamp.year: p for p in series.points}
    assert by_year[2020].value is None
    assert by_year[2020].quality == QualityFlag.MISSING
    # The finite baseline month is still GOOD-valued.
    assert by_year[2005].quality == QualityFlag.GOOD


# Import the connector's cm->mm constant for the parity arithmetic above.
from cos.connectors.cnes_grgs_tws import CM_TO_MM as CM_TO_MM_IMPORT  # noqa: E402
