# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CNES/GRGS RL05 GRACE total water storage connector (gridded, basin-reduced).

A second proof of the **gridded spatial-reduction path**, sourced from the CNES/
GRGS RL05 *regularized spherical-harmonic* GRACE solutions distributed by the
ForM@Ter / SEDOO catalogue (anonymous, no Earthdata auth). Unlike NASA mascons
(NetCDF ``lwe_thickness`` in cm), CNES/GRGS ships monthly **ASCII EWH grids**:
one ``GWH-2_YYYYDDD-YYYYDDD_*.txt`` file per month, ``#``-comment header lines,
then ``lon lat value`` rows with longitude on 0-360 and value in **cm** of
equivalent water height. These solutions are already stabilized, so no mascon
post-processing is needed.

This connector mirrors the native SYMFLUENCE ``cnes_grgs`` handler:

1. parse one-or-more monthly ASCII grids (a directory, or a single file) into a
   ``(time, lat, lon)`` numpy cube — the filename's day-of-year span gives the
   month-start timestamp, exactly as the native acquirer derives it;
2. reduce to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones;
3. convert cm -> **mm** (the canonical ``tws`` unit) at the boundary and subtract
   the anomaly-baseline mean (default 2004-01-01..2009-12-31, matching native).

The live ForM@Ter fetch is wired only behind a config path; the parse + reduce +
canonicalize core is hermetically tested on a synthetic ASCII grid (no network).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError, DataFormatError, ReductionError
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationSeries,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

CM_TO_MM = 10.0
#: Native handler's default anomaly baseline (process() re-referencing window).
DEFAULT_BASELINE = ("2004-01-01", "2009-12-31")
#: <= this area (km²) defaults to point sampling, mirroring the GRACE size policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: Filename day-of-year span -> month-start, e.g. GWH-2_2005001-2005031_...
_FNAME_DATE_RE = re.compile(r"GWH-2_(\d{4})(\d{3})-(\d{4})(\d{3})_")


@register("cnes_grgs_tws")
class CNESGRGSConnector(BaseObservationConnector):
    slug = "cnes_grgs_tws"
    display_name = "CNES/GRGS RL05 GRACE TWS"
    kind = ObservationKind.TWS
    structural_class = "gridded"
    base_url = "https://api.sedoo.fr/formater-catalogue-prod/datasetcontent/v1_0"
    auth = frozenset()  # ForM@Ter / SEDOO is anonymous

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """One reduced region: the basin (or its centroid cell)."""
        reduction = self._choose_reduction(spec)
        return [self._site_for(spec, reduction)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        path = self.config.get("nc_path") or self.config.get("path")
        if not path:
            raise ConnectorError(
                self.slug,
                "CNES/GRGS live fetch needs a path (config 'path'/'nc_path') to a "
                "monthly ASCII EWH grid file or a directory of GWH-2_*.txt grids "
                "from ForM@Ter. The parse+reduce path is the proven part; supply a "
                "downloaded CNES/GRGS RL05 grid to reduce it.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Parse CNES/GRGS ASCII grid(s), reduce to the basin, canonicalize to mm anomaly."""

        from cos.core.reduce import reduce_grid

        lats, lons, times, values = parse_grids(path)  # cm EWH, (time, lat, lon)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("CNES/GRGS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("CNES/GRGS nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values * CM_TO_MM,  # cm -> mm at the boundary
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)) then anomaly baseline.
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]
        points = self._apply_baseline(points, spec)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "CNES/GRGS RL05 GRACE",
                "catalogue": "ForM@Ter / SEDOO",
                "reference": "Lemoine et al. (2007); Bruinsma et al. (2010)",
                "baseline": "-".join(spec.options.get("baseline", DEFAULT_BASELINE)),
            },
            fetched_at=datetime.now(UTC),
        )

    def _apply_baseline(self, points: list, spec: ReductionSpec) -> list:
        """Subtract the baseline-window mean to make a TWS anomaly (mm)."""
        b_start, b_end = spec.options.get("baseline", DEFAULT_BASELINE)
        b0 = _utc(datetime.fromisoformat(b_start))
        b1 = _utc(datetime.fromisoformat(b_end))
        vals = [p.value for p in points if p.value is not None and b0 <= p.timestamp <= b1]
        if not vals:
            vals = [p.value for p in points if p.value is not None]
        if not vals:
            return points
        mean = sum(vals) / len(vals)
        for p in points:
            if p.value is not None:
                p.value = p.value - mean
        return points

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"cnes_grgs:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"cnes_grgs:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"CNES/GRGS TWS over {spec.domain_name}",
        )


# --- pure, network-free parsing -------------------------------------------


def parse_one_grid(text: str):
    """Parse one CNES/GRGS ASCII EWH grid into ``(lats, lons, values2d)``.

    Format: ``#``-comment lines, then ``lon lat value`` rows. Longitude is on
    0-360, value is cm of equivalent water height. The flat point list is folded
    into a regular ``(lat, lon)`` grid keyed by the sorted unique coordinates;
    cells missing from the file become NaN. Returns sorted-ascending lat/lon
    1-D axes and a 2-D ``(nlat, nlon)`` value array (numpy float64, cm).
    """
    import numpy as np

    lons: list[float] = []
    lats: list[float] = []
    vals: list[float] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
        vals.append(float(parts[2]))

    if not vals:
        raise DataFormatError("cnes_grgs_tws", "ASCII grid had no 'lon lat value' rows")

    ulons = np.array(sorted(set(lons)), dtype="float64")
    ulats = np.array(sorted(set(lats)), dtype="float64")
    lon_ix = {round(v, 6): i for i, v in enumerate(ulons)}
    lat_ix = {round(v, 6): i for i, v in enumerate(ulats)}

    grid = np.full((ulats.size, ulons.size), np.nan, dtype="float64")
    for lo, la, v in zip(lons, lats, vals):
        grid[lat_ix[round(la, 6)], lon_ix[round(lo, 6)]] = v
    return ulats, ulons, grid


def date_from_filename(name: str):
    """Month-start :class:`datetime` (UTC) from a ``GWH-2_YYYYDDD-YYYYDDD_`` name.

    Mirrors the native acquirer: the start/end day-of-year span is converted to
    its mid-point, then floored to the first of that month.
    """
    m = _FNAME_DATE_RE.search(name)
    if not m:
        return None
    start = datetime(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)
    end = datetime(int(m.group(3)), 1, 1) + timedelta(days=int(m.group(4)) - 1)
    mid = start + (end - start) / 2
    return datetime(mid.year, mid.month, 1, tzinfo=UTC)


def parse_grids(path: Path):
    """Parse a file or directory of CNES/GRGS ASCII grids into a time cube.

    *path* may be a single ``GWH-2_*.txt`` grid or a directory of them. Returns
    ``(lats, lons, times, values)`` where ``values`` is ``(time, nlat, nlon)`` in
    cm EWH and *times* is a list of UTC :class:`datetime` sorted ascending. All
    grids must share the same lat/lon axes (the CNES product is a fixed 1° grid).
    """
    import numpy as np

    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.txt") if _FNAME_DATE_RE.search(p.name))
        if not files:
            raise DataFormatError("cnes_grgs_tws", f"No GWH-2_*.txt grids in {path}")
    else:
        files = [path]

    ref_lats: np.ndarray | None = None
    ref_lons: np.ndarray | None = None
    times: list[datetime] = []
    layers: list = []
    for f in files:
        ts = date_from_filename(f.name)
        if ts is None:
            raise DataFormatError("cnes_grgs_tws", f"Cannot parse date from filename: {f.name}")
        lats, lons, grid = parse_one_grid(f.read_text())
        if ref_lats is None or ref_lons is None:
            ref_lats, ref_lons = lats, lons
        elif not (np.array_equal(lats, ref_lats) and np.array_equal(lons, ref_lons)):
            raise DataFormatError(
                "cnes_grgs_tws", f"Grid axes of {f.name} differ from the first grid"
            )
        times.append(ts)
        layers.append(grid)

    order = np.argsort(np.array([t.timestamp() for t in times]))
    times = [times[i] for i in order]
    values = np.stack([layers[i] for i in order], axis=0)  # (time, lat, lon)
    return ref_lats, ref_lons, times, values


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
