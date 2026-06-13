# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Gridded spatial-reduction kernels — the gridded → time-series path.

These are the kernels that turn a gridded product (a lat/lon DataArray with a
time axis) into a single per-region time series, the operation that distinguishes
COS's gridded connectors from CSFS's point gauges. Two policies, both required
(GRACE alone uses both depending on basin size — see ``papers/cos_design.md``
§2):

* :func:`basin_mean` — area-weighted mean over the cells whose centers fall in
  the basin bbox (cosine-latitude weighting; a documented approximation of full
  polygon-weighted zonal stats — basin-mean parity is tolerance-based, not
  bitwise, exactly as CAS attributes are);
* :func:`nearest_cell` / :func:`point_sample` — the single cell nearest a point
  (centroid for ``nearest_cell``, explicit lat/lon for ``point_sample``).

Longitude convention is normalized here: products served on 0–360 (e.g. GRACE)
are matched against negative request longitudes by shifting, mirroring the native
``grace.py`` handler.

Kept dependency-light: operates on numpy arrays so a connector passes
``lats, lons, time, values`` extracted from xarray/NetCDF without this module
importing xarray.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from cos.core.exceptions import ReductionError
from cos.core.models import (
    ObservationKind,
    ObservationPoint,
    QualityFlag,
    SpatialReduction,
)


def _normalize_lons(lons: np.ndarray, lon_min: float, lon_max: float) -> tuple[float, float]:
    """Shift request longitudes into the grid's convention if it is 0–360."""
    if lons.size and float(np.nanmax(lons)) > 180.0:
        if lon_min < 0:
            lon_min += 360.0
        if lon_max < 0:
            lon_max += 360.0
    return lon_min, lon_max


def _normalize_point_lon(lons: np.ndarray, lon: float) -> float:
    if lons.size and float(np.nanmax(lons)) > 180.0 and lon < 0:
        return lon + 360.0
    return lon


def basin_mean(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Area-weighted (cos-lat) mean over cells inside *bbox*, per timestep.

    *values* is shaped ``(time, lat, lon)``. ``bbox`` is
    ``(lat_min, lon_min, lat_max, lon_max)``. Returns a length-``time`` vector;
    timesteps with no in-box finite cells become NaN.

    Raises:
        ReductionError: if no grid cell centers fall inside the bbox at all.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    lon_min, lon_max = _normalize_lons(lons, lon_min, lon_max)

    lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    if lat_sel.size == 0 or lon_sel.size == 0:
        raise ReductionError(
            f"No grid cells inside bbox {bbox} (grid lat {lats.min():.2f}..{lats.max():.2f}, "
            f"lon {lons.min():.2f}..{lons.max():.2f})"
        )

    sub = values[:, lat_sel[:, None], lon_sel[None, :]]          # (time, nlat, nlon)
    weights = np.cos(np.deg2rad(lats[lat_sel]))                  # (nlat,)
    w2d = np.broadcast_to(weights[:, None], sub.shape[1:])       # (nlat, nlon)

    out = np.full(sub.shape[0], np.nan, dtype="float64")
    for t in range(sub.shape[0]):
        layer = sub[t]
        finite = np.isfinite(layer)
        if not finite.any():
            continue
        wsum = float(np.sum(w2d[finite]))
        if wsum > 0:
            out[t] = float(np.sum(layer[finite] * w2d[finite]) / wsum)
    return out


def nearest_cell(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    point: tuple[float, float],
) -> np.ndarray:
    """Series at the single grid cell nearest *point* = ``(lat, lon)``."""
    lat, lon = point
    lon = _normalize_point_lon(lons, lon)
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    return values[:, i, j].astype("float64")


# point_sample is the same kernel as nearest_cell; the distinction is semantic
# (centroid vs explicit user lat/lon) and recorded on the series' reduction field.
point_sample = nearest_cell


def reduce_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    *,
    reduction: SpatialReduction,
    bbox: tuple[float, float, float, float] | None,
    point: tuple[float, float] | None,
    kind: ObservationKind,
    unit: str,
) -> list[ObservationPoint]:
    """Reduce a gridded product to canonical :class:`ObservationPoint`s.

    Dispatches on *reduction*. ``times`` is a sequence convertible to
    ``datetime``; values are assumed already in *unit* (converted upstream).
    Quality is GOOD where finite, MISSING where NaN.
    """
    if reduction == SpatialReduction.BASIN_MEAN:
        if bbox is None:
            raise ReductionError("basin_mean reduction requires a bbox")
        series = basin_mean(lats, lons, values, bbox)
    elif reduction in (SpatialReduction.NEAREST_CELL, SpatialReduction.POINT_SAMPLE):
        if point is None:
            raise ReductionError(f"{reduction} reduction requires a point (lat, lon)")
        series = nearest_cell(lats, lons, values, point)
    else:
        raise ReductionError(f"{reduction} is not a gridded reduction")

    points: list[ObservationPoint] = []
    for t, v in zip(times, series):
        ts = t if isinstance(t, datetime) else _as_datetime(t)
        finite = v is not None and np.isfinite(v)
        points.append(
            ObservationPoint(
                timestamp=ts,
                value=float(v) if finite else None,
                quality=QualityFlag.GOOD if finite else QualityFlag.MISSING,
            )
        )
    return points


def _as_datetime(value: object) -> datetime:
    """Coerce a numpy datetime64 / str / pandas Timestamp to a UTC datetime."""
    from datetime import UTC

    dt: datetime
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()  # type: ignore[union-attr]
    elif isinstance(value, np.datetime64):
        dt = value.astype("datetime64[s]").astype(datetime)
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
