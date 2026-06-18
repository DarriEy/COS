# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NOAA IMS snow-cover-area connector (gridded, basin-reduced, anonymous).

Exercises the **gridded spatial-reduction path** for a *binary-classification*
raster. NOAA's IMS (Interactive Multisensor Snow and Ice Mapping System, product
G02156) is a cloud-free daily Northern-Hemisphere snow/ice map served as ASCII
grids behind NSIDC HTTPS (anonymous). Cells carry value codes, not a fraction:

    0 = outside NH   1 = open water   2 = land (no snow)
    3 = sea ice      4 = snow-covered land

The native SYMFLUENCE handler (``observation/handlers/ims_snow.py`` +
``acquisition/handlers/ims_snow.py``) reduces a basin to a daily snow-covered-area
*fraction* = ``count(code == 4) / count(code in {2, 4})`` — snow land pixels over
all land pixels inside the basin bbox, NaN when the basin has no land pixels.
That pixel-ratio is the canonical ``snow_cover`` unit (``fraction``) directly, so
there is no scalar unit conversion — only the code → fraction reduction.

This connector mirrors that semantics exactly and supports two NetCDF layouts,
matching the native two-path handler:

* a **value-code grid** ``(time, y, x)`` (variable ``IMS_Surface_Values`` /
  ``ims`` / anything snow-coded) → code-aware basin reduction (native parity);
* a **pre-reduced** ``snow_fraction`` time series (the acquirer's NetCDF output),
  passed straight through (clipped to ``[0, 1]``).

The fetch/download path against NSIDC is wired per-connector only where trivial;
as with ``grace.py`` the architecture-critical reduce + canonicalize path is
hermetically tested on a synthetic in-memory NetCDF (no network, no auth).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

# IMS surface value codes (G02156 user guide).
CODE_OUTSIDE = 0
CODE_WATER = 1
CODE_LAND = 2
CODE_SEA_ICE = 3
CODE_SNOW = 4

#: candidate variable names for a raw value-code grid (first match wins).
_CODE_VAR_HINTS = ("ims", "snow", "sca", "surface", "values")
#: the pre-reduced fraction variable emitted by the SYMFLUENCE IMS acquirer.
_FRACTION_VAR = "snow_fraction"


@register("ims_sca")
class IMSSnowCoverConnector(BaseObservationConnector):
    slug = "ims_sca"
    display_name = "NOAA IMS Snow Cover (G02156)"
    kind = ObservationKind.SNOW_COVER
    structural_class = "gridded"
    base_url = "https://noaadata.apps.nsidc.org/NOAA/G02156"
    auth = frozenset()  # NSIDC HTTPS is anonymous

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """One reduced region: the basin SCA (IMS is always basin-mean)."""
        return [self._site(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "IMS live fetch needs a NetCDF path (config 'nc_path'/'path') or an "
                "NSIDC ASCII download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded/acquired IMS NetCDF to reduce it.",
            )
        return [self.reduce_file(Path(nc_path), spec, start, end)]

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open an IMS NetCDF, reduce to the basin SCA fraction, canonicalize.

        Supports both the value-code grid (native code-aware reduction) and the
        pre-reduced ``snow_fraction`` time series. Output is the canonical
        ``snow_cover`` unit (``fraction``), trimmed to half-open UTC
        ``[start, end)``.
        """
        import numpy as np
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            if _FRACTION_VAR in ds.data_vars:
                da = ds[_FRACTION_VAR]
                times = np.asarray(da["time"].values)
                fractions = np.asarray(da.values, dtype="float64")
                points = self.fraction_series(times, fractions)
            else:
                grid_var = self._find_code_var([str(v) for v in ds.data_vars])
                if grid_var is None:
                    raise ConnectorError(
                        self.slug,
                        f"NetCDF has neither '{_FRACTION_VAR}' nor an identifiable IMS "
                        f"value-code grid variable (vars: {list(ds.data_vars)})",
                    )
                da = ds[grid_var]
                times = np.asarray(da["time"].values)
                lats = np.asarray(ds["lat"].values, dtype="float64") if "lat" in ds else None
                lons = np.asarray(ds["lon"].values, dtype="float64") if "lon" in ds else None
                codes = np.asarray(da.values)  # (time, y, x)
                points = self.reduce_codes(times, codes, lats, lons, spec.bbox)

        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site(spec),
            reduction=SpatialReduction.BASIN_MEAN,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "NOAA IMS (G02156)",
                "institution": "NOAA/NSIDC",
                "url": "https://nsidc.org/data/g02156",
                "sca_definition": "snow_land_pixels / all_land_pixels in basin bbox",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- pure reducers (hermetically tested, no network) ---------------------

    @staticmethod
    def reduce_codes(
        times,
        codes,
        lats,
        lons,
        bbox: tuple[float, float, float, float] | None,
    ) -> list[ObservationPoint]:
        """Code grid ``(time, y, x)`` → daily basin SCA fraction points.

        Mirrors the native handler exactly: per timestep, SCA =
        ``count(code == SNOW) / count(code in {LAND, SNOW})`` over cells whose
        centers fall in *bbox* (whole grid if *bbox*/coords are absent). When the
        basin has no land pixels the value is None / MISSING.
        """
        import numpy as np

        from cos.core.reduce import _as_datetime, _normalize_lons

        codes = np.asarray(codes)
        # Select the bbox subgrid by cell-center membership (native uses a pixel
        # bbox; here we honor lat/lon coords when present, else use the whole grid).
        if bbox is not None and lats is not None and lons is not None:
            lat_min, lon_min, lat_max, lon_max = bbox
            lon_min, lon_max = _normalize_lons(lons, lon_min, lon_max)
            lat_sel = np.where((lats >= lat_min) & (lats <= lat_max))[0]
            lon_sel = np.where((lons >= lon_min) & (lons <= lon_max))[0]
            if lat_sel.size and lon_sel.size:
                codes = codes[:, lat_sel[:, None], lon_sel[None, :]]

        points: list[ObservationPoint] = []
        for t in range(codes.shape[0]):
            snapshot = codes[t]
            land = int(np.sum((snapshot == CODE_LAND) | (snapshot == CODE_SNOW)))
            snow = int(np.sum(snapshot == CODE_SNOW))
            ts = _as_datetime(times[t])
            if land > 0:
                sca = min(max(snow / land, 0.0), 1.0)
                points.append(ObservationPoint(timestamp=ts, value=sca, quality=QualityFlag.GOOD))
            else:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
        return points

    @staticmethod
    def fraction_series(times, fractions) -> list[ObservationPoint]:
        """Pre-reduced ``snow_fraction`` (already 0-1) → canonical points.

        Clips to ``[0, 1]`` (native ``process`` does the same); NaN → MISSING.
        """
        import numpy as np

        from cos.core.reduce import _as_datetime

        fractions = np.asarray(fractions, dtype="float64")
        points: list[ObservationPoint] = []
        for t in range(fractions.shape[0]):
            v = fractions[t]
            ts = _as_datetime(times[t])
            if np.isfinite(v):
                clipped = min(max(float(v), 0.0), 1.0)
                points.append(ObservationPoint(timestamp=ts, value=clipped, quality=QualityFlag.GOOD))
            else:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
        return points

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _find_code_var(data_vars: list[str]) -> str | None:
        for var in data_vars:
            low = var.lower()
            if any(h in low for h in _CODE_VAR_HINTS):
                return var
        if len(data_vars) == 1:
            return data_vars[0]
        return None

    def _site(self, spec: ReductionSpec) -> SiteRef:
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region",
            site_id=f"ims_sca:domain:{spec.domain_name}",
            latitude=lat,
            longitude=lon,
            name=f"IMS snow cover over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
