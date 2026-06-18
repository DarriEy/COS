# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Canonical data models for COS — the heterogeneous-observation contract.

This is the make-or-break design (see ``papers/cos_design.md`` §2). Both
structural worlds COS spans collapse into ONE tidy time-series model:

* **gridded products** (GRACE TWS, SMAP SM, MODIS snow/ET/LAI/LST, ...) are
  spatially reduced to the evaluation geometry (``basin_mean`` / ``nearest_cell``
  / ``point_sample``) → an :class:`ObservationSeries` whose ``site.kind`` is
  ``"reduced_region"``;
* **point networks & flux towers** (SNOTEL, ISMN, FLUXNET, GGMN, Hub'Eau) select
  stations for the domain → one :class:`ObservationSeries` per station whose
  ``site.kind`` is ``"station"``.

The :class:`ObservationKind` carries the canonical SI unit per kind and maps onto
SYMFLUENCE's per-``obs_type`` ``STANDARD_COLUMNS`` layout; every unit conversion
happens at the connector boundary so the canonical series is always SI + UTC.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field


class QualityFlag(StrEnum):
    GOOD = "good"
    SUSPECT = "suspect"
    MISSING = "missing"
    ESTIMATED = "estimated"
    RAW = "raw"


class ObservationKind(StrEnum):
    """The non-streamflow observation kinds COS serves.

    Streamflow is deliberately absent — it belongs to CSFS (see design §1).
    """

    TWS = "tws"
    SWE = "swe"
    SNOW_COVER = "snow_cover"
    ET = "et"
    SOIL_MOISTURE = "soil_moisture"
    GROUNDWATER = "groundwater"
    LAI = "lai"
    LST = "lst"
    PRECIPITATION = "precipitation"
    SURFACE_WATER = "surface_water"
    WATER_LEVEL = "water_level"
    # Multivariate-breadth kinds (orthogonal constraints on the water-energy-
    # carbon cycle for multi-objective model evaluation).
    VEGETATION_INDEX = "vegetation_index"  # NDVI/EVI, dimensionless
    ALBEDO = "albedo"                      # broadband surface albedo, dimensionless
    SNOW_DEPTH = "snow_depth"              # physical snow depth (distinct from SWE)
    GPP = "gpp"                            # gross primary productivity


#: Frozen canonical SI unit per kind (design §2 unit table). The connector MUST
#: deliver values in this unit; SYMFLUENCE never re-interprets them.
KIND_UNITS: dict[ObservationKind, str] = {
    ObservationKind.TWS: "mm",
    ObservationKind.SWE: "mm",
    ObservationKind.SNOW_COVER: "fraction",
    ObservationKind.ET: "mm/day",
    ObservationKind.SOIL_MOISTURE: "m3/m3",
    ObservationKind.GROUNDWATER: "m",
    ObservationKind.LAI: "1",
    ObservationKind.LST: "K",
    ObservationKind.PRECIPITATION: "mm",
    ObservationKind.SURFACE_WATER: "fraction",
    ObservationKind.WATER_LEVEL: "m",
    ObservationKind.VEGETATION_INDEX: "1",       # NDVI/EVI ratio (~ -1..1)
    ObservationKind.ALBEDO: "1",                 # albedo fraction (0..1)
    ObservationKind.SNOW_DEPTH: "m",             # physical snow depth, metres
    ObservationKind.GPP: "gC/m2/day",            # gross primary productivity
}

#: COS kind -> SYMFLUENCE ``obs_type`` (they coincide today, but the indirection
#: keeps the canonical vocabulary independent of the framework's column table).
KIND_TO_SYMFLUENCE_OBS_TYPE: dict[ObservationKind, str] = {k: k.value for k in ObservationKind}


class SpatialReduction(StrEnum):
    """How a value relates to the evaluation geometry."""

    BASIN_MEAN = "basin_mean"        # area-weighted mean over the catchment polygon
    NEAREST_CELL = "nearest_cell"    # value at the cell nearest the basin centroid
    POINT_SAMPLE = "point_sample"    # gridded value at an explicit lat/lon
    STATION = "station"              # native point observation — no reduction


class SiteRef(BaseModel):
    """What a series is *about*: a real station, or a reduced region.

    One type serves both worlds so :class:`ObservationSeries` need not branch.
    """

    kind: Literal["station", "reduced_region"]
    site_id: str = Field(
        description="canonical site id, e.g. 'snotel:679' (station) or "
        "'grace:domain:<name>' / 'grace:cell:<lat>_<lon>' (reduced region)"
    )
    latitude: float | None = None
    longitude: float | None = None
    name: str | None = None
    #: provider-native metadata: state, network, datum, depth_m (for soil
    #: moisture / groundwater), et model name (OpenET ensemble), ...
    extra: dict[str, str] = Field(default_factory=dict)


class ObservationPoint(BaseModel):
    """One timestep of one series, in the kind's canonical SI unit, UTC."""

    timestamp: datetime
    value: float | None = None
    quality: QualityFlag = QualityFlag.RAW
    #: optional same-unit uncertainty — carries e.g. TWS ``uncertainty_mm``.
    uncertainty: float | None = None


class ObservationSeries(BaseModel):
    """The canonical heterogeneous-observation time series.

    Both gridded reductions and point stations are this single type, tagged by
    :attr:`kind` (→ SI unit + symfluence obs_type) and :attr:`site` (→ station
    vs reduced region). ``unit`` is frozen by :data:`KIND_UNITS` and validated.
    """

    provider: str
    kind: ObservationKind
    site: SiteRef
    reduction: SpatialReduction
    unit: str
    points: list[ObservationPoint] = Field(default_factory=list)
    source_info: dict[str, str] = Field(default_factory=dict)
    fetched_at: datetime

    def model_post_init(self, _ctx: object) -> None:  # noqa: D401
        expected = KIND_UNITS[self.kind]
        if self.unit != expected:
            raise ValueError(
                f"ObservationSeries for kind {self.kind!r} must use canonical unit "
                f"{expected!r}, got {self.unit!r}. Convert at the connector boundary."
            )


class ReductionSpec(BaseModel):
    """The geometry + reduction policy a connector needs to produce a series.

    Gridded connectors reduce a raster to the geometry; point connectors use it
    only for station selection (bbox / explicit ids). The catchment geometry is
    a GeoJSON-like polygon mapping in EPSG:4326 (the manager's discretized HRU /
    catchment outline), with the bbox + centroid + area precomputed so a
    connector need not pull in geopandas just to read them.
    """

    domain_name: str = "domain"
    #: GeoJSON-like polygon geometry (EPSG:4326); None when only a bbox is known.
    geometry: dict | None = None
    #: (lat_min, lon_min, lat_max, lon_max), EPSG:4326.
    bbox: tuple[float, float, float, float] | None = None
    #: (lat, lon) representative point — basin centroid or station seed.
    centroid: tuple[float, float] | None = None
    #: catchment area in km² (drives the basin-mean vs point-sample default).
    area_km2: float | None = None
    #: requested reduction; None = connector's size-based default policy.
    reduction: SpatialReduction | None = None
    #: explicit station ids (point networks); empty = select by bbox/centroid.
    station_ids: tuple[str, ...] = ()
    #: connector-specific options (e.g. GRACE anomaly baseline, OpenET model).
    options: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Arrow schema for the optional reduced-series store / parquet cache.
# Long/tidy: one row per (series, timestep). ``unit`` and ``kind`` are carried
# so a parquet file is self-describing.
# ---------------------------------------------------------------------------

OBSERVATION_SCHEMA = pa.schema([
    pa.field("provider", pa.string(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),
    pa.field("site_id", pa.string(), nullable=False),
    pa.field("site_kind", pa.string(), nullable=False),
    pa.field("reduction", pa.string(), nullable=False),
    pa.field("unit", pa.string(), nullable=False),
    pa.field("timestamp", pa.timestamp("s", tz="UTC"), nullable=False),
    pa.field("value", pa.float64(), nullable=True),
    pa.field("uncertainty", pa.float64(), nullable=True),
    pa.field("quality", pa.string(), nullable=False),
])
