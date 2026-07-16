"""Catalog-backed feature discovery shared by SWOT Hydrocron connectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cos.core.exceptions import ConnectorError
from cos.core.models import ReductionSpec, SiteRef
from cos.core.spatial_catalog import CatalogRecord, load_catalog, query_catalog


class SWOTConnector(Protocol):
    config: dict

    def _feature(self, spec: ReductionSpec) -> str: ...

    def _feature_ids(self, spec: ReductionSpec) -> list[str]: ...

    def _site(self, feature_id: str, spec: ReductionSpec, feature: str) -> SiteRef: ...


def discover_swot_sites(connector: SWOTConnector, spec: ReductionSpec) -> list[SiteRef]:
    """Resolve explicit IDs or query a local PLD/SWORD feature catalog."""
    feature = connector._feature(spec)
    feature_ids = connector._feature_ids(spec)
    if feature_ids:
        return [connector._site(feature_id, spec, feature) for feature_id in feature_ids]
    if spec.bbox is None and spec.centroid is None:
        return []
    catalog_path = (
        spec.options.get("catalog_path")
        or connector.config.get("catalog_path")
        or connector.config.get("pld_catalog_path" if feature == "PriorLake" else "sword_catalog_path")
    )
    if not catalog_path:
        raise ConnectorError(
            "swot_catalog",
            f"{feature} spatial discovery requires config 'catalog_path' "
            f"({'PLD' if feature == 'PriorLake' else 'SWORD'} CSV/GeoJSON/Parquet/vector catalog)",
        )
    records = _load_swot_catalog(catalog_path, feature)
    selected = query_catalog(
        records,
        bbox=spec.bbox,
        centroid=spec.centroid,
        limit=_limit(spec),
    )
    collection = str(spec.options.get("collection_name") or connector.config.get("collection_name") or "")
    return [_site_from_record(connector, spec, feature, record, collection) for record in selected]


def resolve_swot_feature_ids(connector: SWOTConnector, spec: ReductionSpec) -> list[str]:
    """Resolve IDs for fetch, including catalog discovery for spatial specs."""
    explicit = connector._feature_ids(spec)
    if explicit:
        return explicit
    return [site.site_id.split(":", 1)[-1] for site in discover_swot_sites(connector, spec)]


def catalog_reference_area(connector: SWOTConnector, spec: ReductionSpec, feature_id: str) -> float | None:
    """Return a PLD reference area for one feature when a catalog supplies it."""
    path = (
        spec.options.get("catalog_path")
        or connector.config.get("catalog_path")
        or connector.config.get("pld_catalog_path")
    )
    if not path:
        return None
    record = next((row for row in _load_swot_catalog(path, "PriorLake") if row.feature_id == feature_id), None)
    if record is None:
        return None
    for field in ("reference_area_km2", "ref_area", "area_total", "lake_area", "area_km2"):
        value = record.properties.get(field)
        if value is None:
            continue
        try:
            area = float(value)
        except (TypeError, ValueError):
            continue
        if area > 0:
            return area
    return None


def _load_swot_catalog(path: str, feature: str) -> list[CatalogRecord]:
    if feature == "PriorLake":
        ids: Sequence[str] = ("lake_id", "prior_lake_id", "feature_id", "id")
    elif feature == "Reach":
        ids = ("reach_id", "feature_id", "id")
    else:
        ids = ("node_id", "feature_id", "id")
    return load_catalog(
        path,
        id_fields=ids,
        latitude_fields=("latitude", "lat", "lake_lat", "p_lat", "reach_lat", "node_lat"),
        longitude_fields=("longitude", "lon", "lake_lon", "p_lon", "reach_lon", "node_lon"),
        name_fields=("name", "lake_name", "reach_name"),
    )


def _site_from_record(
    connector: SWOTConnector,
    spec: ReductionSpec,
    feature: str,
    record: CatalogRecord,
    collection: str,
) -> SiteRef:
    site = connector._site(record.feature_id, spec, feature)
    extra = dict(site.extra)
    extra["catalog"] = "PLD" if feature == "PriorLake" else "SWORD"
    if collection:
        extra["collection_name"] = collection
    for field in ("reference_area_km2", "ref_area", "basin_id"):
        if record.properties.get(field) not in (None, ""):
            extra[field] = str(record.properties[field])
    return site.model_copy(update={
        "latitude": record.latitude,
        "longitude": record.longitude,
        "name": record.name or site.name,
        "extra": extra,
    })


def _limit(spec: ReductionSpec) -> int:
    try:
        return max(1, int(spec.options.get("max_sites", 25)))
    except (TypeError, ValueError):
        return 25
