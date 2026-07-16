"""Small, provider-neutral spatial catalog reader and bbox query helpers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from cos.core.exceptions import DataFormatError


@dataclass(frozen=True)
class CatalogRecord:
    """One normalized point/feature in a provider catalog."""

    feature_id: str
    latitude: float
    longitude: float
    name: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)


def load_catalog(
    path: str | Path,
    *,
    id_fields: Sequence[str],
    latitude_fields: Sequence[str] = ("latitude", "lat", "LOCATION_LAT"),
    longitude_fields: Sequence[str] = ("longitude", "lon", "long", "LOCATION_LONG"),
    name_fields: Sequence[str] = ("name", "site_name", "SITE_NAME"),
) -> list[CatalogRecord]:
    """Load a CSV, JSON/GeoJSON, Parquet, or optional GDAL vector catalog."""
    source = Path(path)
    if not source.exists():
        raise DataFormatError("spatial_catalog", f"Catalog does not exist: {source}")
    suffix = source.suffix.lower()
    rows: list[Mapping[str, Any]]
    if suffix in {".csv", ".txt"}:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".json", ".geojson"}:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = _json_rows(payload)
    elif suffix in {".parquet", ".geoparquet"}:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = pq.read_table(pa.BufferReader(source.read_bytes())).to_pylist()
    elif suffix in {".gpkg", ".shp"}:
        rows = _vector_rows(source)
    else:
        raise DataFormatError("spatial_catalog", f"Unsupported catalog format: {suffix}")
    return normalize_records(
        rows,
        id_fields=id_fields,
        latitude_fields=latitude_fields,
        longitude_fields=longitude_fields,
        name_fields=name_fields,
    )


def normalize_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_fields: Sequence[str],
    latitude_fields: Sequence[str],
    longitude_fields: Sequence[str],
    name_fields: Sequence[str],
) -> list[CatalogRecord]:
    """Normalize provider rows, dropping entries without a usable id/location."""
    out: list[CatalogRecord] = []
    seen: set[str] = set()
    for row in rows:
        feature_id = _first(row, id_fields)
        latitude = _number(_first(row, latitude_fields))
        longitude = _number(_first(row, longitude_fields))
        if feature_id is None or latitude is None or longitude is None:
            continue
        feature_id = str(feature_id).strip()
        if not feature_id or feature_id in seen or not (-90 <= latitude <= 90):
            continue
        longitude = ((longitude + 180) % 360) - 180
        name = _first(row, name_fields)
        seen.add(feature_id)
        out.append(
            CatalogRecord(
                feature_id=feature_id,
                latitude=latitude,
                longitude=longitude,
                name=str(name).strip() if name not in (None, "") else None,
                properties=dict(row),
            )
        )
    return out


def query_catalog(
    records: Iterable[CatalogRecord],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    centroid: tuple[float, float] | None = None,
    limit: int | None = None,
) -> list[CatalogRecord]:
    """Filter by COS bbox order and rank by centroid when one is supplied."""
    selected = list(records)
    if bbox is not None:
        lat_min, lon_min, lat_max, lon_max = bbox
        selected = [
            row for row in selected
            if lat_min <= row.latitude <= lat_max and _longitude_in_bbox(row.longitude, lon_min, lon_max)
        ]
    if centroid is not None:
        selected.sort(key=lambda row: _haversine_km(centroid, (row.latitude, row.longitude)))
    else:
        selected.sort(key=lambda row: row.feature_id)
    return selected[:limit] if limit is not None else selected


def _json_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping) and payload.get("type") == "FeatureCollection":
        rows: list[Mapping[str, Any]] = []
        for feature in payload.get("features", []):
            if not isinstance(feature, Mapping):
                continue
            row = dict(feature.get("properties") or {})
            geometry = feature.get("geometry") or {}
            if geometry.get("type") == "Point" and len(geometry.get("coordinates", [])) >= 2:
                row.setdefault("longitude", geometry["coordinates"][0])
                row.setdefault("latitude", geometry["coordinates"][1])
            rows.append(row)
        return rows
    if isinstance(payload, Mapping):
        for key in ("results", "sites", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
    raise DataFormatError("spatial_catalog", "JSON catalog is not a row list or FeatureCollection")


def _vector_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        import pyogrio  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DataFormatError(
            "spatial_catalog", "GeoPackage/Shapefile catalogs require the 'spatial' extra"
        ) from exc
    _, table = pyogrio.read_arrow(path)
    return cast(list[Mapping[str, Any]], table.to_pylist())


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _longitude_in_bbox(longitude: float, west: float, east: float) -> bool:
    west = ((west + 180) % 360) - 180
    east = ((east + 180) % 360) - 180
    return west <= longitude <= east if west <= east else longitude >= west or longitude <= east


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))
