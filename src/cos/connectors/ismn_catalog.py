"""Index station locations from user-downloaded ISMN archives."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cos.core.exceptions import DataFormatError
from cos.core.spatial_catalog import CatalogRecord, load_catalog, normalize_records


@dataclass(frozen=True)
class ArchiveText:
    name: str
    text: str


def load_ismn_catalog(path: str | Path) -> list[CatalogRecord]:
    """Load a normalized catalog or index ``.stm`` headers in an ISMN archive."""
    source = Path(path)
    catalog_suffixes = {".csv", ".json", ".geojson", ".parquet", ".geoparquet", ".gpkg", ".shp"}
    if source.suffix.lower() in catalog_suffixes:
        return load_catalog(
            source,
            id_fields=("station_id", "site_id", "station", "id"),
            latitude_fields=("latitude", "lat"),
            longitude_fields=("longitude", "lon"),
            name_fields=("name", "station_name"),
        )
    rows = [_stm_header_row(item) for item in _archive_texts(source)]
    return normalize_records(
        [row for row in rows if row is not None],
        id_fields=("station_id",),
        latitude_fields=("latitude",),
        longitude_fields=("longitude",),
        name_fields=("station_name",),
    )


def _archive_texts(path: Path) -> Iterable[ArchiveText]:
    if path.is_dir():
        for member in sorted(path.rglob("*.stm")):
            yield ArchiveText(str(member.relative_to(path)), member.read_text(errors="replace"))
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(member for member in archive.namelist() if member.lower().endswith(".stm")):
                yield ArchiveText(name, archive.read(name).decode("utf-8", errors="replace"))
        return
    raise DataFormatError("ismn_sm", f"Expected an ISMN archive/directory or station catalog: {path}")


def _stm_header_row(item: ArchiveText) -> dict[str, str] | None:
    """Parse the standard ISMN header+values metadata line."""
    line = next((line.strip() for line in item.text.splitlines() if line.strip()), "")
    fields = line.split()
    if len(fields) < 8:
        return None
    try:
        float(fields[2])
        float(fields[3])
    except ValueError:
        return None
    network, station = fields[0], fields[1]
    return {
        "station_id": f"{network}:{station}",
        "station_name": station,
        "network": network,
        "latitude": fields[2],
        "longitude": fields[3],
        "elevation_m": fields[4],
        "depth_from_m": fields[5],
        "depth_to_m": fields[6],
        "variable": fields[7],
        "archive_member": item.name,
    }
