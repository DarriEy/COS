import json

import pyarrow as pa
import pyarrow.parquet as pq

from cos.core.spatial_catalog import load_catalog, query_catalog


def test_csv_catalog_bbox_centroid_limit_and_dedup(tmp_path):
    path = tmp_path / "sites.csv"
    path.write_text(
        "SITE_ID,SITE_NAME,LOCATION_LAT,LOCATION_LONG\n"
        "A,Alpha,50,-115\nB,Beta,51,-114\nA,Duplicate,50,-115\nC,Far,10,20\n"
    )
    rows = load_catalog(path, id_fields=("SITE_ID",))
    found = query_catalog(rows, bbox=(49, -116, 52, -113), centroid=(51, -114), limit=1)
    assert [row.feature_id for row in found] == ["B"]


def test_geojson_point_catalog(tmp_path):
    path = tmp_path / "sites.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "properties": {"lake_id": "42", "name": "Lake"},
            "geometry": {"type": "Point", "coordinates": [-115, 50]},
        }],
    }))
    rows = load_catalog(path, id_fields=("lake_id",))
    assert rows[0].feature_id == "42"
    assert rows[0].latitude == 50


def test_parquet_catalog_and_dateline_bbox(tmp_path):
    path = tmp_path / "sites.parquet"
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"id": ["west", "east"], "lat": [0, 0], "lon": [179, -179]}), sink)
    path.write_bytes(sink.getvalue().to_pybytes())
    rows = load_catalog(path, id_fields=("id",))
    found = query_catalog(rows, bbox=(-1, 170, 1, -170))
    assert {row.feature_id for row in found} == {"west", "east"}
