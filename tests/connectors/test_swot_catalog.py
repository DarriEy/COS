import pytest

from cos.connectors.swot_lake_area import SWOTLakeAreaConnector
from cos.connectors.swot_lake_storage import SWOTLakeStorageConnector
from cos.connectors.swot_wse import SWOTWaterLevelConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import ReductionSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_cls", [SWOTLakeAreaConnector, SWOTLakeStorageConnector])
async def test_prior_lake_bbox_discovery_from_shared_catalog(tmp_path, connector_cls):
    catalog = tmp_path / "pld.csv"
    catalog.write_text(
        "lake_id,lake_name,lake_lat,lake_lon,reference_area_km2,basin_id\n"
        "2710046612,In Basin,50.5,-115.0,12.5,27\n"
        "999,Outside,10,10,2,99\n"
    )
    connector = connector_cls(config={"pld_catalog_path": str(catalog), "collection_name": "SWOT_L2_HR_LakeSP_D"})
    sites = await connector.list_sites(ReductionSpec(domain_name="bow", bbox=(50, -116, 51, -114)))
    assert [site.site_id for site in sites] == ["swot:2710046612"]
    assert sites[0].name == "In Basin"
    assert sites[0].extra["reference_area_km2"] == "12.5"
    assert sites[0].extra["collection_name"] == "SWOT_L2_HR_LakeSP_D"


@pytest.mark.asyncio
async def test_sword_reach_discovery(tmp_path):
    catalog = tmp_path / "sword.geojson"
    catalog.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"properties":{"reach_id":"71224100223","reach_name":"Bow"},'
        '"geometry":{"type":"Point","coordinates":[-115,51]}}]}'
    )
    connector = SWOTWaterLevelConnector(config={"sword_catalog_path": str(catalog)})
    sites = await connector.list_sites(ReductionSpec(domain_name="bow", bbox=(50, -116, 52, -114)))
    assert [site.site_id for site in sites] == ["swot:71224100223"]
    assert sites[0].extra["catalog"] == "SWORD"


@pytest.mark.asyncio
async def test_spatial_swot_discovery_requires_catalog():
    with pytest.raises(ConnectorError, match="catalog_path"):
        await SWOTLakeAreaConnector().list_sites(
            ReductionSpec(domain_name="bow", bbox=(50, -116, 52, -114))
        )
