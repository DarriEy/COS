"""CLI smoke tests (no network)."""

import json
from datetime import UTC, datetime

from click.testing import CliRunner

from cos.cli.main import cli
from cos.core.models import (
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    SiteRef,
    SpatialReduction,
)


def test_providers_lists_all_registered():
    from cos.core.registry import discover, list_providers

    discover()
    n = len(list_providers())
    result = CliRunner().invoke(cli, ["providers"])
    assert result.exit_code == 0
    # the proof connectors plus the built-out roster are all listed
    for slug in ("grace", "snotel", "openet", "smap_sm", "chirps_precip", "usgs_gw"):
        assert slug in result.output
    assert f"{n} connectors registered" in result.output


def test_kinds_lists_units():
    result = CliRunner().invoke(cli, ["kinds"])
    assert result.exit_code == 0
    assert "tws" in result.output
    assert "mm/day" in result.output
    # streamflow must not appear — that is CSFS's domain.
    assert "streamflow" not in result.output


def test_health_groups_by_kind():
    result = CliRunner().invoke(cli, ["health"])
    assert result.exit_code == 0
    assert "COS roster" in result.output


def test_validation_json_is_machine_readable():
    result = CliRunner().invoke(cli, ["validation", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["connectors"]) == 50
    assert sum(payload["summary"].values()) == 50
    assert "unvalidated" not in payload["summary"]


def test_sites_handles_explicit_ids_and_unknown_provider():
    runner = CliRunner()
    result = runner.invoke(cli, ["sites", "snotel", "-s", "679"])
    assert result.exit_code == 0
    assert "snotel:679" in result.output
    bad = runner.invoke(cli, ["sites", "does-not-exist", "-s", "x"])
    assert bad.exit_code != 0
    assert "No connector registered" in bad.output


def test_fetch_rejects_reverse_window_without_network():
    result = CliRunner().invoke(cli, [
        "fetch", "snotel", "-s", "679", "--start", "2022-02-01", "--end", "2022-01-01",
    ])
    assert result.exit_code != 0
    assert "end must be after start" in result.output


def test_fetch_json_preserves_provenance(monkeypatch):
    import cos

    series = ObservationSeries(
        provider="snotel", kind=ObservationKind.SWE,
        site=SiteRef(kind="station", site_id="snotel:679"),
        reduction=SpatialReduction.STATION, unit="mm",
        points=[ObservationPoint(timestamp=datetime(2022, 1, 1, tzinfo=UTC), value=10, quality=QualityFlag.GOOD)],
        source_info={"source": "NRCS", "station": "679:WA:SNTL"},
        fetched_at=datetime(2022, 2, 1, tzinfo=UTC),
    )

    async def fake_fetch(*args, **kwargs):
        return [series]

    monkeypatch.setattr(cos, "fetch_series", fake_fetch)
    result = CliRunner().invoke(cli, [
        "fetch", "snotel", "-s", "679", "--start", "2022-01-01", "--end", "2022-02-01",
        "--json-output",
    ])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["source_info"]["station"] == "679:WA:SNTL"
    assert payload[0]["points"][0]["quality"] == "good"
