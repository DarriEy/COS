"""CLI smoke tests (no network)."""

from click.testing import CliRunner

from cos.cli.main import cli


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
