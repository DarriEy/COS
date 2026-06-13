"""CLI smoke tests (no network)."""

from click.testing import CliRunner

from cos.cli.main import cli


def test_providers_lists_three_implemented():
    result = CliRunner().invoke(cli, ["providers"])
    assert result.exit_code == 0
    for slug in ("grace", "snotel", "openet"):
        assert slug in result.output
    assert "3 connectors registered" in result.output


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
