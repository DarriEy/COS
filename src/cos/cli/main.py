# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""COS command-line interface (mirrors the CSFS command shape)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import click
import structlog

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])


@click.group()
@click.version_option(package_name="community-observation-service")
@click.option("--config", "-c", default=None, type=click.Path(), help="Path to YAML config file")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """COS — Community Observation Service (non-streamflow observations)."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.pass_context
def providers(ctx: click.Context) -> None:
    """List registered connectors, their kind, structural class, and auth."""
    from cos.core.health import roster_health

    rows = roster_health()
    click.echo(f"  {'PROVIDER':<14s}  {'KIND':<14s}  {'CLASS':<14s}  {'AUTH':<12s}")
    click.echo(f"  {'─' * 14}  {'─' * 14}  {'─' * 14}  {'─' * 12}")
    for r in rows:
        click.echo(
            f"  {r['provider']:<14s}  {(r['kind'] or '?'):<14s}  "
            f"{(r['structural_class'] or '?'):<14s}  {','.join(r['auth']):<12s}"
        )
    click.echo(f"\n  {len(rows)} connectors registered")


@cli.command()
@click.pass_context
def kinds(ctx: click.Context) -> None:
    """List the canonical observation kinds and their SI units."""
    from cos.core.models import KIND_UNITS

    click.echo(f"  {'KIND':<16s}  UNIT")
    click.echo(f"  {'─' * 16}  {'─' * 12}")
    for kind, unit in KIND_UNITS.items():
        click.echo(f"  {kind.value:<16s}  {unit}")


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Report the connector roster grouped by kind."""
    from cos.core.health import roster_health, summarize_roster

    rows = roster_health()
    summary = summarize_roster(rows)
    parts = "  ".join(f"{k}={v}" for k, v in sorted(summary.items()))
    click.echo(f"\n  COS roster ({len(rows)} connectors)  {parts}")


@cli.command("validation")
@click.option("--json-output", "as_json", is_flag=True, help="Emit the full report as JSON")
def validation(as_json: bool) -> None:
    """Report parity, live-validation, auth, and data-license status."""
    from cos.core.validation import validation_report, validation_summary

    rows = validation_report()
    if as_json:
        click.echo(json.dumps({"summary": validation_summary(rows), "connectors": rows}, indent=2))
        return
    summary = "  ".join(f"{k}={v}" for k, v in validation_summary(rows).items())
    click.echo(f"\n  COS validation ({len(rows)} connectors)  {summary}")
    click.echo(f"  {'PROVIDER':<22s}  {'KIND':<18s}  TIER")
    for row in rows:
        click.echo(f"  {row['provider']:<22s}  {row['kind']:<18s}  {row['tier']}")


@cli.command("sites")
@click.argument("provider")
@click.option("--station-id", "-s", multiple=True, help="Explicit station/feature id(s)")
@click.option("--bbox", default=None, help="lat_min,lon_min,lat_max,lon_max")
@click.option("--centroid", default=None, help="lat,lon")
@click.option("--domain", default="domain")
@click.option("--limit", type=click.IntRange(min=1), default=25, show_default=True,
              help="Maximum discovered sites to return")
@click.pass_context
def sites(ctx: click.Context, provider: str, station_id: tuple[str, ...], bbox: str | None,
          centroid: str | None, domain: str, limit: int) -> None:
    """Discover sites or reduced regions available for a domain."""
    import cos
    from cos.core.config import load_config

    configs = load_config(Path(ctx.obj["config_path"]) if ctx.obj.get("config_path") else None)
    cos.discover()
    try:
        connector_cls = cos.get_connector(provider)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    config = dict(configs.get(provider, {}))
    bbox_t = _parse_tuple(bbox, 4)
    centroid_t = _parse_tuple(centroid, 2)
    spec = cos.ReductionSpec(
        domain_name=domain,
        station_ids=tuple(station_id),
        bbox=(bbox_t[0], bbox_t[1], bbox_t[2], bbox_t[3]) if bbox_t else None,
        centroid=(centroid_t[0], centroid_t[1]) if centroid_t else None,
        options={"max_sites": limit},
    )

    async def _run() -> None:
        async with connector_cls(config=config) as connector:
            found = await connector.list_sites(spec)
        for site in found:
            coords = ""
            if site.latitude is not None and site.longitude is not None:
                coords = f"  {site.latitude:.5f},{site.longitude:.5f}"
            click.echo(f"{site.site_id}{coords}  {site.name or ''}".rstrip())
        click.echo(f"\n{len(found)} site(s)")

    asyncio.run(_run())


@cli.command("doctor")
def doctor() -> None:
    """Check connector registration, credentials, cache, and validation gates."""
    from cos.core.config import resolve_credentials
    from cos.core.fetch import cache_dir
    from cos.core.registry import discover, get_connector, list_providers
    from cos.core.validation import validation_report

    discover()
    missing: dict[str, list[str]] = {}
    for slug in list_providers():
        auth: frozenset[str] = getattr(get_connector(slug), "auth", frozenset())
        absent = sorted(auth - resolve_credentials(auth).keys())
        if absent:
            missing[slug] = absent
    ungraded = [row["provider"] for row in validation_report() if not row["grade"]]
    click.echo(f"connectors: {len(list_providers())} registered")
    click.echo(f"validation: {'ok' if not ungraded else f'missing grades for {ungraded}'}")
    click.echo(f"cache: {cache_dir()}")
    if missing:
        click.echo("optional credentials absent:")
        for slug, auth_ids in missing.items():
            click.echo(f"  {slug}: {', '.join(auth_ids)}")
    else:
        click.echo("credentials: all declared providers resolved")


@cli.command()
@click.argument("provider")
@click.option("--station-id", "-s", multiple=True, help="Station id(s) for point networks")
@click.option("--nc-path", default=None, help="Local NetCDF path for gridded connectors (e.g. GRACE)")
@click.option("--bbox", default=None, help="lat_min,lon_min,lat_max,lon_max for gridded reduction")
@click.option("--centroid", default=None, help="lat,lon centroid for nearest-cell / point networks")
@click.option("--start", required=True, help="UTC start (YYYY-MM-DD)")
@click.option("--end", required=True, help="UTC end (YYYY-MM-DD), half-open [start, end)")
@click.option("--domain", default="domain", help="Domain name (labels reduced regions)")
@click.option("--json-output", "as_json", is_flag=True, help="Emit canonical series as JSON")
@click.pass_context
def fetch(
    ctx: click.Context,
    provider: str,
    station_id: tuple[str, ...],
    nc_path: str | None,
    bbox: str | None,
    centroid: str | None,
    start: str,
    end: str,
    domain: str,
    as_json: bool,
) -> None:
    """Fetch and print a canonical observation series from one connector."""
    import cos
    from cos.core.config import load_config

    configs = load_config(Path(ctx.obj["config_path"]) if ctx.obj.get("config_path") else None)
    conn_cfg = dict(configs.get(provider, {}))
    if nc_path:
        conn_cfg["nc_path"] = nc_path

    bbox_t = _parse_tuple(bbox, 4)
    centroid_t = _parse_tuple(centroid, 2)
    spec = cos.ReductionSpec(
        domain_name=domain,
        station_ids=tuple(station_id),
        bbox=(bbox_t[0], bbox_t[1], bbox_t[2], bbox_t[3]) if bbox_t else None,
        centroid=(centroid_t[0], centroid_t[1]) if centroid_t else None,
    )
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    if end_dt <= start_dt:
        raise click.BadParameter("end must be after start", param_hint="--end")

    async def _run() -> None:
        series_list = await cos.fetch_series(provider, spec, start_dt, end_dt, config=conn_cfg)
        if as_json:
            click.echo(json.dumps([s.model_dump(mode="json") for s in series_list], indent=2))
            return
        for s in series_list:
            click.echo(
                f"\n  {s.site.site_id}  [{s.kind.value} / {s.unit} / "
                f"{s.reduction.value}]  {len(s.points)} points"
            )
            for p in s.points[:5]:
                click.echo(f"    {p.timestamp.isoformat()}  {p.value}  ({p.quality.value})")
            if len(s.points) > 5:
                click.echo(f"    ... and {len(s.points) - 5} more")

    asyncio.run(_run())


def _parse_tuple(raw: str | None, n: int) -> tuple[float, ...] | None:
    if not raw:
        return None
    try:
        parts = tuple(float(x) for x in raw.split(","))
    except ValueError as exc:
        raise click.BadParameter("coordinates must be numbers") from exc
    if len(parts) != n:
        raise click.BadParameter(f"expected {n} comma-separated numbers, got {len(parts)}")
    return parts


def _parse_datetime(raw: str) -> datetime:
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise click.BadParameter(f"invalid ISO datetime: {raw!r}") from exc
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
