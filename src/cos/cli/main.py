# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""COS command-line interface (mirrors the CSFS command shape)."""

from __future__ import annotations

import asyncio
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


@cli.command()
@click.argument("provider")
@click.option("--station-id", "-s", multiple=True, help="Station id(s) for point networks")
@click.option("--nc-path", default=None, help="Local NetCDF path for gridded connectors (e.g. GRACE)")
@click.option("--bbox", default=None, help="lat_min,lon_min,lat_max,lon_max for gridded reduction")
@click.option("--centroid", default=None, help="lat,lon centroid for nearest-cell / point networks")
@click.option("--start", required=True, help="UTC start (YYYY-MM-DD)")
@click.option("--end", required=True, help="UTC end (YYYY-MM-DD), half-open [start, end)")
@click.option("--domain", default="domain", help="Domain name (labels reduced regions)")
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
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)

    async def _run() -> None:
        series_list = await cos.fetch_series(provider, spec, start_dt, end_dt, config=conn_cfg)
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
    parts = tuple(float(x) for x in raw.split(","))
    if len(parts) != n:
        raise click.BadParameter(f"expected {n} comma-separated numbers, got {len(parts)}")
    return parts
