# CLI

```
cos providers                 # registered connectors: kind, class, auth
cos kinds                     # canonical observation kinds + SI units
cos health                    # roster grouped by kind
cos validation               # parity/live/spec validation matrix
cos validation --json-output # machine-readable validation + license report
cos sites <provider> ...      # discover stations/features/reduced regions
cos doctor                    # registration, credential, cache, and gate checks
cos doctor --json-output      # secret-free machine-readable readiness
cos fetch <provider> ...      # fetch + print a canonical series
```

`cos fetch` options: `-s/--station-id` (point networks), `--nc-path` (gridded
local NetCDF), `--bbox lat_min,lon_min,lat_max,lon_max`, `--centroid lat,lon`,
`--start`, `--end` (half-open UTC `[start, end)`), `--domain`. Add
`--json-output` to retain the complete canonical series including quality,
uncertainty, source provenance, and fetch timestamp.

`cos doctor` never prints secrets. Missing credentials are informational because
anonymous connectors remain usable; connector errors identify the specific auth
provider and expected environment/netrc setup.

`cos sites` accepts `--limit` (default 25) to bound discovery results. SNOTEL
fetches made without explicit station IDs preserve discovered station metadata
and fetch at most five stations by default; set `max_fetch_sites` in the provider
configuration when a different fan-out is intentional.

AmeriFlux tower discovery is anonymous:

```bash
cos sites fluxnet_et --bbox 40,-89,42,-87 --limit 10
```

SWOT and ISMN use provider catalogs rather than undocumented portal scraping.
Set `catalog_path` (or `pld_catalog_path` / `sword_catalog_path`) for SWOT and
`archive_path` / `catalog_path` for ISMN in the provider configuration. Catalogs
may be CSV, JSON/GeoJSON, or Parquet; install `community-observation-service[spatial]`
to read GeoPackage and Shapefile catalogs.
