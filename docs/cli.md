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
