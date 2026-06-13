# CLI

```
cos providers                 # registered connectors: kind, class, auth
cos kinds                     # canonical observation kinds + SI units
cos health                    # roster grouped by kind
cos fetch <provider> ...      # fetch + print a canonical series
```

`cos fetch` options: `-s/--station-id` (point networks), `--nc-path` (gridded
local NetCDF), `--bbox lat_min,lon_min,lat_max,lon_max`, `--centroid lat,lon`,
`--start`, `--end` (half-open UTC `[start, end)`), `--domain`.
