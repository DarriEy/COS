# Connector roster

The honest, status-labeled roster lives in `inventory/providers.yaml`. At
scaffold time: **3 implemented**, 19 planned, 9 research, 1 manual — coverage
≈ 9% of the enumerated non-streamflow surface.

| connector | kind | class | auth | status |
|---|---|---|---|---|
| grace | tws | gridded | earthdata | implemented |
| snotel | swe | point_network | none | implemented (live-smoked) |
| openet | et | flux_tower | openet | implemented |

All other connectors (gldas_tws, smap, modis_*, gleam, fluxnet, chirps, gpm,
usgs_gw, ggmn, jrc_water, hubeau_waterlevel, ...) are `planned` / `research` /
`manual` — unbuilt. See `inventory/providers.yaml`.

No connector has a recorded native parity grade yet.
