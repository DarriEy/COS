# SYMFLUENCE integration

COS registers a `CommunityObservationBackend` (contract 0.3.0) under
`R.observation_backends`, declaring one capability per implemented connector with
its non-streamflow `kinds`. `acquire()` runs the canonical fetch+reduce and writes
the OBS_CSV_V1 protocol delivery + sidecar manifest, window-trimmed to half-open
UTC `[start, end)`.

SYMFLUENCE can select COS through the same `ObservationBackend` contract used by
CSFS when native acquisition is unavailable or fails. COS is the fallback, not
the primary acquisition path. When selected, it acquires and reduces the data
and writes the evaluator's canonical observation file. Provider selection order
remains a SYMFLUENCE responsibility; this repository only registers the backend.

| configuration key | COS provider | kind |
|---|---|---|
| `GRACE` | `grace` | TWS |
| `SNOTEL` | `snotel` | SWE |
| `MODIS_SNOW` | `modis_sca` | snow cover |
| `MODIS_ET` | `mod16_et` | ET |
| `FLUXNET_ET` | `fluxnet_et` | ET |
| `USGS_GW` | `usgs_gw` | groundwater |
| `SMAP` | `smap_sm` | soil moisture (staged NetCDF) |
| `CHIRPS` | `chirps_precip` | precipitation (staged NetCDF) |

The SYMFLUENCE suite verifies selection, fallback, output adaptation, all eight
routing entries, and a real SNOTEL acquisition. Expansion to the remaining COS
providers is now an adapter-table/evaluator compatibility task rather than an
architectural wiring gap.
