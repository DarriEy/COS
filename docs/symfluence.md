# SYMFLUENCE integration

COS registers a `CommunityObservationBackend` (contract 0.3.0) under
`R.observation_backends`, declaring one capability per implemented connector with
its non-streamflow `kinds`. `acquire()` runs the canonical fetch+reduce and writes
the OBS_CSV_V1 protocol delivery + sidecar manifest, window-trimmed to half-open
UTC `[start, end)`.

With `DATA_ACCESS: community`, current SYMFLUENCE selects COS through the same
`ObservationBackend` contract used by CSFS. It acquires and reduces the data,
writes the evaluator's canonical observation file, and removes the corresponding
native acquisition task. Selection or acquisition failure falls back to native.

| configuration key | COS provider | kind |
|---|---|---|
| `GRACE` | `grace` | TWS |
| `SNOTEL` | `snotel` | SWE |
| `MODIS_SNOW` | `modis_sca` | snow cover |
| `MODIS_ET` | `mod16_et` | ET |
| `FLUXNET_ET` | `fluxnet_et` | ET |
| `USGS_GW` | `usgs_gw` | groundwater |

The SYMFLUENCE suite verifies selection, fallback, output adaptation, all six
routing entries, and a real SNOTEL acquisition. Expansion to the remaining COS
providers is now an adapter-table/evaluator compatibility task rather than an
architectural wiring gap.
