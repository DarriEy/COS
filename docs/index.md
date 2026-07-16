# COS — Community Observation Service

The community service for **non-streamflow** hydrological observations — TWS,
SWE, snow cover, ET, soil moisture, groundwater, LAI, LST,
precipitation-as-observation, surface water, and water level — harmonized to one
canonical per-site time-series contract for model **evaluation and calibration**.

COS is the non-streamflow sibling of [CSFS](https://github.com/DarriEy/CSFS)
(streamflow), in the SYMFLUENCE community-services family (CSFS · CFS · CAS · COS).

!!! info "Coverage and validation"
    COS implements **50 connectors across 20 kinds**. All have recorded parity
    evidence; 44 have been exercised on real provider data. Validation tiers
    distinguish live native parity, live product-spec checks, and hermetic
    parity-by-construction. See the [Connector Roster](roster.md).

## What COS is NOT

- **Not streamflow** — that is CSFS. COS never duplicates a streamflow connector.
- **Not raw raster / forcing** — rasters are CAS, forcing is CFS. COS delivers
  reduced time series only.
