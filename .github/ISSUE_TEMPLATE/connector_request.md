---
name: Connector request
about: Request a new observation connector (non-streamflow)
title: "[connector] <source name>"
labels: connector
---

**Source / product**: <e.g. SMAP L3, GLEAM, ISMN>

**Observation kind**: <tws | swe | snow_cover | et | soil_moisture | groundwater | lai | lst | precipitation | surface_water | water_level>

> Streamflow is out of scope — that belongs to CSFS.

**Structural class**: <gridded | point_network | flux_tower>

**Access / auth**: <anonymous | Earthdata | CDS | OpenET key | AmeriFlux key | registration-gated | manual staging>

**Endpoint / format**: <REST/OPeNDAP/NetCDF/CSV/...; link to API docs>

**Native units → canonical SI unit**: <e.g. cm w.e. → mm; inches → mm; W/m² LE → mm/day>

**Spatial reduction (if gridded)**: <basin_mean | nearest_cell | point_sample>

**Notes / parity hazards**: <unit landmines, 0–360 longitudes, scale/offset, etc.>
