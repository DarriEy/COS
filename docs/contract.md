# The canonical contract

COS spans two structurally different worlds — gridded products that must be
spatially *reduced* to the basin, and point networks / flux towers that must be
*selected* by domain — and collapses both into one `ObservationSeries`:

- tagged by `ObservationKind` (which carries the **frozen SI unit** per kind);
- tagged by `SiteRef` (`station` for point networks, `reduced_region` for gridded
  reductions);
- every value in the kind's canonical SI unit (`KIND_UNITS`), every timestamp UTC,
  every unit conversion done at the connector boundary.

| kind | unit | kind | unit |
|---|---|---|---|
| tws | mm | groundwater | m |
| swe | mm | lai | dimensionless |
| snow_cover | fraction | lst | K |
| et | mm/day | precipitation | mm |
| soil_moisture | m³/m³ | surface_water | fraction |
| | | water_level | m |

Gridded connectors reduce via `cos.core.reduce` (`basin_mean` area-weighted, or
`nearest_cell` / `point_sample`). The reduction is recorded on every series.

Full reasoning: `papers/cos_design.md` §2.
