# Quick Start

```bash
pip install -e ".[dev]"
cos providers          # registered connectors, kind, structural class, auth
cos kinds              # canonical kinds + SI units
cos fetch snotel -s snotel:679 --start 2022-01-01 --end 2022-03-01
```

```python
import cos
from cos import ReductionSpec
from datetime import datetime, UTC

# Point network (SNOTEL SWE, anonymous):
spec = ReductionSpec(domain_name="paradise", station_ids=("snotel:679",), options={"state": "WA"})
series = cos.fetch_series_sync("snotel", spec, datetime(2022,1,1,tzinfo=UTC), datetime(2022,3,1,tzinfo=UTC))

# Gridded product (GRACE TWS, basin reduction) from a local NetCDF:
spec = ReductionSpec(domain_name="bow", bbox=(50,-116,52,-114), centroid=(51,-115), area_km2=8000)
series = cos.fetch_series_sync("grace", spec, datetime(2003,1,1,tzinfo=UTC), datetime(2021,1,1,tzinfo=UTC),
                               config={"nc_path": "grace_mascons.nc"})
```
