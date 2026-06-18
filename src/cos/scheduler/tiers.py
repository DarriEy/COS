# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Connector tiering for COS.

COS observations are *evaluation pulls* (driven by a SYMFLUENCE experiment over
a fixed window), not a continuously-mirrored real-time gauge store like CSFS. So
"tiers" here group connectors by their natural cadence / latency rather than
scheduling a daemon. The table exists for the same reason CSFS's does: the
roster-integrity tests assert every registered connector belongs to exactly one
tier, catching a connector that ships unclassified.

Tiers:
* ``monthly`` — products updated on a monthly cadence (GRACE TWS).
* ``daily``   — daily point networks and daily gridded products (SNOTEL SWE).
* ``ondemand``— keyed / ensemble products pulled per request (OpenET ET).
"""

from __future__ import annotations

#: tier -> connector slugs. Every registered connector must appear in exactly
#: one tier (asserted by tests/test_connector_integrity.py).
PROVIDER_TIERS: dict[str, list[str]] = {
    # Monthly-cadence products: TWS solutions + 8-day/monthly composites.
    "monthly": [
        "grace", "gldas_tws", "cnes_grgs_tws",
        "mod16_et", "gleam_et", "ssebop_et",
        "modis_lai", "modis_lst",
        "modis_ndvi", "modis_gpp", "swot_wse",
    ],
    # Daily point networks + daily/near-daily gridded products.
    "daily": [
        "snotel", "canswe_swe", "cmc_swe", "norswe_swe", "snodas_swe",
        "modis_sca", "ims_sca", "viirs_sca",
        "smap_sm", "smos_sm", "ascat_sm", "esa_cci_sm", "ismn_sm",
        "usgs_gw", "ggmn_gw",
        "chirps_precip", "gpm_imerg_precip", "mswep_precip", "daymet_precip",
        "jrc_surface_water",
        "modis_albedo", "cmc_snow_depth", "hubeau_waterlevel",
    ],
    # Keyed / per-request ensemble or tower products.
    "ondemand": ["openet", "fluxnet_et"],
}

#: tier -> default lookback window (days) for an evaluation pull.
TIER_LOOKBACK_DAYS: dict[str, int] = {
    "monthly": 365 * 20,
    "daily": 365 * 5,
    "ondemand": 365 * 5,
}
