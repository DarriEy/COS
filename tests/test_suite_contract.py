"""Executable compatibility contract shared by the community data services."""

import json
import re
from pathlib import Path

from cos.core.exceptions import ConnectorError, COSError
from cos.core.models import QualityFlag

ROOT = Path(__file__).parents[1]


def test_suite_contract_metadata() -> None:
    contract = json.loads((ROOT / "suite-contract.json").read_text(encoding="utf-8"))
    assert contract["service"] == "cos"
    assert contract["contract_scope"] == "native_public_api"
    assert re.fullmatch(r"\d+\.\d+\.\d+", contract["contract_version"])
    assert contract["canonical_output"] in {"time_series", "attributes", "gridded_forcing"}
    assert contract["timestamps"] == "UTC"
    assert contract["interval_semantics"] == "point_timestamps"
    assert contract["provenance_fields"] == ['provider','source_info','fetched_at']
    assert set(contract["quality_statuses"]) == {flag.value for flag in QualityFlag}
    assert contract["quality_scope"] == "observation_points"
    assert contract["cli_exit_codes"]["upstream_error"] == 1
    assert contract["symfluence_entrypoint"].endswith(":register")
    assert (ROOT / "src" / contract["service"] / "py.typed").is_file()


def test_suite_connector_error_contract() -> None:
    error = ConnectorError("example", "failed")
    assert isinstance(error, COSError)
    assert error.provider == "example"
    assert str(error) == "[example] failed"
