from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_contract import EXPECTED_NODE_COUNT
from qhdalabs_wildfire_topology_v1 import RDLP_WROCLAW_NODES

SCRIPT = Path(__file__).parents[1] / "qhdalabs_wildfire_ignition_v1.py"


def test_topology_node_contract_matches_dataset_size() -> None:
    assert EXPECTED_NODE_COUNT == len(RDLP_WROCLAW_NODES) == 34


def test_refresh_firms_is_accepted_by_parser() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--refresh-firms", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--refresh-firms" in result.stdout


def test_unknown_cli_argument_is_rejected() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--refresh-firmsc"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
