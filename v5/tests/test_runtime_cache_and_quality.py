from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import effis_validator
import qhdalabs_wildfire_ignition_v1 as ignition
import qhdalabs_wildfire_qte_v1 as qte
import qhdalabs_wildfire_sentinel_v1 as sentinel
import run_all
from pipeline_contract import StepStatus, atomic_write_json


def sentinel_node(index: int) -> dict:
    return {
        "id": f"node-{index:02d}",
        "name": f"Node {index:02d}",
        "lat": 50.0 + index / 100,
        "lon": 16.0 + index / 100,
        "eco": "mixed",
        "latest": {},
    }


def sentinel_result(node: dict) -> dict:
    return {
        "node_id": node["id"],
        "node_name": node["name"],
        "data_source": "sentinel2_L2A",
        "fetch_date": "2026-07-25T00:00:00+00:00",
        "dates": ["2026-07-20"],
        "ndwi_values": [0.2],
        "valid_pixel_fraction": [1.0],
        "ndwi_mean_30d": 0.2,
        "ndwi_min_30d": 0.2,
        "ndwi_latest": 0.2,
        "ndwi_trend_14d": 0.0,
        "ndwi_stress_latest": 0.4,
        "calibration_status": "uncalibrated",
        "n_observations": 1,
    }


def ignition_node(index: int) -> dict:
    return {
        "id": f"node-{index:02d}",
        "name": f"Node {index:02d}",
        "lat": 51.0 + index / 1000,
        "lon": 16.0 + index / 1000,
    }


def ignition_features() -> dict[str, list[tuple[float, float]]]:
    return {
        "roads": [(51.0, 16.0)],
        "railways": [(51.0, 16.0)],
        "powerlines": [(51.0, 16.0)],
        "tourism": [(51.0, 16.0)],
        "agriculture": [(51.0, 16.0)],
    }


def test_all_fresh_sentinel_entries_skip_auth_and_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [sentinel_node(index) for index in range(33)]
    topology = tmp_path / "nodes.json"
    graph = tmp_path / "graph.json"
    atomic_write_json(topology, {"nodes": nodes})
    atomic_write_json(
        graph,
        {"adjacency": {node["id"]: [] for node in nodes}},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sentinel, "CACHE_DIR", tmp_path / "sentinel-cache")
    sentinel.CACHE_DIR.mkdir()
    for node in nodes:
        sentinel._cache_set(node, 30, sentinel_result(node), 10)
    monkeypatch.setattr(
        sentinel,
        "get_access_token",
        lambda: pytest.fail("authentication must be skipped"),
    )
    monkeypatch.setitem(sentinel._network_state, "api_requests", 0)

    results = sentinel.run_sentinel_pipeline(
        str(topology),
        str(graph),
    )

    assert len(results) == 33
    assert sentinel._network_state["api_requests"] == 0
    manifest = json.loads(
        (tmp_path / "topology/pipeline_steps/sentinel.json").read_text(encoding="utf-8")
    )
    assert manifest["cache_hits"] == 33
    assert manifest["api_requests"] == 0


def test_sentinel_fingerprint_changes_with_evalscript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = sentinel_node(1)
    original = sentinel.sentinel_query_fingerprint(node)
    monkeypatch.setattr(
        sentinel,
        "EVALSCRIPT_INDICES",
        sentinel.EVALSCRIPT_INDICES + "\n// changed",
    )
    assert sentinel.sentinel_query_fingerprint(node) != original


def test_sentinel_indices_use_explicit_correct_bands() -> None:
    definitions = sentinel.index_definitions()
    assert definitions["ndwi_surface_water"]["bands"] == ["B03", "B08"]
    assert definitions["vegetation_moisture_index"]["bands"] == ["B8A", "B11"]
    assert "not the Gao" in definitions["ndwi_surface_water"]["methodological_source"]
    assert (
        definitions["vegetation_moisture_index"]["calibration_status"] == "uncalibrated"
    )
    assert "-0.35" not in sentinel.EVALSCRIPT_INDICES
    assert "-0.70" not in sentinel.EVALSCRIPT_INDICES


def test_low_coverage_does_not_replace_last_known_good(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "ignition_scores.json"
    diagnostic_dir = tmp_path / "diagnostics"
    output.write_text('{"last_known_good": true}', encoding="utf-8")
    monkeypatch.setattr(ignition, "IGNITION_OUT", output)
    monkeypatch.setattr(ignition, "DIAGNOSTICS_DIR", diagnostic_dir)
    scores = [
        ignition.compute_ignition_score(
            ignition_node(index),
            {},
            [51.0],
            [16.0],
            ["historical_kde"],
        )
        for index in range(33)
    ]

    published = ignition._save_ignition_scores(scores)

    assert output.read_text(encoding="utf-8") == '{"last_known_good": true}'
    assert Path(published).parent == diagnostic_dir


def test_ninety_percent_coverage_is_published_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "ignition_scores.json"
    monkeypatch.setattr(ignition, "IGNITION_OUT", output)
    coverage = [name for name in ignition.SUBLAYER_NAMES if name != "historical_kde"]
    scores = [
        ignition.compute_ignition_score(
            ignition_node(index),
            ignition_features(),
            [],
            [],
            coverage,
        )
        for index in range(33)
    ]

    ignition._save_ignition_scores(scores)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["scores"]) == 33
    assert all(score["coverage_percent"] == 90.0 for score in payload["scores"])
    assert all(score["valid_for_fusion"] for score in payload["scores"])


def test_derived_gis_cache_avoids_reparsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.geojson"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ignition, "DERIVED_CACHE_DIR", tmp_path / "derived")
    calls = 0

    def parser() -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [[51.0, 16.0]]

    first, first_hit, _ = ignition._load_or_build_derived_cache(
        "roads",
        [source],
        parser,
        refresh=False,
    )
    second, second_hit, _ = ignition._load_or_build_derived_cache(
        "roads",
        [source],
        parser,
        refresh=False,
    )

    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert calls == 1


def test_numpy_qte_is_deterministic_for_seed() -> None:
    angles = (0.2, 0.4, 0.6, 0.8, 1.0)
    first = qte._run_numpy_qte(*angles, n_shots=128, random_seed=42)
    second = qte._run_numpy_qte(*angles, n_shots=128, random_seed=42)
    assert first == second


def test_qiskit_and_numpy_qte_are_compatible() -> None:
    pytest.importorskip("qiskit")
    angles = (0.2, 0.4, 0.6, 0.8, 1.0)
    numpy_result = qte._run_numpy_qte(
        *angles,
        n_shots=4096,
        random_seed=7,
    )
    qiskit_result = qte._run_qiskit_qte(*angles)
    for key in ("zz_01", "zz_12", "zz_23", "zz_34", "bridge_rate"):
        assert numpy_result[key] == pytest.approx(qiskit_result[key], abs=0.08)


def test_effis_loads_current_risk_scores_contract(tmp_path: Path) -> None:
    path = tmp_path / "risk_scores.json"
    atomic_write_json(
        path,
        {
            "scores": [
                {"node_name": "Wroclaw", "final_score": 0.75},
                {"node_name": "Milicz", "final_score": 0.25},
            ]
        },
    )
    assert effis_validator.load_scores(path) == {
        "Wroclaw": 0.75,
        "Milicz": 0.25,
    }


def test_failed_effis_process_does_not_reuse_stale_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_all, "STEP_DIR", tmp_path)
    atomic_write_json(
        tmp_path / "effis.json",
        {"step": "effis", "status": StepStatus.SKIPPED},
    )
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    manifest = run_all.run_step(
        "effis",
        SimpleNamespace(effis_tiff=tmp_path / "severity.tiff"),
    )

    assert manifest["status"] == StepStatus.OPTIONAL_FAILED
    assert manifest["valid_for_downstream"] is True


def test_wrong_interpreter_reexec_uses_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_python = tmp_path / "python.exe"
    project_python.touch()
    monkeypatch.setattr(run_all, "PROJECT_PYTHON", project_python)
    monkeypatch.setattr(run_all.sys, "executable", str(tmp_path / "global.exe"))
    monkeypatch.delenv(run_all.REEXEC_GUARD, raising=False)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    args = SimpleNamespace(no_venv_reexec=False)

    with pytest.raises(SystemExit, match="0"):
        run_all.maybe_reexec_in_project_venv(args)

    assert observed["command"][0] == str(project_python.resolve())
    assert observed["environment"][run_all.REEXEC_GUARD] == "1"


def test_step_status_contract_is_machine_readable() -> None:
    assert {status.value for status in StepStatus} == {
        "SUCCESS",
        "DEGRADED",
        "BLOCKED",
        "SKIPPED",
        "OPTIONAL_FAILED",
        "FAILED",
    }
