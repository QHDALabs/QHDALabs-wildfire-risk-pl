"""Orchestrate the v5 wildfire pipeline with explicit quality contracts."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from pipeline_contract import (
    StepStatus,
    atomic_write_json,
    step_manifest,
    utc_now,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TOPOLOGY_DIR = BASE_DIR / "topology"
STEP_DIR = TOPOLOGY_DIR / "pipeline_steps"
PROJECT_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
REEXEC_GUARD = "QHDALABS_V5_VENV_REEXEC"

CORE_STEPS = ("topology", "sentinel", "qte", "ignition", "fusion")
ALL_STEPS = (*CORE_STEPS, "effis")
STEP_SCRIPTS = {
    "topology": "qhdalabs_wildfire_topology_v1.py",
    "sentinel": "qhdalabs_wildfire_sentinel_v1.py",
    "qte": "qhdalabs_wildfire_qte_v1.py",
    "ignition": "qhdalabs_wildfire_ignition_v1.py",
    "fusion": "qhdalabs_wildfire_fusion_v1.py",
    "effis": "effis_validator.py",
}
STEP_DEPENDENCIES = {
    "topology": (),
    "sentinel": ("topology",),
    "qte": ("sentinel",),
    "ignition": ("sentinel",),
    "fusion": ("qte",),
    "effis": ("fusion",),
}
STEP_IMPORTS = {
    "topology": ("numpy", "requests"),
    "sentinel": ("numpy", "requests"),
    "qte": ("numpy",),
    "ignition": ("numpy", "pandas", "geopandas", "pyogrio", "shapely", "pyproj"),
    "fusion": ("numpy",),
    "effis": ("numpy", "rasterio"),
}
ENVIRONMENT_VARIABLES = (
    "CDSE_CLIENT_ID",
    "CDSE_CLIENT_SECRET",
    "FIRMS_MAP_KEY",
    "NASA_FIRMS_MAP_KEY",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QHDALabs Wildfire Risk PL v5 pipeline runner"
    )
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--skip", metavar="STEP", nargs="+", default=[])
    parser.add_argument("--only", metavar="STEP", choices=ALL_STEPS)
    parser.add_argument("--from", dest="from_step", metavar="STEP", choices=ALL_STEPS)
    parser.add_argument("--until", metavar="STEP", choices=ALL_STEPS)
    parser.add_argument("--refresh-sentinel", action="store_true")
    parser.add_argument("--offline-sentinel", action="store_true")
    parser.add_argument("--sentinel-cache-ttl-days", type=float)
    parser.add_argument("--legacy-sentinel-cache", action="store_true")
    parser.add_argument("--refresh-firms", action="store_true")
    parser.add_argument("--refresh-gis", action="store_true")
    parser.add_argument("--effis-tiff", type=Path)
    parser.add_argument("--with-effis", action="store_true")
    parser.add_argument("--no-venv-reexec", action="store_true")
    parser.add_argument("--migrate", action="store_true")
    return parser


def maybe_reexec_in_project_venv(args: argparse.Namespace) -> None:
    if args.no_venv_reexec or not PROJECT_PYTHON.exists():
        return
    if os.environ.get(REEXEC_GUARD) == "1":
        log.info("Virtual-environment re-exec guard is active")
        return
    current = Path(sys.executable).resolve()
    expected = PROJECT_PYTHON.resolve()
    if current == expected:
        return
    log.warning("Current interpreter: %s", current)
    log.info("Re-executing with project interpreter: %s", expected)
    environment = os.environ.copy()
    environment[REEXEC_GUARD] = "1"
    completed = subprocess.run(
        [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=BASE_DIR,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


def _module_diagnostic(module: str) -> dict[str, Any]:
    try:
        imported = importlib.import_module(module)
        try:
            version = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(imported, "__version__", "unknown")
        return {"available": True, "version": str(version)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _osm_driver_diagnostic() -> dict[str, Any]:
    try:
        import pyogrio

        mode = pyogrio.list_drivers().get("OSM")
        return {
            "available": mode is not None and "r" in mode,
            "backend": "pyogrio",
            "mode": mode,
        }
    except Exception as exc:
        return {"available": False, "backend": "pyogrio", "error": str(exc)}


def run_doctor(args: argparse.Namespace) -> int:
    requested_steps = set(select_steps(args))
    modules = sorted(
        {
            module
            for step in requested_steps
            for module in STEP_IMPORTS[step]
            if module != "rasterio" or "effis" in requested_steps
        }
    )
    libraries = {module: _module_diagnostic(module) for module in modules}
    inputs = {
        "nodes": (TOPOLOGY_DIR / "nodes.json").exists(),
        "graph": (TOPOLOGY_DIR / "graph.json").exists(),
        "nodes_enriched": (TOPOLOGY_DIR / "nodes_enriched.json").exists(),
    }
    caches = {
        "topology": (BASE_DIR / ".cache_topology").exists(),
        "sentinel": (BASE_DIR / ".cache_topology" / "sentinel").exists(),
        "ignition": (TOPOLOGY_DIR / "ignition_cache").exists(),
        "gis_derived": (TOPOLOGY_DIR / "ignition_cache" / "derived").exists(),
    }
    imports: dict[str, Any] = {}
    for filename in STEP_SCRIPTS.values():
        module_name = Path(filename).stem
        try:
            importlib.import_module(module_name)
            imports[module_name] = {"available": True}
        except Exception as exc:
            imports[module_name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    osm_driver = (
        _osm_driver_diagnostic()
        if "ignition" in requested_steps
        else {"requested": False}
    )
    required_missing = [
        module
        for module, result in libraries.items()
        if not result["available"]
        and (module != "rasterio" or "effis" in requested_steps)
    ]
    if "ignition" in requested_steps and not osm_driver.get("available"):
        required_missing.append("GDAL OSM driver")
    report = {
        "generated_at": utc_now(),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "project_interpreter": str(PROJECT_PYTHON),
        "using_project_interpreter": (
            PROJECT_PYTHON.exists()
            and Path(sys.executable).resolve() == PROJECT_PYTHON.resolve()
        ),
        "requested_steps": list(requested_steps),
        "libraries": libraries,
        "environment": {
            name: {"present": bool(os.environ.get(name))}
            for name in ENVIRONMENT_VARIABLES
        },
        "cache": caches,
        "inputs": inputs,
        "module_imports": imports,
        "osm_driver": osm_driver,
        "errors": [f"Missing requirement: {name}" for name in required_missing],
    }
    atomic_write_json(TOPOLOGY_DIR / "doctor.json", report)
    log.info("Interpreter: %s", sys.executable)
    log.info("Python: %s", sys.version.splitlines()[0])
    for module, result in libraries.items():
        log.info(
            "%-12s %s",
            module,
            f"OK ({result.get('version')})" if result["available"] else "MISSING",
        )
    for name, state in report["environment"].items():
        log.info(
            "Environment %-22s %s", name, "present" if state["present"] else "absent"
        )
    log.info("GDAL/Pyogrio OSM driver: %s", osm_driver)
    if report["errors"]:
        for error in report["errors"]:
            log.error("%s", error)
        return 2
    log.info("Doctor checks passed")
    return 0


def select_steps(args: argparse.Namespace) -> list[str]:
    available = list(CORE_STEPS)
    if args.with_effis or args.effis_tiff or args.only == "effis":
        available.append("effis")
    if args.only:
        return [args.only]
    start = available.index(args.from_step) if args.from_step in available else 0
    end = available.index(args.until) + 1 if args.until in available else len(available)
    if start >= end:
        raise ValueError("--from must precede or equal --until")
    return available[start:end]


def _step_command(step: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(BASE_DIR / STEP_SCRIPTS[step])]
    if step == "sentinel":
        if args.refresh_sentinel:
            command.append("--refresh")
        if args.offline_sentinel:
            command.append("--offline")
        if args.sentinel_cache_ttl_days is not None:
            command.extend(["--cache-ttl-days", str(args.sentinel_cache_ttl_days)])
        if args.legacy_sentinel_cache:
            command.append("--legacy-cache")
    elif step == "ignition":
        if args.refresh_firms:
            command.append("--refresh-firms")
        if args.refresh_gis:
            command.append("--refresh-gis")
    elif step == "effis":
        if not args.effis_tiff:
            raise ValueError("--with-effis requires --effis-tiff PATH")
        command.extend(
            [
                str(args.effis_tiff.resolve()),
                "--scores",
                str(TOPOLOGY_DIR / "risk_scores.json"),
            ]
        )
    return command


def _read_step_manifest(step: str) -> dict[str, Any] | None:
    path = STEP_DIR / f"{step}.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def _dependency_block(step: str, manifests: dict[str, dict[str, Any]]) -> str | None:
    for dependency in STEP_DEPENDENCIES[step]:
        manifest = manifests.get(dependency)
        if not manifest:
            continue
        status = manifest["status"]
        valid = manifest.get("valid_for_downstream", True)
        if status in {StepStatus.BLOCKED, StepStatus.FAILED} or not valid:
            return f"upstream {dependency} is {status} (valid_for_downstream={valid})"
    return None


def run_step(step: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    command = _step_command(step, args)
    manifest_path = STEP_DIR / f"{step}.json"
    previous_signature = (
        (manifest_path.stat().st_mtime_ns, manifest_path.stat().st_size)
        if manifest_path.exists()
        else None
    )
    log.info("Running %s: %s", step, STEP_SCRIPTS[step])
    completed = subprocess.run(command, cwd=BASE_DIR, check=False)
    duration = time.monotonic() - started
    current_signature = (
        (manifest_path.stat().st_mtime_ns, manifest_path.stat().st_size)
        if manifest_path.exists()
        else None
    )
    manifest = (
        _read_step_manifest(step)
        if current_signature is not None and current_signature != previous_signature
        else None
    )
    if manifest is None:
        if completed.returncode == 0:
            status = StepStatus.SUCCESS
        elif step == "effis":
            status = StepStatus.OPTIONAL_FAILED
        else:
            status = StepStatus.FAILED
        manifest = step_manifest(
            step,
            status,
            duration_seconds=duration,
            valid_for_downstream=completed.returncode == 0 or step == "effis",
            errors=[]
            if completed.returncode == 0
            else [f"exit code {completed.returncode}"],
        )
        atomic_write_json(manifest_path, manifest)
    else:
        manifest["duration_seconds"] = round(duration, 3)
        atomic_write_json(manifest_path, manifest)
    log.info("%s status=%s duration=%.1fs", step, manifest["status"], duration)
    return manifest


def _git_sha() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _data_versions(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        step: manifest.get("data_version", manifest.get("schema_version"))
        for step, manifest in manifests.items()
    }


def execute_pipeline(args: argparse.Namespace) -> int:
    if args.with_effis and not args.effis_tiff:
        raise ValueError("--with-effis requires --effis-tiff PATH")
    steps = select_steps(args)
    unknown_skips = set(args.skip) - set(ALL_STEPS)
    if unknown_skips:
        raise ValueError(f"Unknown --skip steps: {sorted(unknown_skips)}")

    started_at = utc_now()
    started = time.monotonic()
    run_id = str(uuid.uuid4())
    manifests: dict[str, dict[str, Any]] = {}
    operational = True

    for step in steps:
        if step in args.skip:
            manifest = step_manifest(
                step,
                StepStatus.SKIPPED,
                valid_for_downstream=step not in {"topology", "sentinel", "qte"},
                warnings=["Explicitly skipped by --skip"],
            )
            atomic_write_json(STEP_DIR / f"{step}.json", manifest)
            manifests[step] = manifest
            continue
        block = _dependency_block(step, manifests)
        if block:
            manifest = step_manifest(
                step,
                StepStatus.BLOCKED,
                valid_for_downstream=False,
                errors=[block],
            )
            atomic_write_json(STEP_DIR / f"{step}.json", manifest)
            manifests[step] = manifest
            operational = False
            log.error("%s blocked: %s", step, block)
            continue
        try:
            manifest = run_step(step, args)
        except Exception as exc:
            status = (
                StepStatus.OPTIONAL_FAILED if step == "effis" else StepStatus.FAILED
            )
            manifest = step_manifest(
                step,
                status,
                valid_for_downstream=step == "effis",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            atomic_write_json(STEP_DIR / f"{step}.json", manifest)
        manifests[step] = manifest
        status = manifest["status"]
        if manifest.get("operational_validity") is False:
            operational = False
        if status == StepStatus.DEGRADED:
            operational = False
            if args.strict:
                log.error("Strict mode rejects degraded step %s", step)
                break
        if status in {StepStatus.BLOCKED, StepStatus.FAILED}:
            operational = False
        if status == StepStatus.OPTIONAL_FAILED:
            log.warning("Optional EFFIS validation failed; core result is preserved")

    if "effis" not in manifests and not args.only:
        manifest = step_manifest(
            "effis",
            StepStatus.SKIPPED,
            valid_for_downstream=True,
            warnings=["EFFIS is optional; use --with-effis --effis-tiff PATH"],
        )
        atomic_write_json(STEP_DIR / "effis.json", manifest)
        manifests["effis"] = manifest
        log.info("EFFIS status=SKIPPED")

    statuses = [manifest["status"] for manifest in manifests.values()]
    if any(status in {StepStatus.FAILED, StepStatus.BLOCKED} for status in statuses):
        quality = StepStatus.FAILED
    elif any(
        status in {StepStatus.DEGRADED, StepStatus.OPTIONAL_FAILED}
        for status in statuses
    ):
        quality = StepStatus.DEGRADED
    else:
        quality = StepStatus.SUCCESS
    cache_hits = sum(
        int(manifest.get("cache_hits", 0)) for manifest in manifests.values()
    )
    api_requests = sum(
        int(manifest.get("api_requests", 0)) for manifest in manifests.values()
    )
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "started_at": started_at,
        "generated_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "git_sha": _git_sha(),
        "steps": manifests,
        "data_versions": _data_versions(manifests),
        "cache_hits": cache_hits,
        "api_requests": api_requests,
        "quality": str(quality),
        "operational_validity": bool(operational and quality == StepStatus.SUCCESS),
        "allow_degraded": args.allow_degraded,
        "warnings": [
            warning
            for manifest in manifests.values()
            for warning in manifest.get("warnings", [])
        ],
    }
    atomic_write_json(TOPOLOGY_DIR / "pipeline_run.json", payload)
    log.info(
        "Pipeline quality=%s operational_validity=%s",
        quality,
        payload["operational_validity"],
    )
    if quality == StepStatus.FAILED or (args.strict and quality == StepStatus.DEGRADED):
        return 1
    return 0


def migrate() -> int:
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / "db.py"), str(TOPOLOGY_DIR)],
        cwd=BASE_DIR,
        check=False,
    )
    return completed.returncode


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    maybe_reexec_in_project_venv(args)
    try:
        if args.migrate:
            raise SystemExit(migrate())
        if args.doctor:
            raise SystemExit(run_doctor(args))
        raise SystemExit(execute_pipeline(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
