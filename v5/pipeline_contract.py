"""Shared runtime, manifest, and atomic-output helpers for the v5 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

EXPECTED_NODE_COUNT = 34
SCHEMA_VERSION = 2


class StepStatus(StrEnum):
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    OPTIONAL_FAILED = "OPTIONAL_FAILED"
    FAILED = "FAILED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def ensure_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_finite(child, f"{location}[{index}]")


def sanitize_nonfinite(value: Any) -> Any:
    """Return a JSON-safe copy with NaN and infinities represented as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_nonfinite(child) for child in value]
    if isinstance(value, tuple):
        return [sanitize_nonfinite(child) for child in value]
    return value


def validate_node_payload(
    payload: dict[str, Any],
    collection_key: str,
    expected_count: int = EXPECTED_NODE_COUNT,
) -> None:
    records = payload.get(collection_key)
    if not isinstance(records, list):
        raise ValueError(f"{collection_key!r} must be a list")
    if len(records) != expected_count:
        raise ValueError(
            f"{collection_key!r} must contain {expected_count} nodes; got {len(records)}"
        )
    identifiers = [
        record.get("node_id", record.get("id"))
        for record in records
        if isinstance(record, dict)
    ]
    if len(set(identifiers)) != expected_count or None in identifiers:
        raise ValueError(f"{collection_key!r} contains missing or duplicate node IDs")
    ensure_finite(payload)


def step_manifest(
    step: str,
    status: StepStatus | str,
    *,
    duration_seconds: float = 0.0,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    coverage_percent: float | None = None,
    valid_for_downstream: bool = True,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "status": str(status),
        "generated_at": utc_now(),
        "duration_seconds": round(duration_seconds, 3),
        "inputs": inputs or {},
        "outputs": outputs or {},
        "coverage_percent": coverage_percent,
        "valid_for_downstream": valid_for_downstream,
        "warnings": warnings or [],
        "errors": errors or [],
    }
    manifest.update(metadata)
    ensure_finite(manifest)
    return manifest


def write_step_manifest(
    output_dir: str | Path,
    step: str,
    status: StepStatus | str,
    **kwargs: Any,
) -> dict[str, Any]:
    manifest = step_manifest(step, status, **kwargs)
    atomic_write_json(Path(output_dir) / "pipeline_steps" / f"{step}.json", manifest)
    return manifest


def stable_fingerprint(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def source_fingerprint(
    path: str | Path,
    *,
    parser_version: str,
    backend: str,
    crs: str = "EPSG:4326",
    include_sha256: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "parser_version": parser_version,
        "backend": backend,
        "crs": crs,
    }
    if include_sha256:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    result["fingerprint"] = stable_fingerprint(result)
    return result
