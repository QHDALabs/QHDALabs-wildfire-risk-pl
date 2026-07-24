"""Validated, atomic data acquisition for the ignition pipeline."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIRMS_AREA_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_MAX_DAY_RANGE = 5
FIRMS_REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
}
FIRMS_SOURCES = ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT")
TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
USER_AGENT = "QHDALabs-Wildfire/1.1 (research; contact@qhdalabs.pl)"

CLC_QUERY_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_WM/MapServer/0/query"
)
CLC_AGRICULTURE_CODES = (
    "211",
    "212",
    "213",
    "221",
    "222",
    "223",
    "231",
    "241",
    "242",
    "243",
    "244",
)
CLC_PAGE_SIZE = 1000


class DataAcquisitionError(RuntimeError):
    """Base class for a data source failure."""


class PermanentHTTPError(DataAcquisitionError):
    """An HTTP response that must not be retried."""


class TransientHTTPError(DataAcquisitionError):
    """An exhausted transient HTTP failure."""


class ResponseValidationError(DataAcquisitionError):
    """A response is not the advertised data format."""


@dataclass(frozen=True)
class HTTPResponse:
    """Small transport-neutral HTTP response used by tests and downloaders."""

    body: bytes
    content_type: str
    status: int = 200


Transport = Callable[[str, float], HTTPResponse]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace *path* atomically without exposing a partial production file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize JSON deterministically and replace the destination atomically."""
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)


def build_firms_area_url(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start_date: date,
    day_range: int,
) -> str:
    """Build the official FIRMS Area API path in its documented segment order."""
    if not map_key.strip():
        raise ValueError("FIRMS MAP_KEY must not be empty")
    if source not in FIRMS_SOURCES:
        raise ValueError(f"Unsupported FIRMS source: {source}")
    if not 1 <= day_range <= FIRMS_MAX_DAY_RANGE:
        raise ValueError(f"FIRMS day_range must be 1..{FIRMS_MAX_DAY_RANGE}")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid FIRMS bounding box")
    area = ",".join(f"{value:g}" for value in bbox)
    return (
        f"{FIRMS_AREA_BASE_URL}/{urllib.parse.quote(map_key, safe='')}/{source}/"
        f"{area}/{day_range}/{start_date.isoformat()}"
    )


def mask_secret(value: str, secret: str) -> str:
    """Remove a secret and its URL-encoded representation from loggable text."""
    if not secret:
        return value
    return value.replace(secret, "***").replace(
        urllib.parse.quote(secret, safe=""), "***"
    )


def split_date_windows(
    start: date,
    end: date,
    max_days: int = FIRMS_MAX_DAY_RANGE,
) -> list[tuple[date, int]]:
    """Split an inclusive date interval into deterministic API windows."""
    if end < start:
        raise ValueError("end must not precede start")
    if not 1 <= max_days <= FIRMS_MAX_DAY_RANGE:
        raise ValueError(f"max_days must be 1..{FIRMS_MAX_DAY_RANGE}")
    windows: list[tuple[date, int]] = []
    cursor = start
    while cursor <= end:
        days = min(max_days, (end - cursor).days + 1)
        windows.append((cursor, days))
        cursor += timedelta(days=days)
    return windows


def _default_transport(url: str, timeout: float) -> HTTPResponse:
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/csv,application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                body=response.read(),
                content_type=response.headers.get("Content-Type", ""),
                status=response.status,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return HTTPResponse(body=body, content_type=content_type, status=exc.code)


def request_with_retry(
    url: str,
    *,
    timeout: float = 30.0,
    attempts: int = 3,
    backoff_seconds: float = 0.5,
    transport: Transport | None = None,
    log_url: str | None = None,
) -> HTTPResponse:
    """Fetch a URL and retry only timeouts, network failures, and transient HTTP."""
    sender = transport or _default_transport
    safe_url = log_url or url
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = sender(url, timeout)
            if 200 <= response.status < 300:
                return response
            message = _error_message(response.body)
            if response.status not in TRANSIENT_HTTP_STATUS:
                raise PermanentHTTPError(f"HTTP {response.status}: {message}")
            last_error = TransientHTTPError(f"HTTP {response.status}: {message}")
        except PermanentHTTPError:
            raise
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
        if attempt < attempts:
            delay = backoff_seconds * (2 ** (attempt - 1))
            log.warning(
                "Transient download failure for %s (attempt %d/%d); retrying in %.1fs",
                safe_url,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise TransientHTTPError(f"Download failed for {safe_url}: {last_error}")


def _error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return " ".join(text.split())[:200] or "empty response"


def parse_firms_csv_response(
    response: HTTPResponse,
) -> tuple[list[str], list[dict[str, str]]]:
    """Validate and parse a FIRMS CSV response, rejecting API error documents."""
    body = response.body
    if not body.strip():
        raise ResponseValidationError("FIRMS returned an empty response")
    prefix = body.lstrip()[:32].lower()
    content_type = response.content_type.lower()
    if (
        prefix.startswith((b"<html", b"<!doctype", b"<?xml"))
        or "text/html" in content_type
    ):
        raise ResponseValidationError("FIRMS returned HTML/XML instead of CSV")
    if prefix.startswith((b"{", b"[")) or "application/json" in content_type:
        raise ResponseValidationError("FIRMS returned JSON instead of CSV")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ResponseValidationError("FIRMS CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [name.strip() for name in (reader.fieldnames or [])]
    normalized = {name.lower() for name in fieldnames}
    missing = FIRMS_REQUIRED_COLUMNS - normalized
    if missing:
        raise ResponseValidationError(
            f"FIRMS CSV schema is missing columns: {', '.join(sorted(missing))}"
        )
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {(key or "").strip(): (value or "").strip() for key, value in raw.items()}
        lower = {key.lower(): value for key, value in row.items()}
        try:
            latitude = float(lower["latitude"])
            longitude = float(lower["longitude"])
            date.fromisoformat(lower["acq_date"])
            int(lower["acq_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResponseValidationError(
                "FIRMS CSV contains an invalid data row"
            ) from exc
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ResponseValidationError("FIRMS CSV contains invalid coordinates")
        rows.append(row)
    return fieldnames, rows


def merge_firms_csv_parts(parts: Iterable[HTTPResponse]) -> bytes:
    """Validate, combine, de-duplicate, and deterministically sort FIRMS CSV parts."""
    canonical_fields: list[str] | None = None
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for part in parts:
        fields, rows = parse_firms_csv_response(part)
        if canonical_fields is None:
            canonical_fields = fields
        elif {field.lower() for field in fields} != {
            field.lower() for field in canonical_fields
        }:
            raise ResponseValidationError("FIRMS CSV windows use inconsistent schemas")
        assert canonical_fields is not None
        field_lookup = {field.lower(): field for field in fields}
        for row in rows:
            normalized_row = {
                canonical: row.get(field_lookup.get(canonical.lower(), canonical), "")
                for canonical in canonical_fields
            }
            key = tuple(normalized_row.get(field, "") for field in canonical_fields)
            unique[key] = normalized_row
    if canonical_fields is None:
        raise ResponseValidationError("No FIRMS CSV parts were supplied")

    lower_to_field = {field.lower(): field for field in canonical_fields}

    def sort_key(row: dict[str, str]) -> tuple[str, ...]:
        preferred = ("acq_date", "acq_time", "latitude", "longitude", "satellite")
        return tuple(
            row.get(lower_to_field.get(name, ""), "") for name in preferred
        ) + tuple(row.get(field, "") for field in canonical_fields)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=canonical_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(unique.values(), key=sort_key))
    return output.getvalue().encode("utf-8")


def download_firms_year(
    destination: Path,
    *,
    map_key: str,
    year: int,
    bbox: tuple[float, float, float, float],
    source: str,
    transport: Transport | None = None,
    timeout: float = 30.0,
) -> int:
    """Download one complete year in five-day windows and atomically publish it."""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    windows = split_date_windows(start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts: list[HTTPResponse] = []
    with tempfile.TemporaryDirectory(
        prefix=".firms-", dir=destination.parent
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for index, (window_start, day_range) in enumerate(windows):
            url = build_firms_area_url(map_key, source, bbox, window_start, day_range)
            safe_url = mask_secret(url, map_key)
            response = request_with_retry(
                url,
                timeout=timeout,
                transport=transport,
                log_url=safe_url,
            )
            parse_firms_csv_response(response)
            part_path = temp_root / f"{index:03d}-{window_start.isoformat()}.csv"
            part_path.write_bytes(response.body)
            parts.append(response)
        merged = merge_firms_csv_parts(parts)
        _, rows = parse_firms_csv_response(HTTPResponse(merged, "text/csv"))
        atomic_write_bytes(destination, merged)
    return len(rows)


def validate_geojson_response(
    response: HTTPResponse, *, required_property: str | None = None
) -> dict:
    """Validate a GeoJSON FeatureCollection and an optional required property."""
    if not response.body.strip():
        raise ResponseValidationError("GeoJSON response is empty")
    if "html" in response.content_type.lower() or response.body.lstrip().startswith(
        b"<"
    ):
        raise ResponseValidationError("Endpoint returned HTML/XML instead of GeoJSON")
    try:
        payload = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseValidationError("Endpoint returned invalid JSON") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise ResponseValidationError(
            f"Endpoint returned an API error: {payload['error']}"
        )
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ResponseValidationError("JSON is not a GeoJSON FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ResponseValidationError("GeoJSON has no features array")
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ResponseValidationError("GeoJSON contains an invalid feature")
        if not isinstance(feature.get("geometry"), dict):
            raise ResponseValidationError("GeoJSON feature has no geometry")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ResponseValidationError("GeoJSON feature has no properties")
        if required_property and required_property not in properties:
            raise ResponseValidationError(
                f"GeoJSON feature is missing property {required_property}"
            )
    return payload


def build_clc_query_url(
    bbox: tuple[float, float, float, float],
    offset: int,
    page_size: int = CLC_PAGE_SIZE,
) -> str:
    """Build one deterministic, spatially filtered CLC ArcGIS query page."""
    where = (
        "Code_18 IN (" + ",".join(f"'{code}'" for code in CLC_AGRICULTURE_CODES) + ")"
    )
    params = {
        "where": where,
        "geometry": ",".join(f"{value:g}" for value in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,Code_18,Remark,Area_Ha",
        "returnGeometry": "true",
        "orderByFields": "OBJECTID ASC",
        "resultOffset": str(offset),
        "resultRecordCount": str(page_size),
        "f": "geojson",
    }
    return f"{CLC_QUERY_URL}?{urllib.parse.urlencode(params)}"


def download_clc_geojson(
    destination: Path,
    *,
    bbox: tuple[float, float, float, float],
    transport: Transport | None = None,
    timeout: float = 60.0,
) -> int:
    """Download every CLC query page, validate it, and atomically publish GeoJSON."""
    features: list[dict] = []
    offset = 0
    while True:
        url = build_clc_query_url(bbox, offset)
        response = request_with_retry(url, timeout=timeout, transport=transport)
        page = validate_geojson_response(response, required_property="Code_18")
        page_features = page["features"]
        features.extend(page_features)
        if len(page_features) < CLC_PAGE_SIZE:
            break
        offset += CLC_PAGE_SIZE
    payload = {
        "type": "FeatureCollection",
        "name": "CLC2018_agriculture_dolnoslaskie",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    validate_geojson_response(
        HTTPResponse(json.dumps(payload).encode("utf-8"), "application/geo+json"),
        required_property="Code_18",
    )
    atomic_write_json(destination, payload)
    return len(features)
