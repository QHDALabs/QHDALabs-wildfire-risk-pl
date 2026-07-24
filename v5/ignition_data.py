"""Validated, atomic data acquisition for the ignition pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIRMS_AREA_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_MAPKEY_STATUS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"
)
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
FIRMS_DOWNLOAD_STATUSES = {
    "complete",
    "resumable_partial",
    "authorization_failed",
    "rate_limited",
    "source_unavailable",
}
TRANSIENT_HTTP_STATUS = {408, 500, 502, 503, 504}
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


class AuthorizationHTTPError(PermanentHTTPError):
    """HTTP 401, which must stop the current series immediately."""

    def __init__(self, response: HTTPResponse, message: str):
        super().__init__(message)
        self.response = response


class RateLimitHTTPError(PermanentHTTPError):
    """HTTP 429 with an optional server-provided resume delay."""

    def __init__(
        self,
        response: HTTPResponse,
        retry_after_seconds: float | None,
        detail: str,
    ):
        suffix = (
            f"; Retry-After={retry_after_seconds:g}s"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__(f"HTTP 429: {detail}{suffix}")
        self.response = response
        self.retry_after_seconds = retry_after_seconds


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
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FirmsDownloadResult:
    """Outcome of a complete or safely resumable FIRMS product download."""

    status: str
    source: str
    completed_windows: int
    total_windows: int
    row_count: int = 0
    authorization_reason: str | None = None
    retry_after_seconds: float | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in FIRMS_DOWNLOAD_STATUSES:
            raise ValueError(f"Unsupported FIRMS download status: {self.status}")


@dataclass(frozen=True)
class MapKeyStatus:
    """Sanitized interpretation of the FIRMS map-key status endpoint."""

    state: str
    current_transactions: int | None = None
    transaction_limit: int | None = None
    interval_seconds: float | None = None


@dataclass
class RequestRateLimiter:
    """Minimum-spacing limiter shared by all requests in one download series."""

    min_interval_seconds: float
    sleep_fn: Callable[[float], None] = time.sleep
    monotonic_fn: Callable[[], float] = time.monotonic
    _last_request_at: float | None = None

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = self.monotonic_fn()
        if self._last_request_at is not None:
            remaining = self.min_interval_seconds - (now - self._last_request_at)
            if remaining > 0:
                self.sleep_fn(remaining)
                now = self.monotonic_fn()
        self._last_request_at = now


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
        for replace_attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if replace_attempt == 4:
                    raise
                time.sleep(0.05 * (replace_attempt + 1))
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
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        headers = dict(exc.headers.items()) if exc.headers else {}
        return HTTPResponse(
            body=body,
            content_type=content_type,
            status=exc.code,
            headers=headers,
        )


def request_with_retry(
    url: str,
    *,
    timeout: float = 30.0,
    attempts: int = 3,
    backoff_seconds: float = 0.5,
    transport: Transport | None = None,
    log_url: str | None = None,
    secret: str = "",
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> HTTPResponse:
    """Retry timeouts and 5xx responses; stop immediately on 401 and 429."""
    sender = transport or _default_transport
    safe_url = log_url or url
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = sender(url, timeout)
            if 200 <= response.status < 300:
                return response
            message = mask_secret(_error_message(response.body), secret)
            if response.status == 401:
                raise AuthorizationHTTPError(response, f"HTTP 401: {message}")
            if response.status == 429:
                raise RateLimitHTTPError(
                    response,
                    parse_retry_after(response.headers),
                    message,
                )
            if response.status not in TRANSIENT_HTTP_STATUS:
                raise PermanentHTTPError(f"HTTP {response.status}: {message}")
            last_error = TransientHTTPError(f"HTTP {response.status}: {message}")
        except (AuthorizationHTTPError, RateLimitHTTPError, PermanentHTTPError):
            raise
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
        if attempt < attempts:
            base_delay = backoff_seconds * (2 ** (attempt - 1))
            delay = base_delay + (base_delay * 0.25 * random_fn())
            log.warning(
                "Transient download failure for %s (attempt %d/%d); retrying in %.1fs",
                safe_url,
                attempt,
                attempts,
                delay,
            )
            sleep_fn(delay)
    safe_error = mask_secret(str(last_error), secret)
    raise TransientHTTPError(f"Download failed for {safe_url}: {safe_error}")


def parse_retry_after(
    headers: dict[str, str], *, now: datetime | None = None
) -> float | None:
    """Parse Retry-After seconds or an HTTP date without trusting header casing."""
    value = next(
        (
            header_value
            for key, header_value in headers.items()
            if key.lower() == "retry-after"
        ),
        None,
    )
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            resume_at = parsedate_to_datetime(value)
            if resume_at.tzinfo is None:
                resume_at = resume_at.replace(tzinfo=timezone.utc)
            reference = now or datetime.now(timezone.utc)
            return max(0.0, (resume_at - reference).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


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


def validate_firms_window(
    response: HTTPResponse, window_start: date, day_range: int
) -> tuple[list[str], list[dict[str, str]]]:
    """Validate FIRMS CSV structure and constrain every row to its request window."""
    fields, rows = parse_firms_csv_response(response)
    window_end = window_start + timedelta(days=day_range - 1)
    date_field = next(field for field in fields if field.lower() == "acq_date")
    for row in rows:
        acquired = date.fromisoformat(row[date_field])
        if not window_start <= acquired <= window_end:
            raise ResponseValidationError(
                "FIRMS CSV contains a row outside the requested date window"
            )
    return fields, rows


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


def _part_metadata_path(part_path: Path) -> Path:
    return part_path.with_suffix(".meta.json")


def _firms_part_metadata(
    response: HTTPResponse,
    *,
    source: str,
    bbox: tuple[float, float, float, float],
    window_start: date,
    day_range: int,
    row_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": source,
        "bbox": list(bbox),
        "window_start": window_start.isoformat(),
        "window_end": (window_start + timedelta(days=day_range - 1)).isoformat(),
        "day_range": day_range,
        "content_type": response.content_type,
        "byte_count": len(response.body),
        "row_count": row_count,
        "sha256": hashlib.sha256(response.body).hexdigest(),
    }


def _write_firms_part(
    part_path: Path,
    response: HTTPResponse,
    *,
    source: str,
    bbox: tuple[float, float, float, float],
    window_start: date,
    day_range: int,
) -> None:
    _, rows = validate_firms_window(response, window_start, day_range)
    metadata = _firms_part_metadata(
        response,
        source=source,
        bbox=bbox,
        window_start=window_start,
        day_range=day_range,
        row_count=len(rows),
    )
    atomic_write_bytes(part_path, response.body)
    atomic_write_json(_part_metadata_path(part_path), metadata)


def _load_firms_part(
    part_path: Path,
    *,
    source: str,
    bbox: tuple[float, float, float, float],
    window_start: date,
    day_range: int,
) -> HTTPResponse:
    try:
        body = part_path.read_bytes()
    except OSError as exc:
        raise ResponseValidationError(f"Cannot read FIRMS checkpoint: {exc}") from exc

    metadata_path = _part_metadata_path(part_path)
    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseValidationError(
                "FIRMS checkpoint metadata is unreadable"
            ) from exc
        if not isinstance(raw_metadata, dict):
            raise ResponseValidationError("FIRMS checkpoint metadata is invalid")
        metadata = raw_metadata
        expected = {
            "source": source,
            "bbox": list(bbox),
            "window_start": window_start.isoformat(),
            "window_end": (window_start + timedelta(days=day_range - 1)).isoformat(),
            "day_range": day_range,
            "byte_count": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ResponseValidationError(
                "FIRMS checkpoint metadata or checksum does not match"
            )

    content_type = (
        str(metadata.get("content_type", "text/csv"))
        if metadata is not None
        else "text/csv"
    )
    response = HTTPResponse(body, content_type)
    _, rows = validate_firms_window(response, window_start, day_range)
    if metadata is not None and metadata.get("row_count") != len(rows):
        raise ResponseValidationError("FIRMS checkpoint row count does not match")
    if metadata is None:
        atomic_write_json(
            metadata_path,
            _firms_part_metadata(
                response,
                source=source,
                bbox=bbox,
                window_start=window_start,
                day_range=day_range,
                row_count=len(rows),
            ),
        )
    return response


def _transaction_interval_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if not isinstance(value, str):
        return None
    pieces = value.lower().split()
    if not pieces:
        return None
    try:
        amount = float(pieces[0])
    except ValueError:
        return None
    if "minute" in value.lower():
        return amount * 60
    if "hour" in value.lower():
        return amount * 3600
    return amount


def inspect_firms_map_key(
    map_key: str,
    *,
    transport: Transport | None = None,
    timeout: float = 15.0,
    rate_limiter: RequestRateLimiter | None = None,
) -> MapKeyStatus:
    """Query mapkey_status once and return only sanitized status information."""
    sender = transport or _default_transport
    query = urllib.parse.urlencode({"MAP_KEY": map_key})
    url = f"{FIRMS_MAPKEY_STATUS_URL}?{query}"
    try:
        if rate_limiter is not None:
            rate_limiter.wait()
        response = sender(url, timeout)
    except (TimeoutError, urllib.error.URLError, OSError):
        return MapKeyStatus("unknown")

    text = response.body.decode("utf-8", errors="replace")
    lowered = text.lower()
    invalid_markers = (
        "invalid map",
        "invalid key",
        "map_key is invalid",
        "map key is invalid",
        "unknown map",
        "not a valid",
    )
    if any(marker in lowered for marker in invalid_markers):
        return MapKeyStatus("invalid_key")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return MapKeyStatus("unknown")
    if not isinstance(payload, dict):
        return MapKeyStatus("unknown")

    error_text = " ".join(
        str(payload.get(key, "")) for key in ("error", "message", "detail")
    ).lower()
    if any(marker in error_text for marker in invalid_markers):
        return MapKeyStatus("invalid_key")
    try:
        current = int(payload["current_transactions"])
        limit = int(payload["transaction_limit"])
    except (KeyError, TypeError, ValueError):
        return MapKeyStatus("unknown")
    interval = _transaction_interval_seconds(payload.get("transaction_interval"))
    state = "exhausted_transaction_window" if current >= limit else "active"
    return MapKeyStatus(state, current, limit, interval)


def _write_firms_download_status(
    path: Path, result: FirmsDownloadResult, *, year: int
) -> None:
    payload = {
        "status": result.status,
        "source": result.source,
        "year": year,
        "completed_windows": result.completed_windows,
        "total_windows": result.total_windows,
        "row_count": result.row_count,
        "authorization_reason": result.authorization_reason,
        "retry_after_seconds": result.retry_after_seconds,
        "message": result.message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)


def download_firms_year(
    destination: Path,
    *,
    map_key: str,
    year: int,
    bbox: tuple[float, float, float, float],
    source: str,
    transport: Transport | None = None,
    timeout: float = 30.0,
    parts_root: Path | None = None,
    attempts: int = 3,
    backoff_seconds: float = 0.5,
    min_request_interval_seconds: float = 0.25,
    transaction_check_every: int | None = None,
    transaction_reserve: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    random_fn: Callable[[], float] = random.random,
) -> FirmsDownloadResult:
    """Resume one source-specific FIRMS year and publish only when complete."""
    if source not in FIRMS_SOURCES:
        raise ValueError(f"Unsupported FIRMS source: {source}")
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    if transaction_check_every is not None and transaction_check_every < 1:
        raise ValueError("transaction_check_every must be positive or None")
    if transaction_reserve < 0:
        raise ValueError("transaction_reserve must not be negative")

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    windows = split_date_windows(start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_root = parts_root or destination.parent / "firms_parts"
    product_root = cache_root / source
    product_root.mkdir(parents=True, exist_ok=True)
    status_path = product_root / "status.json"
    limiter = RequestRateLimiter(
        min_request_interval_seconds,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )

    valid_parts: dict[date, HTTPResponse] = {}
    for window_start, day_range in windows:
        part_path = product_root / f"{window_start.isoformat()}.csv"
        if not part_path.exists():
            continue
        try:
            valid_parts[window_start] = _load_firms_part(
                part_path,
                source=source,
                bbox=bbox,
                window_start=window_start,
                day_range=day_range,
            )
        except ResponseValidationError as exc:
            log.warning("Ignoring invalid FIRMS checkpoint %s: %s", part_path, exc)

    def finish(
        status: str,
        *,
        row_count: int = 0,
        authorization_reason: str | None = None,
        retry_after_seconds: float | None = None,
        message: str | None = None,
    ) -> FirmsDownloadResult:
        result = FirmsDownloadResult(
            status=status,
            source=source,
            completed_windows=len(valid_parts),
            total_windows=len(windows),
            row_count=row_count,
            authorization_reason=authorization_reason,
            retry_after_seconds=retry_after_seconds,
            message=message,
        )
        _write_firms_download_status(status_path, result, year=year)
        return result

    if len(valid_parts) < len(windows):
        finish(
            "resumable_partial",
            message="Missing windows can be resumed from source-specific checkpoints",
        )

    network_requests = 0
    for window_start, day_range in windows:
        if window_start in valid_parts:
            continue

        if (
            transaction_check_every is not None
            and network_requests % transaction_check_every == 0
        ):
            key_status = inspect_firms_map_key(
                map_key,
                transport=transport,
                timeout=min(timeout, 15.0),
                rate_limiter=limiter,
            )
            if key_status.state == "invalid_key":
                return finish(
                    "authorization_failed",
                    authorization_reason="invalid_key",
                    message="FIRMS map-key status rejected the key",
                )
            if key_status.state == "exhausted_transaction_window" or (
                key_status.state == "active"
                and key_status.current_transactions is not None
                and key_status.transaction_limit is not None
                and key_status.current_transactions
                >= key_status.transaction_limit - transaction_reserve
            ):
                return finish(
                    "rate_limited",
                    retry_after_seconds=key_status.interval_seconds,
                    message="FIRMS transaction window is exhausted or reserved",
                )
            if key_status.state == "active":
                log.info(
                    "FIRMS transactions: %d/%d",
                    key_status.current_transactions,
                    key_status.transaction_limit,
                )
            else:
                log.warning("FIRMS transaction counter could not be read")

        url = build_firms_area_url(map_key, source, bbox, window_start, day_range)
        safe_url = mask_secret(url, map_key)
        try:
            limiter.wait()
            response = request_with_retry(
                url,
                timeout=timeout,
                attempts=attempts,
                backoff_seconds=backoff_seconds,
                transport=transport,
                log_url=safe_url,
                secret=map_key,
                sleep_fn=sleep_fn,
                random_fn=random_fn,
            )
            validate_firms_window(response, window_start, day_range)
        except AuthorizationHTTPError:
            key_status = inspect_firms_map_key(
                map_key,
                transport=transport,
                timeout=min(timeout, 15.0),
                rate_limiter=limiter,
            )
            reason = (
                key_status.state
                if key_status.state in {"invalid_key", "exhausted_transaction_window"}
                else "unknown_authorization_failure"
            )
            return finish(
                "authorization_failed",
                authorization_reason=reason,
                message=f"FIRMS authorization failed: {reason}",
            )
        except RateLimitHTTPError as exc:
            return finish(
                "rate_limited",
                retry_after_seconds=exc.retry_after_seconds,
                message="FIRMS returned HTTP 429; no further requests were sent",
            )
        except TransientHTTPError as exc:
            status = "resumable_partial" if valid_parts else "source_unavailable"
            return finish(status, message=str(exc))
        except (PermanentHTTPError, ResponseValidationError) as exc:
            status = "resumable_partial" if valid_parts else "source_unavailable"
            return finish(status, message=str(exc))

        part_path = product_root / f"{window_start.isoformat()}.csv"
        _write_firms_part(
            part_path,
            response,
            source=source,
            bbox=bbox,
            window_start=window_start,
            day_range=day_range,
        )
        valid_parts[window_start] = response
        network_requests += 1

    ordered_parts: list[HTTPResponse] = []
    for window_start, day_range in windows:
        part_path = product_root / f"{window_start.isoformat()}.csv"
        try:
            ordered_parts.append(
                _load_firms_part(
                    part_path,
                    source=source,
                    bbox=bbox,
                    window_start=window_start,
                    day_range=day_range,
                )
            )
        except ResponseValidationError as exc:
            valid_parts.pop(window_start, None)
            return finish(
                "resumable_partial",
                message=f"Checkpoint failed final validation: {exc}",
            )

    merged = merge_firms_csv_parts(ordered_parts)
    _, rows = parse_firms_csv_response(HTTPResponse(merged, "text/csv"))
    atomic_write_bytes(destination, merged)
    atomic_write_json(
        destination.with_suffix(".meta.json"),
        {
            "schema_version": 1,
            "source": source,
            "year": year,
            "bbox": list(bbox),
            "window_count": len(windows),
            "row_count": len(rows),
            "byte_count": len(merged),
            "sha256": hashlib.sha256(merged).hexdigest(),
        },
    )
    return finish("complete", row_count=len(rows))


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
