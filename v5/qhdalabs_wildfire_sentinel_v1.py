# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 2 — Sentinel-2 Water and Moisture Indices
# File          : qhdalabs_wildfire_sentinel_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Fetches a Green-NIR surface-water index and a separate NIR-SWIR vegetation
# moisture index for each nadleśnictwo in the RDLP Wrocław pilot network.
#
# Raw indices and quality flags are retained separately from an explicitly
# uncalibrated stress mapping.
#
# Indices
# -----------------------------------------------------------------------------
# ndwi_surface_water = (B03 - B08) / (B03 + B08)
# vegetation_moisture_index = (B8A - B11) / (B8A + B11)
# Green-NIR is not the Gao 1996 vegetation liquid-water index. Sentinel-2 B11
# is at 1.61 µm, so B8A/B11 is not Gao's exact 0.86/1.24 µm formulation.
#
# API
# -----------------------------------------------------------------------------
# Copernicus Data Space — Sentinel Hub Statistical API
# Endpoint: https://sh.dataspace.copernicus.eu/statistics/v1
# Auth:     OAuth2 client credentials
#           Set env vars: CDSE_CLIENT_ID, CDSE_CLIENT_SECRET
#           (create at: dataspace.copernicus.eu → Dashboard → User Settings → OAuth)
#
# For each nadleśnictwo centroid we use a 5 km radius bounding box.
# The Statistical API returns mean NDWI over the bounding box per time period.
# Cloud-masked (SCL bands 3,8,9,10,11 excluded).
#
# Time series: last 30 days, one value per available cloud-free acquisition.
# Sentinel-2 revisit time: ~5 days per point in Poland.
# Expected: ~4–6 data points per 30-day window.
#
# Outputs
# -----------------------------------------------------------------------------
#   topology/ndwi_sentinel.json    — NDWI time series per node
#   topology/nodes_enriched.json   — Step 1 nodes + real NDWI merged
#   topology/network_map_v2.html   — updated map with satellite data
#
# Dependencies
# -----------------------------------------------------------------------------
#   numpy, requests
#   Step 1 output: topology/nodes.json must exist
#
# Usage
# -----------------------------------------------------------------------------
#   export CDSE_CLIENT_ID=your_client_id
#   export CDSE_CLIENT_SECRET=your_client_secret
#   python qhdalabs_wildfire_sentinel_v1.py
#
# Author        : Krzysztof W. Banasiewicz / QHDALabs
# License       : MIT
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline_contract import (
    StepStatus,
    atomic_write_json,
    atomic_write_text,
    stable_fingerprint,
    validate_node_payload,
    write_step_manifest,
)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"
CDSE_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")

NDWI_DAYS = 30
BOX_RADIUS_DEG = 0.045
MAX_WORKERS = 4
CACHE_DIR = Path(".cache_topology") / "sentinel"
LEGACY_CACHE_DIR = Path(".cache_topology")
DEFAULT_CACHE_TTL_DAYS = 10
SCHEMA_VERSION = 2
OUTPUT_DIR = "topology"
HTTP_TIMEOUT = 30
COLLECTION = "sentinel-2-l2a"
AGGREGATION_INTERVAL = "P10D"
RESOLUTION_DEGREES = 0.001
MAX_CLOUD_COVERAGE = 85
CIRCUIT_BREAKER_THRESHOLD = 5

CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CACHE
# =========================
def resolve_cache_ttl_days(cli_value: float | None = None) -> float:
    if cli_value is not None:
        return cli_value
    raw = os.environ.get("SENTINEL_CACHE_TTL_DAYS")
    if raw:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError("SENTINEL_CACHE_TTL_DAYS must be numeric") from exc
    return float(DEFAULT_CACHE_TTL_DAYS)


def index_definitions() -> dict[str, dict[str, Any]]:
    return {
        "ndwi_surface_water": {
            "name": "ndwi_surface_water",
            "formula": "(B03 - B08) / (B03 + B08)",
            "bands": ["B03", "B08"],
            "resolution_m": 10,
            "methodological_source": (
                "Green-NIR surface-water index; this is not the Gao (1996) "
                "vegetation liquid-water index"
            ),
            "unit": "dimensionless",
            "algorithm_version": "2.0.0",
            "calibration_status": "uncalibrated",
        },
        "vegetation_moisture_index": {
            "name": "vegetation_moisture_index",
            "formula": "(B8A - B11) / (B8A + B11)",
            "bands": ["B8A", "B11"],
            "resolution_m": 20,
            "methodological_source": (
                "NIR-SWIR canopy-moisture formulation motivated by Gao (1996), "
                "DOI 10.1016/S0034-4257(96)00067-3; Sentinel-2 B11 at 1.61 µm "
                "is an approximation, not Gao's 1.24 µm channel"
            ),
            "unit": "dimensionless",
            "algorithm_version": "2.0.0",
            "calibration_status": "uncalibrated",
        },
    }


def sentinel_query_fingerprint(node: dict, days: int = NDWI_DAYS) -> str:
    return stable_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "node_id": node["id"],
            "latitude": round(float(node["lat"]), 7),
            "longitude": round(float(node["lon"]), 7),
            "period_days": days,
            "radius_degrees": BOX_RADIUS_DEG,
            "collection": COLLECTION,
            "aggregation_interval": AGGREGATION_INTERVAL,
            "resolution_degrees": RESOLUTION_DEGREES,
            "max_cloud_coverage": MAX_CLOUD_COVERAGE,
            "evalscript_sha256": hashlib.sha256(
                EVALSCRIPT_INDICES.encode("utf-8")
            ).hexdigest(),
            "indices": index_definitions(),
        }
    )


def _cache_path(node_id: str) -> Path:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in node_id
    )
    return CACHE_DIR / f"{safe}.json"


def _legacy_cache_path(node: dict, days: int) -> Path:
    key = f"ndwi_sentinel_{node['id']}_{days}d"
    return LEGACY_CACHE_DIR / f"{key.replace('/', '_').replace(':', '_')}.pkl"


def _load_cache_entry(
    node: dict,
    days: int,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any] | None]:
    now = now or datetime.now(timezone.utc)
    path = _cache_path(node["id"])
    if not path.exists():
        return (
            ("legacy", None)
            if _legacy_cache_path(node, days).exists()
            else ("missing", None)
        )
    try:
        with path.open(encoding="utf-8") as stream:
            entry = json.load(stream)
        if entry.get("schema_version") != SCHEMA_VERSION:
            return "invalid", entry
        if entry.get("query_fingerprint") != sentinel_query_fingerprint(node, days):
            return "invalid", entry
        if not isinstance(entry.get("result"), dict):
            return "invalid", entry
        expires = datetime.fromisoformat(entry["expires_at"])
        return ("fresh" if expires > now else "stale"), entry
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "invalid", None


def _cache_set(
    node: dict,
    days: int,
    data: dict[str, Any],
    ttl_days: float,
) -> None:
    created = datetime.now(timezone.utc)
    entry = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(days=ttl_days)).isoformat(),
        "query_fingerprint": sentinel_query_fingerprint(node, days),
        "cache_status": "fresh",
        "result": data,
    }
    atomic_write_json(_cache_path(node["id"]), entry)


def inspect_sentinel_cache(
    nodes: list[dict],
    days: int = NDWI_DAYS,
    *,
    refresh: bool = False,
) -> dict[str, list[tuple[dict, dict[str, Any] | None]]]:
    classified = {
        name: [] for name in ("fresh", "stale", "missing", "invalid", "legacy")
    }
    for node in nodes:
        status, entry = _load_cache_entry(node, days)
        if refresh and status == "fresh":
            status = "stale"
        classified[status].append((node, entry))
    return classified


def _load_legacy_cache_result(node: dict, days: int) -> dict[str, Any] | None:
    path = _legacy_cache_path(node, days)
    if not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            _created_timestamp, result = pickle.load(stream)  # noqa: S301
        if not isinstance(result, dict):
            return None
        migrated = dict(result)
        migrated["cache_status"] = "legacy"
        migrated["legacy_mode"] = True
        migrated["calibration_status"] = "legacy_provisional"
        migrated.setdefault("quality_flags", []).extend(
            [
                "legacy_green_nir_mislabelled_as_vegetation_moisture",
                "legacy_thresholds_provisional",
            ]
        )
        return migrated
    except (OSError, ValueError, TypeError, pickle.UnpicklingError):
        return None


# =========================
# AUTHENTICATION
# =========================
_token_cache: dict = {"token": None, "expires_at": 0.0}


def get_access_token() -> str:
    """
    Fetch OAuth2 access token from Copernicus Identity Service.
    Tokens are valid for 600 seconds; cached to avoid repeated requests.
    """
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 30:
        return _token_cache["token"]

    if not CDSE_CLIENT_ID or not CDSE_CLIENT_SECRET:
        raise EnvironmentError(
            "Missing Copernicus credentials.\n"
            "Set environment variables:\n"
            "  CDSE_CLIENT_ID=your_client_id\n"
            "  CDSE_CLIENT_SECRET=your_client_secret\n"
            "Create credentials at: dataspace.copernicus.eu → Dashboard → User Settings → OAuth"
        )

    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CDSE_CLIENT_ID,
            "client_secret": CDSE_CLIENT_SECRET,
        },
        timeout=15,
    )
    resp.raise_for_status()
    token_data = resp.json()
    _token_cache["token"] = token_data["access_token"]
    _token_cache["expires_at"] = now + int(token_data.get("expires_in", 600))
    log.debug(
        "Copernicus token refreshed, valid for %ds", token_data.get("expires_in", 600)
    )
    return _token_cache["token"]


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "QHDALabs-WildfirePL-Sentinel/1.0"})
    return session


HTTP = _build_session()
_network_state = {
    "api_requests": 0,
    "consecutive_failures": 0,
    "circuit_open": False,
}
_network_lock = threading.Lock()


def _retry_after_seconds(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(raw)
                    return max(
                        0.0,
                        (parsed - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError):
                    pass
    return min(30.0, (2**attempt) + random.uniform(0.0, 0.5))


def _post_statistics(body: dict[str, Any], token: str) -> requests.Response:
    last_error: BaseException | None = None
    for attempt in range(4):
        with _network_lock:
            if _network_state["circuit_open"]:
                raise RuntimeError("Sentinel circuit breaker is open")
            _network_state["api_requests"] += 1
        response: requests.Response | None = None
        try:
            response = HTTP.post(
                CDSE_STATS_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=HTTP_TIMEOUT,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                with _network_lock:
                    _network_state["consecutive_failures"] = 0
                return response
            last_error = requests.HTTPError(
                f"transient Sentinel HTTP {response.status_code}",
                response=response,
            )
        except (
            requests.Timeout,
            requests.ConnectionError,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = exc
        with _network_lock:
            _network_state["consecutive_failures"] += 1
            if _network_state["consecutive_failures"] >= CIRCUIT_BREAKER_THRESHOLD:
                _network_state["circuit_open"] = True
                break
        if attempt < 3:
            time.sleep(_retry_after_seconds(response, attempt))
    if last_error:
        raise last_error
    raise RuntimeError("Sentinel request failed")


# =========================
# EVALSCRIPT
# =========================
# Two explicitly named indices are returned. Green-NIR is a surface-water
# index and must never be described as Gao's vegetation liquid-water index.
EVALSCRIPT_INDICES = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08", "B8A", "B11", "SCL", "dataMask"] }],
    output: [
      { id: "data",     bands: 2, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8"   }
    ]
  };
}
function evaluatePixel(s) {
  // CDSE Statistical API requires explicit dataMask output.
  // Cloud/shadow mask via SCL bands.
  if (s.dataMask === 0) return { data: [NaN], dataMask: [0] };
  var scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10||scl===11)
    return { data: [NaN], dataMask: [0] };
  var surfaceDenominator = s.B03 + s.B08;
  var moistureDenominator = s.B8A + s.B11;
  var surfaceWater = (surfaceDenominator === 0)
    ? 0.0 : (s.B03 - s.B08) / surfaceDenominator;
  var vegetationMoisture = (moistureDenominator === 0)
    ? 0.0 : (s.B8A - s.B11) / moistureDenominator;
  return { data: [surfaceWater, vegetationMoisture], dataMask: [1] };
}
"""
EVALSCRIPT_NDWI = EVALSCRIPT_INDICES


# =========================
# NDWI FETCH
# =========================
def _node_bbox(lat: float, lon: float, radius_deg: float = BOX_RADIUS_DEG) -> dict:
    """Build bounding box geometry for a node centroid."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - radius_deg, lat - radius_deg],
                [lon + radius_deg, lat - radius_deg],
                [lon + radius_deg, lat + radius_deg],
                [lon - radius_deg, lat + radius_deg],
                [lon - radius_deg, lat - radius_deg],
            ]
        ],
    }


def fetch_ndwi_timeseries(
    node: dict,
    days: int = NDWI_DAYS,
    *,
    token: str | None = None,
    cache_ttl_days: float = DEFAULT_CACHE_TTL_DAYS,
) -> dict | None:
    """
    Fetch NDWI time series for a node from Sentinel Hub Statistical API.

    Returns dict:
      node_id, node_name, dates, ndwi_values, ndwi_valid_pixels,
      ndwi_mean_30d, ndwi_min_30d, ndwi_trend_14d, data_source="sentinel2_L2A"
    Returns None on failure (API error, no cloud-free data, etc.)
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    request_body = {
        "input": {
            "bounds": {
                "geometry": _node_bbox(node["lat"], node["lon"]),
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": COLLECTION,
                    "dataFilter": {
                        "mosaickingOrder": "leastCC",  # least cloud cover first
                        "maxCloudCoverage": MAX_CLOUD_COVERAGE,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": start_dt.strftime("%Y-%m-%dT00:00:00Z"),
                "to": end_dt.strftime("%Y-%m-%dT23:59:59Z"),
            },
            "aggregationInterval": {"of": AGGREGATION_INTERVAL},
            "evalscript": EVALSCRIPT_INDICES,
            "resx": RESOLUTION_DEGREES,
            "resy": RESOLUTION_DEGREES,
        },
    }

    try:
        token = token or get_access_token()
        resp = _post_statistics(request_body, token)
        raw = resp.json()
        # DEBUG: log raw response structure on first successful call
        if not getattr(fetch_ndwi_timeseries, "_logged_structure", False):
            fetch_ndwi_timeseries._logged_structure = True
            import json as _json

            log.debug(
                "Sentinel API response structure:\n%s",
                _json.dumps(raw, indent=2)[:1200],
            )
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response else ""
        log.warning(
            "Sentinel API HTTP error for %s: %s\nResponse body: %s",
            node["name"],
            exc,
            body[:1000],
        )
        return None
    except Exception as exc:
        log.warning("Sentinel API failed for %s: %s", node["name"], exc)
        return None

    # Parse response
    intervals = raw.get("data", [])
    if not intervals:
        log.warning("No Sentinel data for %s (all cloudy?)", node["name"])
        return None

    dates: list[str] = []
    surface_values: list[float] = []
    moisture_values: list[float] = []
    valid_px: list[float] = []
    for interval in intervals:
        # Sentinel Hub Statistical API response structure (without "calculations" field):
        # outputs -> data -> bands -> B0 -> stats
        outputs = interval.get("outputs", {})

        # Try standard bands path first
        # CDSE Statistical API response path:
        # outputs -> data -> bands -> B0 -> stats
        try:
            surface_stats = outputs["data"]["bands"]["B0"]["stats"]
            moisture_stats = outputs["data"]["bands"]["B1"]["stats"]
        except (KeyError, TypeError):
            continue

        surface_mean = surface_stats.get("mean")
        moisture_mean = moisture_stats.get("mean")
        count = min(
            surface_stats.get("sampleCount", 0),
            moisture_stats.get("sampleCount", 0),
        )
        no_data = max(
            surface_stats.get("noDataCount", 0),
            moisture_stats.get("noDataCount", 0),
        )

        # Skip intervals with < 10% valid (cloud-free) pixels
        valid_fraction = (count - no_data) / count if count > 0 else 0.0
        if (
            valid_fraction < 0.10
            or surface_mean is None
            or moisture_mean is None
            or surface_mean != surface_mean
            or moisture_mean != moisture_mean
        ):
            continue

        date_str = interval.get("interval", {}).get("from", "")[:10]
        dates.append(date_str)
        surface_values.append(round(float(surface_mean), 4))
        moisture_values.append(round(float(moisture_mean), 4))
        valid_px.append(round(valid_fraction, 3))

    if not dates:
        log.warning("All intervals cloudy for %s — no valid NDWI data", node["name"])
        return None

    # Compute summary statistics
    moisture_array = np.array(moisture_values)
    moisture_mean = round(float(np.mean(moisture_array)), 4)
    moisture_min = round(float(np.min(moisture_array)), 4)

    # 14-day trend: slope of NDWI (negative = drying out = rising fire risk)
    if len(moisture_array) >= 2:
        x = np.arange(len(moisture_array), dtype=float)
        trend = float(np.polyfit(x, moisture_array, 1)[0])
        ndwi_trend_14d = round(trend, 5)
    else:
        ndwi_trend_14d = 0.0

    # Threshold-free affine mapping of the raw [-1, 1] moisture index. It is
    # explicitly uncalibrated and must not be interpreted as an operational
    # drought threshold.
    ndwi_stress_latest = float(np.clip((1.0 - moisture_values[-1]) / 2.0, 0.0, 1.0))
    definitions = index_definitions()
    quality_flags = [
        "cloud_masked",
        "vegetation_moisture_index_uncalibrated",
        "sentinel_b11_is_not_gao_1_24um",
    ]

    result = {
        "node_id": node["id"],
        "node_name": node["name"],
        "data_source": "sentinel2_L2A",
        "fetch_date": datetime.now(timezone.utc).isoformat(),
        "indices": {
            "ndwi_surface_water": {
                **definitions["ndwi_surface_water"],
                "raw_series": surface_values,
                "dates": dates,
                "quality_flags": quality_flags,
            },
            "vegetation_moisture_index": {
                **definitions["vegetation_moisture_index"],
                "raw_series": moisture_values,
                "dates": dates,
                "quality_flags": quality_flags,
            },
        },
        "dates": dates,
        "ndwi_values": moisture_values,
        "valid_pixel_fraction": valid_px,
        "ndwi_mean_30d": moisture_mean,
        "ndwi_min_30d": moisture_min,
        "ndwi_latest": moisture_values[-1],
        "ndwi_trend_14d": ndwi_trend_14d,  # neg = drying, pos = recovering
        "ndwi_stress_latest": round(ndwi_stress_latest, 4),
        "stress_mapping": {
            "formula": "(1 - vegetation_moisture_index) / 2",
            "thresholds": {},
            "calibration_status": "uncalibrated",
            "operational": False,
        },
        "calibration_status": "uncalibrated",
        "quality_flags": quality_flags,
        "n_observations": len(dates),
    }
    _cache_set(node, days, result, cache_ttl_days)
    return result


# =========================
# STRESS NORMALISATION
# =========================
def normalise_stress_across_network(
    ndwi_results: dict[str, dict],
) -> dict[str, float]:
    """
    Normalise NDWI stress scores relative to the network.

    Raw NDWI stress is computed per-node independently. Normalisation
    makes within-network comparisons meaningful: a node at 0.8 is
    drier than 80% of the network, not just 80% of the absolute scale.

    Uses min-max normalisation across the pilot network.
    Returns dict: node_id -> normalised_stress [0, 1]
    """
    stresses = {
        nid: r["ndwi_stress_latest"] for nid, r in ndwi_results.items() if r is not None
    }
    if not stresses:
        return {}

    vals = list(stresses.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo

    if span < 0.01:
        # All nodes equally stressed — return raw values
        return {nid: v for nid, v in stresses.items()}

    return {nid: round((v - lo) / span, 4) for nid, v in stresses.items()}


# =========================
# MERGE WITH STEP 1 DATA
# =========================
def merge_with_topology(
    nodes_json_path: str,
    ndwi_results: dict[str, dict | None],
    normalised: dict[str, float],
) -> list[dict]:
    """
    Load Step 1 nodes.json and replace ndwi_proxy with real Sentinel-2 data.
    Falls back to ndwi_proxy for nodes where satellite data is unavailable.
    """
    with open(nodes_json_path, encoding="utf-8") as f:
        topology = json.load(f)

    nodes = topology["nodes"]
    for node in nodes:
        nid = node["id"]
        result = ndwi_results.get(nid)

        if result is not None:
            # Real satellite data available
            node["ndwi_sentinel"] = result
            node["ndwi_stress_source"] = "sentinel2_L2A"
            node["ndwi_stress_latest"] = result["ndwi_stress_latest"]
            node["ndwi_stress_normalised"] = normalised.get(
                nid, result["ndwi_stress_latest"]
            )
            node["ndwi_trend_14d"] = result["ndwi_trend_14d"]

            # Update the "latest" dict with satellite-based stress
            if "latest" in node:
                node["latest"]["ndwi_stress"] = result["ndwi_stress_latest"]
                node["latest"]["ndwi_source"] = "sentinel2"
                node["latest"]["ndwi_latest"] = result["ndwi_latest"]
        else:
            # Fallback to weather proxy from Step 1
            proxy = node.get("ndwi_proxy", [])
            node["ndwi_stress_source"] = "weather_proxy_fallback"
            node["ndwi_stress_latest"] = proxy[-1] if proxy else 0.0
            node["ndwi_stress_normalised"] = node["ndwi_stress_latest"]
            if "latest" in node:
                node["latest"]["ndwi_source"] = "weather_proxy"

    return nodes


# =========================
# MAP GENERATION V2
# =========================
def generate_map_v2(
    nodes: list[dict],
    graph: dict,
    output_path: str,
) -> None:
    """Updated map showing real Sentinel-2 NDWI data."""

    node_js = []
    for n in nodes:
        latest = n.get("latest", {})
        ndwi_s = n.get("ndwi_sentinel", {})
        node_js.append(
            {
                "id": n["id"],
                "name": n["name"],
                "lat": n["lat"],
                "lon": n["lon"],
                "eco": n["eco"],
                "drought_days": n.get("drought_days", 0),
                "ndwi_stress": n.get("ndwi_stress_latest", 0.0),
                "ndwi_norm": n.get("ndwi_stress_normalised", 0.0),
                "ndwi_source": n.get("ndwi_stress_source", "proxy"),
                "ndwi_trend": n.get("ndwi_trend_14d", 0.0),
                "ndwi_latest": ndwi_s.get("ndwi_latest"),
                "n_obs": ndwi_s.get("n_observations", 0),
                "network_stress": n.get("network_stress", 0.0),
                "temp_max": latest.get("temp_max"),
                "rh_min": latest.get("rh_min"),
                "wind_max": latest.get("wind_max"),
                "neighbors": [nb["id"] for nb in graph.get(n["id"], [])],
            }
        )

    node_data = json.dumps(node_js, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>QHDALabs — Sieć Grzybni v2 (Sentinel-2)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html,body,#map {{ margin:0; height:100vh; font-family:'Courier New',monospace; background:#060c06; }}
    #panel {{
      position:absolute; top:12px; left:12px; z-index:1000;
      background:rgba(6,18,6,0.93); color:#7fff7f; padding:14px 18px;
      border:1px solid #1a4a1a; border-radius:4px; font-size:12px;
      max-width:280px; box-shadow:0 0 24px rgba(0,180,0,0.12);
    }}
    #panel h2 {{ margin:0 0 4px; font-size:13px; color:#afffaf; letter-spacing:2px; text-transform:uppercase; }}
    #panel .sub {{ color:#3a8a3a; font-size:10px; margin-bottom:10px; }}
    #panel .row {{ display:flex; align-items:center; gap:6px; margin:3px 0; font-size:11px; }}
    #panel .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
    #toggle {{ margin-top:10px; font-size:10px; color:#3a8a3a; cursor:pointer; border:1px solid #1a4a1a; padding:3px 7px; border-radius:2px; }}
    #toggle:hover {{ color:#7fff7f; border-color:#3a8a3a; }}
  </style>
</head>
<body>
<div id="panel">
  <h2>🛰 Sieć Grzybni v2</h2>
  <div class="sub">RDLP Wrocław · Sentinel-2 NDWI</div>
  <div class="row"><div class="dot" style="background:#ff1111"></div>KRYTYCZNY stres (&gt;0.70)</div>
  <div class="row"><div class="dot" style="background:#ff7700"></div>WYSOKI (0.50–0.70)</div>
  <div class="row"><div class="dot" style="background:#ffcc00"></div>UMIARKOWANY (0.30–0.50)</div>
  <div class="row"><div class="dot" style="background:#22cc44"></div>NISKI (&lt;0.30)</div>
  <div class="row" style="margin-top:8px">
    <div class="dot" style="background:#888;border-radius:0"></div>
    <span style="color:#3a8a3a">— dane proxy (brak satelity)</span>
  </div>
  <button id="toggle" onclick="toggleEdges()">Ukryj połączenia</button>
</div>
<div id="map"></div>
<script>
const NODES = {node_data};
let showEdges = true;
const edgeLayer = L.layerGroup();

const map = L.map('map').setView([51.0, 16.5], 8);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'© OpenStreetMap © CARTO', maxZoom:19
}}).addTo(map);

function stressColor(s, isSentinel) {{
  if (!isSentinel) return '#666666';
  if (s > 0.70) return '#ff1111';
  if (s > 0.50) return '#ff7700';
  if (s > 0.30) return '#ffcc00';
  return '#22cc44';
}}

function ecoLabel(e) {{
  return {{pine:'Bór sosnowy',pine_wetland:'Bór/mokradło',
           mixed:'Las mieszany',spruce_mountain:'Świerk górski'}}[e]||e;
}}

function trendArrow(t) {{
  if (t < -0.002) return '↓ wysycha';
  if (t >  0.002) return '↑ odradza';
  return '→ stabilny';
}}

// Index nodes
const nodeMap = {{}};
NODES.forEach(n => nodeMap[n.id] = n);

// Draw edges
const drawnEdges = new Set();
NODES.forEach(n => {{
  n.neighbors.forEach(nbId => {{
    const key = [n.id, nbId].sort().join('--');
    if (drawnEdges.has(key)) return;
    drawnEdges.add(key);
    const nb = nodeMap[nbId];
    if (!nb) return;
    const s = (n.network_stress + nb.network_stress) / 2;
    const isSat = n.ndwi_source==='sentinel2_L2A' && nb.ndwi_source==='sentinel2_L2A';
    L.polyline([[n.lat,n.lon],[nb.lat,nb.lon]], {{
      color: stressColor(s, isSat),
      weight: 1.2,
      opacity: isSat ? 0.30 + s*0.45 : 0.15,
    }}).addTo(edgeLayer);
  }});
}});
edgeLayer.addTo(map);

function toggleEdges() {{
  showEdges = !showEdges;
  showEdges ? edgeLayer.addTo(map) : map.removeLayer(edgeLayer);
  document.getElementById('toggle').textContent = showEdges ? 'Ukryj połączenia' : 'Pokaż połączenia';
}}

// Draw nodes
NODES.forEach(n => {{
  const s      = n.ndwi_norm;
  const isSat  = n.ndwi_source === 'sentinel2_L2A';
  const color  = stressColor(s, isSat);
  const r      = 7 + s * 13;

  const ndwiRow = n.ndwi_latest !== null
    ? `<tr><td>NDWI (sat)</td><td><b>${{n.ndwi_latest}}</b></td></tr>`
    : `<tr><td>NDWI</td><td><i>brak danych sat.</i></td></tr>`;

  const popup = `
    <b style="color:${{color}};font-family:'Courier New'">${{n.name}}</b><br>
    <small style="color:#888">${{ecoLabel(n.eco)}} · ${{isSat ? '🛰 Sentinel-2' : '🌦 proxy pogodowy'}}</small>
    <table style="margin:6px 0;font-size:11px;border-collapse:collapse;width:100%">
      <tr><td style="padding:1px 4px 1px 0">Stres NDWI</td>
          <td><b>${{(n.ndwi_stress*100).toFixed(1)}}%</b></td></tr>
      <tr><td>Stres (norm.)</td>
          <td><b>${{(n.ndwi_norm*100).toFixed(1)}}%</b></td></tr>
      ${{ndwiRow}}
      <tr><td>Trend 14d</td><td>${{trendArrow(n.ndwi_trend)}}</td></tr>
      <tr><td>Susza</td><td>${{n.drought_days}} dni</td></tr>
      <tr><td>Serwacje sat.</td><td>${{n.n_obs}}</td></tr>
      <tr><td>Temp max</td><td>${{n.temp_max!==null?n.temp_max+'°C':'N/A'}}</td></tr>
      <tr><td>RH min</td><td>${{n.rh_min!==null?n.rh_min+'%':'N/A'}}</td></tr>
      <tr><td>Wiatr max</td><td>${{n.wind_max!==null?n.wind_max+' m/s':'N/A'}}</td></tr>
    </table>
  `;

  L.circleMarker([n.lat,n.lon], {{
    radius: r, color: color, fillColor: color,
    fillOpacity: isSat ? 0.75 + s*0.20 : 0.40,
    weight: isSat ? 1.5 : 1,
    dashArray: isSat ? null : '4,3',
  }}).addTo(map).bindPopup(popup);
}});
</script>
</body></html>"""

    atomic_write_text(output_path, html)


# =========================
# MAIN PIPELINE
# =========================
def run_sentinel_pipeline(
    nodes_json_path: str = os.path.join(OUTPUT_DIR, "nodes.json"),
    graph_json_path: str = os.path.join(OUTPUT_DIR, "graph.json"),
    ndwi_days: int = NDWI_DAYS,
    *,
    cache_ttl_days: float = DEFAULT_CACHE_TTL_DAYS,
    refresh: bool = False,
    offline: bool = False,
    legacy_cache: bool = False,
) -> dict[str, dict | None]:
    """
    Full Step 2 pipeline:
      1. Load topology from Step 1
      2. Fetch Sentinel-2 NDWI for all nodes (parallel, rate-limited)
      3. Normalise stress scores across the network
      4. Merge with Step 1 data
      5. Save enriched outputs

    Returns dict: node_id -> ndwi_result (None if unavailable)
    """
    with _network_lock:
        _network_state["api_requests"] = 0
        _network_state["consecutive_failures"] = 0
        _network_state["circuit_open"] = False
    log.info("=== QHDALabs Wildfire — Step 2: Sentinel-2 NDWI ===")

    # ── Load topology ──────────────────────────────────────────────────────
    if not os.path.exists(nodes_json_path):
        raise FileNotFoundError(
            f"Step 1 output not found: {nodes_json_path}\n"
            "Run qhdalabs_wildfire_topology_v1.py first."
        )
    with open(nodes_json_path, encoding="utf-8") as f:
        topology = json.load(f)
    with open(graph_json_path, encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = topology["nodes"]
    graph = graph_data["adjacency"]
    log.info("Loaded %d nodes from topology", len(nodes))

    classified = inspect_sentinel_cache(nodes, ndwi_days, refresh=refresh)
    log.info(
        "Sentinel cache: fresh=%d stale=%d missing=%d invalid=%d legacy=%d",
        *(
            len(classified[name])
            for name in ("fresh", "stale", "missing", "invalid", "legacy")
        ),
    )
    log.info("Sentinel cache fresh=%d", len(classified["fresh"]))
    ndwi_results: dict[str, dict | None] = {
        node["id"]: entry["result"]
        for node, entry in classified["fresh"]
        if entry is not None
    }
    stale_fallback = {
        node["id"]: entry["result"]
        for node, entry in classified["stale"]
        if entry is not None and isinstance(entry.get("result"), dict)
    }
    legacy_results = {
        node["id"]: result
        for node, _entry in classified["legacy"]
        if legacy_cache and (result := _load_legacy_cache_result(node, ndwi_days))
    }
    ndwi_results.update(legacy_results)
    pending = [
        node
        for status in ("stale", "missing", "invalid", "legacy")
        for node, _entry in classified[status]
        if node["id"] not in legacy_results
    ]

    if not pending:
        log.info("Copernicus authentication skipped")
    elif offline:
        log.info("Offline Sentinel mode: network access disabled")
        for node in pending:
            fallback = stale_fallback.get(node["id"])
            if fallback:
                fallback = dict(fallback)
                fallback.setdefault("quality_flags", []).append("stale_if_offline")
                fallback["cache_status"] = "stale"
            ndwi_results[node["id"]] = fallback
    else:
        token = get_access_token()
        log.info("Copernicus authentication OK")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    fetch_ndwi_timeseries,
                    node,
                    ndwi_days,
                    token=token,
                    cache_ttl_days=cache_ttl_days,
                ): node
                for node in pending
            }
            for future in as_completed(futures):
                node = futures[future]
                nid = node["id"]
                try:
                    result = future.result()
                except Exception as exc:
                    log.warning(
                        "Sentinel request failed for %s: %s",
                        node["name"],
                        exc,
                    )
                    result = None
                if result is None and nid in stale_fallback:
                    result = dict(stale_fallback[nid])
                    result.setdefault("quality_flags", []).append("stale_if_error")
                    result["cache_status"] = "stale"
                ndwi_results[nid] = result

    ok = sum(1 for v in ndwi_results.values() if v is not None)
    cloudy = len(nodes) - ok
    log.info(
        "NDWI fetch complete: %d/%d nodes with satellite data (%d cloudy/failed)",
        ok,
        len(nodes),
        cloudy,
    )

    if ok == 0:
        log.error("No Sentinel-2 data retrieved. Check credentials and date range.")
        raise RuntimeError("Zero nodes with valid NDWI data.")

    # ── Normalise ──────────────────────────────────────────────────────────
    normalised = normalise_stress_across_network(
        {k: v for k, v in ndwi_results.items() if v}
    )

    # ── Merge with topology ────────────────────────────────────────────────
    enriched_nodes = merge_with_topology(nodes_json_path, ndwi_results, normalised)

    # ── Save outputs ───────────────────────────────────────────────────────
    # ndwi_sentinel.json — raw satellite results
    ndwi_path = os.path.join(OUTPUT_DIR, "ndwi_sentinel.json")
    atomic_write_json(
        ndwi_path,
        {
            "version": "2.0.0",
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ndwi_days": ndwi_days,
            "nodes_ok": ok,
            "nodes_cloudy": cloudy,
            "index_definitions": index_definitions(),
            "calibration_status": "uncalibrated",
            "operational": False,
            "results": {k: v for k, v in ndwi_results.items() if v},
        },
    )
    log.info("Saved %s", ndwi_path)

    # nodes_enriched.json — merged topology + satellite
    enriched_path = os.path.join(OUTPUT_DIR, "nodes_enriched.json")
    enriched_payload = {
        "version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_sources": ["open_meteo_archive", "sentinel2_L2A"],
        "calibration_status": "uncalibrated",
        "nodes": enriched_nodes,
    }
    validate_node_payload(enriched_payload, "nodes")
    atomic_write_json(enriched_path, enriched_payload)
    log.info("Saved %s", enriched_path)

    # network_map_v2.html
    map_path = os.path.join(OUTPUT_DIR, "network_map_v2.html")
    generate_map_v2(enriched_nodes, graph, map_path)
    log.info("Saved %s", map_path)

    # ── Console summary ────────────────────────────────────────────────────
    _print_summary(enriched_nodes)

    cache_hits = len(classified["fresh"])
    api_requests = int(_network_state["api_requests"])
    log.info("Copernicus API requests: %d", api_requests)
    status = (
        StepStatus.SUCCESS
        if ok == len(nodes) and not legacy_results
        else StepStatus.DEGRADED
    )
    warnings = ["Satellite-to-stress mapping is uncalibrated"]
    if legacy_results:
        warnings.append(
            f"Explicit legacy Sentinel cache used for {len(legacy_results)} nodes"
        )
    if ok < len(nodes):
        warnings.append(f"Satellite coverage is {ok}/{len(nodes)} nodes")
    write_step_manifest(
        OUTPUT_DIR,
        "sentinel",
        status,
        inputs={"nodes": nodes_json_path, "graph": graph_json_path},
        outputs={
            "indices": ndwi_path,
            "nodes_enriched": enriched_path,
            "map": map_path,
        },
        coverage_percent=round(ok / len(nodes) * 100.0, 2),
        valid_for_downstream=ok > 0,
        warnings=warnings,
        cache_hits=cache_hits,
        cache_classification={name: len(items) for name, items in classified.items()},
        api_requests=api_requests,
        data_version="sentinel-indices-2.0.0",
        calibration_status="uncalibrated",
        operational_validity=False,
        legacy_cache_nodes=len(legacy_results),
    )
    return ndwi_results


def _print_summary(nodes: list[dict]) -> None:
    log.info("\n%s", "=" * 72)
    log.info("NDWI Sentinel-2 Summary — RDLP Wrocław")
    log.info("%s", "=" * 72)
    log.info(
        "%-24s %-8s %-8s %-10s %-10s %s",
        "Nadleśnictwo",
        "NDWI",
        "Stres%",
        "Trend14d",
        "Susza",
        "Źródło",
    )
    log.info("%s", "-" * 72)

    sorted_nodes = sorted(
        nodes,
        key=lambda n: n.get("ndwi_stress_normalised", 0),
        reverse=True,
    )
    for n in sorted_nodes:
        ndwi_s = n.get("ndwi_sentinel", {})
        ndwi = ndwi_s.get("ndwi_latest", "N/A")
        stress = n.get("ndwi_stress_normalised", 0) * 100
        trend = n.get("ndwi_trend_14d", 0)
        trend_s = f"{trend:+.4f}" if trend else "N/A"
        src = "SAT" if n.get("ndwi_stress_source") == "sentinel2_L2A" else "proxy"
        tier = (
            "KRYT"
            if stress > 70
            else "WYS "
            if stress > 50
            else "UMIA"
            if stress > 30
            else "LOW "
        )
        log.info(
            "%-24s %-8s %6.1f%%  %-10s %4dd [%s] %s",
            n["name"][:24],
            str(ndwi)[:7],
            stress,
            trend_s,
            n.get("drought_days", 0),
            tier,
            src,
        )
    log.info("%s", "=" * 72)
    log.info(
        "Next: run qhdalabs_wildfire_qte_v1.py (Step 3 — Quantum Temporal Encoder)"
    )


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel-2 dual-index enrichment")
    parser.add_argument("--cache-ttl-days", type=float)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--legacy-cache", action="store_true")
    parsed = parser.parse_args()
    run_sentinel_pipeline(
        cache_ttl_days=resolve_cache_ttl_days(parsed.cache_ttl_days),
        refresh=parsed.refresh,
        offline=parsed.offline,
        legacy_cache=parsed.legacy_cache,
    )
