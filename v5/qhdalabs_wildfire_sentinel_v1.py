# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 2 — Sentinel-2 NDWI Time Series
# File          : qhdalabs_wildfire_sentinel_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Fetches real vegetation water stress (NDWI) from Sentinel-2 satellite data
# for each nadleśnictwo in the RDLP Wrocław pilot network.
#
# Replaces the ndwi_proxy (weather-based approximation) from Step 1 with
# actual satellite measurements of forest moisture content.
#
# What is NDWI?
# -----------------------------------------------------------------------------
# Normalized Difference Water Index = (B03 - B08) / (B03 + B08)
#   B03 = green band (550 nm)  — reflects when vegetation is moist
#   B08 = NIR band  (842 nm)   — absorbed when canopy water content drops
#
# Range: -1 to +1
#   > 0.3  = well-watered vegetation
#   0.0–0.3 = moderate stress
#   < 0.0  = significant drought stress → fire risk elevated
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

import json
import logging
import math
import os
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
CDSE_TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATS_URL  = "https://sh.dataspace.copernicus.eu/statistics/v1"
CDSE_CLIENT_ID  = os.environ.get("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")

NDWI_DAYS       = 30          # days of NDWI history to fetch
BOX_RADIUS_DEG  = 0.045       # ~5 km bounding box radius around centroid
MAX_WORKERS     = 4           # conservative — Sentinel Hub has rate limits
CACHE_DIR       = ".cache_topology"
CACHE_TTL       = 43200       # 12 hours (satellite data doesn't change intraday)
OUTPUT_DIR      = "topology"
HTTP_TIMEOUT    = 30          # Sentinel Hub can be slow for large areas

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CACHE
# =========================
def _cache_get(key: str) -> Any | None:
    path = os.path.join(CACHE_DIR, key.replace("/","_").replace(":","_") + ".pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            ts, data = pickle.load(f)
        if time.time() - ts < CACHE_TTL:
            return data
    except Exception:
        pass
    return None


def _cache_set(key: str, data: Any) -> None:
    path = os.path.join(CACHE_DIR, key.replace("/","_").replace(":","_") + ".pkl")
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump((time.time(), data), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as exc:
        log.debug("Cache write failed: %s", exc)
        try:
            os.remove(tmp)
        except OSError:
            pass

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
            "grant_type":    "client_credentials",
            "client_id":     CDSE_CLIENT_ID,
            "client_secret": CDSE_CLIENT_SECRET,
        },
        timeout=15,
    )
    resp.raise_for_status()
    token_data = resp.json()
    _token_cache["token"]      = token_data["access_token"]
    _token_cache["expires_at"] = now + int(token_data.get("expires_in", 600))
    log.debug("Copernicus token refreshed, valid for %ds", token_data.get("expires_in", 600))
    return _token_cache["token"]


def _build_session() -> requests.Session:
    retry = Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "QHDALabs-WildfirePL-Sentinel/1.0"})
    return session

HTTP = _build_session()

# =========================
# EVALSCRIPT
# =========================
# Sentinel-2 L2A evalscript for NDWI (Gao 1996)
# NDWI = (B03 - B08) / (B03 + B08)
# Excludes clouds, cloud shadows, snow, saturated pixels via SCL mask.
# Returns NDWI value per valid pixel.
EVALSCRIPT_NDWI = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "data",     bands: 1, sampleType: "FLOAT32" },
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
  var d = s.B03 + s.B08;
  var ndwi = (d === 0) ? 0.0 : (s.B03 - s.B08) / d;
  return { data: [ndwi], dataMask: [1] };
}
"""

# =========================
# NDWI FETCH
# =========================
def _node_bbox(lat: float, lon: float, radius_deg: float = BOX_RADIUS_DEG) -> dict:
    """Build bounding box geometry for a node centroid."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - radius_deg, lat - radius_deg],
            [lon + radius_deg, lat - radius_deg],
            [lon + radius_deg, lat + radius_deg],
            [lon - radius_deg, lat + radius_deg],
            [lon - radius_deg, lat - radius_deg],
        ]]
    }


def fetch_ndwi_timeseries(node: dict, days: int = NDWI_DAYS) -> dict | None:
    """
    Fetch NDWI time series for a node from Sentinel Hub Statistical API.

    Returns dict:
      node_id, node_name, dates, ndwi_values, ndwi_valid_pixels,
      ndwi_mean_30d, ndwi_min_30d, ndwi_trend_14d, data_source="sentinel2_L2A"
    Returns None on failure (API error, no cloud-free data, etc.)
    """
    cache_key = f"ndwi_sentinel_{node['id']}_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        log.debug("NDWI cache hit: %s", node["name"])
        return cached

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    request_body = {
        "input": {
            "bounds": {
                "geometry": _node_bbox(node["lat"], node["lon"]),
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "mosaickingOrder": "leastCC",   # least cloud cover first
                    "maxCloudCoverage": 85,
                },
            }],
        },
        "aggregation": {
            "timeRange": {
                "from": start_dt.strftime("%Y-%m-%dT00:00:00Z"),
                "to":   end_dt.strftime("%Y-%m-%dT23:59:59Z"),
            },
            "aggregationInterval": {"of": "P10D"},  # 10-day intervals (matches Sentinel-2 revisit)
            "evalscript": EVALSCRIPT_NDWI,
            "resx": 0.001,  # ~70m in CRS84 degrees — sufficient for district averages
            "resy": 0.001,
        },

    }

    try:
        token = get_access_token()
        resp = HTTP.post(
            CDSE_STATS_URL,
            json=request_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
        # DEBUG: log raw response structure on first successful call
        if not getattr(fetch_ndwi_timeseries, '_logged_structure', False):
            fetch_ndwi_timeseries._logged_structure = True
            import json as _json
            log.debug("Sentinel API response structure:\n%s",
                      _json.dumps(raw, indent=2)[:1200])
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response else ""
        log.warning("Sentinel API HTTP error for %s: %s\nResponse body: %s",
                    node["name"], exc, body[:1000])
        return None
    except Exception as exc:
        log.warning("Sentinel API failed for %s: %s", node["name"], exc)
        return None

    # Parse response
    intervals = raw.get("data", [])
    if not intervals:
        log.warning("No Sentinel data for %s (all cloudy?)", node["name"])
        return None

    dates, ndwi_vals, valid_px = [], [], []
    for interval in intervals:
        # Sentinel Hub Statistical API response structure (without "calculations" field):
        # outputs -> data -> bands -> B0 -> stats
        outputs = interval.get("outputs", {})

        # Try standard bands path first
        # CDSE Statistical API response path:
        # outputs -> data -> bands -> B0 -> stats
        try:
            band_stats = outputs["data"]["bands"]["B0"]["stats"]
        except (KeyError, TypeError):
            continue

        mean    = band_stats.get("mean")
        count   = band_stats.get("sampleCount", 0)
        no_data = band_stats.get("noDataCount", 0)

        # Skip intervals with < 10% valid (cloud-free) pixels
        total = count + no_data
        valid_fraction = (count - no_data) / count if count > 0 else 0.0
        if valid_fraction < 0.10 or mean is None or mean != mean:  # NaN check
            continue

        date_str = interval.get("interval", {}).get("from", "")[:10]
        dates.append(date_str)
        ndwi_vals.append(round(float(mean), 4))
        valid_px.append(round(valid_fraction, 3))

    if not dates:
        log.warning("All intervals cloudy for %s — no valid NDWI data", node["name"])
        return None

    # Compute summary statistics
    arr = np.array(ndwi_vals)
    ndwi_mean = round(float(np.mean(arr)), 4)
    ndwi_min  = round(float(np.min(arr)), 4)

    # 14-day trend: slope of NDWI (negative = drying out = rising fire risk)
    if len(arr) >= 2:
        x = np.arange(len(arr), dtype=float)
        trend = float(np.polyfit(x, arr, 1)[0])
        ndwi_trend_14d = round(trend, 5)
    else:
        ndwi_trend_14d = 0.0

    # Convert NDWI to stress score [0,1]: lower NDWI = higher stress.
    # Calibrated to actual Sentinel-2 values for Polish mixed/pine forests:
    #   ndwi >= -0.35  -> stress = 0.0  (well watered, healthy canopy)
    #   ndwi <= -0.70  -> stress = 1.0  (severely dry, elevated fire risk)
    # Values from operational data (RDLP Wrocław, May 2026).
    NDWI_HEALTHY  = -0.35
    NDWI_CRITICAL = -0.70
    ndwi_range = NDWI_HEALTHY - NDWI_CRITICAL   # 0.35
    ndwi_stress_latest = float(np.clip(
        (NDWI_HEALTHY - ndwi_vals[-1]) / ndwi_range, 0.0, 1.0
    ))

    result = {
        "node_id":             node["id"],
        "node_name":           node["name"],
        "data_source":         "sentinel2_L2A",
        "fetch_date":          datetime.now(timezone.utc).isoformat(),
        "dates":               dates,
        "ndwi_values":         ndwi_vals,
        "valid_pixel_fraction":valid_px,
        "ndwi_mean_30d":       ndwi_mean,
        "ndwi_min_30d":        ndwi_min,
        "ndwi_latest":         ndwi_vals[-1],
        "ndwi_trend_14d":      ndwi_trend_14d,   # neg = drying, pos = recovering
        "ndwi_stress_latest":  round(ndwi_stress_latest, 4),
        "n_observations":      len(dates),
    }
    _cache_set(cache_key, result)
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
        nid: r["ndwi_stress_latest"]
        for nid, r in ndwi_results.items()
        if r is not None
    }
    if not stresses:
        return {}

    vals = list(stresses.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo

    if span < 0.01:
        # All nodes equally stressed — return raw values
        return {nid: v for nid, v in stresses.items()}

    return {
        nid: round((v - lo) / span, 4)
        for nid, v in stresses.items()
    }

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
        nid    = node["id"]
        result = ndwi_results.get(nid)

        if result is not None:
            # Real satellite data available
            node["ndwi_sentinel"] = result
            node["ndwi_stress_source"] = "sentinel2_L2A"
            node["ndwi_stress_latest"] = result["ndwi_stress_latest"]
            node["ndwi_stress_normalised"] = normalised.get(nid, result["ndwi_stress_latest"])
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
        node_js.append({
            "id":              n["id"],
            "name":            n["name"],
            "lat":             n["lat"],
            "lon":             n["lon"],
            "eco":             n["eco"],
            "drought_days":    n.get("drought_days", 0),
            "ndwi_stress":     n.get("ndwi_stress_latest", 0.0),
            "ndwi_norm":       n.get("ndwi_stress_normalised", 0.0),
            "ndwi_source":     n.get("ndwi_stress_source", "proxy"),
            "ndwi_trend":      n.get("ndwi_trend_14d", 0.0),
            "ndwi_latest":     ndwi_s.get("ndwi_latest"),
            "n_obs":           ndwi_s.get("n_observations", 0),
            "network_stress":  n.get("network_stress", 0.0),
            "temp_max":        latest.get("temp_max"),
            "rh_min":          latest.get("rh_min"),
            "wind_max":        latest.get("wind_max"),
            "neighbors":       [nb["id"] for nb in graph.get(n["id"], [])],
        })

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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

# =========================
# MAIN PIPELINE
# =========================
def run_sentinel_pipeline(
    nodes_json_path: str = os.path.join(OUTPUT_DIR, "nodes.json"),
    graph_json_path: str = os.path.join(OUTPUT_DIR, "graph.json"),
    ndwi_days: int = NDWI_DAYS,
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

    # ── Verify credentials before starting ────────────────────────────────
    try:
        token = get_access_token()
        log.info("Copernicus authentication OK")
    except EnvironmentError as exc:
        log.error("%s", exc)
        raise

    # ── Fetch NDWI (parallel, MAX_WORKERS=4 to respect rate limits) ────────
    log.info("Fetching Sentinel-2 NDWI for %d nodes (%dd window)...", len(nodes), ndwi_days)
    ndwi_results: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_ndwi_timeseries, node, ndwi_days): node["id"]
            for node in nodes
        }
        for future in as_completed(futures):
            nid = futures[future]
            try:
                ndwi_results[nid] = future.result()
            except Exception as exc:
                log.warning("NDWI failed for %s: %s", nid, exc)
                ndwi_results[nid] = None

    ok      = sum(1 for v in ndwi_results.values() if v is not None)
    cloudy  = len(nodes) - ok
    log.info("NDWI fetch complete: %d/%d nodes with satellite data (%d cloudy/failed)",
             ok, len(nodes), cloudy)

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
    with open(ndwi_path, "w", encoding="utf-8") as f:
        json.dump({
            "version":      "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ndwi_days":    ndwi_days,
            "nodes_ok":     ok,
            "nodes_cloudy": cloudy,
            "results":      {k: v for k, v in ndwi_results.items() if v},
        }, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", ndwi_path)

    # nodes_enriched.json — merged topology + satellite
    enriched_path = os.path.join(OUTPUT_DIR, "nodes_enriched.json")
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump({
            "version":      "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_sources": ["open_meteo_archive", "sentinel2_L2A"],
            "nodes":        enriched_nodes,
        }, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", enriched_path)

    # network_map_v2.html
    map_path = os.path.join(OUTPUT_DIR, "network_map_v2.html")
    generate_map_v2(enriched_nodes, graph, map_path)
    log.info("Saved %s", map_path)

    # ── Console summary ────────────────────────────────────────────────────
    _print_summary(enriched_nodes)

    return ndwi_results


def _print_summary(nodes: list[dict]) -> None:
    log.info("\n%s", "=" * 72)
    log.info("NDWI Sentinel-2 Summary — RDLP Wrocław")
    log.info("%s", "=" * 72)
    log.info("%-24s %-8s %-8s %-10s %-10s %s",
             "Nadleśnictwo", "NDWI", "Stres%", "Trend14d", "Susza", "Źródło")
    log.info("%s", "-" * 72)

    sorted_nodes = sorted(
        nodes,
        key=lambda n: n.get("ndwi_stress_normalised", 0),
        reverse=True,
    )
    for n in sorted_nodes:
        ndwi_s = n.get("ndwi_sentinel", {})
        ndwi   = ndwi_s.get("ndwi_latest", "N/A")
        stress = n.get("ndwi_stress_normalised", 0) * 100
        trend  = n.get("ndwi_trend_14d", 0)
        trend_s= f"{trend:+.4f}" if trend else "N/A"
        src    = "SAT" if n.get("ndwi_stress_source") == "sentinel2_L2A" else "proxy"
        tier   = "KRYT" if stress > 70 else "WYS " if stress > 50 else "UMIA" if stress > 30 else "LOW "
        log.info("%-24s %-8s %6.1f%%  %-10s %4dd [%s] %s",
                 n["name"][:24], str(ndwi)[:7], stress,
                 trend_s, n.get("drought_days", 0), tier, src)
    log.info("%s", "=" * 72)
    log.info("Next: run qhdalabs_wildfire_qte_v1.py (Step 3 — Quantum Temporal Encoder)")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_sentinel_pipeline()
