# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 1 — Forest Network Topology & Weather History
# File          : qhdalabs_wildfire_topology_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Builds the "mycelium network" — the spatial graph of 33 nadleśnictwa
# (forest districts) of RDLP Wrocław (Lower Silesia pilot).
#
# Each node = one nadleśnictwo with:
#   - centroid coordinates (lat/lon)
#   - dominant ecosystem type
#   - list of geographic neighbors (adjacency graph)
#   - 14-day weather history from Open-Meteo archive
#   - daily summaries: temp, RH, wind, rain, VPD, soil moisture
#
# Outputs
# -----------------------------------------------------------------------------
#   topology/nodes.json     — all nodes with metadata
#   topology/graph.json     — adjacency list with distances
#   topology/weather_YYYYMMDD.json — weather snapshots per run
#   topology/network_map.html — interactive Leaflet visualization
#
# Data sources
# -----------------------------------------------------------------------------
#   Nadleśnictwa: embedded centroids (CC0 data from gis.openforestdata.pl)
#     To use actual polygon boundaries: download Nadleśnictwa.zip from
#     https://gis.openforestdata.pl/layers/geonode:nadlesnictwa_wgs84
#     and pass --shapefile path/to/nadlesnictwa.shp
#   Weather: Open-Meteo archive API (free, no key required)
#             https://archive-api.open-meteo.com/v1/archive
#   Future: Sentinel-2 NDVI/NDWI via Copernicus Data Space API (Step 2)
#
# Dependencies
# -----------------------------------------------------------------------------
#   numpy, requests
#   Optional: geopandas, shapely (for shapefile-based adjacency)
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
from dataclasses import dataclass, field, asdict
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
RDLP_NAME        = "Wrocław"
WEATHER_DAYS     = 14          # days of history to fetch
NEIGHBOR_KM      = 60.0        # max centroid distance to be considered neighbor
MAX_WORKERS      = 8
CACHE_DIR        = ".cache_topology"
CACHE_TTL        = 21600       # 6 hours
OUTPUT_DIR       = "topology"
HTTP_TIMEOUT     = 15

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# ECOSYSTEM FIRE RISK FACTORS
# =========================
# Multiplier applied to base FWI score based on dominant ecosystem.
# Source: LP fire statistics, Zabkiewicz 1988, EFFIS EU forest type data.
# pine / pine_wetland: highest risk (resinous, dry litter)
# mixed: moderate
# spruce_mountain: lower (wetter microclimate) BUT catastrophic when ignited
ECO_RISK_MULTIPLIER: dict[str, float] = {
    "pine":             1.30,
    "pine_wetland":     1.10,
    "mixed":            1.00,
    "spruce_mountain":  0.80,
}

# =========================
# NADLEŚNICTWA — RDLP WROCŁAW
# =========================
# Source: LP official portal + BDL (bdl.lasy.gov.pl), encoded as centroids.
# Coordinate accuracy: ±2 km (sufficient for weather API queries).
# For polygon-exact adjacency: load shapefile from openforestdata.pl.
#
# eco types: pine | pine_wetland | mixed | spruce_mountain
RDLP_WROCLAW_NODES: list[dict] = [
    {"id": "bardo_slaskie",  "name": "Bardo Śląskie",    "lat": 50.514, "lon": 16.749, "eco": "mixed"},
    {"id": "boleslawiec",    "name": "Bolesławiec",       "lat": 51.264, "lon": 15.559, "eco": "pine"},
    {"id": "bystrzyca",      "name": "Bystrzyca Kłodzka", "lat": 50.298, "lon": 16.649, "eco": "spruce_mountain"},
    {"id": "gory_stolowe",   "name": "Góry Stołowe",      "lat": 50.431, "lon": 16.364, "eco": "spruce_mountain"},
    {"id": "jawor",          "name": "Jawor",             "lat": 51.041, "lon": 16.197, "eco": "mixed"},
    {"id": "jugow",          "name": "Jugów",             "lat": 50.573, "lon": 16.555, "eco": "spruce_mountain"},
    {"id": "kamienna_gora",  "name": "Kamienna Góra",     "lat": 50.778, "lon": 16.039, "eco": "spruce_mountain"},
    {"id": "klodzko",        "name": "Kłodzko",           "lat": 50.437, "lon": 16.659, "eco": "spruce_mountain"},
    {"id": "ladek",          "name": "Lądek-Zdrój",       "lat": 50.342, "lon": 16.884, "eco": "spruce_mountain"},
    {"id": "legnica",        "name": "Legnica",           "lat": 51.211, "lon": 16.156, "eco": "mixed"},
    {"id": "lesna",          "name": "Leśna",             "lat": 51.001, "lon": 15.268, "eco": "pine"},
    {"id": "lubin",          "name": "Lubin",             "lat": 51.401, "lon": 16.201, "eco": "mixed"},
    {"id": "lwowek",         "name": "Lwówek Śląski",     "lat": 51.107, "lon": 15.593, "eco": "mixed"},
    {"id": "miekinia",       "name": "Miękinia",          "lat": 51.163, "lon": 16.713, "eco": "mixed"},
    {"id": "milicz",         "name": "Milicz",            "lat": 51.531, "lon": 17.284, "eco": "pine_wetland"},
    {"id": "mysliborskie",   "name": "Myśliborskie",      "lat": 51.383, "lon": 15.128, "eco": "pine"},
    {"id": "olesnica",       "name": "Oleśnica Śląska",   "lat": 50.880, "lon": 16.880, "eco": "mixed"},
    {"id": "olawa",          "name": "Oława",             "lat": 50.939, "lon": 17.300, "eco": "mixed"},
    {"id": "piszowice",      "name": "Piszowice",         "lat": 51.026, "lon": 15.867, "eco": "pine"},
    {"id": "prudnik",        "name": "Prudnik",           "lat": 50.323, "lon": 17.580, "eco": "mixed"},
    {"id": "ruszow",         "name": "Ruszów",            "lat": 51.449, "lon": 14.955, "eco": "pine"},
    {"id": "rychtal",        "name": "Rychtal",           "lat": 51.096, "lon": 17.773, "eco": "pine_wetland"},
    {"id": "sycow",          "name": "Syców",             "lat": 51.307, "lon": 17.718, "eco": "mixed"},
    {"id": "szklarska",      "name": "Szklarska Poręba",  "lat": 50.828, "lon": 15.523, "eco": "spruce_mountain"},
    {"id": "swidnica",       "name": "Świdnica",          "lat": 50.842, "lon": 16.488, "eco": "mixed"},
    {"id": "swietoszow",     "name": "Świętoszów",        "lat": 51.607, "lon": 15.353, "eco": "pine"},
    {"id": "walbrzych",      "name": "Wałbrzych",         "lat": 50.771, "lon": 16.284, "eco": "spruce_mountain"},
    {"id": "wegliniec",      "name": "Węgliniec",         "lat": 51.274, "lon": 15.207, "eco": "pine"},
    {"id": "wolow",          "name": "Wołów",             "lat": 51.338, "lon": 16.637, "eco": "mixed"},
    {"id": "wroclaw",        "name": "Wrocław",           "lat": 51.097, "lon": 17.033, "eco": "mixed"},
    {"id": "zlotoryja",      "name": "Złotoryja",         "lat": 51.125, "lon": 15.919, "eco": "mixed"},
    {"id": "zmigrod",        "name": "Żmigród",           "lat": 51.475, "lon": 16.906, "eco": "pine_wetland"},
    {"id": "zgorzelec",      "name": "Zgorzelec",         "lat": 51.155, "lon": 14.999, "eco": "pine"},
]

# =========================
# HTTP SESSION
# =========================
def _build_session() -> requests.Session:
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "QHDALabs-WildfirePL-Topology/1.0"})
    return session

HTTP = _build_session()

# =========================
# CACHE
# =========================
def _cache_get(key: str) -> Any | None:
    path = os.path.join(CACHE_DIR, key.replace("/", "_").replace(":", "_") + ".pkl")
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
    path = os.path.join(CACHE_DIR, key.replace("/", "_").replace(":", "_") + ".pkl")
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
# GEOMETRY
# =========================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.asin(math.sqrt(a))


def build_adjacency_graph(
    nodes: list[dict],
    threshold_km: float = NEIGHBOR_KM,
) -> dict[str, list[dict]]:
    """
    Build spatial adjacency graph from centroids.

    Two nadleśnictwa are neighbors if their centroids are within threshold_km.
    Default 60 km covers typical nadleśnictwo radius (15-30 km) with overlap.

    Returns dict: node_id -> [{id, name, dist_km}, ...]  sorted by distance.

    Note: For polygon-exact adjacency (shared boundary detection), load a
    shapefile and use shapely.touches() instead. This centroid approximation
    is sufficient for signal propagation in the mycelium network.
    """
    graph: dict[str, list[dict]] = {}
    for a in nodes:
        neighbors = []
        for b in nodes:
            if a["id"] == b["id"]:
                continue
            d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= threshold_km:
                neighbors.append({
                    "id": b["id"],
                    "name": b["name"],
                    "dist_km": round(d, 1),
                    "eco": b["eco"],
                })
        neighbors.sort(key=lambda x: x["dist_km"])
        graph[a["id"]] = neighbors

    # Log connectivity stats
    counts = [len(v) for v in graph.values()]
    log.info(
        "Adjacency graph: %d nodes, avg %.1f neighbors (min %d, max %d)",
        len(nodes), sum(counts) / len(counts), min(counts), max(counts),
    )
    isolated = [nid for nid, nbrs in graph.items() if len(nbrs) == 0]
    if isolated:
        log.warning("Isolated nodes (no neighbors): %s", isolated)

    return graph

# =========================
# WEATHER HISTORY
# =========================
def fetch_weather_history(node: dict, days: int = WEATHER_DAYS) -> dict | None:
    """
    Fetch daily weather summary for the past `days` days from Open-Meteo archive.

    Returns dict with daily arrays: dates, temp_max, temp_mean, rh_min, rh_mean,
    wind_mean, wind_max, rain_total, soil_min, vpd_mean.
    Returns None on failure.
    """
    lat, lon = node["lat"], node["lon"]
    key = f"weather_history_{lat:.3f}_{lon:.3f}_{days}d"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    end_date   = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days - 1)

    # Open-Meteo archive API — free, no key required, covers 1940-present
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,precipitation,"
            "soil_moisture_0_to_1cm,vapour_pressure_deficit"
        ),
        "timezone": "Europe/Warsaw",
    }
    try:
        resp = HTTP.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        log.warning("Weather history failed for %s: %s", node["name"], exc)
        return None

    hourly = raw.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        log.warning("Empty weather response for %s", node["name"])
        return None

    # Aggregate hourly → daily summaries
    def safe(v: Any) -> float:
        return float(v) if v is not None else float("nan")

    # Group by date
    from collections import defaultdict
    daily: dict[str, dict] = defaultdict(lambda: {
        "temps": [], "rhs": [], "winds": [], "rains": [],
        "soils": [], "vpds": []
    })

    for i, t in enumerate(times):
        date = t[:10]  # "YYYY-MM-DD"
        daily[date]["temps"].append(safe(hourly.get("temperature_2m", [None]*len(times))[i]))
        daily[date]["rhs"].append(safe(hourly.get("relative_humidity_2m", [None]*len(times))[i]))
        daily[date]["winds"].append(safe(hourly.get("wind_speed_10m", [None]*len(times))[i]))
        daily[date]["rains"].append(safe(hourly.get("precipitation", [None]*len(times))[i]))
        daily[date]["soils"].append(safe(hourly.get("soil_moisture_0_to_1cm", [None]*len(times))[i]))
        daily[date]["vpds"].append(safe(hourly.get("vapour_pressure_deficit", [None]*len(times))[i]))

    def nanmean(lst: list) -> float:
        vals = [v for v in lst if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def nanmax(lst: list) -> float:
        vals = [v for v in lst if not math.isnan(v)]
        return max(vals) if vals else float("nan")

    def nanmin(lst: list) -> float:
        vals = [v for v in lst if not math.isnan(v)]
        return min(vals) if vals else float("nan")

    sorted_dates = sorted(daily.keys())
    result = {
        "node_id":    node["id"],
        "node_name":  node["name"],
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "dates":      sorted_dates,
        "temp_max":   [round(nanmax(daily[d]["temps"]), 1) for d in sorted_dates],
        "temp_mean":  [round(nanmean(daily[d]["temps"]), 1) for d in sorted_dates],
        "rh_min":     [round(nanmin(daily[d]["rhs"]), 1) for d in sorted_dates],
        "rh_mean":    [round(nanmean(daily[d]["rhs"]), 1) for d in sorted_dates],
        "wind_mean":  [round(nanmean(daily[d]["winds"]), 2) for d in sorted_dates],
        "wind_max":   [round(nanmax(daily[d]["winds"]), 2) for d in sorted_dates],
        "rain_total": [round(sum(v for v in daily[d]["rains"] if not math.isnan(v)), 2) for d in sorted_dates],
        "soil_min":   [round(nanmin(daily[d]["soils"]), 4) for d in sorted_dates],
        "soil_mean":  [round(nanmean(daily[d]["soils"]), 4) for d in sorted_dates],
        "vpd_mean":   [round(nanmean(daily[d]["vpds"]), 3) for d in sorted_dates],
    }
    _cache_set(key, result)
    return result


def compute_drought_days(weather: dict) -> int:
    """
    Estimate consecutive dry days ending today.
    A day is "dry" if rain_total < 2 mm (LP threshold for meaningful precipitation).
    Counts backwards from the most recent day.
    """
    rains = list(reversed(weather.get("rain_total", [])))
    drought = 0
    for r in rains:
        if math.isnan(r) or r < 2.0:
            drought += 1
        else:
            break
    return drought


def compute_ndwi_proxy(weather: dict) -> list[float]:
    """
    Proxy for vegetation water stress per day, range [0, 1].
    1 = maximally stressed/dry.

    Formula mirrors the physical basis of Sentinel-2 NDWI:
    lower soil moisture + higher VPD + lower RH = higher stress.

    This will be REPLACED by real Sentinel-2 NDWI in Step 2.
    Kept here as fallback when satellite data is unavailable.
    """
    result = []
    for i in range(len(weather.get("dates", []))):
        soil = weather["soil_mean"][i] if i < len(weather.get("soil_mean", [])) else 0.25
        vpd  = weather["vpd_mean"][i]  if i < len(weather.get("vpd_mean", []))  else 0.5
        rh   = weather["rh_mean"][i]   if i < len(weather.get("rh_mean", []))   else 60.0
        rain = weather["rain_total"][i] if i < len(weather.get("rain_total", [])) else 0.0

        if math.isnan(soil): soil = 0.25
        if math.isnan(vpd):  vpd  = 0.5
        if math.isnan(rh):   rh   = 60.0
        if math.isnan(rain): rain = 0.0

        vpd_n   = min(1.0, vpd / 3.5)
        soil_n  = max(0.0, min(1.0, 1.0 - soil / 0.35))
        rh_n    = max(0.0, (72.0 - rh) / 72.0)
        rain_r  = max(0.0, 1.0 - rain / 5.0)
        stress  = 0.35 * vpd_n + 0.35 * soil_n + 0.20 * rh_n + 0.10 * rain_r
        result.append(round(float(np.clip(stress, 0.0, 1.0)), 4))
    return result

# =========================
# NODE ENRICHMENT
# =========================
def enrich_node(node: dict, weather: dict | None) -> dict:
    """
    Combine static node metadata with dynamic weather history.
    Returns enriched node dict ready for JSON serialization.
    """
    enriched = dict(node)
    enriched["eco_risk_multiplier"] = ECO_RISK_MULTIPLIER.get(node["eco"], 1.0)

    if weather is not None:
        enriched["weather_history"] = weather
        enriched["drought_days"]    = compute_drought_days(weather)
        enriched["ndwi_proxy"]      = compute_ndwi_proxy(weather)
        # Latest-day summary (most recent data)
        enriched["latest"] = {
            "date":       weather["dates"][-1]  if weather["dates"]  else None,
            "temp_max":   weather["temp_max"][-1]  if weather["temp_max"]  else None,
            "rh_min":     weather["rh_min"][-1]    if weather["rh_min"]    else None,
            "wind_max":   weather["wind_max"][-1]  if weather["wind_max"]  else None,
            "rain_total": weather["rain_total"][-1] if weather["rain_total"] else None,
            "soil_min":   weather["soil_min"][-1]  if weather["soil_min"]  else None,
            "vpd_mean":   weather["vpd_mean"][-1]  if weather["vpd_mean"]  else None,
            "ndwi_stress": enriched["ndwi_proxy"][-1] if enriched["ndwi_proxy"] else None,
        }
        # 7-day NDWI trend: positive = stress increasing (drying out)
        ndwi = enriched["ndwi_proxy"]
        if len(ndwi) >= 7:
            enriched["ndwi_trend_7d"] = round(ndwi[-1] - ndwi[-7], 4)
        else:
            enriched["ndwi_trend_7d"] = None
    else:
        enriched["weather_history"] = None
        enriched["drought_days"]    = 0
        enriched["ndwi_proxy"]      = []
        enriched["latest"]          = {}
        enriched["ndwi_trend_7d"]   = None

    return enriched

# =========================
# MYCELIUM SIGNAL PROPAGATION
# =========================
def compute_network_stress(
    nodes_enriched: list[dict],
    graph: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Compute propagated stress signal through the mycelium network.

    Each node starts with its own ndwi_stress. Neighbors share 15% of their
    stress signal — if your neighbor is critically dry, your effective risk
    threshold lowers slightly even if local conditions are moderate.

    This models the biological reality: a dry front approaching from a
    neighboring forest increases fire risk before it arrives locally.

    Returns dict: node_id -> propagated_stress [0, 1]
    """
    # Base stress from latest NDWI proxy
    base: dict[str, float] = {}
    for n in nodes_enriched:
        latest = n.get("latest", {})
        base[n["id"]] = float(latest.get("ndwi_stress") or 0.0)

    # One propagation pass (biological: signal travels one hop per time step)
    propagated: dict[str, float] = {}
    for n in nodes_enriched:
        nid  = n["id"]
        own  = base[nid]
        nbrs = graph.get(nid, [])
        if not nbrs:
            propagated[nid] = own
            continue
        # Weighted average of neighbor stresses (inverse distance weighting)
        total_weight = 0.0
        neighbor_signal = 0.0
        for nb in nbrs:
            w = 1.0 / max(nb["dist_km"], 1.0)
            total_weight += w
            neighbor_signal += base.get(nb["id"], 0.0) * w
        avg_neighbor = neighbor_signal / total_weight if total_weight > 0 else 0.0

        # Propagated = own stress + 15% of mean neighbor stress
        # Capped at 1.0
        propagated[nid] = float(np.clip(own + 0.15 * avg_neighbor, 0.0, 1.0))

    return propagated

# =========================
# MAIN PIPELINE
# =========================
def run_topology_pipeline(
    nodes: list[dict] | None = None,
    weather_days: int = WEATHER_DAYS,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Full Step 1 pipeline:
      1. Build adjacency graph
      2. Fetch weather history for all nodes (parallel)
      3. Enrich nodes with weather + NDWI proxy + drought days
      4. Compute mycelium propagation
      5. Save outputs

    Returns (enriched_nodes, adjacency_graph)
    """
    if nodes is None:
        nodes = RDLP_WROCLAW_NODES

    log.info("=== QHDALabs Wildfire — Step 1: Topology & Weather ===")
    log.info("Region: RDLP %s | Nodes: %d | Weather window: %d days",
             RDLP_NAME, len(nodes), weather_days)

    # ── Step 1a: Adjacency graph ──────────────────────────────────────────
    graph = build_adjacency_graph(nodes)

    # ── Step 1b: Weather history (parallel) ──────────────────────────────
    log.info("Fetching %d-day weather history for %d nodes ...", weather_days, len(nodes))
    weather_map: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_weather_history, node, weather_days): node["id"]
            for node in nodes
        }
        for future in as_completed(futures):
            nid = futures[future]
            try:
                weather_map[nid] = future.result()
            except Exception as exc:
                log.warning("Weather failed for %s: %s", nid, exc)
                weather_map[nid] = None

    ok  = sum(1 for v in weather_map.values() if v is not None)
    log.info("Weather fetch complete: %d/%d nodes OK", ok, len(nodes))

    # ── Step 1c: Enrich nodes ─────────────────────────────────────────────
    enriched = [enrich_node(n, weather_map.get(n["id"])) for n in nodes]

    # ── Step 1d: Mycelium propagation ─────────────────────────────────────
    propagated_stress = compute_network_stress(enriched, graph)
    for n in enriched:
        n["network_stress"] = round(propagated_stress[n["id"]], 4)

    # ── Step 1e: Save outputs ─────────────────────────────────────────────
    _save_outputs(enriched, graph)

    # ── Step 1f: Console summary ──────────────────────────────────────────
    _print_summary(enriched, graph)

    return enriched, graph


def _save_outputs(enriched: list[dict], graph: dict) -> None:
    """Save nodes.json, graph.json, and HTML map."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    # nodes.json — full enriched data
    nodes_path = os.path.join(OUTPUT_DIR, "nodes.json")
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0.0",
            "rdlp": RDLP_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "weather_days": WEATHER_DAYS,
            "nodes": enriched,
        }, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", nodes_path)

    # graph.json — adjacency list
    graph_path = os.path.join(OUTPUT_DIR, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0.0",
            "rdlp": RDLP_NAME,
            "neighbor_threshold_km": NEIGHBOR_KM,
            "adjacency": graph,
        }, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", graph_path)

    # HTML map
    map_path = os.path.join(OUTPUT_DIR, "network_map.html")
    _generate_map(enriched, graph, map_path)
    log.info("Saved %s", map_path)


def _generate_map(
    enriched: list[dict],
    graph: dict,
    path: str,
) -> None:
    """Generate interactive Leaflet map of the mycelium network."""

    # Prepare node data for JS
    node_js = []
    for n in enriched:
        latest = n.get("latest", {})
        node_js.append({
            "id":            n["id"],
            "name":          n["name"],
            "lat":           n["lat"],
            "lon":           n["lon"],
            "eco":           n["eco"],
            "eco_risk":      n["eco_risk_multiplier"],
            "drought_days":  n.get("drought_days", 0),
            "ndwi_stress":   latest.get("ndwi_stress") or 0.0,
            "network_stress": n.get("network_stress", 0.0),
            "ndwi_trend":    n.get("ndwi_trend_7d") or 0.0,
            "temp_max":      latest.get("temp_max"),
            "rh_min":        latest.get("rh_min"),
            "wind_max":      latest.get("wind_max"),
            "rain":          latest.get("rain_total"),
            "neighbors":     [nb["id"] for nb in graph.get(n["id"], [])],
        })

    node_data = json.dumps(node_js, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>QHDALabs — Sieć Grzybni RDLP Wrocław</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html,body,#map {{ margin:0; height:100vh; font-family:'Courier New',monospace; background:#0a0f0a; }}
    #panel {{
      position:absolute; top:12px; left:12px; z-index:1000;
      background:rgba(10,20,10,0.92); color:#7fff7f; padding:14px 18px;
      border:1px solid #2a5a2a; border-radius:4px; font-size:12px;
      max-width:260px; box-shadow:0 0 20px rgba(0,180,0,0.15);
    }}
    #panel h2 {{ margin:0 0 8px; font-size:13px; color:#afffaf; letter-spacing:2px; text-transform:uppercase; }}
    #panel .legend-row {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
    #panel .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
    .eco-badge {{
      display:inline-block; padding:1px 5px; border-radius:2px;
      font-size:10px; font-family:'Courier New',monospace;
      background:rgba(0,80,0,0.5); color:#7fff7f; border:1px solid #2a5a2a;
    }}
  </style>
</head>
<body>
<div id="panel">
  <h2>🍄 Sieć Grzybni</h2>
  <div style="color:#5fa; margin-bottom:8px; font-size:11px">RDLP Wrocław — Pilot</div>
  <div class="legend-row"><div class="dot" style="background:#ff2222"></div> KRYTYCZNY stres (&gt;0.70)</div>
  <div class="legend-row"><div class="dot" style="background:#ff8800"></div> WYSOKI (0.50–0.70)</div>
  <div class="legend-row"><div class="dot" style="background:#ffcc00"></div> UMIARKOWANY (0.30–0.50)</div>
  <div class="legend-row"><div class="dot" style="background:#33cc55"></div> NISKI (&lt;0.30)</div>
  <div style="margin-top:8px; color:#5a7a5a; font-size:10px">
    Linie = połączenia sieciowe<br>
    Kliknij węzeł po dane
  </div>
</div>
<div id="map"></div>
<script>
const NODES = {node_data};

const map = L.map('map', {{zoomControl:true}}).setView([51.0, 16.5], 8);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'© OpenStreetMap © CARTO', maxZoom:18
}}).addTo(map);

function stressColor(s) {{
  if (s > 0.70) return '#ff2222';
  if (s > 0.50) return '#ff8800';
  if (s > 0.30) return '#ffcc00';
  return '#33cc55';
}}

function ecoLabel(eco) {{
  const labels = {{
    pine:'Bór sosnowy', pine_wetland:'Bór/mokradło',
    mixed:'Las mieszany', spruce_mountain:'Świerkowy górski'
  }};
  return labels[eco] || eco;
}}

// Draw network edges first (below nodes)
const nodeMap = {{}};
NODES.forEach(n => nodeMap[n.id] = n);

const drawnEdges = new Set();
NODES.forEach(n => {{
  n.neighbors.forEach(nbId => {{
    const edgeKey = [n.id, nbId].sort().join('--');
    if (drawnEdges.has(edgeKey)) return;
    drawnEdges.add(edgeKey);
    const nb = nodeMap[nbId];
    if (!nb) return;
    const avgStress = (n.network_stress + nb.network_stress) / 2;
    L.polyline([[n.lat, n.lon],[nb.lat, nb.lon]], {{
      color: stressColor(avgStress),
      weight: 1.5,
      opacity: 0.35 + avgStress * 0.4,
    }}).addTo(map);
  }});
}});

// Draw nodes
NODES.forEach(n => {{
  const s = n.network_stress;
  const r = 8 + s * 14;
  const color = stressColor(s);

  const popup = `
    <b style="color:${{color}}">${{n.name}}</b><br>
    <span class="eco-badge">${{ecoLabel(n.eco)}}</span>
    <table style="margin-top:6px;font-size:11px;border-collapse:collapse">
      <tr><td>Stres NDWI</td><td><b>${{(n.ndwi_stress*100).toFixed(1)}}%</b></td></tr>
      <tr><td>Stres sieciowy</td><td><b>${{(n.network_stress*100).toFixed(1)}}%</b></td></tr>
      <tr><td>Trend 7d</td><td>${{n.ndwi_trend>=0?'+':''}}${{(n.ndwi_trend*100).toFixed(1)}}%</td></tr>
      <tr><td>Susza</td><td>${{n.drought_days}} dni</td></tr>
      <tr><td>Temp max</td><td>${{n.temp_max !== null ? n.temp_max+'°C' : 'N/A'}}</td></tr>
      <tr><td>RH min</td><td>${{n.rh_min !== null ? n.rh_min+'%' : 'N/A'}}</td></tr>
      <tr><td>Wiatr max</td><td>${{n.wind_max !== null ? n.wind_max+' m/s' : 'N/A'}}</td></tr>
      <tr><td>Deszcz</td><td>${{n.rain !== null ? n.rain+' mm' : 'N/A'}}</td></tr>
    </table>
  `;

  L.circleMarker([n.lat, n.lon], {{
    radius: r,
    color: color,
    fillColor: color,
    fillOpacity: 0.7 + s * 0.25,
    weight: 1.5,
  }}).addTo(map).bindPopup(popup);
}});
</script>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _print_summary(enriched: list[dict], graph: dict) -> None:
    """Print a compact summary table to console."""
    log.info("\n%s", "=" * 65)
    log.info("RDLP %s — Network Summary", RDLP_NAME)
    log.info("%s", "=" * 65)
    log.info("%-24s %-14s %6s %8s %8s", "Nadleśnictwo", "Ekosystem", "Susza", "NDWI", "Network")
    log.info("%s", "-" * 65)

    sorted_nodes = sorted(enriched, key=lambda n: n.get("network_stress", 0), reverse=True)
    for n in sorted_nodes:
        latest = n.get("latest", {})
        ndwi   = (latest.get("ndwi_stress") or 0.0) * 100
        net    = n.get("network_stress", 0.0) * 100
        tier   = ("KRYT" if net > 70 else "WYS " if net > 50 else "UMIA" if net > 30 else "LOW ")
        log.info(
            "%-24s %-14s %5dd %7.1f%% %7.1f%% [%s]",
            n["name"][:24], n["eco"][:14],
            n.get("drought_days", 0), ndwi, net, tier,
        )
    log.info("%s", "=" * 65)
    log.info("Outputs: %s/", OUTPUT_DIR)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    enriched_nodes, adjacency_graph = run_topology_pipeline()
    log.info("Step 1 complete. Next: run qhdalabs_wildfire_sentinel_v1.py (Step 2 — Sentinel-2 NDWI)")
