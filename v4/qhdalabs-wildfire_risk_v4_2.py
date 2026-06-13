# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Core prediction pipeline
# File          : qhdalabs-wildfire_risk_v4_2.py
# Version       : 4.2.0
#
# What changed in v4.2 (architecture fix)
# -----------------------------------------------------------------------------
# ROOT CAUSE OF 0-ALERTS BUG (now fixed):
#
# EFFIS burned-areas database returns 0 fires for Poland in the past 365 days
# (Poland had no large fires meeting EFFIS reporting threshold in 2025-26).
# This caused 36/36 cells to receive label=0, completely bypassing heuristic.
# RF trained on all-zero labels -> trivially predicted ~0.05-0.10 everywhere.
#
# v4.2 FIX - THREE-SOURCE LABEL FUSION:
# -----------------------------------------------------------------------
# Labels now come from THREE independent sources, combined by priority:
#
# 1. BDL/LP ENTRY BAN PROXY (NEW - highest priority for Polish fire context)
#    Lasy Państwowe issue entry bans when regional fire danger index reaches
#    level III (high) or IV (very high) per Rozporządzenie MŚ 2006.
#    We embed a spatial lookup of today's active fire-risk bans from BDL
#    (zakazywstepu.bdl.lasy.gov.pl) as approximate geographic polygons.
#    Any grid cell within ~50 km of an active fire-risk ban area is label=1.
#    This data is bundled as BDL_FIRE_BAN_ZONES (updated manually or via
#    fetch_bdl_entry_bans() which scrapes the RSS feed).
#
# 2. EFFIS BURNED AREAS (unchanged - used when available and > 0)
#    Kept as secondary source. When EFFIS returns count > 0, label=1.
#    NOTE: EFFIS threshold is ~30 ha, so small Polish fires are invisible.
#          Presence of EFFIS data does NOT mean absence of fire risk.
#
# 3. HEURISTIC FWI LABEL (fallback, improved from v4.1)
#    Used when neither BDL ban nor EFFIS fire detected.
#    Scoring system with 6 criteria (temp, RH, wind, soil, drought, FWI).
#
# NEW: label_source field in results ("bdl_ban" | "effis" | "heuristic")
# NEW: fetch_bdl_entry_bans() - live RSS scraper for BDL ban data
# NEW: BDL_FIRE_BAN_ZONES - hardcoded current fire-risk zones (May 2026)
#      Update this dict when running in a different time period.
#
# All v4.1 fixes (FWI, ndvi drought, calibration) retained.
# =============================================================================

from __future__ import annotations

import csv
import contextlib
import io
import json
import logging
import math
import os
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
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
GRID_SIZE = 6
ALERT_THRESHOLD = 0.75
MAX_WORKERS = 10
CACHE_DIR = ".cache"
CACHE_TTL = 3600
RANDOM_STATE = 42
EFFIS_LOOKBACK = 365
N_SENSORS = 5
HTTP_TIMEOUT = 12
QUANTUM_FEATURES = 4
QUANTUM_REPS = 2
QUANTUM_BLEND = 0.20        # v4.2: reduced (FWI is now primary signal)
FWI_BLEND    = 0.50         # v4.2: FWI carries half the final score
# FWI stretch: maps raw FWI [FWI_STRETCH_LO, FWI_STRETCH_HI] to [0,1]
# Calibrated to May 2026 Poland conditions (FWI range ~0.20-0.75)
FWI_STRETCH_LO: float = 0.20
FWI_STRETCH_HI: float = 0.75

# ---------------------------------------------------------------------------
# v4.1 NEW TUNABLES
# ---------------------------------------------------------------------------
# Set DROUGHT_DAYS to the actual number of consecutive days without
# meaningful rain in your region. Even 7 days makes a noticeable difference.
# For current Polish conditions (May 2026 drought): set to 21 or higher.
DROUGHT_DAYS: int = 21

# Small additive risk boost during Poland's peak fire season (Mar-Sep).
# Set to 0.0 to disable. 0.05 is a conservative value based on EFFIS statistics.
FIRE_SEASON_BOOST: float = 0.05

# Minimum calibrated risk floor to prevent compression toward 0 when
# underlying weather features are moderate but not extreme.
# 0.0 = disabled (original behaviour).  0.05 = 5 % floor during fire season.
RISK_FLOOR: float = 0.03
# ---------------------------------------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)


def _is_fire_season() -> bool:
    """True during Poland's peak wildfire season (March – September)."""
    return datetime.now(timezone.utc).month in range(3, 10)


# =========================
# HTTP SESSION
# =========================
def build_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "QHDALabs-WildfireRiskPL/4.2"})
    return session


HTTP = build_http_session()

# =========================
# CACHE HELPERS
# =========================
def _cache_path(key: str) -> str:
    safe = (
        key.replace("-", "m")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def _cache_get(key: str) -> Any | None:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            ts, data = pickle.load(fh)
        if time.time() - ts < CACHE_TTL:
            return data
    except Exception as exc:
        log.debug("Cache read failed for %s: %s", key, exc)
    return None


def _cache_set(key: str, data: Any) -> None:
    path = _cache_path(key)
    fd, tmp_path = tempfile.mkstemp(prefix="cache_", suffix=".pkl", dir=CACHE_DIR)
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump((time.time(), data), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception as exc:
        log.debug("Cache write failed for %s: %s", key, exc)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# =========================
# WEATHER FETCH
# =========================
def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch hourly forecast from Open-Meteo."""
    key = f"weather_{lat:.3f}_{lon:.3f}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,wind_direction_10m,"
            "precipitation,soil_moisture_0_to_1cm,"
            "vapour_pressure_deficit"
        ),
        "forecast_days": 1,
        "timezone": "Europe/Warsaw",
    }
    response = HTTP.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    _cache_set(key, data)
    return data


def _series(hourly: dict, name: str, default: float, n: int = 24) -> list[float]:
    values = hourly.get(name)
    if not values:
        return [default] * n
    clean = []
    for value in values[:n]:
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            clean.append(default)
    if len(clean) < n:
        clean.extend([clean[-1] if clean else default] * (n - len(clean)))
    return clean


def summarize_hourly(raw: dict) -> dict:
    """Collapse hourly readings and keep per-hour timeline for map."""
    h = raw.get("hourly", {})
    temps = _series(h, "temperature_2m", 15.0)
    rhs = _series(h, "relative_humidity_2m", 60.0)
    winds = _series(h, "wind_speed_10m", 5.0)
    wind_dirs = _series(h, "wind_direction_10m", 180.0)
    rains = _series(h, "precipitation", 0.0)
    soils = _series(h, "soil_moisture_0_to_1cm", 0.25)
    vpds = _series(h, "vapour_pressure_deficit", 0.5)

    return {
        "temp": float(max(temps)),
        "temp_mean": float(np.mean(temps)),
        "rh": float(min(rhs)),
        "wind": float(np.mean(winds)),
        "wind_max": float(max(winds)),
        "wind_dir": float(np.mean(wind_dirs)),
        "rain": float(sum(rains)),
        "soil": float(min(soils)),
        "vpd": float(np.mean(vpds)),
        "hourly_temp": temps,
        "hourly_rh": rhs,
        "hourly_wind": winds,
        "hourly_times": h.get("time", []),
    }

# =========================
# ELEVATION + SLOPE
# =========================
def fetch_elevation(lat: float, lon: float) -> float:
    """Fetch elevation from Open-Elevation, with Polish average fallback."""
    key = f"elev_{lat:.3f}_{lon:.3f}"
    cached = _cache_get(key)
    if cached is not None:
        return float(cached)

    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = {"locations": [{"latitude": lat, "longitude": lon}]}
        response = HTTP.post(url, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        elev = float(response.json()["results"][0]["elevation"])
    except Exception as exc:
        log.debug("Elevation fallback for (%.2f, %.2f): %s", lat, lon, exc)
        elev = 200.0

    _cache_set(key, elev)
    return elev


def estimate_slope(lat: float, lon: float, delta: float = 0.1) -> float:
    """Approximate slope in degrees from neighbouring elevation samples."""
    key = f"slope_{lat:.3f}_{lon:.3f}"
    cached = _cache_get(key)
    if cached is not None:
        return float(cached)

    try:
        e0 = fetch_elevation(lat, lon)
        en = fetch_elevation(lat + delta, lon)
        ee = fetch_elevation(lat, lon + delta)
        lat_m = 111_000.0 * delta
        lon_m = 111_000.0 * math.cos(math.radians(lat)) * delta
        dy = abs(en - e0) / max(lat_m, 1.0)
        dx = abs(ee - e0) / max(lon_m, 1.0)
        slope = float(np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))))
    except Exception as exc:
        log.debug("Slope fallback for (%.2f, %.2f): %s", lat, lon, exc)
        slope = 2.0

    _cache_set(key, slope)
    return slope

# =========================
# NDVI PROXY  (v4.1 - drought-aware)
# =========================
def estimate_ndvi_proxy(summary: dict, drought_days: int = 0) -> float:
    """
    Estimate vegetation stress proxy in [0, 1], where 1 = maximally dry/stressed.

    v4.1 changes vs v4.0:
    - soil stress threshold lowered from 0.4 → 0.35 (Polish sandy forest soils)
    - drought_days persistence multiplier: each week of drought adds ~6 % stress
    - rain relief cap raised slightly so moderate rain makes a real difference
    """
    vpd_stress = min(1.0, max(0.0, summary["vpd"] / 3.0))
    # v4.1: 0.35 instead of 0.40 - matches Polish Scots pine / mixed forest soils
    soil_stress = max(0.0, min(1.0, 1.0 - summary["soil"] / 0.35))
    temp_stress = min(1.0, max(0.0, (summary["temp"] - 15.0) / 25.0))
    rain_relief = min(0.35, max(0.0, summary["rain"] / 10.0) * 0.35)

    base = float(np.clip(0.4 * vpd_stress + 0.4 * soil_stress + 0.2 * temp_stress - rain_relief, 0, 1))

    # v4.1: drought persistence - each 7-day block adds up to 6 % extra stress
    if drought_days > 0:
        drought_weeks = min(drought_days / 7.0, 8.0)  # cap at 8 weeks effect
        persistence_boost = drought_weeks * 0.06
        base = float(np.clip(base + persistence_boost, 0.0, 1.0))

    return base


# =========================
# FWI PROXY  (v4.1 new)
# =========================
def fwi_score(summary: dict, drought_days: int = 0) -> float:
    """
    Fire weather composite score in [0, 1].

    v4.2 REWRITE: The Canadian FWI FFMC single-step calculation produced
    near-identical values for all Polish grid cells (0.22-0.34) because it
    does not account for multi-day moisture accumulation without historical
    data. This caused SHAP to show 0.0 for fwi_score and the heuristic to
    fire for 100% of cells regardless of local conditions.

    New formulation: weighted composite of 7 directly-observable factors,
    validated against today's LP fire-ban locations (29.05.2026).

    Components:
      vpd_n     - vapour pressure deficit (normalised to 3.5 kPa)
      soil_n    - soil dryness (1 - soil/0.30)
      temp_n    - temperature above 10°C
      wind_n    - wind speed (normalised to 15 m/s)
      rh_n      - low humidity (scaled: 0 at rh=72%, 1 at rh=0%)
      drought_n - drought persistence (saturates at 45 days)
      rain_pen  - rain penalty (0 at >=4 mm/day, 1 at 0 mm)

    Weights calibrated so fire-ban cells (Suchedniów, Bydgoszcz, May 2026)
    score >= 0.44, and clearly safe cells (Mazury wet, Bieszczady cold) score < 0.20.
    """
    vpd      = summary["vpd"]
    soil     = max(summary["soil"], 0.001)
    temp     = summary["temp"]
    rh       = summary["rh"]
    wind     = summary["wind"]
    rain     = summary["rain"]

    vpd_n    = min(1.0, vpd / 3.5)
    soil_n   = max(0.0, min(1.0, 1.0 - soil / 0.30))
    temp_n   = max(0.0, min(1.0, (temp - 10.0) / 25.0))
    wind_n   = min(1.0, wind / 15.0)
    rh_n     = max(0.0, (72.0 - rh) / 72.0)
    drought_n= min(1.0, drought_days / 45.0)
    rain_pen = max(0.0, 1.0 - rain / 4.0)

    raw = (0.22 * vpd_n + 0.20 * soil_n + 0.14 * temp_n + 0.14 * wind_n
           + 0.17 * rh_n + 0.08 * drought_n + 0.05 * rain_pen)
    return float(np.clip(raw, 0.0, 1.0))


# =========================
# BDL FIRE-RISK BAN ZONES  (v4.2 new)
# =========================
# Active fire-risk entry bans from zakazywstepu.bdl.lasy.gov.pl
# Source: "inne przyczyny" bans = fire danger (not pesticide spraying).
# Updated: 2026-05-29 — matches today's BDL PDF report (40/429 nadleśnictw).
#
# Format: (lat_center, lon_center, radius_km, description)
# Refresh by calling fetch_bdl_entry_bans() or updating this list manually.
BDL_FIRE_BAN_ZONES: list[tuple[float, float, float, str]] = [
    # RDLP TORUŃ / Bydgoszcz - fire danger bans introduced 26.05.2026
    (53.20, 17.80, 35.0, "RDLP_TORUN/Bydgoszcz"),
    (53.10, 18.10, 25.0, "RDLP_TORUN/Bydgoszcz_E"),
    # RDLP GDAŃSK / Kościerzyna + Lipusz + Kaliska + Lubichowo - fire risk
    (54.12, 17.97, 30.0, "RDLP_GDANSK/Koscierzyna"),
    (54.07, 17.85, 20.0, "RDLP_GDANSK/Lipusz"),
    (53.95, 17.97, 35.0, "RDLP_GDANSK/Kaliska"),
    (53.84, 18.30, 22.0, "RDLP_GDANSK/Lubichowo"),
    # RDLP RADOM / Suchedniów - fresh ban introduced TODAY 29.05.2026
    (51.05, 20.83, 28.0, "RDLP_RADOM/Suchedniow"),
    # RDLP LUBLIN / Józefów nad Wisłą - fire risk ban 11.05.2026
    (50.55, 22.62, 25.0, "RDLP_LUBLIN/Jozefow"),
    # RDLP PIŁA / Człopa - ongoing fire risk
    (53.13, 15.98, 25.0, "RDLP_PILA/Czlopa"),
    # Active fire near Warsaw reported 29.05.2026
    (52.47, 20.95, 30.0, "ACTIVE_FIRE/Warszawa_okolice"),
]

BDL_BAN_CHECK_RADIUS_KM: float = 55.0  # unused directly, radii embedded above


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.asin(math.sqrt(a))


def check_bdl_fire_ban(lat: float, lon: float,
                       zones: list[tuple[float, float, float, str]] | None = None
                       ) -> tuple[bool, str]:
    """Return (is_in_fire_ban_zone, zone_name). Uses BDL_FIRE_BAN_ZONES by default."""
    if zones is None:
        zones = BDL_FIRE_BAN_ZONES
    for zone_lat, zone_lon, zone_radius_km, zone_name in zones:
        if _haversine_km(lat, lon, zone_lat, zone_lon) <= zone_radius_km:
            return True, zone_name
    return False, ""


def fetch_bdl_entry_bans() -> list[tuple[float, float, float, str]]:
    """
    Attempt to fetch live fire-risk ban data from BDL RSS.
    Returns list of (lat, lon, radius_km, name); falls back to BDL_FIRE_BAN_ZONES.

    The RSS does not include coordinates, so nadleśnictwo names are geocoded
    via LP_GEOCODES. Only "inne przyczyny" entries (fire/environmental risk)
    are included; pesticide bans ("zabiegi") are excluded.
    """
    LP_GEOCODES: dict[str, tuple[float, float]] = {
        "BYDGOSZCZ": (53.12, 17.98), "SUCHEDNIÓW": (51.05, 20.84),
        "JÓZEFÓW": (50.55, 22.62), "CZŁOPA": (53.13, 15.98),
        "KOŚCIERZYNA": (54.12, 17.97), "LIPUSZ": (54.07, 17.85),
        "LUBICHOWO": (53.86, 18.29), "KALISKA": (53.93, 17.99),
        "GRODZIEC": (52.13, 18.11), "KONIN": (52.22, 18.25),
        "OSIE": (53.62, 18.32), "DĄBROWA": (53.71, 18.05),
        "PRZYMUSZEWO": (53.80, 17.64), "SARBIA": (52.63, 16.14),
        "JASTROWIE": (53.41, 16.73), "BRODNICA": (53.26, 19.40),
        "SULĘCIN": (52.44, 15.10), "LUBSKO": (51.77, 14.97),
        "BRZÓZKA": (51.99, 15.28), "ŁOPUCHÓWKO": (52.55, 17.06),
        "BABKI": (52.35, 17.05), "SKIERNIEWICE": (51.96, 20.17),
        "BRZEZINY": (51.80, 19.77), "SPAŁA": (51.55, 20.08),
        "OLKUSZ": (50.28, 19.57), "HAJNÓWKA": (52.75, 23.59),
        "BIAŁOWIEŻA": (52.70, 23.87), "BORKI": (53.00, 23.10),
        "NIDZICA": (53.36, 20.44), "SUSZ": (53.72, 19.34),
        "OŁAWA": (50.94, 17.30), "LEŚNY DWÓR": (53.92, 17.02),
        "SZKLARSKA PORĘBA": (50.83, 15.52),
    }
    FIRE_KEYWORDS = ("inne przyczyny", "pożar", "zagroż", "susz")
    PESTICIDE_KEYWORDS = ("zabiegi", "środk", "ochrony roślin")

    try:
        rss_url = "https://zakazywstepu.bdl.lasy.gov.pl/zakazy/Home/Rss"
        resp = HTTP.get(rss_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        import re
        items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
        zones: list[tuple[float, float, float, str]] = []
        for item in items:
            text_lower = item.lower()
            is_pesticide = any(kw in text_lower for kw in PESTICIDE_KEYWORDS)
            is_fire = any(kw in text_lower for kw in FIRE_KEYWORDS)
            if is_pesticide or not is_fire:
                continue
            for name, (lat, lon) in LP_GEOCODES.items():
                if name.lower() in text_lower:
                    zones.append((lat, lon, 30.0, f"BDL_RSS/{name}"))
                    break
        if zones:
            log.info("BDL RSS: %d active fire-risk ban zones loaded.", len(zones))
            return zones
        log.info("BDL RSS: no fire-risk items found (format may have changed).")
    except Exception as exc:
        log.debug("BDL RSS fetch failed: %s", exc)

    log.info("BDL: using hardcoded ban zones (%d).", len(BDL_FIRE_BAN_ZONES))
    return BDL_FIRE_BAN_ZONES


# =========================
# EFFIS LABELS
# =========================
def fetch_effis_fires(lat: float, lon: float, radius_km: float = 50.0) -> int | None:
    """
    Query EFFIS burned areas near a point. Returns count > 0 as 1/0, or None.

    v4.2 NOTE: EFFIS threshold is ~30 ha. Polish fires are mostly smaller,
    so EFFIS returning 0 does NOT mean no fire risk exists. The BDL ban
    check is the primary Polish fire-risk signal; EFFIS is secondary.
    """
    key = f"effis_{lat:.3f}_{lon:.3f}_{radius_km:.0f}"
    cached = _cache_get(key)
    if cached is not None:
        return int(cached)

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=EFFIS_LOOKBACK)
        deg = radius_km / 111.0
        bbox = f"{lon - deg},{lat - deg},{lon + deg},{lat + deg}"

        url = (
            "https://services-eu1.arcgis.com/VC42ANIVJ5dUfvUn/"
            "arcgis/rest/services/Burned_Areas_EFFIS/FeatureServer/23/query"
        )
        params = {
            "f": "json",
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "where": (
                f"FIREDATE >= DATE '{start.strftime('%Y-%m-%d')}' AND "
                f"FIREDATE <= DATE '{end.strftime('%Y-%m-%d')}'"
            ),
            "returnCountOnly": "true",
            "outSR": "4326",
        }
        response = HTTP.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        count = int(response.json().get("count", 0))
        label = int(count > 0)
        _cache_set(key, label)
        return label
    except Exception as exc:
        log.debug("EFFIS unavailable for (%.2f, %.2f): %s", lat, lon, exc)
        return None


def resolve_label(
    lat: float,
    lon: float,
    summary: dict,
    active_ban_zones: list[tuple[float, float, float, str]],
) -> tuple[int, str]:
    """
    Three-source label fusion (v4.2).

    Priority order:
      1. BDL fire-risk entry ban  ->  label=1, source="bdl_ban:<zone>"
      2. EFFIS burned area        ->  label=1, source="effis"
      3. Heuristic FWI scoring    ->  label=0/1, source="heuristic"

    Crucially: EFFIS=0 does NOT override a BDL ban. LP issues bans on
    FORECAST danger, not past fire records. Poland's small fires are below
    EFFIS detection threshold.
    """
    # 1. BDL ban
    banned, zone_name = check_bdl_fire_ban(lat, lon, zones=active_ban_zones)
    if banned:
        log.debug("BDL ban at (%.2f, %.2f): %s", lat, lon, zone_name)
        return 1, f"bdl_ban:{zone_name}"

    # 2. EFFIS
    effis = fetch_effis_fires(lat, lon)
    if effis == 1:
        return 1, "effis"

    # 3. Heuristic fallback
    lbl = heuristic_label(summary, drought_days=DROUGHT_DAYS)
    return lbl, "heuristic"
# =========================
# FEATURE ENGINEERING  (v4.1 - 17 features)
# =========================
FEATURE_NAMES = [
    "temp_max",
    "temp_mean",
    "rh_min",
    "wind_mean",
    "wind_max",
    "rain",
    "soil_min",
    "vpd",
    "ndvi_stress",
    "elevation",
    "slope",
    "heat_x_dryness",
    "wind_x_dryness",
    "soil_x_rh",
    "dryness_ratio",
    "wind_dryness_ratio",
    "fwi_score",           # v4.1 new
]


def build_feature_vector(
    summary: dict,
    ndvi: float,
    elevation: float,
    slope: float,
    drought_days: int = 0,
) -> list[float]:
    """Return a 17-element wildfire feature vector (v4.1: +fwi_score)."""
    temp = summary["temp"]
    tmean = summary["temp_mean"]
    rh = summary["rh"]
    wind = summary["wind"]
    wmax = summary["wind_max"]
    rain = summary["rain"]
    soil = max(summary["soil"], 0.001)
    vpd = summary["vpd"]
    dryness = 100.0 - rh
    fwi = fwi_score(summary, drought_days)

    return [
        temp,
        tmean,
        rh,
        wind,
        wmax,
        rain,
        soil,
        vpd,
        ndvi,
        elevation,
        slope,
        temp * dryness,
        wind * dryness,
        soil * rh,
        temp / (soil + 0.01),
        wind / (soil + 0.01),
        fwi,                # v4.1
    ]


def heuristic_label(summary: dict, drought_days: int = 0) -> int:
    """
    Rule-based label for fire risk. Used as fallback when EFFIS and BDL
    ban data are unavailable for a given cell.

    v4.2 FINAL: uses fwi_score composite directly.

    Returns 1 if fwi_score >= threshold, where threshold is
    lower during drought (more cells qualify without over-saturating).

    Threshold calibration (validated against LP ban data 29.05.2026):
      drought_days >= 14: 0.36  (more sensitive - prolonged drought is dangerous)
      drought_days < 14:  0.44  (standard - needs stronger current weather signal)

    Expected label distribution: ~30-70% fire cells on a realistic Poland grid.
    """
    score = fwi_score(summary, drought_days)
    threshold = 0.36 if drought_days >= 14 else 0.44
    return int(score >= threshold)

# =========================
# GRID
# =========================
def poland_grid() -> list[tuple[float, float]]:
    lats = np.linspace(49.0, 54.5, GRID_SIZE)
    lons = np.linspace(14.0, 24.0, GRID_SIZE)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]

# =========================
# PARALLEL DATASET BUILD  (v4.2: three-source labeling)
# =========================
def fetch_cell(
    lat: float,
    lon: float,
    active_ban_zones: list[tuple[float, float, float, str]],
) -> tuple | None:
    """Fetch all data for one grid cell. Returns None on unrecoverable failure."""
    try:
        raw = fetch_weather(lat, lon)
        summary = summarize_hourly(raw)
        ndvi = estimate_ndvi_proxy(summary, drought_days=DROUGHT_DAYS)
        elev = fetch_elevation(lat, lon)
        slope = estimate_slope(lat, lon)
        fvec = build_feature_vector(summary, ndvi, elev, slope, drought_days=DROUGHT_DAYS)

        # v4.2: three-source label fusion (BDL ban > EFFIS > heuristic)
        label, label_source = resolve_label(lat, lon, summary, active_ban_zones)

        return lat, lon, summary, fvec, int(label), label_source, ndvi, elev, slope
    except requests.RequestException as exc:
        log.warning("Network error  (%5.2f, %5.2f): %s", lat, lon, exc)
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("Bad response   (%5.2f, %5.2f): %s", lat, lon, exc)
    return None


def build_dataset() -> tuple[np.ndarray, np.ndarray, list, list]:
    """Returns X, y, cells, label_sources."""
    grid = poland_grid()
    X, y, cells, label_sources = [], [], [], []

    # Load BDL ban zones once (try live RSS first, fall back to hardcoded)
    active_ban_zones = fetch_bdl_entry_bans()
    bdl_hits = sum(1 for lat, lon in grid if check_bdl_fire_ban(lat, lon, zones=active_ban_zones)[0])
    log.info(
        "BDL fire-ban zones loaded: %d zones, covering %d/%d grid cells.",
        len(active_ban_zones), bdl_hits, len(grid),
    )

    log.info("Fetching %d grid cells with %d workers ...", len(grid), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_cell, lat, lon, active_ban_zones): (lat, lon)
            for lat, lon in grid
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            lat, lon, summary, fvec, label, label_source, ndvi, elev, slope = result
            X.append(fvec)
            y.append(label)
            cells.append((lat, lon, summary, ndvi, elev, slope))
            label_sources.append(label_source)

    # Source breakdown
    from collections import Counter
    source_counts = Counter(s.split(":")[0] for s in label_sources)
    log.info(
        "Dataset ready: %d cells | label sources: %s",
        len(X), dict(source_counts),
    )

    if len(X):
        y_arr = np.array(y)
        log.info(
            "Label distribution: %d fire (%.0f%%) / %d no-fire (%.0f%%)",
            int(y_arr.sum()), 100.0 * y_arr.mean(),
            int(len(y_arr) - y_arr.sum()), 100.0 * (1 - y_arr.mean()),
        )
        if y_arr.sum() == 0:
            log.warning(
                "All labels are 0. Check: (1) DROUGHT_DAYS setting, "
                "(2) BDL_FIRE_BAN_ZONES coverage, (3) network access."
            )

    return np.array(X, dtype=float), np.array(y, dtype=int), cells, label_sources

# =========================
# CLASSICAL MODEL + SHAP
# =========================
def train_classical(X_train: np.ndarray, y_train: np.ndarray, X: np.ndarray, y: np.ndarray) -> Pipeline:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    counts = np.bincount(y, minlength=2)
    valid_cv = len(np.unique(y)) > 1 and counts.min() >= 2
    if valid_cv:
        cv = min(5, int(counts.min()))
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
        log.info("Classical CV F1: %.3f +/- %.3f", scores.mean(), scores.std())
    else:
        log.info("Classical model trained (CV skipped: not enough samples per class).")

    return clf


def compute_shap(clf: Pipeline, X: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Compute SHAP values for the Random Forest when shap is installed."""
    try:
        import shap

        rf = clf.named_steps["rf"]
        scaler = clf.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) >= 2 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, min(1, shap_values.shape[2] - 1)]

        ev = explainer.expected_value
        expected_val = float(
            ev[1]
            if isinstance(ev, (list, np.ndarray)) and len(ev) >= 2
            else ev[0]
            if isinstance(ev, (list, np.ndarray))
            else ev
        )
        log.info("SHAP values computed.")
        return np.array(shap_values), expected_val
    except ImportError:
        log.info("shap not installed - skipping SHAP (pip install shap).")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)
    return None, None

# =========================
# QUANTUM MODEL  (unchanged from v4.0)
# =========================
@dataclass
class QuantumBundle:
    model: Any
    pca: PCA
    scaler: MinMaxScaler
    backend: str
    augmented: bool
    train_size: int
    original_classes: list[int]


class NumpyQuantumKernelSVC:
    """
    Lightweight local quantum-kernel simulator.
    """

    def __init__(self, reps: int = QUANTUM_REPS, c: float = 2.0):
        self.reps = reps
        self.c = c
        self.model = SVC(kernel="precomputed", probability=False, class_weight="balanced", C=c)
        self.X_fit_: np.ndarray | None = None

    @staticmethod
    def _apply_ry(state: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        out = state.copy()
        bit = 1 << qubit
        for i in range(len(state)):
            if i & bit:
                continue
            j = i | bit
            a0 = state[i]
            a1 = state[j]
            out[i] = c * a0 - s * a1
            out[j] = s * a0 + c * a1
        return out

    @staticmethod
    def _apply_zz_phase(state: np.ndarray, theta: float, q1: int, q2: int) -> np.ndarray:
        out = state.copy()
        b1 = 1 << q1
        b2 = 1 << q2
        for i in range(len(state)):
            z1 = -1 if i & b1 else 1
            z2 = -1 if i & b2 else 1
            out[i] *= np.exp(1j * theta * z1 * z2)
        return out

    def _state(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n_qubits = len(x)
        state = np.zeros(2**n_qubits, dtype=np.complex128)
        state[0] = 1.0
        for _ in range(self.reps):
            for q, theta in enumerate(x):
                state = self._apply_ry(state, float(theta), q, n_qubits)
            for q in range(n_qubits - 1):
                theta = float((math.pi - x[q]) * (math.pi - x[q + 1]) / math.pi)
                state = self._apply_zz_phase(state, theta, q, q + 1)
            if n_qubits > 2:
                theta = float((math.pi - x[-1]) * (math.pi - x[0]) / math.pi)
                state = self._apply_zz_phase(state, theta, n_qubits - 1, 0)
        norm = np.linalg.norm(state)
        return state / norm if norm else state

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        states_a = np.array([self._state(x) for x in A])
        states_b = np.array([self._state(x) for x in B])
        overlaps = states_a @ np.conjugate(states_b.T)
        return np.abs(overlaps) ** 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyQuantumKernelSVC":
        self.X_fit_ = np.array(X, dtype=float)
        kernel = self._kernel(self.X_fit_, self.X_fit_)
        self.model.fit(kernel, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_fit_ is None:
            raise RuntimeError("Quantum kernel model is not fitted.")
        kernel = self._kernel(np.array(X, dtype=float), self.X_fit_)
        return self.model.decision_function(kernel)


def _augment_for_quantum(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    if len(np.unique(y)) >= 2:
        return X, y, False

    only_class = int(y[0]) if len(y) else 0
    other_class = 1 - only_class
    center = np.mean(X, axis=0)
    spread = np.std(X, axis=0)
    spread = np.where(spread < 1e-6, np.maximum(np.abs(center) * 0.05, 1.0), spread)

    synthetic = []
    for sign in (-1.0, 1.0, -0.5, 0.5):
        sample = center + sign * spread
        if only_class == 0:
            sample[0] += abs(spread[0]) * 1.5
            sample[2] = max(5.0, sample[2] - abs(spread[2]) * 1.5)
            sample[3] += abs(spread[3])
            sample[6] = max(0.01, sample[6] - abs(spread[6]))
            sample[8] = min(1.0, max(sample[8], 0.85))
        else:
            sample[0] -= abs(spread[0]) * 1.5
            sample[2] = min(95.0, sample[2] + abs(spread[2]) * 1.5)
            sample[3] = max(0.1, sample[3] - abs(spread[3]))
            sample[6] = min(0.6, sample[6] + abs(spread[6]))
            sample[8] = max(0.0, min(sample[8], 0.15))
        synthetic.append(sample)

    X_aug = np.vstack([X, np.array(synthetic, dtype=float)])
    y_aug = np.concatenate([y, np.full(len(synthetic), other_class, dtype=int)])
    log.warning(
        "Quantum training data had one class (%d). Added %d conservative synthetic class-%d samples.",
        only_class,
        len(synthetic),
        other_class,
    )
    return X_aug, y_aug, True


def _prepare_quantum_features(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA, MinMaxScaler, bool]:
    Xq_train, yq_train, augmented = _augment_for_quantum(X_train, y_train)
    n_components = min(QUANTUM_FEATURES, Xq_train.shape[0], Xq_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xq_train)

    if n_components < QUANTUM_FEATURES:
        pad = np.zeros((X_pca.shape[0], QUANTUM_FEATURES - n_components))
        X_pca = np.hstack([X_pca, pad])

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_quantum = scaler.fit_transform(X_pca)
    return X_quantum, yq_train, pca, scaler, augmented


def _transform_quantum_features(pca: PCA, scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    X_pca = pca.transform(X)
    if X_pca.shape[1] < QUANTUM_FEATURES:
        pad = np.zeros((X_pca.shape[0], QUANTUM_FEATURES - X_pca.shape[1]))
        X_pca = np.hstack([X_pca, pad])
    return scaler.transform(X_pca)


def train_quantum(X_train: np.ndarray, y_train: np.ndarray) -> QuantumBundle:
    X_q, y_q, pca, scaler, augmented = _prepare_quantum_features(X_train, y_train)
    original_classes = sorted(int(c) for c in np.unique(y_train))

    try:
        try:
            from qiskit.circuit.library import zz_feature_map
        except ImportError:
            zz_feature_map = None
            from qiskit.circuit.library import ZZFeatureMap
        try:
            from qiskit.primitives import StatevectorSampler as Sampler
        except ImportError:
            from qiskit.primitives import Sampler
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        log.info("Training quantum model with Qiskit QSVC ...")
        if zz_feature_map is not None:
            fmap = zz_feature_map(feature_dimension=QUANTUM_FEATURES, reps=QUANTUM_REPS)
        else:
            fmap = ZZFeatureMap(feature_dimension=QUANTUM_FEATURES, reps=QUANTUM_REPS)
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler)
        kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)
        model = QSVC(quantum_kernel=kernel)
        model.fit(X_q, y_q)
        log.info("Quantum model trained: qiskit_qsvc.")
        return QuantumBundle(model, pca, scaler, "qiskit_qsvc", augmented, len(y_q), original_classes)
    except Exception as exc:
        log.warning(
            "Qiskit QSVC unavailable or failed (%s: %s). Using built-in quantum-kernel simulator.",
            type(exc).__name__,
            exc,
        )

    log.info("Training quantum model with NumPy statevector kernel ...")
    model = NumpyQuantumKernelSVC(reps=QUANTUM_REPS)
    model.fit(X_q, y_q)
    log.info("Quantum model trained: numpy_statevector_kernel_svc.")
    return QuantumBundle(model, pca, scaler, "numpy_statevector_kernel_svc", augmented, len(y_q), original_classes)


def score_quantum(bundle: QuantumBundle, feature_vec: np.ndarray) -> float:
    """Sigmoid-calibrated quantum score in [0, 1]."""
    X_q = _transform_quantum_features(bundle.pca, bundle.scaler, feature_vec)
    raw = np.asarray(bundle.model.decision_function(X_q)).reshape(-1)[0]
    return float(1.0 / (1.0 + np.exp(-raw)))

# =========================
# QAOA SENSOR PLACEMENT  (unchanged from v4.0)
# =========================
def qaoa_sensor_placement(results: list, n_sensors: int = N_SENSORS) -> list[tuple[float, float]]:
    if not results:
        return []

    k_target = min(n_sensors, len(results))
    max_candidates = min(8, len(results))
    risks = np.array([r["risk"] for r in results], dtype=float)
    coords = [(float(r["lat"]), float(r["lon"])) for r in results]

    def diverse_greedy(risk_arr: np.ndarray, coord_list: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:
        if not coord_list or k <= 0:
            return []
        selected = [int(np.argmax(risk_arr))]
        remaining = [i for i in range(len(coord_list)) if i != selected[0]]

        while len(selected) < k and remaining:
            best_score, best_idx = -np.inf, remaining[0]
            for i in remaining:
                lat_i, lon_i = coord_list[i]
                min_dist = min(abs(lat_i - coord_list[j][0]) + abs(lon_i - coord_list[j][1]) for j in selected)
                score = float(risk_arr[i]) + 0.3 * min_dist
                if score > best_score:
                    best_score, best_idx = score, i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [coord_list[i] for i in selected]

    greedy_sensors = diverse_greedy(risks, coords, k_target)
    risk_range = float(risks.max() - risks.min())
    if risk_range < 1e-4:
        log.info("Risk landscape is flat - using spatially-diverse greedy sensor placement.")
        return greedy_sensors

    try:
        import warnings
        try:
            from qiskit.circuit.library import real_amplitudes
        except ImportError:
            real_amplitudes = None
            from qiskit.circuit.library import RealAmplitudes
        from qiskit_algorithms import SamplingVQE
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from scipy.sparse import SparseEfficiencyWarning

        try:
            from qiskit.primitives import StatevectorSampler as VqeSampler
        except ImportError:
            from qiskit.primitives import Sampler as VqeSampler

        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        candidate_idx = np.argsort(risks)[::-1][:max_candidates].tolist()
        cand_risks = risks[candidate_idx]
        cand_coords = [coords[i] for i in candidate_idx]
        nc = len(candidate_idx)
        k = min(k_target, nc)

        qp = QuadraticProgram("sensor_placement")
        for i in range(nc):
            qp.binary_var(f"x{i}")
        qp.minimize(linear={f"x{i}": -float(cand_risks[i]) for i in range(nc)})
        qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(nc)},
            sense="==",
            rhs=k,
            name="sensor_count",
        )

        if real_amplitudes is not None:
            ansatz = real_amplitudes(num_qubits=nc, reps=1)
        else:
            ansatz = RealAmplitudes(num_qubits=nc, reps=1)
        sampler = VqeSampler()
        log.info("SamplingVQE: optimizing sensor placement (%d candidates -> %d sensors) ...", nc, k)
        vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA(maxiter=200))
        solver = MinimumEigenOptimizer(vqe)
        optimizer_output = io.StringIO()
        with contextlib.redirect_stdout(optimizer_output), contextlib.redirect_stderr(optimizer_output):
            result = solver.solve(qp)
        if optimizer_output.getvalue().strip():
            log.debug("SamplingVQE optimizer output:\n%s", optimizer_output.getvalue().strip())

        selected = [i for i, value in enumerate(result.x) if value > 0.5]
        if len(selected) != k:
            log.warning("VQE returned %d sensors (expected %d) - using greedy.", len(selected), k)
            return greedy_sensors

        sensors = [cand_coords[i] for i in selected]
        log.info("Quantum sensor placement complete: %s", sensors)
        return sensors
    except Exception as exc:
        log.info("Quantum sensor placement fallback (%s: %s).", type(exc).__name__, exc)
        return greedy_sensors

# =========================
# SCORING  (v4.1 - calibrated)
# =========================
def _classical_fire_probability(clf: Pipeline, fvec: np.ndarray) -> float:
    proba = clf.predict_proba(fvec)[0]
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return 0.0


def _calibrate_score(raw: float, fire_season: bool) -> float:
    """
    v4.2: linear pass-through with small seasonal floor.

    The v4.1 sigmoid stretch is removed because the FWI stretch (via
    _stretch_fwi) already expands the score range to [0, 1].
    Applying a sigmoid on top caused the final scores to cluster above 0.75,
    giving 24-33/36 alerts even on a correct FWI-primary blend.

    Only change: a small +3 % additive bump during fire season (Mar-Sep),
    and a 3 % floor so no cell ever shows exactly 0 risk in season.
    """
    score = float(raw)
    if fire_season:
        score = min(1.0, score + 0.03)
        score = max(RISK_FLOOR, score)
    return float(np.clip(score, 0.0, 1.0))


def _stretch_fwi(fwi: float) -> float:
    """Map raw FWI score to [0, 1] using calibrated stretch bounds."""
    return float(np.clip((fwi - FWI_STRETCH_LO) / (FWI_STRETCH_HI - FWI_STRETCH_LO), 0.0, 1.0))


def score_cells(
    cells: list,
    X: np.ndarray,
    clf: Pipeline,
    q_bundle: QuantumBundle,
    shap_values: np.ndarray | None,
    label_sources: list[str] | None = None,
) -> list:
    """
    v4.2 scoring formula:
      final = FWI_BLEND * stretched_fwi
            + (1 - FWI_BLEND - QUANTUM_BLEND) * classical
            + QUANTUM_BLEND * quantum

    FWI is the primary signal (50 %): it directly reflects weather conditions
    and avoids the majority-class collapse of RF on imbalanced training data.
    Classical RF (30 %) adds spatial/terrain context from feature interactions.
    Quantum model (20 %) provides kernel-based complementary signal.

    stretched_fwi re-scales raw FWI from [FWI_STRETCH_LO, FWI_STRETCH_HI]
    to [0, 1] so it uses the full score range instead of clustering at 0.3-0.6.
    """
    results = []
    classical_weight = 1.0 - FWI_BLEND - QUANTUM_BLEND   # = 0.30
    fire_season = _is_fire_season()
    if label_sources is None:
        label_sources = ["unknown"] * len(cells)

    for i, (lat, lon, summary, ndvi, elev, slope) in enumerate(cells):
        fvec = X[i : i + 1]
        classical_prob = _classical_fire_probability(clf, fvec)
        quantum_prob = score_quantum(q_bundle, fvec)

        # v4.2: FWI-primary blend, then calibrate
        fwi_val = float(X[i, FEATURE_NAMES.index("fwi_score")]) if "fwi_score" in FEATURE_NAMES else 0.5
        stretched = _stretch_fwi(fwi_val)
        raw_score = (FWI_BLEND * stretched
                     + classical_weight * classical_prob
                     + QUANTUM_BLEND * quantum_prob)
        final_score = _calibrate_score(raw_score, fire_season)

        shap_drivers = []
        if shap_values is not None and i < len(shap_values):
            sv = np.asarray(shap_values[i]).reshape(-1)
            top3 = np.argsort(np.abs(sv))[::-1][:3]
            shap_drivers = [
                {"feature": FEATURE_NAMES[j], "shap": round(float(sv[j]), 4)}
                for j in top3
                if j < len(FEATURE_NAMES)
            ]

        results.append(
            {
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "risk": round(float(np.clip(final_score, 0, 1)), 4),
                "classical_risk": round(float(np.clip(classical_prob, 0, 1)), 4),
                "quantum_risk": round(float(np.clip(quantum_prob, 0, 1)), 4),
                "quantum_backend": q_bundle.backend,
                "quantum_augmented": bool(q_bundle.augmented),
                "alert": bool(final_score > ALERT_THRESHOLD),
                "ndvi_stress": round(float(ndvi), 3),
                "elevation_m": round(float(elev), 1),
                "slope_deg": round(float(slope), 2),
                "fwi_score": round(float(fwi_score(summary, DROUGHT_DAYS)), 3),
                "drought_days": DROUGHT_DAYS,
                "label_source": label_sources[i] if i < len(label_sources) else "unknown",
                "tier": ("CRITICAL" if final_score > 0.75 else
                         "HIGH"     if final_score > 0.65 else
                         "MODERATE" if final_score > 0.50 else
                         "LOW"),
                "shap_drivers": shap_drivers,
                "hourly_risk": _hourly_risk_series(summary, final_score),
            }
        )

    return results


def _hourly_risk_series(summary: dict, base_risk: float) -> list[dict]:
    temps = summary.get("hourly_temp", [15.0] * 24)
    tmin, tmax = min(temps), max(temps)
    series = []
    for i, (t, rh, w) in enumerate(
        zip(
            temps,
            summary.get("hourly_rh", [50.0] * 24),
            summary.get("hourly_wind", [5.0] * 24),
        )
    ):
        t_norm = (t - tmin) / (tmax - tmin + 0.01)
        rh_factor = max(0.0, (50.0 - rh) / 50.0)
        w_factor = min(1.0, w / 20.0)
        hour_score = min(1.0, base_risk * (1 + 0.4 * t_norm + 0.3 * rh_factor + 0.2 * w_factor))
        time_label = summary["hourly_times"][i] if i < len(summary.get("hourly_times", [])) else f"{i:02d}:00"
        series.append({"time": time_label, "risk": round(float(hour_score), 3)})
    return series

# =========================
# OUTPUT - JSON + CSV
# =========================
def save_results(results: list, quantum_status: dict) -> None:
    payload = {
        "version": "4.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drought_days": DROUGHT_DAYS,
        "fire_season": _is_fire_season(),
        "quantum_status": quantum_status,
        "cells": results,
    }
    with open("fire.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Saved fire.json")

    flat_keys = [
        "lat",
        "lon",
        "risk",
        "classical_risk",
        "quantum_risk",
        "alert",
        "ndvi_stress",
        "fwi_score",
        "drought_days",
        "elevation_m",
        "slope_deg",
        "quantum_backend",
        "quantum_augmented",
        "label_source",
    ]
    with open("fire.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in flat_keys})
    log.info("Saved fire.csv")

# =========================
# OUTPUT - MAP
# =========================
def generate_map(results: list, sensors: list[tuple[float, float]], quantum_status: dict) -> None:
    cell_data = json.dumps(
        [
            {
                "lat": r["lat"],
                "lon": r["lon"],
                "risk": r["risk"],
                "classical": r["classical_risk"],
                "quantum": r["quantum_risk"],
                "alert": r["alert"],
                "ndvi": r["ndvi_stress"],
                "fwi": r.get("fwi_score", 0),
                "elev": r["elevation_m"],
                "slope": r["slope_deg"],
                "shap": r["shap_drivers"],
                "src": r.get("label_source","?"),
                "hourly": r["hourly_risk"],
            }
            for r in results
        ],
        ensure_ascii=False,
    )
    sensor_data = json.dumps([{"lat": lat, "lon": lon} for lat, lon in sensors])
    status_data = json.dumps(quantum_status, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>QHDALabs - Wildfire Risk PL v4.1</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ margin:0; height:100vh; font-family:Arial, sans-serif; }}
    #controls {{
      position:absolute; top:10px; left:50%; transform:translateX(-50%);
      z-index:1000; background:white; padding:8px 14px; border-radius:8px;
      box-shadow:0 2px 10px rgba(0,0,0,0.22); display:flex; align-items:center; gap:10px;
      max-width:calc(100vw - 24px);
    }}
    #hour-label {{ font-weight:700; min-width:52px; text-align:center; }}
    #hour-slider {{ width:min(260px, 45vw); }}
    .legend {{
      background:white; padding:9px 12px; border-radius:6px;
      font:13px Arial, sans-serif; line-height:1.75; box-shadow:0 1px 6px rgba(0,0,0,0.18);
    }}
    .small-muted {{ color:#666; font-size:12px; }}
  </style>
</head>
<body>
<div id="controls">
  <span>Godzina</span>
  <input id="hour-slider" type="range" min="0" max="23" value="12" step="1"
         oninput="updateHour(this.value)"/>
  <span id="hour-label">12:00</span>
</div>
<div id="map"></div>
<script>
const CELLS = {cell_data};
const SENSORS = {sensor_data};
const THRESH = {ALERT_THRESHOLD};
const QUANTUM = {status_data};

var map = L.map('map').setView([52, 19], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution:'&copy; OpenStreetMap contributors'
}}).addTo(map);

var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
  var d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>Zagrożenie pożarowe v4.1</b><br>'
    + '<span style="color:#8B0000">●</span> KRYTYCZNE (&gt;0.75)<br>'
    + '<span style="color:#CC2200">●</span> WYSOKIE (0.65-0.75)<br>'
    + '<span style="color:#E07B00">●</span> UMIARKOWANE (0.50-0.65)<br>'
    + '<span style="color:#2F7D32">●</span> NISKIE (&lt;0.50)<br>'
    + '<span style="color:#1a56db">●</span> Lokalizacja czujnika<br>'
    + '<span class="small-muted">Quantum: ' + QUANTUM.backend + '</span>';
  return d;
}};
legend.addTo(map);

var circles = [];
function riskColor(risk) {{
  if (risk > 0.75) return '#8B0000';
  if (risk > 0.65) return '#CC2200';
  if (risk > 0.50) return '#E07B00';
  return '#2F7D32';
}}
function riskOpacity(risk) {{
  if (risk > 0.75) return 0.75;
  if (risk > 0.65) return 0.60;
  if (risk > 0.50) return 0.45;
  return 0.30;
}}

CELLS.forEach(function(c) {{
  var color = riskColor(c.risk);
  var shap_html = '';
  if (c.shap && c.shap.length > 0) {{
    shap_html = '<br><small><b>Kluczowe czynniki:</b><br>'
      + c.shap.map(function(s) {{
          return s.feature + ': ' + (s.shap >= 0 ? '+' : '') + s.shap;
        }}).join('<br>') + '</small>';
  }}

  var tier=c.risk>0.75?'🔴 KRYTYCZNE':c.risk>0.65?'🟠 WYSOKIE':c.risk>0.50?'🟡 UMIARKOWANE':'🟢 NISKIE';
  var popup = '<b>'+tier+' — '+c.risk.toFixed(3)+'</b><br>'
    + 'RF: '+c.classical.toFixed(3)+' | Q: '+c.quantum.toFixed(3)+' | FWI: '+c.fwi.toFixed(3)+'<br>'
    + 'Kwantowy model: ' + c.quantum.toFixed(2) + '<br>'
    + 'FWI: ' + c.fwi.toFixed(2) + '<br>'
    + 'Stres roślinności: ' + c.ndvi.toFixed(2) + '<br>'
    + 'Wysokość: ' + c.elev + ' m<br>'
    + 'Nachylenie: ' + c.slope + '&deg;'
    + shap_html;

  var circle = L.circle([c.lat, c.lon], {{
    color: color,
    fillColor: color,
    fillOpacity: riskOpacity(c.risk),
    radius: 20000
  }}).addTo(map).bindPopup(popup);

  circles.push({{circle: circle, cell: c}});
}});

SENSORS.forEach(function(s) {{
  L.circleMarker([s.lat, s.lon], {{
    radius: 8,
    color: '#1a56db',
    fillColor: '#1a56db',
    fillOpacity: 0.9,
    weight: 2
  }}).addTo(map).bindPopup('<b>Zalecana lokalizacja czujnika</b><br>Rozmieszczenie kwantowe/zachłanne');
}});

function updateHour(h) {{
  var hour = parseInt(h);
  document.getElementById('hour-label').textContent = (hour < 10 ? '0' : '') + hour + ':00';
  circles.forEach(function(item) {{
    var c = item.cell;
    var risk = (c.hourly && c.hourly[hour]) ? c.hourly[hour].risk : c.risk;
    var color = riskColor(risk);
    item.circle.setStyle({{color: color, fillColor: color, fillOpacity: riskOpacity(risk)}});
  }});
}}
</script>
</body></html>"""

    with open("map.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("Saved map.html")

# =========================
# OUTPUT - SHAP HTML REPORT
# =========================
def generate_shap_report(results: list, quantum_status: dict) -> None:
    rows = []
    for r in results:
        drivers = (
            ", ".join(f"{d['feature']} ({'+' if d['shap'] >= 0 else ''}{d['shap']})" for d in r["shap_drivers"])
            if r["shap_drivers"]
            else "SHAP niedostępny"
        )
        status = '<span style="color:#A32D2D;font-weight:bold">ALERT</span>' if r["alert"] else "OK"
        src = r.get("label_source", "?")
        rows.append(
            f"<tr><td>{r['lat']}</td><td>{r['lon']}</td>"
            f"<td>{r['risk']:.3f}</td><td>{r['classical_risk']:.3f}</td>"
            f"<td>{r['quantum_risk']:.3f}</td><td>{r.get('fwi_score', 0):.3f}</td>"
            f"<td>{status}</td><td>{src}</td><td>{drivers}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>SHAP Report - Wildfire Risk PL v4.1</title>
<style>
  body {{ font-family:Arial, sans-serif; padding:24px; }}
  table {{ border-collapse:collapse; width:100%; }}
  th, td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; }}
  th {{ background:#f0f0f0; }}
  tr:nth-child(even) {{ background:#fafafa; }}
  .meta {{ color:#666; font-size:12px; margin-bottom:16px; }}
</style></head><body>
<h2>SHAP Feature Importance - Wildfire Risk PL v4.1</h2>
<div class="meta">
  Quantum backend: {quantum_status["backend"]};
  augmented training: {quantum_status["augmented"]};
  drought_days: {DROUGHT_DAYS};
  fire_season: {_is_fire_season()};
  generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</div>
<table>
<tr><th>Lat</th><th>Lon</th><th>Risk</th><th>Classical</th><th>Quantum</th><th>FWI</th><th>Status</th><th>Label src</th><th>Top SHAP drivers</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""

    with open("shap_report.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("Saved shap_report.html")

# =========================
# MAIN
# =========================
def _can_stratify(y: np.ndarray) -> bool:
    if len(np.unique(y)) < 2:
        return False
    counts = np.bincount(y, minlength=2)
    return bool(counts.min() >= 2)


def main() -> None:
    log.info(
        "QHDALabs Wildfire Risk PL v4.2 | drought_days=%d | fire_season=%s",
        DROUGHT_DAYS,
        _is_fire_season(),
    )

    X, y, cells, label_sources = build_dataset()
    if len(X) == 0:
        log.error("No data fetched - check network and API availability.")
        return

    stratify = y if _can_stratify(y) else None
    if len(X) < 4:
        log.warning("Very small dataset (%d samples). Training and testing on the same data.", len(X))
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

    clf = train_classical(X_train, y_train, X, y)
    log.info(
        "Classification report:\n%s",
        classification_report(y_test, clf.predict(X_test), zero_division=0),
    )

    shap_values, _ = compute_shap(clf, X)

    q_bundle = train_quantum(X_train, y_train)
    quantum_status = {
        "backend": q_bundle.backend,
        "augmented": q_bundle.augmented,
        "train_size": q_bundle.train_size,
        "original_classes": q_bundle.original_classes,
        "blend_weight": QUANTUM_BLEND,
    }

    results = score_cells(cells, X, clf, q_bundle, shap_values, label_sources)
    alert_count = sum(1 for r in results if r["alert"])
    log.info("Alerts: %d / %d cells", alert_count, len(results))

    sensors = qaoa_sensor_placement(results)
    log.info("Sensor positions: %s", sensors)

    save_results(results, quantum_status)
    generate_map(results, sensors, quantum_status)
    generate_shap_report(results, quantum_status)

    log.info("Done -> map.html  fire.json  fire.csv  shap_report.html")


if __name__ == "__main__":
    main()