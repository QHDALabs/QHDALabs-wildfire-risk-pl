# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 4b — Ignition Pressure Layer
# File          : qhdalabs_wildfire_ignition_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Computes ignition_score per nadleśnictwo node — a measure of how likely
# a fire is to be *started*, independent of how ready the forest is to burn.
#
# Conceptually separates two orthogonal questions:
#   readiness_score  — "Does the forest want to burn?"  (fusion_v1.py)
#   ignition_score   — "Is there a realistic ignition source nearby?"  (this)
#
# Both feed the Fire Emergence Index (FEI) computed in fusion_v1.py:
#   FEI = sqrt(R * I) + extremity_bonus
#
# Sublayers
# -----------------------------------------------------------------------------
#   roads          proximity to roads (BDOT10k / OSM)           weight: 0.25
#   railways       proximity to railway lines (BDOT10k / OSM)   weight: 0.20
#   powerlines     proximity to HV/MV power lines (BDOT10k)     weight: 0.15
#   tourism        density of hiking/cycling trails (OSM)        weight: 0.15
#   agriculture    LPIS/CLC arable land edge density             weight: 0.15
#   historical_kde kernel density of past ignition points        weight: 0.10
#                  source: NASA FIRMS VIIRS 375m (2025 cache)
#
# Data sources & auto-download
# -----------------------------------------------------------------------------
#   OSM roads/railways/trails  — Geofabrik PBF (dolnoslaskie extract)
#   BDOT10k power lines        — GIS-Support SHP bundles (SULN02/SULN03)
#   ARiMR LPIS                 — WFS endpoint (public since XII 2025)
#   CORINE Land Cover 2018     — Copernicus GeoTIFF
#   NASA FIRMS VIIRS 2025      — CSV archive (cached once, immutable)
#
# All raw data lands in:  topology/ignition_cache/
# Derived ignition scores written to: topology/ignition_scores.json
#
# Integration with fusion_v1.py
# -----------------------------------------------------------------------------
# fusion_v1.py reads topology/ignition_scores.json as an optional input.
# If the file is absent, fusion_v1 skips FEI computation and logs a warning.
# Run this module first (or schedule separately — GIS data changes rarely).
#
# Outputs
# -----------------------------------------------------------------------------
#   topology/ignition_scores.json   — ignition_score + sublayer breakdown
#                                     per node_id, plus flags & QIES input
#
# Usage
# -----------------------------------------------------------------------------
#   python qhdalabs_wildfire_ignition_v1.py                # normal run
#   python qhdalabs_wildfire_ignition_v1.py --refresh-firms # re-download FIRMS
#   python qhdalabs_wildfire_ignition_v1.py --stub          # synthetic data, no GIS
#
# Author        : Krzysztof W. Banasiewicz / QHDALabs
# License       : MIT
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ignition_data import (
    CLC_AGRICULTURE_CODES,
    DataAcquisitionError,
    HTTPResponse,
    ResponseValidationError,
    atomic_write_json,
    download_clc_geojson,
    download_firms_year,
    parse_firms_csv_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR = Path("topology")
CACHE_DIR = OUTPUT_DIR / "ignition_cache"
FIRMS_CSV = CACHE_DIR / "firms_viirs_dolnoslaskie_2025.csv"
OSM_PBF = CACHE_DIR / "dolnoslaskie-latest.osm.pbf"
SULN02_GPKG = CACHE_DIR / "wn.gpkg"  # GIS-Support: linie WN ~7 MB
SULN03_GPKG = CACHE_DIR / "sn.gpkg"  # GIS-Support: linie SN ~169 MB
LPIS_CACHE = CACHE_DIR / "lpis_dolnoslaskie.gpkg"
CLC_CACHE = CACHE_DIR / "clc18_dolnoslaskie.geojson"
EFFIS_CSV = CACHE_DIR / "effis_fires_pl_2025.csv"  # JRC EFFIS fallback
IBL_GEOJSON = CACHE_DIR / "ibl_pozary_dolnoslaskie.geojson"

IGNITION_OUT = OUTPUT_DIR / "ignition_scores.json"
NODES_JSON = OUTPUT_DIR / "nodes_enriched.json"

# =============================================================================
# IGNITION CONFIG
# =============================================================================

# Sublayer weights — sum must equal 1.0
# Calibrated for Dolny Śląsk; tune after backtesting against FIRMS 2025
IGNITION_WEIGHTS = {
    "roads": 0.25,  # human access — highest frequency ignition cause
    "railways": 0.20,  # sparks from braking / traction, esp. non-electric
    "powerlines": 0.15,  # arc discharge, conductor sag, fallen lines
    "tourism": 0.15,  # footpaths, campsites — recreational pressure
    "agriculture": 0.15,  # crop/stubble burning, machinery, field edges
    "historical_kde": 0.10,  # past ignition hotspots (FIRMS VIIRS proxy)
}
assert abs(sum(IGNITION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"
SUBLAYER_NAMES = tuple(IGNITION_WEIGHTS)

# This threshold retains most of the architecture: roads + railways (0.45) are
# insufficient by themselves, and a lone power-line layer (0.15) is blocked.
MIN_COVERAGE_FOR_FUSION = 0.70

# Proximity decay: exponential half-distance in metres for each infrastructure type.
# At d=0 → signal=1.0; at d=HALF_DIST → signal=0.5; at d→∞ → signal→0
PROXIMITY_HALF_DIST_M = {
    "roads": 300,  # main roads; ignition peaks within 300 m
    "railways": 200,  # railway corridor: very localised spark risk
    "powerlines": 150,  # arc/fault risk drops off sharply
    "tourism": 500,  # trail density — broader recreational spread
    "agriculture": 800,  # field-edge interface — wider transition zone
}

# KDE bandwidth for historical ignition points (metres)
FIRMS_KDE_BANDWIDTH_M = 5_000  # 5 km Gaussian kernel

# FEI fusion parameters (mirrors what fusion_v1.py will compute)
FEI_EXTREMITY_R_THRESHOLD = 80  # readiness above this triggers extremity bonus
FEI_EXTREMITY_I_THRESHOLD = 80  # ignition above this triggers extremity bonus
FEI_EXTREMITY_R_BONUS = 12  # max bonus points at R=100
FEI_EXTREMITY_I_BONUS = 8  # max bonus points at I=100

# QIES quantum-inspired coherence parameter
QIES_INTERFERENCE_LAMBDA = 0.25  # destructive interference strength

# Alert thresholds for ignition_score standalone (informational, not operational)
IGNITION_HIGH = 75
IGNITION_MODERATE = 50

# =============================================================================
# DATA DOWNLOAD REGISTRY
# =============================================================================
# Each entry: (local_path, url, description, is_zip, zip_target_name)
# URLs verified June 2026; GIS-Support bundles are free, no auth required.

DATA_REGISTRY = [
    # ── Power lines — GIS-Support BDOT10k GeoPackages (free, no auth) ────────
    # Correct format is .gpkg, not .zip — verified gis-support.pl/dane-do-pobrania/
    (
        SULN02_GPKG,
        "https://gis-support.pl/downloads/wn.gpkg",
        "BDOT10k SULN02 — linie wysokiego napięcia WN (~7 MB)",
        False,
        None,
    ),
    (
        SULN03_GPKG,
        "https://gis-support.pl/downloads/sn.gpkg",
        "BDOT10k SULN03 — linie średniego napięcia SN (~169 MB)",
        False,
        None,
    ),
]
# NASA FIRMS archive requires MAP_KEY (free, register at firms.modaps.eosdis.nasa.gov).
# Primary hotspot source is JRC EFFIS (no key needed).
# FIRMS is handled separately in _download_firms_or_effis().

# OSM and LPIS/CLC require dedicated download functions (see below)


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class IgnitionSublayers:
    roads: Optional[float] = None  # [0,100], None means unavailable
    railways: Optional[float] = None
    powerlines: Optional[float] = None
    tourism: Optional[float] = None
    agriculture: Optional[float] = None
    historical_kde: Optional[float] = None


@dataclass
class IgnitionScore:
    node_id: str
    node_name: str
    lat: float
    lon: float
    ignition_score: Optional[float]  # full operational score, else None
    ignition_score_raw: float
    ignition_score_available: Optional[float]
    sublayers: IgnitionSublayers
    extreme_ignition: bool  # I > IGNITION_HIGH
    # Flags for fusion_v1
    dominant_source: str  # sublayer with highest contribution
    data_coverage: list[str]  # which sublayers have real data
    missing_sublayers: list[str]
    coverage_weight: float
    coverage_percent: float
    score_status: str
    valid_for_fusion: bool
    warnings: list[str]
    computed_at: str  # ISO timestamp
    # Populated by fusion_v1 when both R and I are known
    fei: Optional[float] = None
    qies: Optional[float] = None


# =============================================================================
# UTILITY — PROXIMITY SIGNAL
# =============================================================================


def proximity_signal(distance_m: float, half_dist_m: float) -> float:
    """
    Exponential decay: signal=1 at distance=0, signal=0.5 at half_dist_m.
    Models how ignition probability drops with distance from infrastructure.

        signal = exp(-λ * d)   where λ = ln(2) / half_dist

    Returns value in [0, 1].
    """
    if distance_m <= 0:
        return 1.0
    lam = math.log(2) / half_dist_m
    return math.exp(-lam * distance_m)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast haversine distance in metres between two WGS84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def gaussian_kde_score(
    point_lat: float,
    point_lon: float,
    hotspot_lats: list[float],
    hotspot_lons: list[float],
    bandwidth_m: float,
) -> float:
    """
    Gaussian KDE score at (point_lat, point_lon) given a list of ignition hotspots.
    Returns value in [0, 1] normalised against maximum possible contribution.
    """
    if not hotspot_lats:
        return 0.0
    sigma = bandwidth_m
    total = 0.0
    for hl, hln in zip(hotspot_lats, hotspot_lons):
        d = haversine_m(point_lat, point_lon, hl, hln)
        total += math.exp(-0.5 * (d / sigma) ** 2)
    # Normalise: max possible = all hotspots at d=0
    norm = total / len(hotspot_lats)
    # Soft cap at 1.0 (multiple nearby hotspots can push above 1 before cap)
    return float(min(1.0, norm))


# =============================================================================
# UTILITY — FEI & QIES  (standalone, for pre-computation / reporting)
# =============================================================================


def compute_fei(readiness: float, ignition: float) -> float:
    """
    Fire Emergence Index — geometric mean with extremity bonuses.

    Geometry:  FEI_base = sqrt(R * I)          (AND-logic: need both)
    Extremity: quadratic bonus past threshold   (extreme values warrant
               separate operational flag even when the other is low)

    Both inputs in [0, 100]; output clipped to [0, 100].
    """
    r, i = readiness, ignition
    fei_base = math.sqrt(r * i)

    r_extreme = max(0.0, (r - FEI_EXTREMITY_R_THRESHOLD) / 20.0) ** 2
    i_extreme = max(0.0, (i - FEI_EXTREMITY_I_THRESHOLD) / 20.0) ** 2

    fei = (
        fei_base + FEI_EXTREMITY_R_BONUS * r_extreme + FEI_EXTREMITY_I_BONUS * i_extreme
    )
    return float(np.clip(fei, 0.0, 100.0))


def compute_qies(readiness: float, ignition: float) -> float:
    """
    Quantum-Inspired Entanglement Score — diagnostic only, not used for alerts.

    Models destructive interference when R and I are imbalanced:
    coherence = 1 - λ * sin(π * |R - I| / 100)
    QIES = (R * I / 100) * coherence

    Interpretation: a perfectly coherent system (R == I) maximises QIES.
    Imbalanced states (high R, low I or vice versa) suffer interference loss.
    This is the quantum contribution: it doesn't change the alert, but it
    quantifies how 'entangled' the two risk channels are — useful for
    backtesting and understanding false positive / false negative patterns.
    """
    r, i = readiness, ignition
    imbalance = abs(r - i) / 100.0
    coherence = 1.0 - QIES_INTERFERENCE_LAMBDA * math.sin(math.pi * imbalance)
    qies = (r * i / 100.0) * coherence
    return float(np.clip(qies, 0.0, 100.0))


# =============================================================================
# GIS DATA DOWNLOAD
# =============================================================================


def _download_file(
    url: str, dest: Path, description: str, chunk_size: int = 65536
) -> bool:
    """Download url to dest with progress logging. Returns True on success."""
    log.info("Downloading: %s", description)
    log.info("  URL: %s", url[:80] + ("…" if len(url) > 80 else ""))
    log.info("  → %s", dest)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "QHDALabs-Wildfire/1.0 (research; contact@qhdalabs.pl)"
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded % (5 * 1024 * 1024) < chunk_size:  # log every ~5 MB
                    log.info("  … %.1f MB", downloaded / 1_048_576)
        log.info("  Done: %.2f MB", os.path.getsize(dest) / 1_048_576)
        return True
    except (urllib.error.URLError, OSError) as exc:
        log.warning("  Download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        return False


def _unzip_first_shp(zip_path: Path, out_dir: Path) -> Optional[Path]:
    """Extract a ZIP and return path to the first .shp found inside."""
    with zipfile.ZipFile(zip_path) as zf:
        shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        if not shp_names:
            log.warning("No .shp found in %s", zip_path)
            return None
        zf.extractall(out_dir)
        return out_dir / shp_names[0]


def ensure_data_available(refresh_firms: bool = False) -> dict[str, Optional[Path]]:
    """
    Ensure all required GIS data is present in CACHE_DIR.
    Downloads missing files automatically.
    Returns a dict mapping data key → local Path (or None if unavailable).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Optional[Path]] = {}

    # ── Standard registry items (SULN02/03 GeoPackages) ──────────────────
    for local_path, url, desc, _is_zip, _zip_target in DATA_REGISTRY:
        key = local_path.stem  # "wn" or "sn"
        if local_path.exists():
            log.info(
                "Cache hit: %s (%.1f MB)",
                local_path.name,
                local_path.stat().st_size / 1_048_576,
            )
            paths[key] = local_path
        else:
            ok = _download_file(url, local_path, desc)
            paths[key] = local_path if ok else None

    # ── FIRMS / EFFIS — historical ignition points ─────────────────────────
    firms_path = _download_firms_or_effis(refresh_firms=refresh_firms)
    paths["firms"] = firms_path

    # ── OSM Geofabrik (dolnoslaskie) ──────────────────────────────────────
    if not OSM_PBF.exists():
        ok = _download_file(
            "https://download.geofabrik.de/europe/poland/dolnoslaskie-latest.osm.pbf",
            OSM_PBF,
            "OSM Geofabrik — dolnośląskie extract (~167 MB, cached permanently)",
        )
        paths["osm"] = OSM_PBF if ok else None
    else:
        log.info(
            "OSM cache hit: %s (%.1f MB)",
            OSM_PBF.name,
            OSM_PBF.stat().st_size / 1_048_576,
        )
        paths["osm"] = OSM_PBF

    # ── ARiMR LPIS — WFS; fallback to CLC ────────────────────────────────
    if not LPIS_CACHE.exists():
        log.info("LPIS cache not found — attempting ARiMR WFS download")
        lpis_ok = _download_lpis_wfs()
        paths["lpis"] = LPIS_CACHE if lpis_ok else None
    else:
        log.info("LPIS cache hit: %s", LPIS_CACHE.name)
        paths["lpis"] = LPIS_CACHE

    # ── CORINE Land Cover 2018 (fallback agriculture layer) ───────────────
    if not CLC_CACHE.exists():
        log.info("CLC cache not found — attempting EEA WFS download")
        clc_ok = _download_clc()
        paths["clc"] = CLC_CACHE if clc_ok else None
    else:
        log.info("CLC cache hit: %s", CLC_CACHE.name)
        paths["clc"] = CLC_CACHE

    return paths


def _download_firms_or_effis(refresh_firms: bool = False) -> Optional[Path]:
    """Return a validated cache or download the complete FIRMS 2025 year."""
    if FIRMS_CSV.exists() and not refresh_firms:
        try:
            parse_firms_csv_response(HTTPResponse(FIRMS_CSV.read_bytes(), "text/csv"))
            log.info("FIRMS cache hit: %s", FIRMS_CSV.name)
            return FIRMS_CSV
        except (OSError, ResponseValidationError) as exc:
            log.warning("Ignoring invalid FIRMS cache %s: %s", FIRMS_CSV, exc)

    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        if FIRMS_CSV.exists():
            try:
                parse_firms_csv_response(
                    HTTPResponse(FIRMS_CSV.read_bytes(), "text/csv")
                )
                log.warning(
                    "--refresh-firms requested without FIRMS_MAP_KEY; "
                    "retaining the validated cache"
                )
                return FIRMS_CSV
            except (OSError, ResponseValidationError):
                pass
        log.warning(
            "FIRMS_MAP_KEY is not set; historical_kde is unavailable. "
            "Set it in the environment or place a validated CSV at %s",
            FIRMS_CSV,
        )
    else:
        for source in ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"):
            try:
                row_count = download_firms_year(
                    FIRMS_CSV,
                    map_key=map_key,
                    year=2025,
                    bbox=(14.6, 49.9, 17.9, 51.9),
                    source=source,
                )
                log.info(
                    "FIRMS %s download complete: %d unique rows", source, row_count
                )
                return FIRMS_CSV
            except DataAcquisitionError as exc:
                log.warning("FIRMS %s unavailable: %s", source, exc)

    # The previously hard-coded IBL GeoServer endpoint returns HTTP 404.
    # A manually provided FeatureCollection remains supported.
    if IBL_GEOJSON.exists():
        try:
            payload = json.loads(IBL_GEOJSON.read_text(encoding="utf-8"))
            if payload.get("type") == "FeatureCollection" and isinstance(
                payload.get("features"), list
            ):
                log.info("Using manually supplied IBL fire data: %s", IBL_GEOJSON)
                return IBL_GEOJSON
        except (OSError, json.JSONDecodeError):
            pass
        log.warning("Ignoring invalid manual IBL file: %s", IBL_GEOJSON)

    if EFFIS_CSV.exists():
        log.info("Using manually supplied EFFIS fire data: %s", EFFIS_CSV)
        return EFFIS_CSV
    log.warning(
        "Historical ignition source unavailable; missing data remains null, not zero"
    )
    return None


def _download_lpis_wfs() -> bool:
    """Keep LPIS optional until ARiMR exposes a protocol-compliant WFS."""
    log.warning(
        "ARiMR LPIS automatic endpoint is unavailable/protocol-incompatible. "
        "Provide a local GeoPackage at %s; TLS verification is never disabled.",
        LPIS_CACHE,
    )
    return False


def _download_clc() -> bool:
    """Download and validate every page of the official EEA CLC layer."""
    try:
        count = download_clc_geojson(
            CLC_CACHE,
            bbox=(14.6, 49.9, 17.9, 51.9),
        )
        log.info("CLC 2018 downloaded: %d agricultural polygons", count)
        return True
    except DataAcquisitionError as exc:
        log.warning("CLC 2018 unavailable: %s", exc)
        return False


# =============================================================================
# GIS PARSERS
# =============================================================================


def _parse_firms_csv(firms_path: Path) -> tuple[list[float], list[float]]:
    """
    Parse NASA FIRMS VIIRS CSV.
    Returns (lats, lons) of confirmed fire detections (confidence >= nominal).
    """
    lats, lons = [], []
    if not firms_path or not firms_path.exists():
        return lats, lons
    try:
        with open(firms_path, encoding="utf-8") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if header is None:
                    header = [h.lower().strip() for h in line.split(",")]
                    continue
                parts = line.split(",")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                # Filter: confidence = n (nominal) or h (high); skip l (low)
                conf = row.get("confidence", "n").lower().strip()
                if conf == "l":
                    continue
                try:
                    lats.append(float(row["latitude"]))
                    lons.append(float(row["longitude"]))
                except (KeyError, ValueError):
                    pass
    except OSError as exc:
        log.warning("Could not read FIRMS CSV: %s", exc)
    log.info("FIRMS: loaded %d ignition points (confidence ≥ nominal)", len(lats))
    return lats, lons


def _parse_effis_geojson(effis_path: Path) -> tuple[list[float], list[float]]:
    """
    Parse JRC EFFIS GeoJSON fire data.
    Returns (lats, lons) of fire centroids/points.
    """
    lats, lons = [], []
    if not effis_path or not effis_path.exists():
        return lats, lons
    try:
        with open(effis_path, encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        for feat in features:
            geom = feat.get("geometry", {})
            if not geom:
                continue
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "Point" and len(coords) >= 2:
                lons.append(float(coords[0]))
                lats.append(float(coords[1]))
            elif gtype in ("Polygon", "MultiPolygon"):
                # Use centroid approximation
                flat = []
                if gtype == "Polygon":
                    flat = coords[0] if coords else []
                else:
                    for ring in coords:
                        flat.extend(ring[0] if ring else [])
                if flat:
                    mlat = sum(c[1] for c in flat) / len(flat)
                    mlon = sum(c[0] for c in flat) / len(flat)
                    lats.append(mlat)
                    lons.append(mlon)
        log.info("EFFIS: loaded %d fire locations", len(lats))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read EFFIS GeoJSON: %s", exc)
    return lats, lons


def _legacy_parse_osm_features(
    osm_path: Optional[Path],
) -> dict[str, list[tuple[float, float]]]:
    """
    Parse OSM PBF/XML for roads, railways, and tourism trails.
    Returns dict of feature_type → list of (lat, lon) centroid points.

    Uses osmium-tool (osmium) CLI if available, otherwise falls back
    to pure-Python osmread for smaller files, or returns empty on failure.
    The PBF is large; we extract only the way centroids we need.
    """
    features: dict[str, list[tuple[float, float]]] = {
        "roads": [],
        "railways": [],
        "tourism": [],
    }
    if not osm_path or not osm_path.exists():
        log.warning("OSM PBF not available — roads/railways/tourism are missing")
        return features

    # Attempt osmium-tool export to GeoJSON (lightweight, no GDAL needed)
    try:
        import subprocess
        import tempfile

        for feat_key, osm_filter in [
            ("roads", "w/highway"),
            ("railways", "w/railway=rail,narrow_gauge,tram"),
            ("tourism", "w/route=hiking,bicycle,mtb  w/highway=path,footway"),
        ]:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                [
                    "osmium",
                    "export",
                    "--geometry-types=linestring",
                    f"--filter={osm_filter}",
                    "-o",
                    tmp_path,
                    "-f",
                    "geojson",
                    "--overwrite",
                    str(osm_path),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                with open(tmp_path, encoding="utf-8") as f:
                    gj = json.load(f)
                for feat in gj.get("features", []):
                    coords = feat.get("geometry", {}).get("coordinates", [])
                    if coords:
                        # Centroid of linestring
                        mlat = sum(c[1] for c in coords) / len(coords)
                        mlon = sum(c[0] for c in coords) / len(coords)
                        features[feat_key].append((mlat, mlon))
                log.info("OSM %s: %d features", feat_key, len(features[feat_key]))
            os.unlink(tmp_path)
    except (FileNotFoundError, OSError):
        log.info("osmium-tool not found — falling back to geopandas/fiona for OSM")
        features = _parse_osm_geopandas(osm_path, features)

    return features


def _parse_osm_features(
    osm_path: Optional[Path],
) -> dict[str, list[tuple[float, float]]]:
    """Read OSM with osmium first, then Pyogrio, then Fiona."""
    empty = {"roads": [], "railways": [], "tourism": []}
    if not osm_path or not osm_path.exists():
        log.warning("OSM PBF is unavailable; roads/railways/tourism are missing")
        return empty
    if shutil.which("osmium"):
        try:
            return _legacy_parse_osm_features(osm_path)
        except (TimeoutError, subprocess.TimeoutExpired):
            log.warning("osmium timed out while parsing %s", osm_path)
        except OSError as exc:
            log.warning("osmium failed while parsing %s: %s", osm_path, exc)
    else:
        log.info("osmium-tool not found; trying GeoPandas backends")
    return _parse_osm_geopandas(osm_path, empty)


def _parse_osm_geopandas(
    osm_path: Path,
    features: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    """Read the OSM lines layer with explicit backend diagnostics."""
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        log.warning("GeoPandas is not installed; OSM sublayers are unavailable")
        return features

    engine: str | None = None
    try:
        import pyogrio

        if pyogrio.list_drivers().get("OSM") == "r":
            engine = "pyogrio"
        else:
            log.warning("Pyogrio/GDAL does not expose a readable OSM driver")
    except ModuleNotFoundError:
        log.info("Pyogrio is not installed; trying Fiona")
    except RuntimeError as exc:
        log.warning("GDAL driver discovery failed through Pyogrio: %s", exc)

    if engine is None:
        try:
            import fiona

            if "OSM" not in fiona.supported_drivers:
                log.warning("Fiona/GDAL does not expose the OSM driver")
                return features
            engine = "fiona"
        except ModuleNotFoundError:
            log.warning("Neither a usable Pyogrio backend nor Fiona is installed")
            return features

    try:
        log.info("Reading OSM 'lines' layer with %s", engine)
        gdf = gpd.read_file(str(osm_path), layer="lines", engine=engine)
        gdf = gdf.to_crs(epsg=4326)
    except TimeoutError:
        log.warning("%s timed out while reading OSM PBF", engine)
        return features
    except (OSError, ValueError, RuntimeError) as exc:
        message = str(exc)
        if "not recognized" in message.lower() or "unsupported" in message.lower():
            log.warning("%s/GDAL does not support this PBF: %s", engine, exc)
        else:
            log.warning("OSM PBF is unreadable or corrupt via %s: %s", engine, exc)
        return features

    def centroids(frame) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                continue
            centroid = geometry.centroid
            points.append((centroid.y, centroid.x))
        return points

    if "highway" in gdf.columns:
        road_types = {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "service",
            "track",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
        }
        features["roads"] = centroids(gdf[gdf["highway"].isin(road_types)])
    if "railway" in gdf.columns:
        rail_types = {"rail", "narrow_gauge", "tram", "light_rail"}
        features["railways"] = centroids(gdf[gdf["railway"].isin(rail_types)])
    tourism_mask = pd.Series(False, index=gdf.index, dtype=bool)
    if "highway" in gdf.columns:
        tourism_mask |= gdf["highway"].isin(
            {"footway", "path", "cycleway", "bridleway"}
        )
    if "foot" in gdf.columns:
        tourism_mask |= gdf["foot"].isin({"yes", "designated"})
    if "bicycle" in gdf.columns:
        tourism_mask |= gdf["bicycle"].isin({"yes", "designated"})
    features["tourism"] = centroids(gdf[tourism_mask])
    log.info(
        "OSM parsed with %s: roads=%d railways=%d tourism=%d",
        engine,
        len(features["roads"]),
        len(features["railways"]),
        len(features["tourism"]),
    )
    return features


def _read_vector_file(path: Path):
    """Read a vector dataset with explicit GeoPandas/backend diagnostics."""
    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise DataAcquisitionError("GeoPandas is not installed") from exc
    try:
        import pyogrio  # noqa: F401

        engine = "pyogrio"
    except ModuleNotFoundError:
        try:
            import fiona  # noqa: F401

            engine = "fiona"
        except ModuleNotFoundError as exc:
            raise DataAcquisitionError(
                "Neither Pyogrio nor Fiona is installed"
            ) from exc
    try:
        return gpd.read_file(str(path), engine=engine).to_crs(epsg=4326)
    except TimeoutError as exc:
        raise DataAcquisitionError(f"{engine} timed out reading {path.name}") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataAcquisitionError(
            f"{engine}/GDAL could not read {path.name}: {exc}"
        ) from exc


def _parse_power_lines(
    suln02_path: Optional[Path], suln03_path: Optional[Path]
) -> list[tuple[float, float]]:
    """
    Parse BDOT10k SULN02/SULN03 GeoPackage files (wn.gpkg / sn.gpkg).
    Returns list of (lat, lon) centroid points of power line segments.
    Accepts either .gpkg or .shp — geopandas handles both transparently.
    """
    points: list[tuple[float, float]] = []
    for gpkg_path in [suln02_path, suln03_path]:
        if not gpkg_path or not gpkg_path.exists():
            continue
        try:
            gdf = _read_vector_file(gpkg_path)
            before = len(points)
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                c = geom.centroid
                points.append((c.y, c.x))
            log.info(
                "Power lines from %s: %d segments", gpkg_path.name, len(points) - before
            )
        except DataAcquisitionError as exc:
            log.warning("Could not parse %s: %s", gpkg_path.name, exc)
    return points


def _parse_agriculture(
    lpis_path: Optional[Path], clc_path: Optional[Path]
) -> list[tuple[float, float]]:
    """
    Parse agricultural parcel centroids from LPIS (preferred) or CLC (fallback).
    Filters to arable land / grassland classes most associated with burning risk.
    """
    points: list[tuple[float, float]] = []

    # Prefer LPIS (higher resolution, 2025)
    src_path = lpis_path if (lpis_path and lpis_path.exists()) else clc_path
    if not src_path or not src_path.exists():
        return points

    try:
        gdf = _read_vector_file(src_path)

        # CLC: filter high-risk classes (arable 210-220, transitional 324, agri 200-244)
        code_field = next(
            (name for name in ("Code_18", "CODE_18", "code_18") if name in gdf.columns),
            None,
        )
        if code_field:
            gdf = gdf[gdf[code_field].astype(str).isin(CLC_AGRICULTURE_CODES)]

        for geom in gdf.geometry:
            if geom is None:
                continue
            c = geom.centroid
            points.append((c.y, c.x))
        log.info("Agriculture source %s: %d parcels", src_path.name, len(points))
    except DataAcquisitionError as exc:
        log.warning("Could not parse agriculture layer: %s", exc)

    return points


# =============================================================================
# STUB DATA  (only with the explicit --stub flag)
# =============================================================================


def _generate_stub_features(
    nodes: list[dict],
) -> dict[str, list[tuple[float, float]]]:
    """
    Generate synthetic GIS feature points around node centroids.
    Used when real GIS data is unavailable (testing, CI, demo).
    Points are placed at realistic offsets from each node.
    """
    rng = np.random.default_rng(seed=42)
    stub: dict[str, list[tuple[float, float]]] = {
        "roads": [],
        "railways": [],
        "powerlines": [],
        "tourism": [],
        "agriculture": [],
    }

    for node in nodes:
        lat, lon = node["lat"], node["lon"]
        # Roads: dense network — many points within 0-500m
        for _ in range(8):
            dlat = rng.uniform(-0.005, 0.005)
            dlon = rng.uniform(-0.008, 0.008)
            stub["roads"].append((lat + dlat, lon + dlon))
        # Railways: sparse — 1-2 lines per area
        if rng.random() > 0.4:
            stub["railways"].append(
                (lat + rng.uniform(-0.01, 0.01), lon + rng.uniform(-0.02, 0.02))
            )
        # Power lines: moderate density
        for _ in range(3):
            stub["powerlines"].append(
                (lat + rng.uniform(-0.008, 0.008), lon + rng.uniform(-0.012, 0.012))
            )
        # Tourism: trails spread wider
        for _ in range(4):
            stub["tourism"].append(
                (lat + rng.uniform(-0.02, 0.02), lon + rng.uniform(-0.03, 0.03))
            )
        # Agriculture: field edges
        for _ in range(5):
            stub["agriculture"].append(
                (lat + rng.uniform(-0.025, 0.025), lon + rng.uniform(-0.035, 0.035))
            )

    log.info(
        "Stub features generated: %s",
        " | ".join(f"{k}:{len(v)}" for k, v in stub.items()),
    )
    return stub


# =============================================================================
# CORE IGNITION SCORER
# =============================================================================


def _min_distance_m(lat: float, lon: float, points: list[tuple[float, float]]) -> float:
    """Return minimum haversine distance in metres from (lat,lon) to any point in list."""
    if not points:
        return float("inf")
    return min(haversine_m(lat, lon, p[0], p[1]) for p in points)


def _proximity_score_100(
    lat: float, lon: float, points: list[tuple[float, float]], half_dist_m: float
) -> float:
    """
    Proximity-based sublayer score in [0, 100].
    Uses minimum distance to nearest feature + exponential decay.
    """
    if not points:
        return 0.0
    d = _min_distance_m(lat, lon, points)
    return proximity_signal(d, half_dist_m) * 100.0


def compute_ignition_score(
    node: dict,
    features: dict[str, list[tuple[float, float]]],
    firms_lats: list[float],
    firms_lons: list[float],
    data_coverage: list[str],
    warnings: Optional[list[str]] = None,
) -> IgnitionScore:
    """
    Compute full ignition score for one node.

    Parameters
    ----------
    node          : enriched node dict from nodes_enriched.json
    features      : dict of feature_type → list of (lat, lon) points
    firms_lats/lons : FIRMS ignition point coordinates
    data_coverage : list of sublayer names with real (non-stub) data
    """
    lat, lon = node["lat"], node["lon"]
    W = IGNITION_WEIGHTS

    available = {name for name in data_coverage if name in SUBLAYER_NAMES}
    stub_mode = "stub" in data_coverage

    # Compute values only for layers known to be available.  An available
    # layer with no nearby feature is a real 0; an unavailable layer is None.
    raw_roads = _proximity_score_100(
        lat,
        lon,
        features.get("roads", []),
        PROXIMITY_HALF_DIST_M["roads"],
    )
    raw_railways = _proximity_score_100(
        lat,
        lon,
        features.get("railways", []),
        PROXIMITY_HALF_DIST_M["railways"],
    )
    raw_powerlines = _proximity_score_100(
        lat,
        lon,
        features.get("powerlines", []),
        PROXIMITY_HALF_DIST_M["powerlines"],
    )
    raw_tourism = _proximity_score_100(
        lat,
        lon,
        features.get("tourism", []),
        PROXIMITY_HALF_DIST_M["tourism"],
    )
    raw_agriculture = _proximity_score_100(
        lat,
        lon,
        features.get("agriculture", []),
        PROXIMITY_HALF_DIST_M["agriculture"],
    )

    # Historical KDE — Gaussian kernel over FIRMS points
    kde_raw = gaussian_kde_score(
        lat, lon, firms_lats, firms_lons, FIRMS_KDE_BANDWIDTH_M
    )
    raw_kde = kde_raw * 100.0
    values: dict[str, Optional[float]] = {
        "roads": raw_roads if "roads" in available or stub_mode else None,
        "railways": raw_railways if "railways" in available or stub_mode else None,
        "powerlines": raw_powerlines
        if "powerlines" in available or stub_mode
        else None,
        "tourism": raw_tourism if "tourism" in available or stub_mode else None,
        "agriculture": raw_agriculture
        if "agriculture" in available or stub_mode
        else None,
        "historical_kde": raw_kde
        if "historical_kde" in available or stub_mode
        else None,
    }

    coverage_weight = 0.0 if stub_mode else sum(W[name] for name in available)
    ignition_raw = float(
        np.clip(
            sum(W[name] * value for name, value in values.items() if value is not None),
            0.0,
            100.0,
        )
    )
    ignition_available = (
        float(np.clip(ignition_raw / coverage_weight, 0.0, 100.0))
        if coverage_weight > 0
        else None
    )
    valid_for_fusion = (
        not stub_mode and coverage_weight + 1e-12 >= MIN_COVERAGE_FOR_FUSION
    )
    ignition_score = ignition_raw if valid_for_fusion else None
    if stub_mode:
        score_status = "stub"
    elif coverage_weight >= 1.0 - 1e-12:
        score_status = "complete"
    elif coverage_weight > 0:
        score_status = "partial"
    else:
        score_status = "unavailable"
    missing = [name for name in SUBLAYER_NAMES if name not in available]
    score_warnings = list(warnings or [])
    if missing:
        score_warnings.append(
            "Missing sublayers are null and excluded from the available-layer normalization"
        )
    if not valid_for_fusion:
        score_warnings.append(
            f"Coverage is below the {MIN_COVERAGE_FOR_FUSION:.0%} fusion threshold"
        )

    # ── Dominant source ───────────────────────────────────────────────────
    contributions = {
        name: W[name] * value for name, value in values.items() if value is not None
    }
    dominant = max(contributions, key=contributions.get) if contributions else "none"

    return IgnitionScore(
        node_id=node["id"],
        node_name=node["name"],
        lat=lat,
        lon=lon,
        ignition_score=round(ignition_score, 2) if ignition_score is not None else None,
        ignition_score_raw=round(ignition_raw, 2),
        ignition_score_available=(
            round(ignition_available, 2) if ignition_available is not None else None
        ),
        sublayers=IgnitionSublayers(
            **{
                name: round(value, 2) if value is not None else None
                for name, value in values.items()
            }
        ),
        extreme_ignition=(
            ignition_score is not None and ignition_score >= IGNITION_HIGH
        ),
        dominant_source=dominant,
        data_coverage=sorted(available),
        missing_sublayers=missing,
        coverage_weight=round(coverage_weight, 4),
        coverage_percent=round(coverage_weight * 100.0, 2),
        score_status=score_status,
        valid_for_fusion=valid_for_fusion,
        warnings=sorted(set(score_warnings)),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# PIPELINE
# =============================================================================


def run_ignition_pipeline(
    nodes_json: Path = NODES_JSON,
    refresh_firms: bool = False,
    use_stub: bool = False,
) -> list[IgnitionScore]:
    log.info("=" * 82)
    log.info("QHDALabs Wildfire — Step 4b: Ignition Pressure Layer")
    log.info("=" * 82)

    # ── Load nodes ────────────────────────────────────────────────────────
    if not nodes_json.exists():
        raise FileNotFoundError(
            f"nodes_enriched.json not found at {nodes_json}\n"
            "Run Step 2 (enrichment) first."
        )
    with open(nodes_json, encoding="utf-8") as f:
        enriched = json.load(f)
    nodes = enriched["nodes"]
    log.info("Loaded %d nodes from %s", len(nodes), nodes_json)

    # ── GIS data ──────────────────────────────────────────────────────────
    firms_lats: list[float] = []
    firms_lons: list[float] = []
    features: dict[str, list[tuple[float, float]]] = {}
    data_coverage: list[str] = []
    pipeline_warnings: list[str] = []

    if use_stub:
        log.info("Stub mode — generating synthetic GIS features")
        features = _generate_stub_features(nodes)
        data_coverage = ["stub"]
        pipeline_warnings.append(
            "Synthetic --stub mode; result is never valid for fusion"
        )
    else:
        paths = ensure_data_available(refresh_firms=refresh_firms)

        # FIRMS / EFFIS ignition points
        firms_path = paths.get("firms")
        if firms_path and firms_path.exists():
            # EFFIS GeoJSON needs different parser than FIRMS CSV
            if firms_path.suffix == ".csv":
                firms_lats, firms_lons = _parse_firms_csv(firms_path)
            else:
                firms_lats, firms_lons = _parse_effis_geojson(firms_path)
            if firms_lats:
                data_coverage.append("historical_kde")
                log.info(
                    "Ignition points loaded: %d points from %s",
                    len(firms_lats),
                    firms_path.name,
                )
        else:
            log.warning(
                "No ignition point source available — historical_kde is missing"
            )
            pipeline_warnings.append("Historical ignition data is unavailable")

        # OSM (roads, railways, tourism)
        osm_feats = _parse_osm_features(paths.get("osm"))
        features.update(osm_feats)
        for key in ["roads", "railways", "tourism"]:
            if osm_feats.get(key):
                data_coverage.append(key)

        # Power lines — keys are "wn" and "sn" (GeoPackage stems)
        pl_points = _parse_power_lines(
            paths.get("wn"),
            paths.get("sn"),
        )
        features["powerlines"] = pl_points
        if pl_points:
            data_coverage.append("powerlines")

        # Agriculture
        ag_points = _parse_agriculture(paths.get("lpis"), paths.get("clc"))
        features["agriculture"] = ag_points
        if ag_points:
            data_coverage.append("agriculture")

        # Synthetic data is only permitted when --stub is explicit.
        if not any(features.values()):
            log.warning(
                "All GIS layers are unavailable; values remain null. "
                "Use --stub only for explicit synthetic testing."
            )
            pipeline_warnings.append("All GIS feature layers are unavailable")

    # ── Score all nodes ───────────────────────────────────────────────────
    scores = [
        compute_ignition_score(
            node,
            features,
            firms_lats,
            firms_lons,
            data_coverage,
            pipeline_warnings,
        )
        for node in nodes
    ]
    scores.sort(key=lambda s: s.ignition_score_raw, reverse=True)

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(scores)

    # ── Save ──────────────────────────────────────────────────────────────
    _save_ignition_scores(scores)

    return scores


# =============================================================================
# OUTPUT
# =============================================================================


def _print_summary(scores: list[IgnitionScore]) -> None:
    log.info("-" * 82)
    log.info(
        "%-24s %7s %7s %7s %7s %7s %7s %7s  dominant",
        "Nadleśnictwo",
        "IGN",
        "roads",
        "rail",
        "power",
        "tour",
        "agri",
        "kde",
    )
    log.info("-" * 82)

    def shown(value: Optional[float]) -> str:
        return f"{value:7.1f}" if value is not None else "      —"

    for s in scores:
        ext = " !" if s.extreme_ignition else "  "
        log.info(
            "%-24s %6.1f %s %s %s %s %s %s  %s%s",
            s.node_name[:24],
            s.ignition_score_raw,
            shown(s.sublayers.roads),
            shown(s.sublayers.railways),
            shown(s.sublayers.powerlines),
            shown(s.sublayers.tourism),
            shown(s.sublayers.agriculture),
            shown(s.sublayers.historical_kde),
            s.dominant_source,
            ext,
        )
    extreme_n = sum(1 for s in scores if s.extreme_ignition)
    log.info("-" * 82)
    log.info(
        "Extreme ignition pressure (I > %d): %d/%d nodes",
        IGNITION_HIGH,
        extreme_n,
        len(scores),
    )
    log.info(
        "Data coverage: %s",
        ", ".join(sorted(set(c for s in scores for c in s.data_coverage))) or "none",
    )
    if scores:
        log.info(
            "Coverage weight: %.0f%% | status=%s | valid_for_fusion=%s",
            scores[0].coverage_percent,
            scores[0].score_status,
            scores[0].valid_for_fusion,
        )


def _save_ignition_scores(scores: list[IgnitionScore]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.1.0",
        "module": "qhdalabs_wildfire_ignition_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rdlp": "Wrocław",
        "config": {
            "ignition_weights": IGNITION_WEIGHTS,
            "proximity_half_dist_m": PROXIMITY_HALF_DIST_M,
            "firms_kde_bandwidth_m": FIRMS_KDE_BANDWIDTH_M,
            "fei_extremity": {
                "r_threshold": FEI_EXTREMITY_R_THRESHOLD,
                "i_threshold": FEI_EXTREMITY_I_THRESHOLD,
                "r_bonus": FEI_EXTREMITY_R_BONUS,
                "i_bonus": FEI_EXTREMITY_I_BONUS,
            },
            "qies_lambda": QIES_INTERFERENCE_LAMBDA,
            "min_coverage_for_fusion": MIN_COVERAGE_FOR_FUSION,
        },
        "scores": [
            {
                "node_id": s.node_id,
                "node_name": s.node_name,
                "lat": s.lat,
                "lon": s.lon,
                "ignition_score": s.ignition_score,
                "ignition_score_raw": s.ignition_score_raw,
                "ignition_score_available": s.ignition_score_available,
                "extreme_ignition": s.extreme_ignition,
                "dominant_source": s.dominant_source,
                "data_coverage": s.data_coverage,
                "missing_sublayers": s.missing_sublayers,
                "coverage_weight": s.coverage_weight,
                "coverage_percent": s.coverage_percent,
                "score_status": s.score_status,
                "valid_for_fusion": s.valid_for_fusion,
                "warnings": s.warnings,
                "sublayers": {
                    "roads": s.sublayers.roads,
                    "railways": s.sublayers.railways,
                    "powerlines": s.sublayers.powerlines,
                    "tourism": s.sublayers.tourism,
                    "agriculture": s.sublayers.agriculture,
                    "historical_kde": s.sublayers.historical_kde,
                },
                "computed_at": s.computed_at,
            }
            for s in scores
        ],
    }

    atomic_write_json(IGNITION_OUT, payload)
    log.info("Saved %s (%d nodes)", IGNITION_OUT, len(scores))


# =============================================================================
# FUSION_V1 BRIDGE  — called directly by fusion_v1.py
# =============================================================================


def load_ignition_map(
    ignition_json: Path = IGNITION_OUT,
) -> dict[str, IgnitionScore]:
    """
    Load ignition_scores.json and return node_id → IgnitionScore dict.
    Called from fusion_v1.py to enrich each RiskScore with FEI/QIES.

    Usage in fusion_v1.py
    ----------------------
    from qhdalabs_wildfire_ignition_v1 import load_ignition_map, compute_fei, compute_qies

    ignition_map = load_ignition_map()   # returns {} with warning if file absent
    ...
    # inside compute_risk_score():
    ign = ignition_map.get(nid)
    if ign:
        readiness_100 = final_score * 100          # convert fusion [0,1] → [0,100]
        fei   = compute_fei(readiness_100, ign.ignition_score)
        qies  = compute_qies(readiness_100, ign.ignition_score)
        extreme_readiness = readiness_100 > 80
    """
    if not ignition_json.exists():
        log.warning(
            "ignition_scores.json not found — FEI/QIES will be skipped in fusion.\n"
            "Run: python qhdalabs_wildfire_ignition_v1.py"
        )
        return {}

    with open(ignition_json, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, IgnitionScore] = {}
    for rec in data.get("scores", []):
        sl = rec.get("sublayers", {})
        coverage = sorted(
            name for name in rec.get("data_coverage", []) if name in SUBLAYER_NAMES
        )
        coverage_weight = float(
            rec.get(
                "coverage_weight",
                sum(IGNITION_WEIGHTS[name] for name in coverage),
            )
        )
        valid_for_fusion = bool(
            rec.get(
                "valid_for_fusion",
                coverage_weight + 1e-12 >= MIN_COVERAGE_FOR_FUSION,
            )
        )
        raw_score = float(rec.get("ignition_score_raw", rec.get("ignition_score", 0.0)))
        available_score = rec.get("ignition_score_available")
        if available_score is None and coverage_weight > 0:
            available_score = min(100.0, raw_score / coverage_weight)
        operational_score = rec.get("ignition_score") if valid_for_fusion else None
        result[rec["node_id"]] = IgnitionScore(
            node_id=rec["node_id"],
            node_name=rec["node_name"],
            lat=rec["lat"],
            lon=rec["lon"],
            ignition_score=(
                float(operational_score) if operational_score is not None else None
            ),
            ignition_score_raw=raw_score,
            ignition_score_available=(
                float(available_score) if available_score is not None else None
            ),
            sublayers=IgnitionSublayers(
                **{
                    name: sl.get(name) if name in coverage else None
                    for name in SUBLAYER_NAMES
                }
            ),
            extreme_ignition=bool(rec.get("extreme_ignition", False)),
            dominant_source=rec.get("dominant_source", "none"),
            data_coverage=coverage,
            missing_sublayers=rec.get(
                "missing_sublayers",
                [name for name in SUBLAYER_NAMES if name not in coverage],
            ),
            coverage_weight=round(coverage_weight, 4),
            coverage_percent=float(
                rec.get("coverage_percent", coverage_weight * 100.0)
            ),
            score_status=rec.get(
                "score_status",
                "complete" if coverage_weight >= 1.0 else "partial",
            ),
            valid_for_fusion=valid_for_fusion,
            warnings=list(rec.get("warnings", [])),
            computed_at=rec.get("computed_at", data.get("generated_at", "")),
        )
    log.info("Loaded ignition map: %d nodes from %s", len(result), ignition_json)
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QHDALabs Wildfire — Ignition Pressure Layer v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qhdalabs_wildfire_ignition_v1.py              # normal run
  python qhdalabs_wildfire_ignition_v1.py --stub       # synthetic data, no GIS needed
  python qhdalabs_wildfire_ignition_v1.py --refresh-firms  # force re-download FIRMS

Output:
  topology/ignition_scores.json   (read by fusion_v1.py)
  topology/ignition_cache/        (raw GIS data, kept between runs)
        """,
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use synthetic GIS data (for testing without network/GIS access)",
    )
    parser.add_argument(
        "--refresh-firms",
        action="store_true",
        dest="refresh_firms",
        help="Force re-download of NASA FIRMS 2025 cache",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=NODES_JSON,
        help=f"Path to nodes_enriched.json (default: {NODES_JSON})",
    )
    args = parser.parse_args()

    scores = run_ignition_pipeline(
        nodes_json=args.nodes,
        refresh_firms=args.refresh_firms,
        use_stub=args.stub,
    )

    # Quick scenario check — mirrors the table from the design session
    log.info("\n%s", "=" * 60)
    log.info("SCENARIO VERIFICATION (illustrative — from node scores)")
    log.info("%s", "=" * 60)
    sample = scores[:5]
    for s in sample:
        r_synthetic = 70.0  # placeholder readiness for illustration
        if not s.valid_for_fusion or s.ignition_score is None:
            log.info(
                "%-24s  raw=%5.1f  coverage=%4.0f%%  fusion=BLOCKED  dominant=%s",
                s.node_name[:24],
                s.ignition_score_raw,
                s.coverage_percent,
                s.dominant_source,
            )
            continue
        fei = compute_fei(r_synthetic, s.ignition_score)
        qies = compute_qies(r_synthetic, s.ignition_score)
        log.info(
            "%-24s  I=%5.1f  FEI(R=70,I)=%5.1f  QIES=%5.1f  ext=%s  dominant=%s",
            s.node_name[:24],
            s.ignition_score,
            fei,
            qies,
            "YES" if s.extreme_ignition else "no",
            s.dominant_source,
        )
    log.info("%s", "=" * 60)
    log.info(
        "\nNext step: run fusion_v1.py — it will pick up ignition_scores.json automatically."
    )
