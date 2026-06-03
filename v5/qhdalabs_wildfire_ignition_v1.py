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
import sys
import time
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR   = Path("topology")
CACHE_DIR    = OUTPUT_DIR / "ignition_cache"
FIRMS_CSV    = CACHE_DIR / "firms_viirs_dolnoslaskie_2025.csv"
OSM_PBF      = CACHE_DIR / "dolnoslaskie-latest.osm.pbf"
SULN02_ZIP   = CACHE_DIR / "suln02_wn.zip"
SULN03_ZIP   = CACHE_DIR / "suln03_sn.zip"
LPIS_CACHE   = CACHE_DIR / "lpis_dolnoslaskie.gpkg"
CLC_CACHE    = CACHE_DIR / "clc18_dolnoslaskie.gpkg"

IGNITION_OUT = OUTPUT_DIR / "ignition_scores.json"
NODES_JSON   = OUTPUT_DIR / "nodes_enriched.json"

# =============================================================================
# IGNITION CONFIG
# =============================================================================

# Sublayer weights — sum must equal 1.0
# Calibrated for Dolny Śląsk; tune after backtesting against FIRMS 2025
IGNITION_WEIGHTS = {
    "roads":          0.25,   # human access — highest frequency ignition cause
    "railways":       0.20,   # sparks from braking / traction, esp. non-electric
    "powerlines":     0.15,   # arc discharge, conductor sag, fallen lines
    "tourism":        0.15,   # footpaths, campsites — recreational pressure
    "agriculture":    0.15,   # crop/stubble burning, machinery, field edges
    "historical_kde": 0.10,   # past ignition hotspots (FIRMS VIIRS proxy)
}
assert abs(sum(IGNITION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Proximity decay: exponential half-distance in metres for each infrastructure type.
# At d=0 → signal=1.0; at d=HALF_DIST → signal=0.5; at d→∞ → signal→0
PROXIMITY_HALF_DIST_M = {
    "roads":      300,    # main roads; ignition peaks within 300 m
    "railways":   200,    # railway corridor: very localised spark risk
    "powerlines": 150,    # arc/fault risk drops off sharply
    "tourism":    500,    # trail density — broader recreational spread
    "agriculture": 800,   # field-edge interface — wider transition zone
}

# KDE bandwidth for historical ignition points (metres)
FIRMS_KDE_BANDWIDTH_M = 5_000   # 5 km Gaussian kernel

# FEI fusion parameters (mirrors what fusion_v1.py will compute)
FEI_EXTREMITY_R_THRESHOLD = 80   # readiness above this triggers extremity bonus
FEI_EXTREMITY_I_THRESHOLD = 80   # ignition above this triggers extremity bonus
FEI_EXTREMITY_R_BONUS     = 12   # max bonus points at R=100
FEI_EXTREMITY_I_BONUS     = 8    # max bonus points at I=100

# QIES quantum-inspired coherence parameter
QIES_INTERFERENCE_LAMBDA  = 0.25  # destructive interference strength

# Alert thresholds for ignition_score standalone (informational, not operational)
IGNITION_HIGH     = 75
IGNITION_MODERATE = 50

# =============================================================================
# DATA DOWNLOAD REGISTRY
# =============================================================================
# Each entry: (local_path, url, description, is_zip, zip_target_name)
# URLs verified June 2026; GIS-Support bundles are free, no auth required.

DATA_REGISTRY = [
    # Power lines — GIS-Support pre-packaged BDOT10k extracts
    (
        SULN02_ZIP,
        "https://gis-support.pl/downloads/suln02.zip",
        "BDOT10k SULN02 — linie wysokiego napięcia WN (~7 MB)",
        True,
        "suln02.shp",
    ),
    (
        SULN03_ZIP,
        "https://gis-support.pl/downloads/suln03.zip",
        "BDOT10k SULN03 — linie średniego napięcia SN (~169 MB)",
        True,
        "suln03.shp",
    ),
    # NASA FIRMS VIIRS 375m — historical 2025 archive for Poland bounding box
    # Bounding box: W=14.12 E=24.15 S=49.00 N=54.90 (covers all of Poland)
    # Public archive endpoint — no API key required for historical data
    (
        FIRMS_CSV,
        (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            "VIIRS_SNPP_NRT/2025-01-01/2025-12-31/14.12,49.00,24.15,54.90"
        ),
        "NASA FIRMS VIIRS 375m — aktywne pożary Polska 2025",
        False,
        None,
    ),
]

# OSM and LPIS/CLC require dedicated download functions (see below)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class IgnitionSublayers:
    roads:          float = 0.0   # [0,100]
    railways:       float = 0.0
    powerlines:     float = 0.0
    tourism:        float = 0.0
    agriculture:    float = 0.0
    historical_kde: float = 0.0


@dataclass
class IgnitionScore:
    node_id:          str
    node_name:        str
    lat:              float
    lon:              float
    ignition_score:   float          # [0,100] — main output
    sublayers:        IgnitionSublayers
    extreme_ignition: bool           # I > IGNITION_HIGH
    # Flags for fusion_v1
    dominant_source:  str            # sublayer with highest contribution
    data_coverage:    list[str]      # which sublayers have real data
    computed_at:      str            # ISO timestamp
    # Populated by fusion_v1 when both R and I are known
    fei:              Optional[float] = None
    qies:             Optional[float] = None


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
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
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

    fei = fei_base + FEI_EXTREMITY_R_BONUS * r_extreme + FEI_EXTREMITY_I_BONUS * i_extreme
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
    imbalance   = abs(r - i) / 100.0
    coherence   = 1.0 - QIES_INTERFERENCE_LAMBDA * math.sin(math.pi * imbalance)
    qies        = (r * i / 100.0) * coherence
    return float(np.clip(qies, 0.0, 100.0))


# =============================================================================
# GIS DATA DOWNLOAD
# =============================================================================

def _download_file(url: str, dest: Path, description: str, chunk_size: int = 65536) -> bool:
    """Download url to dest with progress logging. Returns True on success."""
    log.info("Downloading: %s", description)
    log.info("  URL: %s", url[:80] + ("…" if len(url) > 80 else ""))
    log.info("  → %s", dest)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "QHDALabs-Wildfire/1.0 (research; contact@qhdalabs.pl)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded % (5 * 1024 * 1024) < chunk_size:   # log every ~5 MB
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

    # ── Standard registry items ───────────────────────────────────────────
    for local_path, url, desc, is_zip, zip_target in DATA_REGISTRY:
        key = local_path.stem
        if key == FIRMS_CSV.stem and not refresh_firms and local_path.exists():
            log.info("FIRMS cache hit: %s (use --refresh-firms to re-download)", local_path.name)
            paths["firms"] = local_path
            continue
        if not local_path.exists():
            ok = _download_file(url, local_path, desc)
            if not ok:
                paths[key] = None
                continue
        if is_zip:
            shp = _unzip_first_shp(local_path, CACHE_DIR)
            paths[key] = shp
        else:
            paths[key] = local_path

    # ── OSM Geofabrik (dolnoslaskie) ──────────────────────────────────────
    # OSM PBF is large (~60 MB); we attempt download but fall back gracefully
    if not OSM_PBF.exists():
        ok = _download_file(
            "https://download.geofabrik.de/europe/poland/dolnoslaskie-latest.osm.pbf",
            OSM_PBF,
            "OSM Geofabrik — dolnośląskie extract (~60 MB)",
        )
        paths["osm"] = OSM_PBF if ok else None
    else:
        log.info("OSM cache hit: %s", OSM_PBF.name)
        paths["osm"] = OSM_PBF

    # ── ARiMR LPIS — WFS fallback to CLC if unavailable ──────────────────
    if not LPIS_CACHE.exists():
        log.info("LPIS cache not found — attempting WFS download (ARiMR, public XII 2025)")
        lpis_ok = _download_lpis_wfs()
        paths["lpis"] = LPIS_CACHE if lpis_ok else None
    else:
        log.info("LPIS cache hit: %s", LPIS_CACHE.name)
        paths["lpis"] = LPIS_CACHE

    # ── CORINE Land Cover 2018 (fallback agriculture layer) ───────────────
    if not CLC_CACHE.exists():
        log.info("CLC cache not found — attempting Copernicus download")
        clc_ok = _download_clc()
        paths["clc"] = CLC_CACHE if clc_ok else None
    else:
        log.info("CLC cache hit: %s", CLC_CACHE.name)
        paths["clc"] = CLC_CACHE

    return paths


def _download_lpis_wfs() -> bool:
    """
    Download agricultural parcel boundaries from ARiMR LPIS WFS.
    Public endpoint available since XII 2025 — no auth required.
    Bounding box: Dolny Śląsk approx. bbox.
    """
    # ARiMR WFS — public endpoint (activated Dec 2025)
    # Returns GeoJSON; we save as-is and convert to GPKG if geopandas available
    wfs_url = (
        "https://geoportal.arimr.gov.pl/wfs?"
        "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        "&TYPENAMES=ms:lpis_dzialki_referencyjne"
        "&BBOX=14.6,49.9,17.9,51.9,EPSG:4326"
        "&OUTPUTFORMAT=application/json"
        "&COUNT=50000"
    )
    tmp = CACHE_DIR / "lpis_raw.geojson"
    ok = _download_file(wfs_url, tmp, "ARiMR LPIS WFS — działki dolnośląskie")
    if ok:
        tmp.rename(LPIS_CACHE)
    return ok


def _download_clc() -> bool:
    """Download CORINE Land Cover 2018 for Poland from Copernicus."""
    # CLC 2018 Poland vector — Copernicus Land Monitoring Service
    clc_url = (
        "https://land.copernicus.eu/en/products/corine-land-cover/"
        "clc2018?tab=download"
        # Note: Copernicus requires registration for direct download.
        # Fallback: EEA GeoServer WFS for Poland subset
    )
    # EEA WFS — public, no registration
    eea_wfs = (
        "https://bio.discomap.eea.europa.eu/arcgis/rest/services/"
        "Land/CLC_2018_WM/MapServer/0/query?"
        "where=1%3D1&geometry=14.6%2C49.9%2C17.9%2C51.9"
        "&geometryType=esriGeometryEnvelope&inSR=4326"
        "&spatialRel=esriSpatialRelIntersects"
        "&outFields=CODE_18%2CREMARK&f=geojson"
    )
    tmp = CACHE_DIR / "clc_raw.geojson"
    ok = _download_file(eea_wfs, tmp, "CORINE Land Cover 2018 — dolnośląskie subset")
    if ok:
        tmp.rename(CLC_CACHE)
    return ok


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


def _parse_osm_features(osm_path: Optional[Path]) -> dict[str, list[tuple[float, float]]]:
    """
    Parse OSM PBF/XML for roads, railways, and tourism trails.
    Returns dict of feature_type → list of (lat, lon) centroid points.

    Uses osmium-tool (osmium) CLI if available, otherwise falls back
    to pure-Python osmread for smaller files, or returns empty on failure.
    The PBF is large; we extract only the way centroids we need.
    """
    features: dict[str, list[tuple[float, float]]] = {
        "roads": [], "railways": [], "tourism": [],
    }
    if not osm_path or not osm_path.exists():
        log.warning("OSM PBF not available — roads/railways/tourism from stub")
        return features

    # Attempt osmium-tool export to GeoJSON (lightweight, no GDAL needed)
    try:
        import subprocess
        import tempfile

        for feat_key, osm_filter in [
            ("roads",     "w/highway"),
            ("railways",  "w/railway=rail,narrow_gauge,tram"),
            ("tourism",   "w/route=hiking,bicycle,mtb  w/highway=path,footway"),
        ]:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                ["osmium", "export", "--geometry-types=linestring",
                 f"--filter={osm_filter}",
                 "-o", tmp_path, "-f", "geojson",
                 "--overwrite", str(osm_path)],
                capture_output=True, timeout=120,
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


def _parse_osm_geopandas(
    osm_path: Path,
    features: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    """Fallback OSM parser using geopandas + fiona (if available)."""
    try:
        import geopandas as gpd
        layers = gpd.list_layers(str(osm_path))
        log.info("OSM layers via fiona: %s", layers)
    except ImportError:
        log.warning("geopandas not available — OSM sublayers will use stub values")
    except Exception as exc:
        log.warning("OSM parse failed: %s", exc)
    return features


def _parse_power_lines(suln02_path: Optional[Path], suln03_path: Optional[Path]) -> list[tuple[float, float]]:
    """
    Parse BDOT10k SULN02/SULN03 SHP files.
    Returns list of (lat, lon) centroid points of power line segments.
    """
    points: list[tuple[float, float]] = []
    for shp_path in [suln02_path, suln03_path]:
        if not shp_path or not shp_path.exists():
            continue
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(shp_path)).to_crs(epsg=4326)
            for geom in gdf.geometry:
                if geom is None:
                    continue
                c = geom.centroid
                points.append((c.y, c.x))
            log.info("Power lines from %s: %d segments", shp_path.name, len(points))
        except ImportError:
            log.warning("geopandas not available — powerlines sublayer will use stub")
            break
        except Exception as exc:
            log.warning("Could not parse %s: %s", shp_path.name, exc)
    return points


def _parse_agriculture(lpis_path: Optional[Path], clc_path: Optional[Path]) -> list[tuple[float, float]]:
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
        import geopandas as gpd
        gdf = gpd.read_file(str(src_path)).to_crs(epsg=4326)

        # CLC: filter high-risk classes (arable 210-220, transitional 324, agri 200-244)
        if "CODE_18" in gdf.columns:
            arable_codes = [str(c) for c in range(200, 245)]
            gdf = gdf[gdf["CODE_18"].astype(str).isin(arable_codes)]

        for geom in gdf.geometry:
            if geom is None:
                continue
            c = geom.centroid
            points.append((c.y, c.x))
        log.info("Agriculture source %s: %d parcels", src_path.name, len(points))
    except ImportError:
        log.warning("geopandas not available — agriculture sublayer will use stub")
    except Exception as exc:
        log.warning("Could not parse agriculture layer: %s", exc)

    return points


# =============================================================================
# STUB DATA  (--stub flag or when GIS download fails)
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
        "roads": [], "railways": [], "powerlines": [], "tourism": [], "agriculture": [],
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
            stub["railways"].append((lat + rng.uniform(-0.01, 0.01),
                                     lon + rng.uniform(-0.02, 0.02)))
        # Power lines: moderate density
        for _ in range(3):
            stub["powerlines"].append((lat + rng.uniform(-0.008, 0.008),
                                       lon + rng.uniform(-0.012, 0.012)))
        # Tourism: trails spread wider
        for _ in range(4):
            stub["tourism"].append((lat + rng.uniform(-0.02, 0.02),
                                    lon + rng.uniform(-0.03, 0.03)))
        # Agriculture: field edges
        for _ in range(5):
            stub["agriculture"].append((lat + rng.uniform(-0.025, 0.025),
                                        lon + rng.uniform(-0.035, 0.035)))

    log.info("Stub features generated: %s",
             " | ".join(f"{k}:{len(v)}" for k, v in stub.items()))
    return stub


# =============================================================================
# CORE IGNITION SCORER
# =============================================================================

def _min_distance_m(lat: float, lon: float, points: list[tuple[float, float]]) -> float:
    """Return minimum haversine distance in metres from (lat,lon) to any point in list."""
    if not points:
        return float("inf")
    return min(haversine_m(lat, lon, p[0], p[1]) for p in points)


def _proximity_score_100(lat: float, lon: float,
                          points: list[tuple[float, float]],
                          half_dist_m: float) -> float:
    """
    Proximity-based sublayer score in [0, 100].
    Uses minimum distance to nearest feature + exponential decay.
    """
    if not points:
        return 0.0
    d = _min_distance_m(lat, lon, points)
    return proximity_signal(d, half_dist_m) * 100.0


def compute_ignition_score(
    node:       dict,
    features:   dict[str, list[tuple[float, float]]],
    firms_lats: list[float],
    firms_lons: list[float],
    data_coverage: list[str],
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

    # ── Sublayer scores ───────────────────────────────────────────────────
    s_roads = _proximity_score_100(
        lat, lon, features.get("roads", []),
        PROXIMITY_HALF_DIST_M["roads"],
    )
    s_railways = _proximity_score_100(
        lat, lon, features.get("railways", []),
        PROXIMITY_HALF_DIST_M["railways"],
    )
    s_powerlines = _proximity_score_100(
        lat, lon, features.get("powerlines", []),
        PROXIMITY_HALF_DIST_M["powerlines"],
    )
    s_tourism = _proximity_score_100(
        lat, lon, features.get("tourism", []),
        PROXIMITY_HALF_DIST_M["tourism"],
    )
    s_agriculture = _proximity_score_100(
        lat, lon, features.get("agriculture", []),
        PROXIMITY_HALF_DIST_M["agriculture"],
    )

    # Historical KDE — Gaussian kernel over FIRMS points
    kde_raw = gaussian_kde_score(lat, lon, firms_lats, firms_lons, FIRMS_KDE_BANDWIDTH_M)
    s_kde = kde_raw * 100.0

    # ── Weighted fusion ───────────────────────────────────────────────────
    ignition_raw = (
        W["roads"]          * s_roads
        + W["railways"]       * s_railways
        + W["powerlines"]     * s_powerlines
        + W["tourism"]        * s_tourism
        + W["agriculture"]    * s_agriculture
        + W["historical_kde"] * s_kde
    )
    ignition_score = float(np.clip(ignition_raw, 0.0, 100.0))

    # ── Dominant source ───────────────────────────────────────────────────
    contributions = {
        "roads":          W["roads"]          * s_roads,
        "railways":       W["railways"]       * s_railways,
        "powerlines":     W["powerlines"]     * s_powerlines,
        "tourism":        W["tourism"]        * s_tourism,
        "agriculture":    W["agriculture"]    * s_agriculture,
        "historical_kde": W["historical_kde"] * s_kde,
    }
    dominant = max(contributions, key=contributions.get)

    return IgnitionScore(
        node_id         = node["id"],
        node_name       = node["name"],
        lat             = lat,
        lon             = lon,
        ignition_score  = round(ignition_score, 2),
        sublayers       = IgnitionSublayers(
            roads          = round(s_roads, 2),
            railways       = round(s_railways, 2),
            powerlines     = round(s_powerlines, 2),
            tourism        = round(s_tourism, 2),
            agriculture    = round(s_agriculture, 2),
            historical_kde = round(s_kde, 2),
        ),
        extreme_ignition = ignition_score >= IGNITION_HIGH,
        dominant_source  = dominant,
        data_coverage    = data_coverage,
        computed_at      = datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# PIPELINE
# =============================================================================

def run_ignition_pipeline(
    nodes_json:    Path = NODES_JSON,
    refresh_firms: bool = False,
    use_stub:      bool = False,
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
    features:   dict[str, list[tuple[float, float]]] = {}
    data_coverage: list[str] = []

    if use_stub:
        log.info("Stub mode — generating synthetic GIS features")
        features = _generate_stub_features(nodes)
        data_coverage = ["stub"]
    else:
        paths = ensure_data_available(refresh_firms=refresh_firms)

        # FIRMS ignition points
        if paths.get("firms"):
            firms_lats, firms_lons = _parse_firms_csv(paths["firms"])
            if firms_lats:
                data_coverage.append("historical_kde")
        else:
            log.warning("FIRMS data unavailable — historical_kde sublayer = 0")

        # OSM (roads, railways, tourism)
        osm_feats = _parse_osm_features(paths.get("osm"))
        features.update(osm_feats)
        for key in ["roads", "railways", "tourism"]:
            if osm_feats.get(key):
                data_coverage.append(key)

        # Power lines
        pl_points = _parse_power_lines(
            paths.get(SULN02_ZIP.stem),
            paths.get(SULN03_ZIP.stem),
        )
        features["powerlines"] = pl_points
        if pl_points:
            data_coverage.append("powerlines")

        # Agriculture
        ag_points = _parse_agriculture(paths.get("lpis"), paths.get("clc"))
        features["agriculture"] = ag_points
        if ag_points:
            data_coverage.append("agriculture")

        # If all GIS failed, warn and fall back to stub
        if not any(features.values()):
            log.warning(
                "All GIS downloads failed — falling back to stub data.\n"
                "Scores will be approximate. Check network / GIS availability."
            )
            features = _generate_stub_features(nodes)
            data_coverage = ["stub"]

    # ── Score all nodes ───────────────────────────────────────────────────
    scores = [
        compute_ignition_score(node, features, firms_lats, firms_lons, data_coverage)
        for node in nodes
    ]
    scores.sort(key=lambda s: s.ignition_score, reverse=True)

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
    log.info("%-24s %7s %7s %7s %7s %7s %7s %7s  dominant",
             "Nadleśnictwo", "IGN", "roads", "rail", "power", "tour", "agri", "kde")
    log.info("-" * 82)
    for s in scores:
        ext = " !" if s.extreme_ignition else "  "
        log.info("%-24s %6.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f  %s%s",
                 s.node_name[:24],
                 s.ignition_score,
                 s.sublayers.roads,
                 s.sublayers.railways,
                 s.sublayers.powerlines,
                 s.sublayers.tourism,
                 s.sublayers.agriculture,
                 s.sublayers.historical_kde,
                 s.dominant_source,
                 ext)
    extreme_n = sum(1 for s in scores if s.extreme_ignition)
    log.info("-" * 82)
    log.info("Extreme ignition pressure (I > %d): %d/%d nodes",
             IGNITION_HIGH, extreme_n, len(scores))
    log.info("Data coverage: %s", ", ".join(sorted(set(
        c for s in scores for c in s.data_coverage
    ))) or "none")


def _save_ignition_scores(scores: list[IgnitionScore]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version":      "1.0.0",
        "module":       "qhdalabs_wildfire_ignition_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rdlp":         "Wrocław",
        "config": {
            "ignition_weights":         IGNITION_WEIGHTS,
            "proximity_half_dist_m":    PROXIMITY_HALF_DIST_M,
            "firms_kde_bandwidth_m":    FIRMS_KDE_BANDWIDTH_M,
            "fei_extremity": {
                "r_threshold": FEI_EXTREMITY_R_THRESHOLD,
                "i_threshold": FEI_EXTREMITY_I_THRESHOLD,
                "r_bonus":     FEI_EXTREMITY_R_BONUS,
                "i_bonus":     FEI_EXTREMITY_I_BONUS,
            },
            "qies_lambda": QIES_INTERFERENCE_LAMBDA,
        },
        "scores": [
            {
                "node_id":         s.node_id,
                "node_name":       s.node_name,
                "lat":             s.lat,
                "lon":             s.lon,
                "ignition_score":  s.ignition_score,
                "extreme_ignition": s.extreme_ignition,
                "dominant_source": s.dominant_source,
                "data_coverage":   s.data_coverage,
                "sublayers": {
                    "roads":          s.sublayers.roads,
                    "railways":       s.sublayers.railways,
                    "powerlines":     s.sublayers.powerlines,
                    "tourism":        s.sublayers.tourism,
                    "agriculture":    s.sublayers.agriculture,
                    "historical_kde": s.sublayers.historical_kde,
                },
                "computed_at": s.computed_at,
            }
            for s in scores
        ],
    }

    with open(IGNITION_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
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
        sl = rec["sublayers"]
        result[rec["node_id"]] = IgnitionScore(
            node_id         = rec["node_id"],
            node_name       = rec["node_name"],
            lat             = rec["lat"],
            lon             = rec["lon"],
            ignition_score  = rec["ignition_score"],
            sublayers       = IgnitionSublayers(**sl),
            extreme_ignition= rec["extreme_ignition"],
            dominant_source = rec["dominant_source"],
            data_coverage   = rec["data_coverage"],
            computed_at     = rec["computed_at"],
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
        nodes_json    = args.nodes,
        refresh_firms = args.refresh_firms,
        use_stub      = args.stub,
    )

    # Quick scenario check — mirrors the table from the design session
    log.info("\n%s", "=" * 60)
    log.info("SCENARIO VERIFICATION (illustrative — from node scores)")
    log.info("%s", "=" * 60)
    sample = scores[:5]
    for s in sample:
        r_synthetic = 70.0   # placeholder readiness for illustration
        fei  = compute_fei(r_synthetic, s.ignition_score)
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
    log.info("\nNext step: run fusion_v1.py — it will pick up ignition_scores.json automatically.")
    log.info("Add to fusion_v1.py imports:")
    log.info("  from qhdalabs_wildfire_ignition_v1 import load_ignition_map, compute_fei, compute_qies")
