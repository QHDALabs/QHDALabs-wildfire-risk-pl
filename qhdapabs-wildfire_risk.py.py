# =============================================================================
# Project       : QHDALabs — Wildfire Risk PL
# Module        : Core prediction pipeline
# File          : qhdаlabs-wildfire_risk.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Real-time wildfire risk prediction system for Poland.
# Fetches hourly weather forecasts across a configurable lat/lon grid,
# engineers fire-relevant features, and scores each cell using a hybrid
# classical/quantum machine-learning pipeline.
#
# Architecture Context
# -----------------------------------------------------------------------------
# Standalone script — no external services required beyond Open-Meteo API.
# Outputs are self-contained files (HTML map, JSON, CSV) suitable for
# downstream dashboards or alert pipelines.
#
#   Open-Meteo API
#         │  (parallel, cached)
#   Feature Engineering (11 features + interactions)
#         │
#   ┌─────┴──────┐
#   RF Classifier   QSVC (optional, Qiskit)
#   └─────┬──────┘
#     Blended score  →  map.html / fire.json / fire.csv
#
# Key Responsibilities
# -----------------------------------------------------------------------------
# - Parallel weather data ingestion with TTL-based disk caching
# - Classical Random Forest classification with 5-fold cross-validation
# - Optional Quantum SVM (QSVC) via Qiskit ZZFeatureMap kernel
# - Sigmoid-calibrated score blending (70% RF + 30% QSVC)
# - Leaflet HTML map generation with three risk tiers
#
# Dependencies
# -----------------------------------------------------------------------------
# Runtime  : numpy, requests, scikit-learn
# Optional : qiskit >= 2.0, qiskit-machine-learning
# Data     : Open-Meteo Forecast API (https://open-meteo.com/) — free, no key
#
# Author        : Krzysztof W. Banasiewicz
# Organisation  : QHDALabs
#
# Created       : 2026
# Last Modified : 07.05.2026
#
# License
# -----------------------------------------------------------------------------
# MIT License — free to use, modify and distribute with attribution.
#
# Notes
# -----------------------------------------------------------------------------
# - Heuristic labels are a placeholder; replace with EFFIS historical fire
#   incident data for production use (https://effis.jrc.ec.europa.eu/).
# - Quantum model is silently skipped when Qiskit is absent or when the
#   training set contains only one class (common outside fire season).
# - Increase GRID_SIZE for finer spatial resolution (API calls scale as N²).
# =============================================================================

import numpy as np
import requests
import json
import csv
import os
import pickle
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

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
GRID_SIZE       = 6
ALERT_THRESHOLD = 0.7
MAX_WORKERS     = 10          # parallel API threads
CACHE_DIR       = ".cache"
CACHE_TTL       = 3600        # seconds — 1 hour
RANDOM_STATE    = 42

os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# DATA FETCH  (with caching)
# =========================
def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch hourly forecast from Open-Meteo for one grid cell."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,precipitation,soil_moisture_0_to_1cm"
        ),
        "forecast_days": 1,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_weather_cached(lat: float, lon: float) -> dict:
    """Return cached response if fresh, otherwise fetch and cache."""
    cache_key  = f"{lat:.2f}_{lon:.2f}".replace("-", "m")
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        try:
            timestamp, data = pickle.load(open(cache_path, "rb"))
            if time.time() - timestamp < CACHE_TTL:
                return data
        except Exception:
            pass  # corrupt cache — re-fetch

    data = fetch_weather(lat, lon)
    pickle.dump((time.time(), data), open(cache_path, "wb"))
    return data


def summarize_hourly(raw_data: dict) -> dict:
    """Collapse 24 hourly readings into a single summary dict."""
    hourly = raw_data["hourly"]
    return {
        "temp":  max(hourly["temperature_2m"]),
        "rh":    min(hourly["relative_humidity_2m"]),
        "wind":  float(np.mean(hourly["wind_speed_10m"])),
        "rain":  sum(hourly["precipitation"]),
        "soil":  min(hourly["soil_moisture_0_to_1cm"]),
    }

# =========================
# FEATURE ENGINEERING
# =========================
def build_feature_vector(summary: dict) -> list:
    """
    Return an 11-element feature vector.
    Interaction terms capture non-linear fire-risk relationships.
    """
    temp = summary["temp"]
    rh   = summary["rh"]
    wind = summary["wind"]
    rain = summary["rain"]
    soil = summary["soil"]

    return [
        temp,
        rh,
        wind,
        rain,
        soil,
        temp * (100 - rh),        # heat × dryness
        wind * (100 - rh),        # wind-driven evaporation
        soil * rh,                # moisture retention
        rain * soil,              # rain × soil moisture
        temp / (soil + 0.01),     # dryness ratio
        wind / (soil + 0.01),     # wind dryness ratio
    ]


def heuristic_label(summary: dict) -> int:
    """
    Rule-based fire-risk label used as a PLACEHOLDER only.

    ⚠️  Replace with real historical fire incident labels
        (e.g. from EFFIS: https://effis.jrc.ec.europa.eu/)
        before using this model in any serious context.
        Training on this label means the model learns to reproduce
        the rule below — not actual fire risk.
    """
    return int(
        summary["temp"]  > 25  and
        summary["rh"]    < 40  and
        summary["wind"]  > 10  and
        summary["soil"]  < 0.2
    )

# =========================
# GRID GENERATION
# =========================
def poland_grid() -> list[tuple[float, float]]:
    """Generate a lat/lon grid covering Poland."""
    lats = np.linspace(49.0, 54.5, GRID_SIZE)
    lons = np.linspace(14.0, 24.0, GRID_SIZE)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]

# =========================
# PARALLEL DATASET BUILD
# =========================
def fetch_cell(lat: float, lon: float) -> tuple | None:
    """Fetch + summarise one grid cell. Returns None on failure."""
    try:
        raw      = fetch_weather_cached(lat, lon)
        summary  = summarize_hourly(raw)
        features = build_feature_vector(summary)
        label    = heuristic_label(summary)
        return lat, lon, summary, features, label
    except requests.RequestException as exc:
        log.warning("Network error  (%5.2f, %5.2f): %s", lat, lon, exc)
    except (KeyError, ValueError) as exc:
        log.warning("Bad response   (%5.2f, %5.2f): %s", lat, lon, exc)
    return None


def build_dataset() -> tuple:
    """
    Fetch all grid cells in parallel.
    Returns (X, y, cell_metadata).
    """
    grid   = poland_grid()
    X, y   = [], []
    cells  = []

    log.info("Fetching %d grid cells with %d workers …", len(grid), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_cell, lat, lon): (lat, lon)
            for lat, lon in grid
        }
        for future in as_completed(future_map):
            result = future.result()
            if result is None:
                continue
            lat, lon, summary, feature_vec, label = result
            X.append(feature_vec)
            y.append(label)
            cells.append((lat, lon, summary))

    log.info("Dataset ready: %d cells", len(X))
    return np.array(X), np.array(y), cells

# =========================
# CLASSICAL MODEL
# =========================
def train_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Pipeline:
    """Train a StandardScaler + RandomForest pipeline and cross-validate."""
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
        )),
    ])
    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
    log.info("Classical CV F1: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    return clf

# =========================
# QUANTUM MODEL  (optional)
# =========================
def train_quantum(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """
    Attempt to train a Quantum SVM via Qiskit.
    Falls back gracefully if Qiskit is not installed.

    Returns (model, pca, scaler) or (None, None, None).

    NOTE: The QSVC outputs hard class labels (0/1), not probabilities.
    We calibrate the score in score_quantum() using the decision_function
    approach so it can be blended with the classical probability on the
    same 0–1 scale.
    """
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC
        try:
            from qiskit.primitives import StatevectorSampler as Sampler
        except ImportError:
            from qiskit.primitives import Sampler
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        # QSVM needs at least 2 classes
        if len(np.unique(y_train)) < 2:
            log.warning("Quantum model skipped: only one class in training data "
                        "(heuristic label threshold not triggered today).")
            return None, None, None

        log.info("Training quantum model …")

        pca = PCA(n_components=4, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_train)

        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_q    = scaler.fit_transform(X_pca)

        # Use new functional API (Qiskit 2.1+), fall back to class for older versions
        try:
            from qiskit.circuit.library import zz_feature_map
            fmap = zz_feature_map(feature_dimension=4, reps=2)
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap
            fmap = ZZFeatureMap(feature_dimension=4, reps=2)

        sampler  = Sampler()
        fidelity = ComputeUncompute(sampler)
        kernel   = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)

        model = QSVC(quantum_kernel=kernel)
        model.fit(X_q, y_train)

        log.info("Quantum model trained.")
        return model, pca, scaler

    except Exception as exc:
        log.warning("Quantum model skipped: %s: %s", type(exc).__name__, exc)

    return None, None, None


def score_quantum(
    model,
    pca: PCA,
    scaler: MinMaxScaler,
    feature_vec: np.ndarray,
) -> float:
    """
    Return a calibrated 0–1 quantum risk score.

    QSVC.predict returns a hard label (0/1). We use decision_function
    and sigmoid-normalise it so the score is on the same scale as the
    classical predict_proba output, making the weighted blend meaningful.
    """
    X_pca = scaler.transform(pca.transform(feature_vec))
    raw   = model.decision_function(X_pca)[0]
    # sigmoid calibration: maps (-∞, +∞) → (0, 1)
    return float(1 / (1 + np.exp(-raw)))

# =========================
# SCORING
# =========================
def score_cells(
    cells: list,
    X: np.ndarray,
    clf: Pipeline,
    q_model,
    q_pca,
    q_scaler,
) -> list:
    """Compute final risk scores for every grid cell."""
    results = []

    for i, (lat, lon, summary) in enumerate(cells):
        feature_vec = X[i : i + 1]

        proba = clf.predict_proba(feature_vec)[0]
        # If only one class was seen during training, proba has shape (1,)
        if len(proba) == 1:
            # The single class tells us which direction to default
            classical_prob = 0.0 if clf.classes_[0] == 0 else 1.0
        else:
            classical_prob = proba[1]

        if q_model is not None:
            quantum_prob = score_quantum(q_model, q_pca, q_scaler, feature_vec)
            final_score  = 0.7 * classical_prob + 0.3 * quantum_prob
        else:
            final_score = classical_prob

        results.append({
            "lat":   round(lat, 4),
            "lon":   round(lon, 4),
            "risk":  round(float(final_score), 4),
            "alert": bool(final_score > ALERT_THRESHOLD),
        })

    return results

# =========================
# OUTPUT  — JSON + CSV
# =========================
def save_results(results: list) -> None:
    with open("fire.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Saved fire.json")

    with open("fire.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info("Saved fire.csv")

# =========================
# OUTPUT  — MAP
# =========================
def generate_map(results: list) -> None:
    """Write a Leaflet HTML map coloured by risk level."""
    circles = []
    for r in results:
        if r["alert"]:
            color, fill_opacity = "#A32D2D", 0.6
        elif r["risk"] > 0.4:
            color, fill_opacity = "#854F0B", 0.45
        else:
            color, fill_opacity = "#3B6D11", 0.3

        risk_str  = f"{r['risk']:.2f}"
        alert_str = "⚠ ALERT" if r["alert"] else "OK"
        circles.append(
            f"L.circle([{r['lat']},{r['lon']}],"
            f"{{color:'{color}',fillColor:'{color}',"
            f"fillOpacity:{fill_opacity},radius:20000}})"
            f".addTo(map)"
            f".bindPopup('Risk: {risk_str} — {alert_str}');"
        )

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <title>Wildfire Risk Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>html,body,#map{{margin:0;height:100vh}}</style>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map').setView([52, 19], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OpenStreetMap contributors'}}).addTo(map);

var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
  var d = L.DomUtil.create('div','legend');
  d.style.cssText='background:white;padding:8px 12px;border-radius:6px;font:13px sans-serif;line-height:1.7';
  d.innerHTML='<b>Fire risk</b><br>'
    +'<span style="color:#A32D2D">●</span> Alert (&gt;{ALERT_THRESHOLD})<br>'
    +'<span style="color:#854F0B">●</span> Elevated (0.4–{ALERT_THRESHOLD})<br>'
    +'<span style="color:#3B6D11">●</span> Low (&lt;0.4)';
  return d;
}};
legend.addTo(map);

{chr(10).join(circles)}
</script>
</body></html>"""

    with open("map.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("Saved map.html")

# =========================
# MAIN
# =========================
def main() -> None:
    # 1. Fetch data in parallel
    X, y, cells = build_dataset()

    if len(X) == 0:
        log.error("No data fetched — check network and API availability.")
        return

    # 2. Train/evaluate classical model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    clf = train_classical(X_train, y_train, X, y)
    log.info("Classification report (held-out test set):\n%s",
             classification_report(y_test, clf.predict(X_test), zero_division=0))

    # 3. Optionally train quantum model
    q_model, q_pca, q_scaler = train_quantum(X_train, y_train)

    # 4. Score every cell
    results = score_cells(cells, X, clf, q_model, q_pca, q_scaler)

    alert_count = sum(1 for r in results if r["alert"])
    log.info("Alerts: %d / %d cells", alert_count, len(results))

    # 5. Save outputs
    save_results(results)
    generate_map(results)

    log.info("Done → map.html  fire.json  fire.csv")


if __name__ == "__main__":
    main()