# =============================================================================
# Project       : QHDALabs — Wildfire Risk PL
# Module        : Core prediction pipeline
# File          : qhdalabs-wildfire_risk_v2.py
# Version       : 2.0.0
#
# Description
# -----------------------------------------------------------------------------
# Real-time wildfire risk prediction system for Poland.
# Fetches hourly weather forecasts across a configurable lat/lon grid,
# engineers fire-relevant features (weather + NDVI vegetation stress +
# terrain slope), and scores each cell using a hybrid classical/quantum
# machine-learning pipeline.
#
# Architecture Context
# -----------------------------------------------------------------------------
#
#   Open-Meteo API + Open-Elevation API + NDVI (Sentinel-2 proxy)
#         │  (parallel, cached, TTL 1h)
#   Feature Engineering (16 features: weather + vegetation + terrain)
#         │
#   Label source: EFFIS API (real fire incidents) → fallback: heuristic
#         │
#   ┌─────┴──────┐
#   RF Classifier   QSVC (optional, Qiskit)
#   └─────┬──────┘
#     Blended score  →  SHAP explanation per cell
#         │
#   map.html (timeline) / fire.json / fire.csv / shap_report.html
#         │
#   QAOA sensor placement (optional, Qiskit)
#
# Key Responsibilities
# -----------------------------------------------------------------------------
# - Parallel weather + elevation + NDVI data ingestion with TTL caching
# - EFFIS historical fire labels (fallback to heuristic when unavailable)
# - Classical Random Forest with 5-fold CV + SHAP feature importance
# - Optional Quantum SVM (QSVC) with sigmoid score calibration
# - Optional QAOA for optimal IoT sensor placement
# - Leaflet HTML map with hourly timeline animation
#
# Dependencies
# -----------------------------------------------------------------------------
# Runtime  : numpy, requests, scikit-learn, shap
# Optional : qiskit >= 2.0, qiskit-machine-learning, qiskit-algorithms
# Data     : Open-Meteo (free), Open-Elevation (free), EFFIS (EU, free)
#
# Install  : pip install numpy requests scikit-learn shap
#            pip install qiskit qiskit-machine-learning qiskit-algorithms
#
# Author        : Krzysztof W. Banasiewicz
# Organisation  : QHDALabs
#
# Created       : 2026
# Last Modified : 10.05.2026
#
# License
# -----------------------------------------------------------------------------
# MIT License — free to use, modify and distribute with attribution.
#
# Notes
# -----------------------------------------------------------------------------
# - EFFIS labels are fetched for the past 365 days; if the API is unreachable
#   the system falls back to heuristic thresholds with a clear warning.
# - NDVI is approximated from Open-Meteo soil + temperature data when
#   Sentinel-2 access is unavailable (no API key required).
# - QAOA sensor placement runs on a simulator; swap to real backend via
#   QiskitRuntimeService for hardware execution.
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
from datetime import datetime, timedelta
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
GRID_SIZE        = 6
ALERT_THRESHOLD  = 0.7
MAX_WORKERS      = 10
CACHE_DIR        = ".cache"
CACHE_TTL        = 3600        # 1 hour
RANDOM_STATE     = 42
EFFIS_LOOKBACK   = 365         # days of historical fire data to fetch
N_SENSORS        = 5           # target sensor count for QAOA placement

os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# CACHE HELPERS
# =========================
def _cache_path(key: str) -> str:
    safe = key.replace("-", "m").replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def _cache_get(key: str):
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            ts, data = pickle.load(open(path, "rb"))
            if time.time() - ts < CACHE_TTL:
                return data
        except Exception:
            pass
    return None


def _cache_set(key: str, data) -> None:
    pickle.dump((time.time(), data), open(_cache_path(key), "wb"))

# =========================
# WEATHER FETCH
# =========================
def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch hourly forecast from Open-Meteo."""
    cached = _cache_get(f"weather_{lat:.2f}_{lon:.2f}")
    if cached:
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
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    _cache_set(f"weather_{lat:.2f}_{lon:.2f}", data)
    return data


def summarize_hourly(raw: dict) -> dict:
    """Collapse 24 hourly readings + keep per-hour timeline for map."""
    h = raw["hourly"]
    return {
        "temp":       max(h["temperature_2m"]),
        "temp_mean":  float(np.mean(h["temperature_2m"])),
        "rh":         min(h["relative_humidity_2m"]),
        "wind":       float(np.mean(h["wind_speed_10m"])),
        "wind_max":   max(h["wind_speed_10m"]),
        "wind_dir":   float(np.mean(h["wind_direction_10m"])),
        "rain":       sum(h["precipitation"]),
        "soil":       min(h["soil_moisture_0_to_1cm"]),
        "vpd":        float(np.mean(h.get("vapour_pressure_deficit",
                                          [0.0] * 24))),
        # Hourly snapshots for timeline map
        "hourly_temp":  h["temperature_2m"],
        "hourly_rh":    h["relative_humidity_2m"],
        "hourly_wind":  h["wind_speed_10m"],
        "hourly_times": h.get("time", []),
    }

# =========================
# ELEVATION + SLOPE
# =========================
def fetch_elevation(lat: float, lon: float) -> float:
    """
    Fetch elevation (m) from Open-Elevation API.
    Slope is approximated from neighbouring grid points.
    Falls back to 200m (Polish average) on failure.
    """
    cached = _cache_get(f"elev_{lat:.2f}_{lon:.2f}")
    if cached is not None:
        return cached

    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = {"locations": [{"latitude": lat, "longitude": lon}]}
        r = requests.post(url, json=payload, timeout=8)
        r.raise_for_status()
        elev = float(r.json()["results"][0]["elevation"])
    except Exception:
        elev = 200.0   # fallback: mean Polish elevation

    _cache_set(f"elev_{lat:.2f}_{lon:.2f}", elev)
    return elev


def estimate_slope(lat: float, lon: float, delta: float = 0.1) -> float:
    """
    Approximate slope (degrees) from elevation difference across ~10 km.
    Uses two neighbouring elevation queries.
    """
    cached = _cache_get(f"slope_{lat:.2f}_{lon:.2f}")
    if cached is not None:
        return cached

    try:
        e0 = fetch_elevation(lat, lon)
        en = fetch_elevation(lat + delta, lon)
        ee = fetch_elevation(lat, lon + delta)
        # ~11 km per 0.1° lat, ~7.5 km per 0.1° lon at 52°N
        dy = abs(en - e0) / 11000
        dx = abs(ee - e0) / 7500
        slope = float(np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))))
    except Exception:
        slope = 2.0   # flat fallback

    _cache_set(f"slope_{lat:.2f}_{lon:.2f}", slope)
    return slope

# =========================
# NDVI PROXY
# =========================
def estimate_ndvi_proxy(summary: dict) -> float:
    """
    Estimate vegetation stress index as NDVI proxy.

    Without Sentinel-2 access we approximate from:
    - Vapour Pressure Deficit (VPD) — high VPD → stressed/dry vegetation
    - Soil moisture           — low soil → dry vegetation
    - Temperature             — high temp → fire-prone vegetation

    Returns a value in [0, 1] where 1 = maximally stressed (fire-prone).
    """
    vpd_stress   = min(1.0, summary["vpd"] / 3.0)          # VPD saturates at 3 kPa
    soil_stress  = max(0.0, 1.0 - summary["soil"] / 0.4)   # dry below 0.4
    temp_stress  = min(1.0, max(0.0, (summary["temp"] - 15) / 25))
    return float(0.4 * vpd_stress + 0.4 * soil_stress + 0.2 * temp_stress)

# =========================
# EFFIS LABELS
# =========================
def fetch_effis_fires(lat: float, lon: float,
                      radius_km: float = 50.0) -> int:
    """
    Query the EFFIS ActiveFire WFS for historical fire incidents
    within radius_km of (lat, lon) in the past EFFIS_LOOKBACK days.

    Returns 1 if ≥1 fire recorded nearby, 0 otherwise.
    Falls back to heuristic_label() on API failure.
    """
    cached = _cache_get(f"effis_{lat:.2f}_{lon:.2f}")
    if cached is not None:
        return cached

    try:
        import datetime as _dt
        end   = _dt.datetime.now(_dt.timezone.utc)
        start = end - timedelta(days=EFFIS_LOOKBACK)
        # Bounding box ≈ radius_km around point
        deg   = radius_km / 111.0
        bbox  = f"{lon-deg},{lat-deg},{lon+deg},{lat+deg}"

        url = (
            "https://effis.jrc.ec.europa.eu/arcgis/rest/services/"
            "EFFIS/StatisticsAdditional/MapServer/2/query"
        )
        params = {
            "f":              "json",
            "geometry":       bbox,
            "geometryType":   "esriGeometryEnvelope",
            "spatialRel":     "esriSpatialRelIntersects",
            "where":          (
                f"FIREDATE >= DATE '{start.strftime('%Y-%m-%d')}' AND "
                f"FIREDATE <= DATE '{end.strftime('%Y-%m-%d')}'"
            ),
            "returnCountOnly": "true",
            "outSR":          "4326",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        count = int(r.json().get("count", 0))
        label = int(count > 0)
        _cache_set(f"effis_{lat:.2f}_{lon:.2f}", label)
        return label

    except Exception as exc:
        log.debug("EFFIS unavailable for (%.2f, %.2f): %s", lat, lon, exc)
        return None   # signals fallback needed

# =========================
# FEATURE ENGINEERING
# =========================
FEATURE_NAMES = [
    "temp_max", "temp_mean", "rh_min", "wind_mean", "wind_max",
    "rain", "soil_min", "vpd",
    "ndvi_stress",
    "elevation", "slope",
    "heat_x_dryness", "wind_x_dryness", "soil_x_rh",
    "dryness_ratio", "wind_dryness_ratio",
]


def build_feature_vector(summary: dict,
                         ndvi: float,
                         elevation: float,
                         slope: float) -> list:
    """
    Return a 16-element feature vector.
    New in v2: VPD, NDVI proxy, elevation, slope, wind_max, temp_mean.
    """
    temp  = summary["temp"]
    tmean = summary["temp_mean"]
    rh    = summary["rh"]
    wind  = summary["wind"]
    wmax  = summary["wind_max"]
    rain  = summary["rain"]
    soil  = summary["soil"]
    vpd   = summary["vpd"]

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
        temp * (100 - rh),        # heat × dryness
        wind * (100 - rh),        # wind-driven evaporation proxy
        soil * rh,                # moisture retention
        temp / (soil + 0.01),     # dryness ratio
        wind / (soil + 0.01),     # wind dryness ratio
    ]


def heuristic_label(summary: dict) -> int:
    """
    Rule-based fallback label when EFFIS is unavailable.
    ⚠ This is a placeholder — not real fire incident data.
    """
    return int(
        summary["temp"]  > 25  and
        summary["rh"]    < 40  and
        summary["wind"]  > 10  and
        summary["soil"]  < 0.2
    )

# =========================
# GRID
# =========================
def poland_grid() -> list:
    lats = np.linspace(49.0, 54.5, GRID_SIZE)
    lons = np.linspace(14.0, 24.0, GRID_SIZE)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]

# =========================
# PARALLEL DATASET BUILD
# =========================
def fetch_cell(lat: float, lon: float) -> tuple | None:
    """Fetch all data for one grid cell. Returns None on failure."""
    try:
        raw      = fetch_weather(lat, lon)
        summary  = summarize_hourly(raw)
        ndvi     = estimate_ndvi_proxy(summary)
        elev     = fetch_elevation(lat, lon)
        slope    = estimate_slope(lat, lon)
        fvec     = build_feature_vector(summary, ndvi, elev, slope)

        # Label: try EFFIS first, fall back to heuristic
        label = fetch_effis_fires(lat, lon)
        effis_used = label is not None
        if label is None:
            label = heuristic_label(summary)

        return lat, lon, summary, fvec, label, effis_used, ndvi, elev, slope
    except requests.RequestException as exc:
        log.warning("Network error  (%5.2f, %5.2f): %s", lat, lon, exc)
    except (KeyError, ValueError) as exc:
        log.warning("Bad response   (%5.2f, %5.2f): %s", lat, lon, exc)
    return None


def build_dataset() -> tuple:
    grid  = poland_grid()
    X, y  = [], []
    cells = []
    effis_count = 0

    log.info("Fetching %d grid cells with %d workers …", len(grid), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_cell, lat, lon): (lat, lon)
            for lat, lon in grid
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            lat, lon, summary, fvec, label, effis_used, ndvi, elev, slope = result
            X.append(fvec)
            y.append(label)
            cells.append((lat, lon, summary, ndvi, elev, slope))
            if effis_used:
                effis_count += 1

    log.info(
        "Dataset ready: %d cells  (EFFIS labels: %d, heuristic: %d)",
        len(X), effis_count, len(X) - effis_count
    )
    if effis_count == 0:
        log.warning(
            "⚠ All labels are heuristic — EFFIS API may be unreachable. "
            "Model learns a rule, not real fire risk."
        )

    return np.array(X), np.array(y), cells

# =========================
# CLASSICAL MODEL + SHAP
# =========================
def train_classical(X_train, y_train, X, y) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
        )),
    ])
    clf.fit(X_train, y_train)

    if len(np.unique(y)) > 1:
        cv_scores = cross_val_score(clf, X, y, cv=min(5, len(X)),
                                    scoring="f1_weighted")
        log.info("Classical CV F1: %.3f ± %.3f",
                 cv_scores.mean(), cv_scores.std())
    else:
        log.info("Classical model trained (single class — CV skipped).")

    return clf


def compute_shap(clf: Pipeline, X: np.ndarray) -> tuple:
    """
    Compute SHAP values for the Random Forest.
    Returns (shap_values, expected_value) or (None, None) if shap unavailable.
    """
    try:
        import shap
        rf       = clf.named_steps["rf"]
        scaler   = clf.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)

        # shap_values shape depends on sklearn version + number of classes:
        #   single class  → ndarray (n, features)
        #   binary        → list [class0_arr, class1_arr]  OR ndarray (n, features, 2)
        if isinstance(shap_values, list):
            if len(shap_values) >= 2:
                shap_values = shap_values[1]   # take class-1 (fire) shap
            else:
                shap_values = shap_values[0]   # only one class present
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1] # (n, features, classes) → class 1

        ev = explainer.expected_value
        expected_val = float(
            ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) >= 2
            else ev[0] if isinstance(ev, (list, np.ndarray))
            else ev
        )
        log.info("SHAP values computed.")
        return shap_values, expected_val
    except ImportError:
        log.info("shap not installed — skipping SHAP (pip install shap).")
        return None, None
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)
        return None, None

# =========================
# QUANTUM MODEL  (QSVC)
# =========================
def train_quantum(X_train, y_train) -> tuple:
    """
    Quantum SVM via Qiskit. Returns (model, pca, scaler) or (None, None, None).
    """
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC
        try:
            from qiskit.primitives import StatevectorSampler as Sampler
        except ImportError:
            from qiskit.primitives import Sampler
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        if len(np.unique(y_train)) < 2:
            log.warning(
                "Quantum model skipped: only one class in training data."
            )
            return None, None, None

        log.info("Training quantum model …")

        pca    = PCA(n_components=4, random_state=RANDOM_STATE)
        X_pca  = pca.fit_transform(X_train)
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_q    = scaler.fit_transform(X_pca)

        try:
            from qiskit.circuit.library import zz_feature_map
            fmap = zz_feature_map(feature_dimension=4, reps=2)
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap
            fmap = ZZFeatureMap(feature_dimension=4, reps=2)

        sampler  = Sampler()
        fidelity = ComputeUncompute(sampler)
        kernel   = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)
        model    = QSVC(quantum_kernel=kernel)
        model.fit(X_q, y_train)

        log.info("Quantum model trained.")
        return model, pca, scaler

    except Exception as exc:
        log.warning("Quantum model skipped: %s: %s", type(exc).__name__, exc)

    return None, None, None


def score_quantum(model, pca, scaler, feature_vec) -> float:
    """Sigmoid-calibrated quantum score → [0, 1]."""
    X_q = scaler.transform(pca.transform(feature_vec))
    raw = model.decision_function(X_q)[0]
    return float(1 / (1 + np.exp(-raw)))

# =========================
# QAOA SENSOR PLACEMENT
# =========================
def qaoa_sensor_placement(results: list, n_sensors: int = N_SENSORS) -> list:
    """
    Use QAOA to select the N_SENSORS grid cells that maximise coverage
    of high-risk zones.

    ⚠ QAOA on a statevector simulator scales as 2^n_qubits in memory.
    With 36 cells that requires 1 TiB RAM — impossible on any laptop.
    Solution: pre-filter to the top MAX_QAOA_CANDIDATES highest-risk cells,
    run QAOA on that small subset, then return selected positions.

    Falls back to greedy top-N if Qiskit unavailable or QAOA fails.
    """
    MAX_QAOA_CANDIDATES = 12   # 2^12 = 4096 states → ~64 KB — fine on any machine

    risks  = np.array([r["risk"] for r in results])
    coords = [(r["lat"], r["lon"]) for r in results]
    n      = len(results)

    # Greedy fallback (always computed)
    top_idx_greedy = np.argsort(risks)[::-1][:n_sensors].tolist()
    greedy_sensors = [coords[i] for i in top_idx_greedy]

    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        try:
            from qiskit.primitives import StatevectorSampler as QaoaSampler
        except ImportError:
            from qiskit.primitives import Sampler as QaoaSampler
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        log.info("QAOA imports OK — building problem …")

        # Pre-filter: keep only top MAX_QAOA_CANDIDATES cells by risk
        candidate_idx = np.argsort(risks)[::-1][:MAX_QAOA_CANDIDATES].tolist()
        cand_risks    = risks[candidate_idx]
        cand_coords   = [coords[i] for i in candidate_idx]
        nc            = len(candidate_idx)

        # Clamp n_sensors to available candidates
        k = min(n_sensors, nc)

        qp = QuadraticProgram("sensor_placement")
        for i in range(nc):
            qp.binary_var(f"x{i}")

        # Objective: maximise sum(risk_i * x_i) → minimise negative
        linear = {f"x{i}": -float(cand_risks[i]) for i in range(nc)}
        qp.minimize(linear=linear)

        # Constraint: exactly k sensors
        qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(nc)},
            sense="==",
            rhs=k,
            name="sensor_count",
        )

        sampler = QaoaSampler()
        log.info("QAOA: running optimizer (%d candidates → %d sensors) …",
                 nc, k)
        qaoa   = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=150), reps=2)
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(qp)

        selected = [i for i, v in enumerate(result.x) if v > 0.5]
        if len(selected) != k:
            log.warning("QAOA returned %d sensors (expected %d) — using greedy.",
                        len(selected), k)
            return greedy_sensors

        sensors = [cand_coords[i] for i in selected]
        log.info("QAOA sensor placement complete: %s", sensors)
        return sensors

    except Exception as exc:
        if "No module" in str(exc) or isinstance(exc, ImportError):
            log.info(
                "qiskit-optimization / qiskit-algorithms not installed — "
                "using greedy sensor placement."
            )
        else:
            log.warning("QAOA skipped: %s: %s — using greedy fallback.",
                        type(exc).__name__, exc)

    return greedy_sensors

# =========================
# SCORING
# =========================
def score_cells(cells, X, clf, q_model, q_pca, q_scaler,
                shap_values) -> list:
    results = []

    for i, (lat, lon, summary, ndvi, elev, slope) in enumerate(cells):
        fvec = X[i : i + 1]

        proba = clf.predict_proba(fvec)[0]
        classical_prob = (
            0.0 if (len(proba) == 1 and clf.classes_[0] == 0)
            else 1.0 if len(proba) == 1
            else float(proba[1])
        )

        if q_model is not None:
            quantum_prob = score_quantum(q_model, q_pca, q_scaler, fvec)
            final_score  = 0.7 * classical_prob + 0.3 * quantum_prob
        else:
            final_score = classical_prob

        # Top-3 SHAP drivers for this cell
        shap_drivers = []
        if shap_values is not None:
            sv   = shap_values[i]
            top3 = np.argsort(np.abs(sv))[::-1][:3]
            shap_drivers = [
                {"feature": FEATURE_NAMES[j], "shap": round(float(sv[j]), 4)}
                for j in top3
            ]

        results.append({
            "lat":          round(lat, 4),
            "lon":          round(lon, 4),
            "risk":         round(float(final_score), 4),
            "alert":        bool(final_score > ALERT_THRESHOLD),
            "ndvi_stress":  round(float(ndvi), 3),
            "elevation_m":  round(float(elev), 1),
            "slope_deg":    round(float(slope), 2),
            "shap_drivers": shap_drivers,
            "hourly_risk":  _hourly_risk_series(summary, classical_prob),
        })

    return results


def _hourly_risk_series(summary: dict, base_risk: float) -> list:
    """
    Approximate per-hour risk variation for the timeline map.
    Scales base_risk by normalised temperature curve.
    """
    temps = summary.get("hourly_temp", [base_risk] * 24)
    tmin, tmax = min(temps), max(temps)
    series = []
    for i, (t, rh, w) in enumerate(zip(
        temps,
        summary.get("hourly_rh",   [50.0] * 24),
        summary.get("hourly_wind", [5.0]  * 24),
    )):
        t_norm = (t - tmin) / (tmax - tmin + 0.01)
        rh_factor  = max(0, (50 - rh) / 50)
        w_factor   = min(1, w / 20)
        hour_score = min(1.0, base_risk * (1 + 0.4 * t_norm
                                           + 0.3 * rh_factor
                                           + 0.2 * w_factor))
        time_label = summary["hourly_times"][i] if i < len(summary["hourly_times"]) else f"{i:02d}:00"
        series.append({"time": time_label, "risk": round(hour_score, 3)})
    return series

# =========================
# OUTPUT — JSON + CSV
# =========================
def save_results(results: list) -> None:
    # JSON (full, including SHAP + hourly)
    with open("fire.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log.info("Saved fire.json")

    # CSV (flat summary only)
    flat_keys = ["lat", "lon", "risk", "alert",
                 "ndvi_stress", "elevation_m", "slope_deg"]
    with open("fire.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in flat_keys})
    log.info("Saved fire.csv")

# =========================
# OUTPUT — MAP (with timeline)
# =========================
def generate_map(results: list, sensors: list) -> None:
    """
    Leaflet map with:
    - Risk circles per cell (colour by tier)
    - Hourly timeline slider (Leaflet.timeline)
    - SHAP tooltip showing top drivers
    - QAOA sensor markers
    """
    # Build per-cell JS data
    cell_data = json.dumps([
        {
            "lat":     r["lat"],
            "lon":     r["lon"],
            "risk":    r["risk"],
            "alert":   r["alert"],
            "ndvi":    r["ndvi_stress"],
            "elev":    r["elevation_m"],
            "slope":   r["slope_deg"],
            "shap":    r["shap_drivers"],
            "hourly":  r["hourly_risk"],
        }
        for r in results
    ], ensure_ascii=False)

    sensor_data = json.dumps([
        {"lat": lat, "lon": lon} for lat, lon in sensors
    ])

    alert_thresh = ALERT_THRESHOLD

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <title>QHDALabs — Wildfire Risk PL v2</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html,body,#map {{ margin:0; height:100vh; font-family:sans-serif; }}
    #controls {{
      position:absolute; top:10px; left:50%; transform:translateX(-50%);
      z-index:1000; background:white; padding:8px 16px; border-radius:8px;
      box-shadow:0 2px 8px rgba(0,0,0,0.2); display:flex; align-items:center; gap:12px;
    }}
    #hour-label {{ font-weight:600; min-width:50px; text-align:center; }}
    #hour-slider {{ width:260px; }}
    .legend {{ background:white; padding:8px 12px; border-radius:6px;
               font:13px sans-serif; line-height:1.8; }}
    .sensor-icon {{ background:#1a56db; border:2px solid white;
                    border-radius:50%; width:14px; height:14px; }}
  </style>
</head>
<body>
<div id="controls">
  <span>🕐 Hour:</span>
  <input id="hour-slider" type="range" min="0" max="23" value="12" step="1"
         oninput="updateHour(this.value)"/>
  <span id="hour-label">12:00</span>
</div>
<div id="map"></div>
<script>
const CELLS   = {cell_data};
const SENSORS = {sensor_data};
const THRESH  = {alert_thresh};

var map = L.map('map').setView([52, 19], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OpenStreetMap contributors'}}).addTo(map);

// Legend
var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
  var d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>Fire risk</b><br>'
    + '<span style="color:#A32D2D">●</span> Alert (&gt;' + THRESH + ')<br>'
    + '<span style="color:#854F0B">●</span> Elevated (0.4–' + THRESH + ')<br>'
    + '<span style="color:#3B6D11">●</span> Low (&lt;0.4)<br><br>'
    + '<span style="color:#1a56db">⬤</span> Sensor (QAOA)';
  return d;
}};
legend.addTo(map);

// Draw circles
var circles = [];
CELLS.forEach(function(c) {{
  var color = c.alert ? '#A32D2D' : (c.risk > 0.4 ? '#854F0B' : '#3B6D11');
  var opacity = c.alert ? 0.6 : (c.risk > 0.4 ? 0.45 : 0.3);

  var shap_html = '';
  if (c.shap && c.shap.length > 0) {{
    shap_html = '<br><small><b>Top drivers:</b><br>'
      + c.shap.map(function(s) {{
          return s.feature + ': ' + (s.shap >= 0 ? '+' : '') + s.shap;
        }}).join('<br>') + '</small>';
  }}

  var popup = '<b>Risk: ' + c.risk.toFixed(2) + (c.alert ? ' ⚠ ALERT' : '') + '</b><br>'
    + 'NDVI stress: ' + c.ndvi.toFixed(2) + '<br>'
    + 'Elevation: ' + c.elev + ' m<br>'
    + 'Slope: ' + c.slope + '°'
    + shap_html;

  var circle = L.circle([c.lat, c.lon], {{
    color: color, fillColor: color,
    fillOpacity: opacity, radius: 20000
  }}).addTo(map).bindPopup(popup);

  circles.push({{ circle: circle, cell: c, base_color: color }});
}});

// QAOA sensor markers
SENSORS.forEach(function(s) {{
  L.circleMarker([s.lat, s.lon], {{
    radius: 8, color: '#1a56db', fillColor: '#1a56db',
    fillOpacity: 0.9, weight: 2
  }}).addTo(map).bindPopup('<b>📡 Recommended sensor</b><br>(QAOA placement)');
}});

// Timeline slider
function updateHour(h) {{
  var hour = parseInt(h);
  var label = (hour < 10 ? '0' : '') + hour + ':00';
  document.getElementById('hour-label').textContent = label;

  circles.forEach(function(item) {{
    var c    = item.cell;
    var risk = (c.hourly && c.hourly[hour]) ? c.hourly[hour].risk : c.risk;
    var color = (risk > THRESH) ? '#A32D2D' : (risk > 0.4 ? '#854F0B' : '#3B6D11');
    var opacity = (risk > THRESH) ? 0.6 : (risk > 0.4 ? 0.45 : 0.3);
    item.circle.setStyle({{ color: color, fillColor: color, fillOpacity: opacity }});
  }});
}}
</script>
</body></html>"""

    with open("map.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("Saved map.html")


# =========================
# OUTPUT — SHAP HTML REPORT
# =========================
def generate_shap_report(results: list) -> None:
    """Write a simple HTML table of SHAP drivers per cell."""
    rows = []
    for r in results:
        if not r["shap_drivers"]:
            continue
        drivers = ", ".join(
            f"{d['feature']} ({'+' if d['shap']>=0 else ''}{d['shap']})"
            for d in r["shap_drivers"]
        )
        alert_badge = (
            '<span style="color:red;font-weight:bold">⚠ ALERT</span>'
            if r["alert"] else "OK"
        )
        rows.append(
            f"<tr><td>{r['lat']}</td><td>{r['lon']}</td>"
            f"<td>{r['risk']:.3f}</td><td>{alert_badge}</td>"
            f"<td>{drivers}</td></tr>"
        )

    if not rows:
        return

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>SHAP Report — Wildfire Risk PL</title>
<style>
  body {{ font-family:sans-serif; padding:24px; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; }}
  th {{ background:#f0f0f0; }}
  tr:nth-child(even) {{ background:#fafafa; }}
</style></head><body>
<h2>SHAP Feature Importance — Top 3 drivers per cell</h2>
<table>
<tr><th>Lat</th><th>Lon</th><th>Risk</th><th>Status</th><th>Top SHAP drivers</th></tr>
""" + "\n".join(rows) + """
</table>
<p style="color:#888;font-size:12px">
  Generated by QHDALabs Wildfire Risk PL v2.0 — """ + datetime.now().strftime("%Y-%m-%d %H:%M UTC") + """
</p>
</body></html>"""

    with open("shap_report.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("Saved shap_report.html")

# =========================
# MAIN
# =========================
def main() -> None:
    # 1. Fetch data (weather + NDVI + elevation + EFFIS labels)
    X, y, cells = build_dataset()

    if len(X) == 0:
        log.error("No data fetched — check network and API availability.")
        return

    # 2. Train/evaluate classical model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    clf = train_classical(X_train, y_train, X, y)
    log.info(
        "Classification report:\n%s",
        classification_report(y_test, clf.predict(X_test), zero_division=0)
    )

    # 3. SHAP explanations
    shap_values, _ = compute_shap(clf, X)

    # 4. Quantum model (optional)
    q_model, q_pca, q_scaler = train_quantum(X_train, y_train)

    # 5. Score all cells
    results = score_cells(cells, X, clf, q_model, q_pca, q_scaler, shap_values)

    alert_count = sum(1 for r in results if r["alert"])
    log.info("Alerts: %d / %d cells", alert_count, len(results))

    # 6. QAOA sensor placement
    sensors = qaoa_sensor_placement(results)
    log.info("Sensor positions: %s", sensors)

    # 7. Save all outputs
    save_results(results)
    generate_map(results, sensors)
    generate_shap_report(results)

    log.info("Done → map.html  fire.json  fire.csv  shap_report.html")


if __name__ == "__main__":
    main()
