# =============================================================================
# QHDALabs - Wildfire Risk PL v5
# config.py — centralna konfiguracja
# =============================================================================

import os
from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "wildfire.db"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR    = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# =========================
# TTL (sekundy)
# =========================
TTL_WEATHER_S    = 6 * 3600        # 6h  — pogoda zmienia się często
TTL_NDWI_S       = 10 * 24 * 3600  # 10d — Sentinel-2 revisit ~5d, limit API
TTL_QTE_S        = 6 * 3600        # 6h  — przeliczamy gdy dane wejściowe świeże
TTL_RISK_S       = 6 * 3600        # 6h

# =========================
# COPERNICUS
# =========================
CDSE_CLIENT_ID     = os.environ.get("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")
CDSE_TOKEN_URL     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATS_URL     = "https://sh.dataspace.copernicus.eu/statistics/v1"
NDWI_DAYS          = 30
NDWI_BOX_DEG       = 0.045   # ~5 km radius bbox
NDWI_RESX          = 0.001   # ~70m w CRS84

# =========================
# FUSION WEIGHTS
# =========================
W_NDWI    = 0.45
W_QTE     = 0.30
W_FWI     = 0.15
W_TREND   = 0.10

BRIDGE_BONUS   = 0.08
NETWORK_COEFF  = 0.05

ALERT_CRITICAL = 0.75
ALERT_HIGH     = 0.60
ALERT_MODERATE = 0.45

ECO_MULTIPLIERS = {
    "pine":            1.30,
    "pine_wetland":    1.10,
    "mixed":           1.00,
    "spruce_mountain": 0.80,
}

# =========================
# QTE
# =========================
QTE_N_SHOTS      = 256
QTE_N_QUBITS     = 5
NDWI_HEALTHY     = -0.35
NDWI_CRITICAL    = -0.70
WIND_LOW         = 4.0
WIND_HIGH        = 10.0
TEMP_LOW         = 15.0
TEMP_HIGH        = 30.0
DROUGHT_MAX      = 30

# =========================
# NETWORK / HTTP
# =========================
MAX_WORKERS      = 8
SENTINEL_WORKERS = 4     # rate limit Copernicus
HTTP_TIMEOUT     = 30
NEIGHBOR_KM      = 60.0
WEATHER_DAYS     = 14

# =========================
# WEATHER THRESHOLDS
# =========================
DROUGHT_RAIN_MM  = 2.0   # <2mm/dzień = dzień suszy
