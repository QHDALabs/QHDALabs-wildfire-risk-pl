# 🔥 QHDALabs  Wildfire Risk PL — System wczesnego ostrzegania przed pożarami lasów

> Hybrydowy system predykcji ryzyka pożarów lasów dla Polski łączący klasyczne uczenie maszynowe (Random Forest), obliczenia kwantowe (Qiskit QSVC + QAOA) oraz wyjaśnialność modelu (SHAP).

---

## 🗂️ Wersje

| Wersja | Plik | Opis |
|--------|------|------|
| **v1** | `qhdalabs-wildfire_risk_v1.py` | MVP — pogoda + RF + QSVC, mapa Leaflet |
| **v2** | `qhdalabs-wildfire_risk_v2.py` | EFFIS labels, NDVI, terrain, SHAP, QAOA (wymaga ~1 TB RAM dla pełnej siatki) |
| **v3** | `qhdalabs-wildfire_risk_v3.py` | QAOA z pre-filtrowaniem kandydatów (12 kubitów → działa na laptopie) *(w trakcie dopracowania)* |

---

## 📋 Opis projektu

Skrypt pobiera dane pogodowe, wegetacyjne i terenowe w czasie rzeczywistym dla 36 punktów siatki pokrywającej Polskę, trenuje hybrydowy klasyfikator i generuje interaktywną mapę ryzyka pożarowego z wyjaśnieniem decyzji modelu.

**Dane wejściowe (v2+, dla każdego punktu siatki):**

| Źródło | Zmienne |
|--------|---------|
| Open-Meteo | temp, wilgotność, wiatr, opady, VPD, wilgotność gleby |
| Open-Elevation | wysokość n.p.m., nachylenie terenu (slope) |
| NDVI proxy | wskaźnik stresu roślinności (z VPD + soil + temp) |
| EFFIS API | historyczne incydenty pożarowe (etykiety treningowe) |

**Dane wyjściowe:**
- `map.html` — interaktywna mapa Leaflet z timelineʼem godzinowym
- `fire.json` — wyniki z SHAP drivers i hourly risk per komórka
- `fire.csv` — płaskie podsumowanie do dalszej analizy
- `shap_report.html` — raport wyjaśnialności: top-3 cechy napędzające ryzyko per komórka

---

## 🗺️ Mapa

Trzy poziomy ryzyka + suwak czasowy (00:00–23:00):

| Kolor | Poziom | Próg |
|-------|--------|------|
| 🔴 Czerwony | **ALERT** | > 0.70 |
| 🟠 Pomarańczowy | Podwyższone | 0.40 – 0.70 |
| 🟢 Zielony | Niskie | < 0.40 |
| 🔵 Niebieski | Sensor (QAOA) | — |

---

## ⚙️ Wymagania

**Python 3.10+**

```bash
# Runtime (wymagane)
pip install numpy requests scikit-learn shap

# Quantum — model QSVC
pip install qiskit qiskit-machine-learning

# Quantum — QAOA sensor placement (v3)
pip install qiskit-algorithms qiskit-optimization
```

> ℹ️ Każdy moduł kwantowy jest opcjonalny — skrypt gracefully fallback'uje do klasycznych odpowiedników gdy Qiskit nie jest zainstalowany lub gdy dane treningowe zawierają tylko jedną klasę.

---

## 🚀 Uruchomienie

```bash
python qhdalabs-wildfire_risk_v3.py
```

Przykładowy output (v3):

```
13:26:24  INFO  Dataset ready: 36 cells  (EFFIS labels: 12, heuristic: 24)
13:26:26  INFO  Classical CV F1: 0.923 ± 0.041
13:26:26  INFO  SHAP values computed.
13:26:28  INFO  Training quantum model …
13:26:29  INFO  QAOA imports OK — building problem …
13:26:29  INFO  QAOA: running optimizer (12 candidates → 5 sensors) …
13:26:31  INFO  QAOA sensor placement complete.
13:26:31  INFO  Alerts: 3 / 36 cells
13:26:32  INFO  Done → map.html  fire.json  fire.csv  shap_report.html
```

---

## 🏗️ Architektura (v3)

```
Open-Meteo + Open-Elevation + NDVI proxy
         │  (równolegle, cache TTL 1h)
         ▼
┌─────────────────────────────┐
│   Feature Engineering       │  16 cech:
│   weather + terrain + NDVI  │  VPD, slope, elevation,
└────────────┬────────────────┘  wind_max, temp_mean …
             │
    EFFIS API labels ──→ fallback: heuristic
             │
        ┌────┴─────┐
        ▼          ▼
  ┌──────────┐  ┌──────────────┐
  │  Random  │  │ Qiskit QSVC  │  (opcjonalnie)
  │  Forest  │  │ ZZFeatureMap │  4 qubity, sigmoid
  │   70%    │  │    30%       │  kalibracja
  └────┬─────┘  └──────┬───────┘
       └────────┬───────┘
                ▼
          wynik końcowy
          (0.0 – 1.0)
                │
       ┌────────┼──────────────┐
       ▼        ▼              ▼
    map.html  fire.json   shap_report.html
    (timeline) (+ SHAP)
                │
         QAOA pre-filter
         top-12 → 5 sensorów
                ▼
         📡 sensor markers
```

---

## 🔧 Konfiguracja

```python
GRID_SIZE            = 6      # rozdzielczość siatki (6×6 = 36 punktów)
ALERT_THRESHOLD      = 0.7    # próg alertu (0.0 – 1.0)
MAX_WORKERS          = 10     # równoległe wątki API
CACHE_TTL            = 3600   # TTL cache w sekundach
EFFIS_LOOKBACK       = 365    # dni historii pożarów z EFFIS
N_SENSORS            = 5      # liczba sensorów do rozmieszczenia (QAOA)
MAX_QAOA_CANDIDATES  = 12     # pre-filtr przed QAOA (2^12 = 4096 stanów)
```

---

## 🧠 SHAP — wyjaśnialność modelu

Każda komórka siatki w `fire.json` zawiera pole `shap_drivers` z top-3 cechami które najbardziej wpłynęły na jej wynik ryzyka:

```json
"shap_drivers": [
  { "feature": "vpd",          "shap":  0.142 },
  { "feature": "ndvi_stress",  "shap":  0.098 },
  { "feature": "slope_deg",    "shap":  0.061 }
]
```

Pełny raport w `shap_report.html`.

---

## ⚛️ Moduły kwantowe

### QSVC — klasyfikacja ryzyka

1. Redukcja do 4 cech przez PCA
2. Skalowanie do `[0, π]`
3. Enkodowanie przez **ZZFeatureMap** (4 qubity, 2 warstwy)
4. Jądro kwantowe **FidelityQuantumKernel**
5. Kalibracja sigmoid → wynik 0–1
6. Blend: **70% RF + 30% QSVC**

### QAOA — rozmieszczenie sensorów IoT

1. Klasyczna pre-selekcja top-12 komórek o najwyższym ryzyku
2. Formułowanie jako QUBO (Quadratic Unconstrained Binary Optimization)
3. QAOA na symulatorze statevector (12 kubitów = 4096 stanów, ~64 KB RAM)
4. Wynik: 5 optymalnych lokalizacji sensorów na mapie

> **Dlaczego pre-filtrowanie?** Symulator statevector wymaga 2^n stanów w pamięci. Bez filtrowania 36 zmiennych = 2^36 = **1 TiB RAM**. Po filtracji do 12 kandydatów = 2^12 = **64 KB**.

---

## ⚠️ Etykiety treningowe

Skrypt próbuje pobrać rzeczywiste dane o pożarach z **EFFIS API** (ostatnie 365 dni, promień 50 km od każdego punktu siatki). Gdy API jest niedostępne, automatycznie przełącza się na etykiety heurystyczne z wyraźnym ostrzeżeniem w logu.

```
⚠ All labels are heuristic — EFFIS API may be unreachable.
```

Docelowe źródła rzeczywistych danych:
- [EFFIS — European Forest Fire Information System](https://effis.jrc.ec.europa.eu/)
- [BDOT10k — Baza Danych Obiektów Topograficznych](https://www.geoportal.gov.pl/)

---

## 📁 Struktura projektu

```
QHDALabs-wildfire-risk-pl/
├── qhdalabs-wildfire_risk_v1.py    # MVP
├── qhdalabs-wildfire_risk_v2.py    # pełny QAOA (wymaga ~1 TB RAM)
├── qhdalabs-wildfire_risk_v3.py    # QAOA z pre-filtrowaniem ✅ zalecana
├── map.html               # wygenerowana mapa
├── fire.json              # wyniki JSON + SHAP
├── fire.csv               # wyniki CSV
├── shap_report.html       # raport wyjaśnialności
├── .cache/                # cache API (TTL 1h)
└── README.md
```

---

## 📜 Licencja

MIT — możesz swobodnie używać, modyfikować i dystrybuować z zachowaniem atrybucji.

---

*QHDALabs — budujemy fundament pod autonomiczną infrastrukturę ochrony środowiska.*
