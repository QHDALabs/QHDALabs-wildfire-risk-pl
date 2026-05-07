# 🔥 QHDALabs Wildfire Risk PL — System wczesnego ostrzegania przed pożarami lasów

> System predykcji ryzyka pożarów lasów dla Polski oparty na klasycznym uczeniu maszynowym (Random Forest) z opcjonalnym modelem kwantowym (Qiskit QSVC).

---

## 📋 Opis projektu

Skrypt pobiera dane pogodowe w czasie rzeczywistym z API [Open-Meteo](https://open-meteo.com/) dla 36 punktów siatki pokrywającej Polskę, oblicza wektory cech, trenuje klasyfikator i generuje interaktywną mapę ryzyka pożarowego.

**Dane wejściowe (dla każdego punktu siatki):**

- Maksymalna temperatura powietrza (°C)
- Minimalna wilgotność względna (%)
- Średnia prędkość wiatru (km/h)
- Suma opadów (mm)
- Minimalna wilgotność gleby

**Dane wyjściowe:**

- `map.html` — interaktywna mapa Leaflet z kolorowymi wskaźnikami ryzyka
- `fire.json` — wyniki w formacie JSON
- `fire.csv` — wyniki w formacie CSV

---

## 🗺️ Przykład mapy

Mapa generuje trzy poziomy ryzyka:

| Kolor | Poziom | Próg |
|-------|--------|------|
| 🔴 Czerwony | **ALERT** | > 0.70 |
| 🟠 Pomarańczowy | Podwyższone | 0.40 – 0.70 |
| 🟢 Zielony | Niskie | < 0.40 |

---

## ⚙️ Wymagania

**Python 3.10+**

Instalacja zależności:

```bash
pip install numpy requests scikit-learn
```

Opcjonalnie — model kwantowy (Qiskit):

```bash
pip install qiskit qiskit-machine-learning
```

> ℹ️ Bez Qiskit skrypt działa normalnie — model kwantowy jest pomijany, a wynik końcowy oparty jest wyłącznie na Random Forest.

---

## 🚀 Uruchomienie

```bash
python qhdalabs-wildfire_risk.py
```

Przykładowy output:

```
13:19:54  INFO  Fetching 36 grid cells with 10 workers …
13:19:54  INFO  Dataset ready: 36 cells
13:19:56  INFO  Classical CV F1: 1.000 ± 0.000
13:19:58  INFO  Alerts: 3 / 36 cells
13:19:58  INFO  Saved fire.json
13:19:58  INFO  Saved fire.csv
13:19:58  INFO  Saved map.html
13:19:58  INFO  Done → map.html  fire.json  fire.csv
```

---

## 🏗️ Architektura

```
Open-Meteo API
      │
      ▼ (36 komórek siatki równolegle)
┌─────────────────┐
│  fetch_weather  │  ThreadPoolExecutor (10 wątków)
│  + cache TTL 1h │  ← przyspiesza ponowne uruchomienia
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ build_feature   │  11 cech + interakcje
│    _vector      │
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐  ┌──────────────┐
│Random  │  │ Qiskit QSVC  │  (opcjonalnie)
│Forest  │  │ ZZFeatureMap │
│  70%   │  │    30%       │
└────┬───┘  └──────┬───────┘
     │             │ sigmoid
     └──────┬──────┘
            ▼
      wynik końcowy
      (0.0 – 1.0)
            │
     ┌──────┴──────┐
     ▼             ▼
  map.html    fire.json
             fire.csv
```

---

## 🔧 Konfiguracja

W górnej części pliku `qhdapabs-wildfire_risk.py` można dostosować:

```python
GRID_SIZE       = 6      # rozdzielczość siatki (6×6 = 36 punktów)
ALERT_THRESHOLD = 0.7    # próg alertu (0.0 – 1.0)
MAX_WORKERS     = 10     # liczba równoległych wątków API
CACHE_TTL       = 3600   # czas życia cache w sekundach (1 godzina)
```

---

## ⚠️ Ważna uwaga — etykiety

Aktualnie etykiety treningowe są generowane **heurystycznie** na podstawie progów pogodowych:

```python
temp > 25°C  AND  wilgotność < 40%  AND  wiatr > 10 km/h  AND  wilgotność gleby < 0.2
```

Oznacza to, że model uczy się odtwarzać tę regułę, a **nie** rzeczywistego ryzyka pożarowego. Aby uzyskać prawdziwy system predykcji, należy zastąpić etykiety historycznymi danymi o pożarach, np.:

- [EFFIS — European Forest Fire Information System](https://effis.jrc.ec.europa.eu/)
- [BDOT10k — Baza Danych Obiektów Topograficznych](https://www.geoportal.gov.pl/)

---

## 📁 Struktura projektu

```
qhdalabs-wildfire-risk-pl/
├── qhdalabs-wildfire_risk.py   # główny skrypt
├── map.html           # wygenerowana mapa (po uruchomieniu)
├── fire.json          # wyniki JSON (po uruchomieniu)
├── fire.csv           # wyniki CSV (po uruchomieniu)
├── .cache/            # cache odpowiedzi API
└── README.md
```

---

## 🔬 Model kwantowy

Jeśli zainstalowany jest Qiskit, skrypt trenuje dodatkowo **Quantum Support Vector Classifier (QSVC)**:

1. Redukcja wymiarów do 4 cech przez **PCA**
2. Skalowanie do zakresu `[0, π]` (MinMaxScaler)
3. Enkodowanie danych przez **ZZFeatureMap** (4 qubity, 2 warstwy)
4. Obliczenie jądra kwantowego przez **FidelityQuantumKernel**
5. Kalibracja wyjścia przez **funkcję sigmoid** → wynik 0–1

Wynik kwantowy jest łączony z klasycznym w proporcji **70% RF + 30% QSVC**.

> Model kwantowy jest automatycznie pomijany jeśli Qiskit nie jest zainstalowany lub dane treningowe zawierają tylko jedną klasę.

---

## 📜 Licencja

MIT — możesz swobodnie używać, modyfikować i dystrybuować.

---

*Projekt stworzony jako demonstracja hybrydowego systemu klasyczno-kwantowego do predykcji ryzyka środowiskowego.*
