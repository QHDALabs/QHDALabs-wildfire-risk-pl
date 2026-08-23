# QHDALabs Wildfire Risk PL v5

Pipeline v5 buduje graf 34 węzłów Dolnego Śląska, oblicza presję zapłonu,
stres roślinności Sentinel-2, sygnał QTE i wynik fusion, a następnie może
porównać wynik z rastrem EFFIS.

## v5.1 — controlled correction

v5.1 nie zmienia architektury. Poprawia sześć findings krytycznych i wysokich
z audytu [FORENSIC_AUDIT_2026-08-23.md](FORENSIC_AUDIT_2026-08-23.md).

| ID | Poprawka | Skutek |
| --- | --- | --- |
| F-01 | Fusion czyta `ndwi_stress_latest` (skala bezwzględna). Ranking min-max nadal jest publikowany jako `ndwi_stress_rank`, ale wyłącznie diagnostycznie. Granice `lo`/`hi` zapisane w `ndwi_sentinel.json`. | koniec gwarantowanego 1.000 |
| F-02 | `MAX_ACQUISITION_AGE_DAYS = 14` liczone od `max(dates)`, nie od TTL. Flaga `stale_acquisition`, status `DEGRADED`, `fresh_acquisition_percent` obok `coverage_percent`. Data akwizycji trafia do `risk_scores.json` i `alerts.json`. | świeżość cache ≠ świeżość zdjęcia |
| F-03 | Etykieta bridge zmieniona na `Previous composite had negative canopy moisture`. Mechanizm bez zmian. | opis zgodny z kodem |
| F-04 | `bridge_rate = sin²(θ/2)` analitycznie w obu backendach; wartość próbkowana zachowana jako `bridge_rate_sampled`. Flaga `bridge_near_threshold` gdy odległość od progu < 0.05. | tier niezależny od symulatora |
| F-05 | `soil_moisture_0_to_1cm` → `soil_moisture_0_to_7cm` (API zwracało HTTP 200 z samymi `null`). Brak danych = flaga i renormalizacja wag, nie podstawienie stałej 0.20. | kanał wilgotności gleby działa |
| F-06 | FWI używa wykładniczo wygaszanej sumy opadów z 14 dni (`RAIN_MEMORY_HALFLIFE_D = 3`), nie tylko ostatniej doby. | tydzień deszczu jest widoczny |

Dodatkowo klucz cache pogody zawiera teraz fingerprint listy zmiennych, więc
zmiana żądanych pasm unieważnia cache zamiast po cichu czytać `null`.

Skutek na danych z 2026-08-23: Jawor `0.7528 CRITICAL` → `0.5261 MODERATE`,
krok sentinel `SUCCESS` → `DEGRADED`, liczba alertów `1` → `0`.

Findings F-07 … F-14 pozostają otwarte i są opisane w raporcie.

## Moduły

| Skrypt | Rola |
| --- | --- |
| `qhdalabs_wildfire_topology_v1.py` | Budowa grafu i wzbogacenie 34 węzłów (nodes) |
| `qhdalabs_wildfire_sentinel_v1.py` | Stres roślinności Sentinel-2 NDWI dla węzłów |
| `qhdalabs_wildfire_qte_v1.py` | Sygnał Quantum Temporal Encoder |
| `qhdalabs_wildfire_ignition_v1.py` | Presja zapłonu i kontrola coverage |
| `qhdalabs_wildfire_fusion_v1.py` | Wynik fusion oraz FEI i QIES |
| `effis_validator.py` | Opcjonalna walidacja z rastrem EFFIS |

## Wymagania

Wymagany jest Python 3.11 lub nowszy. Minimalne zależności obliczeniowe są w
`requirements.txt`. Zalecany jest lokalny interpreter `v5/.venv`; runner
automatycznie uruchomi się ponownie przez ten interpreter, jeżeli został
wywołany przez globalne `py` albo `python`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Pliki zależności mają rozdzielone role:

| Plik | Zakres |
| --- | --- |
| `requirements.txt` | NumPy, Pandas i Requests |
| `requirements-gis.txt` | GeoPandas, Pyogrio, Shapely, PyProj i Fiona |
| `requirements-effis.txt` | Opcjonalny Rasterio |
| `requirements-qte.txt` | Opcjonalny Qiskit |
| `requirements-full.txt` | Cały runtime |
| `requirements-dev.txt` | Runtime, testy i Ruff |

Pełne przetwarzanie lokalnych danych GIS wymaga GeoPandas, Pyogrio, Shapely
i PyProj.

```powershell
python -m pip install -r requirements-gis.txt
```

Fiona jest opcjonalnym drugim backendem. Pipeline preferuje następującą
kolejność dla OSM:

1. program `osmium` z pakietu osmium-tool
2. Pyogrio z czytelnym driverem GDAL OSM
3. Fiona z driverem GDAL OSM
4. jawny status braku warstwy

Brak zależności i błąd drivera są raportowane oddzielnie. Wartości syntetyczne
są używane wyłącznie po podaniu `--stub`.

## Konfiguracja FIRMS

NASA FIRMS Area API wymaga bezpłatnego `MAP_KEY`. Pipeline odczytuje go
wyłącznie ze zmiennej środowiskowej `FIRMS_MAP_KEY`.

```powershell
$env:FIRMS_MAP_KEY = "wartość-klucza"
python qhdalabs_wildfire_ignition_v1.py --refresh-firms
```

Klucza nie należy zapisywać w `.env`, kodzie, logach ani repozytorium. Adresy
URL są maskowane przed logowaniem. Rejestracja i bieżąca dokumentacja są
dostępne na stronie
[NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/).

Domyślnym i jedynym automatycznie wybranym produktem jest
`VIIRS_SNPP_SP`. Zmiana na NRT wymaga jawnej polityki źródłowej:

```powershell
$env:FIRMS_SOURCE = "VIIRS_SNPP_NRT"
python qhdalabs_wildfire_ignition_v1.py --refresh-firms
```

Pipeline nigdy nie dołącza fragmentów SP do NRT ani nie uruchamia NRT po
częściowym pobraniu SP. Każdy produkt ma osobny katalog checkpointów.

`--refresh-firms` pobiera lub wznawia pełny rok 2025 w oknach po najwyżej pięć
dni. Każde okno jest walidowane i atomowo zapisywane jako dane dla węzłów:

```text
topology/ignition_cache/firms_parts/<SOURCE>/2025-01-01.csv
topology/ignition_cache/firms_parts/<SOURCE>/2025-01-01.meta.json
```

Metadane zawierają źródło, zakres dat, bbox, liczbę wierszy i SHA-256. Przy
ponownym uruchomieniu poprawne fragmenty są pomijane, a błędne lub brakujące
pobierane ponownie. Produkcyjny `firms_viirs_dolnoslaskie_2025.csv` jest
łączony, deduplikowany i atomowo zastępowany dopiero po ponownej walidacji
wszystkich 73 okien jednego produktu. Bez klucza poprawny istniejący cache
pozostaje używany.

Żądania są domyślnie rozdzielone odstępem 0,25 sekundy. Opcjonalna kontrola
licznika transakcji może być wykonywana co N nowych okien:

```powershell
$env:FIRMS_TRANSACTION_CHECK_EVERY = "10"
```

Stan serii znajduje się w `firms_parts/<SOURCE>/status.json` i ma jedną z
wartości:

- `complete` — wszystkie okna scalono i opublikowano;
- `resumable_partial` — checkpointy są poprawne, brakuje części okien;
- `authorization_failed` — HTTP 401; status klucza rozróżnia `invalid_key`,
  `exhausted_transaction_window` lub `unknown_authorization_failure`;
- `rate_limited` — HTTP 429 albo osiągnięty limit transakcji; pole
  `retry_after_seconds` określa najwcześniejsze zalecane wznowienie;
- `source_unavailable` — nie udało się zapisać żadnego poprawnego okna.

HTTP 401 natychmiast zatrzymuje serię i uruchamia pojedyncze, maskowane
sprawdzenie `mapkey_status`. HTTP 429 nie jest ponawiane w tej samej serii i
zatrzymuje dalsze żądania zgodnie z `Retry-After`. Timeouty oraz HTTP 500, 502,
503 i 504 są ponawiane z exponential backoff i jitter; wyczerpanie prób
pozostawia pobrane checkpointy do następnego uruchomienia.

## Uruchomienie

Podstawowe polecenia można wykonywać z katalogu repozytorium. `EFFIS` nie jest
częścią domyślnego core pipeline.

```powershell
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --doctor
.\v5\.venv\Scripts\python.exe .\v5\run_all.py
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --offline-sentinel
```

Dostępne sterowanie przebiegiem:

```powershell
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --skip sentinel
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --only fusion
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --from qte --until fusion
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --refresh-sentinel
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --refresh-firms --refresh-gis
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --strict
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --allow-degraded
```

`--strict` odrzuca każdy wynik `DEGRADED`. `--allow-degraded` pozwala na
kontynuację badawczą, ale `operational_validity` pozostaje równe `false`.
`--no-venv-reexec` wyłącza automatyczne przejście do `.venv`.

## Statusy pipeline

Każdy krok zapisuje manifest w `topology/pipeline_steps/<step>.json`, a runner
zapisuje podsumowanie w `topology/pipeline_run.json`.

| Status | Znaczenie |
| --- | --- |
| `SUCCESS` | Wynik pełny |
| `DEGRADED` | Wynik częściowy, jawnie oznaczony |
| `BLOCKED` | Wynik nie może zasilać downstream |
| `SKIPPED` | Krok pominięty |
| `OPTIONAL_FAILED` | Opcjonalna walidacja nie powiodła się |
| `FAILED` | Błąd kroku core |

`BLOCKED` i `FAILED` blokują kroki zależne. `DEGRADED` przechodzi dalej tylko,
gdy manifest ma `valid_for_downstream=true`.

## Cache Sentinel

Cache Sentinel jest trwałym JSON-em schema v2 w `.cache_topology/sentinel`.
Domyślny TTL wynosi 10 dni. Priorytet konfiguracji to argument CLI, zmienna
`SENTINEL_CACHE_TTL_DAYS`, a następnie wartość domyślna.

```powershell
.\v5\.venv\Scripts\python.exe .\v5\run_all.py `
  --sentinel-cache-ttl-days 10
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --refresh-sentinel
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --offline-sentinel
```

Cache jest klasyfikowany przed uwierzytelnieniem jako `fresh`, `stale`,
`missing`, `invalid` albo `legacy`. Pełny zestaw `fresh` nie wykonuje
uwierzytelnienia ani zapytań HTTP. Fingerprint obejmuje pozycję węzła, okres,
promień, kolekcję, interwał agregacji, rozdzielczość, zachmurzenie, indeksy
i hash evalscriptu. Przy błędzie sieci może zostać użyty jawnie oznaczony
`stale-if-error`.

Stary cache pickle można odczytać wyłącznie po świadomym użyciu
`--legacy-sentinel-cache`. Taki przebieg jest zawsze oznaczony jako
`DEGRADED`, `legacy`, `provisional` i `research_only`; nie migruje danych po
cichu ani nie nadaje im ważności operacyjnej.

## Indeksy satelitarne

Pipeline zapisuje dwa odrębne indeksy:

| Nazwa | Wzór | Pasma Sentinel-2 | Rozdzielczość |
| --- | --- | --- | ---: |
| `ndwi_surface_water` | `(B03 - B08) / (B03 + B08)` | Green, NIR | 10 m |
| `vegetation_moisture_index` | `(B8A - B11) / (B8A + B11)` | NIR, SWIR | 20 m |

Green–NIR jest indeksem powierzchniowej wody i nie jest nazywany indeksem
Gao dla wody w roślinności. Gao 1996 zdefiniował indeks dla około 0,86 µm
i 1,24 µm. Sentinel-2 nie ma pasma 1,24 µm, dlatego B8A/B11 jest jawną
aproksymacją NIR/SWIR, a nie dokładną implementacją Gao. Źródła:
[Gao 1996](https://doi.org/10.1016/S0034-4257(96)00067-3) oraz
[Sentinel-2 L2A bands](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html).

Surowe serie, daty i quality flags są oddzielone od mapowania na stress.
Mapowanie ma status `uncalibrated`, nie ma odziedziczonych progów `-0.35`
i `-0.70`, a alerty nie wydają operacyjnej rekomendacji drona.

## Last-known-good i cache GIS

Ignition przechodzi preflight GeoPandas, Pyogrio, Shapely, PyProj i drivera
GDAL OSM przed pobieraniem. Coverage poniżej 70 procent ma status `BLOCKED`
i jest zapisywane wyłącznie jako
`topology/diagnostics/ignition_scores_partial_<timestamp>.json`.
Nie zastępuje ostatniego poprawnego `topology/ignition_scores.json`.

Pochodne OSM, powerlines i agriculture dla węzłów są przechowywane w
`topology/ignition_cache/derived`. Fingerprint zawiera ścieżkę, rozmiar,
mtime, parser, backend, CRS i wersję schematu. Niezmienione dane źródłowe nie
są ponownie parsowane.

## Cache i artefakty

Katalog `topology/ignition_cache/` jest ignorowany przez Git. Może zawierać:

| Plik | Źródło | Tryb |
| --- | --- | --- |
| `firms_viirs_dolnoslaskie_2025.csv` | NASA FIRMS | Auto lub ręczny |
| `firms_parts/<SOURCE>/*.csv` | NASA FIRMS | Trwałe checkpointy |
| `dolnoslaskie-latest.osm.pbf` | Geofabrik OpenStreetMap | Automatyczny |
| `wn.gpkg` i `sn.gpkg` | GIS-Support BDOT10k | Automatyczny |
| `clc18_dolnoslaskie.geojson` | EEA CLC 2018 REST | Automatyczny |
| `lpis_dolnoslaskie.gpkg` | ARiMR LPIS | Ręczny |
| `ibl_pozary_dolnoslaskie.geojson` | IBL KSIPL | Ręczny |
| `effis_fires_pl_2025.csv` | EFFIS | Ręczny |

Pliki tymczasowe powstają w katalogu cache i nie są publikowane pod nazwą
produkcyjną przed pełną walidacją.

## Status źródeł

Stan zweryfikowany 16 sierpnia 2026 (runtime: `run_all.py` / `.venv`):

| Źródło | Status | Uwagi |
| --- | --- | --- |
| NASA FIRMS VIIRS | Dostępne z `MAP_KEY` | `historical_kde` oraz cache `firms_viirs_dolnoslaskie_2025.csv`; okna 1–5 dni, checkpointy i walidacja checksum |
| OSM Geofabrik | Dostępne | `roads`, `railways`, `tourism` z extractu dolnośląskiego; parser wymaga `osmium`, Pyogrio lub Fiona |
| BDOT10k WN i SN | Dostępne | `powerlines` z lokalnych plików `wn.gpkg` i `sn.gpkg`; firmy GIS-Support, brak uwierzytelniania |
| EEA CLC 2018 | Dostępne | `agriculture` fallback z oficjalnej warstwy ArcGIS, paginacja po 1000 obiektów |
| ARiMR LPIS | Opcjonalne / ręczne | Publiczny endpoint nie jest kompatybilny z obecnym WFS; pipeline automatycznie przechodzi do CLC |
| IBL KSIPL | Opcjonalne / ręczne | Dawny GeoServer WFS zwraca 404; wymagany lokalny GeoJSON, brak automatycznego pobierania |
| EFFIS | Opcjonalne / ręczne | Brak auto-pobierania hotspotów; obsługiwany jako raster manualny z `--with-effis` |
| Open-Meteo Archive | Dostępne publicznie | `weather_history` dla 14 dni z throttlingiem, aby uniknąć HTTP 429 i limitów API |

Nie należy zakładać automatycznej dostępności LPIS ani IBL. Pipeline nie
wyłącza weryfikacji TLS i nie utrzymuje martwego URL jako fallbacku. Dla
`powerlines` oraz `roads`/`railways`/`tourism` źródła są aktywne i weryfikowane
lokalnie w cache GIS, a dla `agriculture` preferowany jest LPIS, a fallback
CLC. Open-Meteo ma dodatkowy limiter zapytań, bo 34 węzły pobierające pogodę
równolegle mogły zakończyć się 429.

## Model coverage

Wagi pełnego modelu wynoszą:

| Warstwa | Waga |
| --- | ---: |
| Drogi | 0.25 |
| Kolej | 0.20 |
| Linie energetyczne | 0.15 |
| Turystyka | 0.15 |
| Rolnictwo | 0.15 |
| Historyczne hotspoty | 0.10 |

Brak danych ma wartość JSON `null`. Rzeczywisty wynik warstwy dla danego
węzła równy zero ma wartość `0.0`.

Każdy rekord węzła zawiera:

- `data_coverage`
- `missing_sublayers`
- `coverage_weight`
- `coverage_percent`
- `score_status`
- `valid_for_fusion`
- `warnings`
- `ignition_score_raw`
- `ignition_score_available`
- `ignition_score`

`ignition_score_raw` zachowuje wagi pełnego modelu. Pole
`ignition_score_available` normalizuje wynik wyłącznie względem dostępnych
warstw i nie jest wynikiem pełnym. `ignition_score` ma wartość `null`, gdy
coverage jest mniejsze niż 70 procent.

Próg `MIN_COVERAGE_FOR_FUSION = 0.70` blokuje wynik oparty wyłącznie na dwóch
największych warstwach o łącznej wadze 0.45 i zdecydowanie blokuje pojedynczą
warstwę linii energetycznych o wadze 0.15. Fusion oblicza FEI i QIES wyłącznie
dla rekordu z `valid_for_fusion` równym `true`.

## Dane dostarczane ręcznie

Ręczny plik FIRMS musi być prawdziwym CSV z co najmniej kolumnami
`latitude`, `longitude`, `acq_date`, `acq_time`, `satellite`, `instrument` i
`confidence`.

LPIS należy dostarczyć jako czytelny GeoPackage w układzie rozpoznawanym przez
GDAL. IBL należy dostarczyć jako poprawny GeoJSON `FeatureCollection`. CLC jest
pobierany automatycznie z oficjalnej warstwy EEA, ale można również umieścić
zweryfikowany cache w oczekiwanej lokalizacji.

Raster EFFIS jest opcjonalny i nie należy do repozytorium.

```powershell
.\v5\.venv\Scripts\python.exe .\v5\run_all.py `
  --with-effis `
  --effis-tiff C:\data\severity_2025.tiff
```

Bez `--with-effis` manifest ma `EFFIS=SKIPPED`. Rasterio jest importowany
leniwo i nie jest wymagany dla core pipeline.

## Diagnostyka

`FIRMS HTTP 400` zwykle oznacza błędny produkt lub składnię URL. Błędy 400,
401, 403 i 404 nie są ponawiane. Po HTTP 401 należy sprawdzić klasyfikację w
`firms_parts/<SOURCE>/status.json`; po 429 wznowić nie wcześniej niż wskazuje
`retry_after_seconds`.

`osmium-tool not found` oznacza przejście do Pyogrio lub Fiona. Informacja o
braku GeoPandas pojawia się tylko wtedy, gdy nie można zaimportować samego
pakietu.

`driver OSM unavailable` oznacza, że zainstalowany GDAL nie obsługuje PBF.
Należy zainstalować osmium-tool albo dystrybucję GDAL z driverem OSM.

Brak LPIS lub IBL nie zatrzymuje pipeline, ale obniża coverage. Wynik częściowy
nie jest używany przez fusion jako operacyjny ignition score.

Jeżeli `py` i `.venv` wskazują inne interpretery, należy porównać:

```powershell
py -c "import sys; print(sys.executable)"
.\v5\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\v5\.venv\Scripts\python.exe .\v5\run_all.py --doctor
```

Runner domyślnie wykryje różnicę i bezpiecznie uruchomi się ponownie. Zmienna
ochronna zapobiega pętli re-exec.

## Testy

Podstawowy zestaw testów nie wymaga sieci.

```powershell
.\v5\.venv\Scripts\python.exe -m pytest
.\v5\.venv\Scripts\python.exe -m ruff check v5
.\v5\.venv\Scripts\python.exe -m ruff format --check v5
.\v5\.venv\Scripts\python.exe -m compileall v5
```
