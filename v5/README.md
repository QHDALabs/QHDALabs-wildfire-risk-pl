# QHDALabs Wildfire Risk PL v5

Pipeline v5 buduje graf nadleśnictw Dolnego Śląska, oblicza presję zapłonu,
stres roślinności Sentinel-2, sygnał QTE i wynik fusion, a następnie może
porównać wynik z rastrem EFFIS.

## Moduły

| Skrypt | Rola |
| --- | --- |
| `qhdalabs_wildfire_topology_v1.py` | Budowa grafu i wzbogacenie 33 węzłów |
| `qhdalabs_wildfire_sentinel_v1.py` | Stres roślinności Sentinel-2 NDWI |
| `qhdalabs_wildfire_qte_v1.py` | Sygnał Quantum Temporal Encoder |
| `qhdalabs_wildfire_ignition_v1.py` | Presja zapłonu i kontrola coverage |
| `qhdalabs_wildfire_fusion_v1.py` | Wynik fusion oraz FEI i QIES |
| `effis_validator.py` | Opcjonalna walidacja z rastrem EFFIS |

## Wymagania

Wymagany jest Python 3.11 lub nowszy. Minimalne zależności obliczeniowe są w
`requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

Pełne przetwarzanie lokalnych danych GIS wymaga również GeoPandas, Pyogrio,
Shapely i PyProj.

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
dni. Każde okno jest walidowane i atomowo zapisywane jako:

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

Polecenia należy wykonywać z katalogu `v5`.

```powershell
python qhdalabs_wildfire_ignition_v1.py --help
python qhdalabs_wildfire_ignition_v1.py
python qhdalabs_wildfire_fusion_v1.py
```

Jawny tryb syntetyczny służy wyłącznie do testów:

```powershell
python qhdalabs_wildfire_ignition_v1.py --stub
```

Wynik `--stub` ma `score_status` równy `stub` oraz
`valid_for_fusion` równy `false`.

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

Stan zweryfikowany 24 lipca 2026:

| Źródło | Status | Uwagi |
| --- | --- | --- |
| NASA FIRMS | Dostępne z `MAP_KEY` | Area API, okna od 1 do 5 dni |
| OSM Geofabrik | Dostępne | Parser wymaga osmium lub drivera GDAL OSM |
| BDOT10k WN i SN | Dostępne | Lokalne GeoPackage |
| EEA CLC 2018 | Dostępne | Oficjalna warstwa ArcGIS, paginacja po 1000 |
| ARiMR LPIS | Opcjonalne ręczne | Publiczny URL nie zwraca poprawnego HTTP |
| IBL KSIPL | Opcjonalne ręczne | Dawny GeoServer WFS zwraca HTTP 404 |
| EFFIS | Opcjonalne ręczne | Brak automatycznego źródła hotspotów |

Nie należy zakładać automatycznej dostępności LPIS ani IBL. Pipeline nie
wyłącza weryfikacji TLS i nie utrzymuje martwego URL jako fallbacku.

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

Brak danych ma wartość JSON `null`. Rzeczywisty wynik warstwy równy zero ma
wartość `0.0`.

Każdy rekord zawiera:

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
python effis_validator.py severity_2025.tiff --scores topology/risk_scores.json
```

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

## Testy

Podstawowy zestaw testów nie wymaga sieci.

```powershell
python -m pytest
python -m compileall .
```
