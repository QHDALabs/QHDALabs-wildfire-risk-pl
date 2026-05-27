# QHDALabs - Wildfire Risk PL v4

System do szacowania ryzyka pozarowego dla Polski na podstawie prognozy pogody, wilgotnosci gleby, stresu roslinnosci, uksztaltowania terenu oraz hybrydowego modelu klasyczno-kwantowego.

Plik glownego programu:

```text
qhdalabs-wildfire_risk_v4.py
```

## Opis dla laika

Ten program sprawdza, gdzie w Polsce warunki pogodowe moga sprzyjac powstawaniu pozarow.

Dziala mniej wiecej tak:

1. Dzieli Polske na siatke punktow.
2. Dla kazdego punktu pobiera prognoze pogody, m.in. temperature, wilgotnosc, wiatr, opady i wilgotnosc gleby.
3. Sprawdza przyblizone dane o terenie, np. wysokosc i nachylenie.
4. Oblicza, czy roslinnosc moze byc przesuszona.
5. Probuje pobrac historyczne dane o pozarach z EFFIS.
6. Trenuje model klasyczny oraz model kwantowy.
7. Laczy wyniki i zapisuje mape oraz raporty.

Wynikiem jest mapa `map.html`, na ktorej widac punkty ryzyka. Kolor zielony oznacza niskie ryzyko, pomaranczowy podwyzszone, a czerwony alarmowe.

Wersja v4 pilnuje, aby czesc kwantowa uruchamiala sie zawsze:

- jesli Qiskit jest zainstalowany, uzywany jest model `qiskit_qsvc`,
- jesli Qiskit nie zadziala, program uzywa lokalnego symulatora kwantowego w NumPy,
- jesli dane maja tylko jedna klase, program dodaje ostrozne przyklady pomocnicze, zeby model kwantowy mogl sie nauczyc rozroznienia.

To nie jest oficjalny system alarmowy. Wyniki nalezy traktowac jako analize eksperymentalna i pomocnicza.

## Co nowego w v4

- Model kwantowy jest zawsze trenowany albo przez Qiskit, albo przez fallback NumPy.
- Dodano `quantum_status` do `fire.json`.
- Dodano osobne kolumny `classical_risk` i `quantum_risk`.
- Poprawiono odporne pobieranie danych z API przez retry i cache.
- Poprawiono trenowanie przy danych zawierajacych tylko jedna klase.
- Wyciszono techniczne komunikaty optymalizatora Qiskit.
- Zaktualizowano wywolania Qiskit pod nowsze wersje biblioteki.

## Wymagania

Minimalne:

```bash
pip install numpy requests scikit-learn
```

Zalecane:

```bash
pip install shap
```

Dla pelnej sciezki kwantowej Qiskit:

```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-optimization
```

Jesli pakiety Qiskit nie sa dostepne, program nadal dziala, bo ma wbudowany fallback kwantowy.

## Uruchomienie

W folderze z plikiem uruchom:

```bash
py qhdalabs-wildfire_risk_v4.py
```

Albo:

```bash
python qhdalabs-wildfire_risk_v4.py
```

Program pobierze dane z internetu, wytrenuje modele i zapisze wyniki w tym samym katalogu.

## Pliki wynikowe

Po uruchomieniu powstaja:

```text
map.html
fire.json
fire.csv
shap_report.html
```

`map.html`  
Interaktywna mapa Polski z punktami ryzyka i suwakiem godzinowym.

`fire.json`  
Pelny wynik w formacie JSON. Zawiera ryzyko, dane kwantowe, SHAP, dane godzinowe i metadane `quantum_status`.

`fire.csv`  
Tabela z najwazniejszymi wynikami dla kazdego punktu siatki.

`shap_report.html`  
Raport pokazujacy, ktore cechy najbardziej wplynely na wynik modelu.

## Jak czytac wynik

Najwazniejsze pola:

```text
risk            - laczne ryzyko po zmieszaniu modelu klasycznego i kwantowego
classical_risk  - wynik modelu Random Forest
quantum_risk    - wynik modelu kwantowego
alert           - true, jesli risk przekracza ALERT_THRESHOLD
ndvi_stress     - przyblizony stres roslinnosci
elevation_m     - wysokosc terenu
slope_deg       - nachylenie terenu
quantum_backend - uzyta sciezka kwantowa
```

Domyslnie alarm pojawia sie, gdy:

```text
risk > 0.7
```

Prog mozna zmienic w pliku:

```python
ALERT_THRESHOLD = 0.7
```

## Najwazniejsze ustawienia

W pliku `qhdalabs-wildfire_risk_v4.py` mozna zmienic:

```python
GRID_SIZE = 6
ALERT_THRESHOLD = 0.7
MAX_WORKERS = 10
CACHE_TTL = 3600
EFFIS_LOOKBACK = 365
N_SENSORS = 5
QUANTUM_BLEND = 0.30
```

`GRID_SIZE`  
Rozdzielczosc siatki. Wartosc 6 oznacza 36 punktow. Wieksza wartosc daje dokladniejsza mape, ale wiecej zapytan do API.

`ALERT_THRESHOLD`  
Prog ryzyka, powyzej ktorego punkt jest oznaczony jako alarmowy.

`QUANTUM_BLEND`  
Udzial modelu kwantowego w wyniku koncowym. Domyslnie 30%.

`N_SENSORS`  
Liczba rekomendowanych lokalizacji czujnikow.

## Dane zewnetrzne

Program korzysta z:

- Open-Meteo - prognoza pogody,
- Open-Elevation - wysokosc terenu,
- EFFIS - europejskie dane o pozarach,
- OpenStreetMap - podklad mapy w `map.html`.

Jesli EFFIS jest niedostepny, program uzywa etykiet heurystycznych. Jesli Open-Elevation nie odpowiada, podstawiana jest srednia wartosc wysokosci.

## Uwagi o modelu kwantowym

W v4 model kwantowy nie jest juz tylko opcjonalnym dodatkiem.

Program probuje kolejno:

1. uruchomic Qiskit QSVC,
2. jesli to sie nie uda, uruchomic lokalny symulator kwantowego kernela,
3. jesli dane treningowe maja tylko jedna klase, dodac syntetyczne kontrprzyklady.

Dzieki temu `quantum_risk` jest zawsze obliczany, a `fire.json` pokazuje, ktora sciezka zostala uzyta.

Przyklad:

```json
"quantum_status": {
  "backend": "qiskit_qsvc",
  "augmented": true,
  "train_size": 31,
  "original_classes": [0],
  "blend_weight": 0.3
}
```

`augmented: true` oznacza, ze program dodal pomocnicze przyklady treningowe, bo dane zawieraly tylko jedna klase.

## Ograniczenia

- To narzedzie analityczne i eksperymentalne, nie oficjalny system ostrzegania.
- Jakosc wyniku zalezy od dostepnosci API i jakosci danych.
- NDVI jest tylko przyblizeniem, nie bezposrednim odczytem satelitarnym.
- Przy malej liczbie klas model uczy sie raczej wzorca ryzyka niz pelnej historii pozarow.
- Wyniki powinny byc interpretowane razem z lokalna wiedza terenowa.

## Licencja

MIT License - wolno uzywac, modyfikowac i rozpowszechniac z zachowaniem informacji o autorze.

Autor: Krzysztof W. Banasiewicz  
Organizacja: QHDALabs
