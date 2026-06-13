"""
QHDALabs Wildfire — EFFIS TIFF Validator
Krok 1: Inspekcja pliku / Step 1: File inspection
Krok 2: Wytnij Dolny Śląsk / Step 2: Crop Lower Silesia
Krok 3: Wyciągnij wartości per nadleśnictwo / Step 3: Extract per nadleśnictwo
Krok 4: Porównaj z SCORE / Step 4: Compare with fusion SCORE

Użycie / Usage:
    py effis_validator.py severity_2025.tiff
    py effis_validator.py severity_2025.tiff --scores topology/risk_scores.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


# ============================================================
# Współrzędne centrów nadleśnictw Dolnego Śląska
# Coordinates of Lower Silesian forest district centers
# ============================================================

NADLESNICTWA = {
    "Piszowice"       : (51.0012,  15.8900),
    "Leśna"           : (51.0010,  15.2800),
    "Myśliborskie"    : (51.5200,  15.7400),
    "Bolesławiec"     : (51.2650,  15.5650),
    "Ruszów"          : (51.4200,  15.3500),
    "Bardo Śląskie"   : (50.5050,  16.7400),
    "Węgliniec"       : (51.2900,  15.2200),
    "Złotoryja"       : (51.1250,  15.9200),
    "Zgorzelec"       : (51.1500,  15.0300),
    "Jawor"           : (51.0500,  16.1800),
    "Lwówek Śląski"   : (51.1050,  15.5900),
    "Miękinia"        : (51.0700,  16.8200),
    "Rychtal"         : (51.3800,  17.7500),
    "Lądek-Zdrój"     : (50.3400,  16.8900),
    "Milicz"          : (51.5250,  17.2800),
    "Wołów"           : (51.3400,  16.6400),
    "Góry Stołowe"    : (50.4800,  16.3600),
    "Jugów"           : (50.5800,  16.4900),
    "Legnica"         : (51.2100,  16.1600),
    "Świętoszów"      : (51.5500,  15.6700),
    "Wałbrzych"       : (50.7700,  16.2600),
    "Szklarska Poręba": (50.8300,  15.5200),
    "Prudnik"         : (50.3200,  17.5700),
    "Żmigród"         : (51.4700,  16.9000),
    "Bystrzyca Kłodzka": (50.3000,  16.6400),
    "Świdnica"        : (50.8500,  16.4900),
    "Syców"           : (51.3100,  17.7200),
    "Kłodzko"         : (50.4350,  16.6600),
    "Oleśnica Śląska" : (50.9200,  17.3800),
    "Kamienna Góra"   : (50.7800,  16.0300),
    "Lubin"           : (51.4000,  16.2000),
    "Oława"           : (50.9400,  17.2900),
    "Wrocław"         : (51.1000,  17.0300),
}

# Bbox Dolnego Śląska / Lower Silesia bbox (lon_min, lat_min, lon_max, lat_max)
DS_BBOX = (14.5, 49.9, 18.1, 51.9)


# ============================================================
# KROK 1: Inspekcja pliku / Step 1: File inspection
# ============================================================

def inspect_tiff(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  EFFIS TIFF Inspection: {path.name}")
    print(f"{'='*60}")

    with rasterio.open(path) as src:
        print(f"  CRS          : {src.crs}")
        print(f"  Bounds       : {src.bounds}")
        print(f"  Shape        : {src.height} x {src.width} px")
        print(f"  Bands        : {src.count}")
        print(f"  Dtype        : {src.dtypes[0]}")
        print(f"  NoData       : {src.nodata}")
        print(f"  Resolution   : {src.res}")

        # Podgląd zakresu wartości / Value range preview
        data = src.read(1)
        valid = data[data != src.nodata] if src.nodata is not None else data.flatten()
        valid = valid[valid > 0]  # tylko piksele z ogniem / only fire pixels

        if len(valid) > 0:
            print(f"\n  Wartości / Values (fire pixels only):")
            print(f"    Min    : {valid.min():.4f}")
            print(f"    Max    : {valid.max():.4f}")
            print(f"    Mean   : {valid.mean():.4f}")
            print(f"    Nonzero: {len(valid):,} px")
        else:
            print(f"\n  Brak pikseli z ogniem w tym pliku / No fire pixels found")
            print(f"  Data range: {data.min()} – {data.max()}")


# ============================================================
# KROK 2+3: Wyciągnij wartości per nadleśnictwo
# Step 2+3: Extract values per forest district
# ============================================================

def extract_per_nadlesnictwo(path: Path) -> dict:
    """
    Dla każdego nadleśnictwa wyciąga wartość severity z TIFF.
    Używa okna 5x5 px wokół centrum i bierze max (worst case).
    For each district extracts severity value from TIFF.
    Uses 5x5 px window around center, takes max (worst case).
    """
    results = {}

    with rasterio.open(path) as src:
        # Sprawdź czy potrzebna reprojekcja / Check if reprojection needed
        tiff_crs = src.crs
        is_wgs84 = tiff_crs and tiff_crs.to_epsg() == 4326

        for name, (lat, lon) in NADLESNICTWA.items():
            try:
                if is_wgs84 or tiff_crs is None:
                    # Bezpośrednie użycie lon/lat / Direct lon/lat use
                    row, col = src.index(lon, lat)
                else:
                    # Reprojekcja współrzędnych / Reproject coordinates
                    from rasterio.warp import transform
                    xs, ys = transform(
                        "EPSG:4326", tiff_crs, [lon], [lat]
                    )
                    row, col = src.index(xs[0], ys[0])

                # Okno 5x5 px / 5x5 px window
                r0 = max(0, row - 2)
                r1 = min(src.height, row + 3)
                c0 = max(0, col - 2)
                c1 = min(src.width,  col + 3)

                window = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
                patch  = src.read(1, window=window).astype(float)

                nodata = src.nodata if src.nodata is not None else -9999
                patch[patch == nodata] = np.nan
                patch[patch <= 0]      = np.nan

                val = float(np.nanmax(patch)) if not np.all(np.isnan(patch)) else 0.0
                results[name] = val

            except Exception as e:
                results[name] = 0.0

    return results


# ============================================================
# KROK 4: Porównanie z SCORE fuzji / Compare with fusion SCORE
# ============================================================

def load_scores(scores_path: Path) -> dict:
    """Wczytaj score z pliku JSON systemu fuzji / Load fusion scores from JSON."""
    try:
        with open(scores_path) as f:
            data = json.load(f)
        # Obsługuje różne struktury / Handle different JSON structures
        if isinstance(data, list):
            return {item.get("name", item.get("nadlesnictwo", "")): item.get("score", 0.0)
                    for item in data}
        if isinstance(data, dict):
            # {name: score} lub {name: {score: ...}}
            out = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    out[k] = v.get("score", v.get("SCORE", 0.0))
                else:
                    out[k] = float(v)
            return out
    except Exception as e:
        print(f"  Ostrzeżenie: nie można wczytać {scores_path}: {e}")
    return {}


def normalize(values: dict) -> dict:
    """Normalizuj wartości do [0,1] / Normalize values to [0,1]."""
    vals = list(values.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def compare_and_print(effis: dict, fusion: dict) -> None:
    """
    Wydruk porównawczy EFFIS severity vs Fusion SCORE.
    Kluczowe pytanie: czy ranking się zgadza?
    Print comparison of EFFIS severity vs Fusion SCORE.
    Key question: does the ranking agree?
    """
    print(f"\n{'='*78}")
    print(f"  PORÓWNANIE / COMPARISON — EFFIS Severity 2025 vs Fusion SCORE")
    print(f"{'='*78}")
    print(f"  {'Nadleśnictwo':<22} {'EFFIS_raw':>10} {'EFFIS_norm':>10} "
          f"{'Fusion':>8} {'Δ':>8}  {'Zgodność':>9}")
    print(f"  {'-'*75}")

    # Normalizuj EFFIS / Normalize EFFIS
    effis_norm = normalize(effis)

    rows = []
    for name in NADLESNICTWA:
        en   = effis_norm.get(name, 0.0)
        er   = effis.get(name, 0.0)
        fs   = fusion.get(name, None)
        if fs is not None:
            delta = en - fs
            agree = "✓" if abs(delta) < 0.25 else ("~" if abs(delta) < 0.40 else "✗")
        else:
            delta = None
            agree = "—"
        rows.append((name, er, en, fs, delta, agree))

    # Sortuj po EFFIS severity / Sort by EFFIS severity
    rows.sort(key=lambda x: x[2], reverse=True)

    for name, er, en, fs, delta, agree in rows:
        fs_str    = f"{fs:8.3f}" if fs is not None else "     n/a"
        delta_str = f"{delta:+8.3f}" if delta is not None else "     n/a"
        print(f"  {name:<22} {er:>10.3f} {en:>10.3f} {fs_str} {delta_str}  {agree:>9}")

    # Korelacja Spearmana / Spearman correlation
    if fusion:
        common = [n for n in NADLESNICTWA if n in fusion]
        if len(common) >= 5:
            effis_ranks  = sorted(common, key=lambda n: effis_norm.get(n, 0), reverse=True)
            fusion_ranks = sorted(common, key=lambda n: fusion.get(n, 0),     reverse=True)
            n = len(common)
            d2 = sum(
                (effis_ranks.index(name) - fusion_ranks.index(name))**2
                for name in common
            )
            rho = 1 - (6 * d2) / (n * (n**2 - 1))
            print(f"\n  Korelacja Spearmana (ranking) / Spearman rank correlation: ρ = {rho:.4f}")
            if rho > 0.7:
                print(f"  ✓ Silna zgodność rankingów — walidacja historyczna potwierdza system")
            elif rho > 0.4:
                print(f"  ~ Umiarkowana zgodność — sprawdź węzły z największym Δ")
            else:
                print(f"  ✗ Słaba zgodność — wymaga analizy (lub 2025 nie miał pożarów na DS)")

    print(f"\n  Uwaga: EFFIS severity 2025 może być zerowe jeśli sezon jeszcze trwa")
    print(f"  lub pożary na Dolnym Śląsku były poza progiem detekcji satelitarnej.")
    print(f"  Rozważ pobranie 2020-2024 dla pełnej walidacji historycznej.")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="EFFIS TIFF validator for QHDALabs Wildfire system"
    )
    parser.add_argument("tiff", help="Ścieżka do pliku TIFF / Path to TIFF file")
    parser.add_argument(
        "--scores",
        default="topology/risk_scores.json",
        help="Ścieżka do JSON z wynikami fuzji / Path to fusion scores JSON"
    )
    args = parser.parse_args()

    tiff_path   = Path(args.tiff)
    scores_path = Path(args.scores)

    if not tiff_path.exists():
        print(f"BŁĄD: Nie znaleziono pliku {tiff_path}")
        sys.exit(1)

    # Krok 1
    inspect_tiff(tiff_path)

    # Kroki 2+3
    print(f"\n{'='*60}")
    print(f"  Wyciągam wartości per nadleśnictwo...")
    print(f"{'='*60}")
    effis_values = extract_per_nadlesnictwo(tiff_path)

    nonzero = sum(1 for v in effis_values.values() if v > 0)
    print(f"  Nadleśnictw z wartością > 0: {nonzero}/{len(effis_values)}")

    if nonzero == 0:
        print(f"\n  Wszystkie wartości = 0. Możliwe przyczyny:")
        print(f"  1. Plik TIFF obejmuje cały świat ale fire pixels są rzadkie")
        print(f"  2. Rok 2025 — sezon w toku, mało danych")
        print(f"  3. CRS pliku wymaga sprawdzenia (patrz wyżej)")
        print(f"\n  → Pobierz severity_2022.tiff lub severity_2023.tiff")
        print(f"    (lata z udokumentowanymi pożarami na Dolnym Śląsku)")

    # Krok 4
    fusion_scores = {}
    if scores_path.exists():
        fusion_scores = load_scores(scores_path)
        print(f"\n  Wczytano {len(fusion_scores)} wyników fuzji z {scores_path}")
    else:
        print(f"\n  Brak pliku {scores_path} — pominięto porównanie z fuzją")
        print(f"  Użyj: py effis_validator.py tiff --scores topology/risk_scores.json")

    compare_and_print(effis_values, fusion_scores)


if __name__ == "__main__":
    main()
