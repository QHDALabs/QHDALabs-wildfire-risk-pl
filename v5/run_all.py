#!/usr/bin/env python3
# =============================================================================
# QHDALabs - Wildfire Risk PL v5
# run_all.py — uruchamia cały pipeline w kolejności
#
# Kolejność:
#   1. topology    — nadleśnictwa, graf sąsiedztwa, pogoda 14d
#   2. sentinel    — Sentinel-2 NDWI (z TTL 10d — pomija jeśli świeże)
#   3. qte         — Quantum Temporal Encoder
#   4. ignition    — moduł ciśnienia zapłonu
#   5. fusion      — fuzja sygnałów, scoring końcowy
#   6. effis       — walidacja z bazą EFFIS
#
# Użycie:
#   py run_all.py              # pełny pipeline
#   py run_all.py --skip-sentinel  # pomija Sentinel (np. gdy dane świeże)
#   py run_all.py --only fusion    # tylko jeden moduł
#   py run_all.py --migrate    # migracja JSON->SQLite i wyjście
# =============================================================================

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# =========================
# PIPELINE STEPS
# =========================
STEPS = [
    {
        "name":   "topology",
        "script": "qhdalabs_wildfire_topology_v1.py",
        "desc":   "Nadleśnictwa + graf sąsiedztwa + pogoda 14d",
    },
    {
        "name":   "sentinel",
        "script": "qhdalabs_wildfire_sentinel_v1.py",
        "desc":   "Sentinel-2 NDWI (TTL 10d)",
        "skippable": True,
    },
    {
        "name":   "qte",
        "script": "qhdalabs_wildfire_qte_v1.py",
        "desc":   "Quantum Temporal Encoder",
    },
    {
        "name":   "ignition",
        "script": "qhdalabs_wildfire_ignition_v1.py",
        "desc":   "Moduł ciśnienia zapłonu",
    },
    {
        "name":   "fusion",
        "script": "qhdalabs_wildfire_fusion_v1.py",
        "desc":   "Fuzja sygnałów + scoring końcowy + alerty",
    },
    {
        "name":   "effis",
        "script": "effis_validator.py",
        "desc":   "Walidacja z bazą EFFIS",
        "skippable": True,
    },
]


def run_step(step: dict) -> bool:
    """Uruchom jeden krok pipeline. Zwraca True jeśli sukces."""
    script = BASE_DIR / step["script"]
    if not script.exists():
        log.warning("Pominięto (brak pliku): %s", step["script"])
        return True  # nie przerywamy pipeline

    log.info("─" * 60)
    log.info("▶  %s — %s", step["name"].upper(), step["desc"])
    log.info("─" * 60)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        log.info("✓  %s — OK (%.1fs)", step["name"], elapsed)
        return True
    else:
        log.error("✗  %s — BŁĄD (kod %d, %.1fs)",
                  step["name"], result.returncode, elapsed)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QHDALabs Wildfire Risk PL v5 — pipeline runner"
    )
    parser.add_argument(
        "--skip", metavar="STEP", nargs="+",
        help="Pomiń kroki (np. --skip sentinel effis)"
    )
    parser.add_argument(
        "--only", metavar="STEP",
        help="Uruchom tylko jeden krok"
    )
    parser.add_argument(
        "--migrate", action="store_true",
        help="Migruj dane JSON->SQLite i zakończ"
    )
    args = parser.parse_args()

    # ── Migracja ──────────────────────────────────────────────────────────
    if args.migrate:
        log.info("Migracja JSON -> SQLite ...")
        topology_dir = BASE_DIR / "topology"
        if not topology_dir.exists():
            log.error("Brak katalogu topology/")
            sys.exit(1)
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "db.py"), str(topology_dir)],
            cwd=str(BASE_DIR),
        )
        sys.exit(result.returncode)

    # ── Wybór kroków ──────────────────────────────────────────────────────
    skip = set(args.skip or [])
    steps = STEPS

    if args.only:
        steps = [s for s in STEPS if s["name"] == args.only]
        if not steps:
            log.error("Nieznany krok: %s. Dostępne: %s",
                      args.only, [s["name"] for s in STEPS])
            sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("QHDALabs Wildfire Risk PL v5")
    log.info("Kroki: %s", [s["name"] for s in steps if s["name"] not in skip])
    log.info("=" * 60)

    t_total = time.time()
    failed = []

    for step in steps:
        if step["name"] in skip:
            log.info("⏭  %s — pominięty (--skip)", step["name"])
            continue

        ok = run_step(step)
        if not ok:
            failed.append(step["name"])
            log.error("Pipeline przerwany na kroku: %s", step["name"])
            break

    # ── Podsumowanie ──────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    log.info("=" * 60)
    if not failed:
        log.info("✓  Pipeline zakończony pomyślnie (%.1fs)", elapsed)

        # Pokaż skrót alertów jeśli fusion był uruchomiony
        if not args.only or args.only == "fusion":
            _print_alert_summary()
    else:
        log.error("✗  Pipeline zakończony z błędami: %s", failed)
        sys.exit(1)
    log.info("=" * 60)


def _print_alert_summary() -> None:
    """Wyświetl skrót aktywnych alertów z bazy."""
    try:
        import db
        from pathlib import Path
        summary = db.db_summary()
        scores  = db.get_all_risk_scores()
        alerts  = [s for s in scores
                   if s.get("tier") in ("CRITICAL", "HIGH")]

        log.info("")
        log.info("AKTYWNE ALERTY: %d  (CRITICAL: %d  HIGH: %d)",
                 len(alerts),
                 sum(1 for a in alerts if a.get("tier") == "CRITICAL"),
                 sum(1 for a in alerts if a.get("tier") == "HIGH"))

        for a in alerts[:10]:  # max 10 w podsumowaniu
            tier   = a.get("tier", "?")
            name   = a.get("node_name", a.get("node_id", "?"))
            score  = a.get("final_score", 0)
            bridge = "🔥" if a.get("signals", {}).get("bridge_fired") else "  "
            drone  = " → DRON" if tier == "CRITICAL" else ""
            log.info("  [%s] %s %.3f %s%s", tier, name, score, bridge, drone)

        if len(alerts) > 10:
            log.info("  ... i %d więcej (patrz topology/alerts.json)", len(alerts)-10)

        log.info("DB: %s", summary)
    except Exception as exc:
        log.debug("Nie udało się wczytać podsumowania alertów: %s", exc)


if __name__ == "__main__":
    main()
