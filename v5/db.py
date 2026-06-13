# =============================================================================
# QHDALabs - Wildfire Risk PL v5
# db.py — warstwa dostępu do SQLite
#
# Schemat:
#   nadlesnictwa     — statyczne dane geograficzne
#   weather_history  — dane Open-Meteo (TTL 6h)
#   ndwi_sentinel    — dane Sentinel-2 (TTL 10d)
#   qte_results      — wyniki Quantum Temporal Encoder (TTL 6h)
#   risk_scores      — wyniki fuzji (TTL 6h)
#   alerts           — log alertów (append-only, nigdy nie kasowane)
# =============================================================================

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import DB_PATH, TTL_WEATHER_S, TTL_NDWI_S, TTL_QTE_S, TTL_RISK_S

log = logging.getLogger(__name__)

# =========================
# SCHEMA
# =========================
SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nadlesnictwa (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    eco         TEXT NOT NULL,
    rdlp        TEXT NOT NULL DEFAULT 'Wrocław',
    wojewodztwo TEXT NOT NULL DEFAULT 'dolnośląskie',
    created_at  REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS weather_history (
    node_id     TEXT NOT NULL,
    fetched_at  REAL NOT NULL,
    data        TEXT NOT NULL,   -- JSON
    PRIMARY KEY (node_id)
);

CREATE TABLE IF NOT EXISTS ndwi_sentinel (
    node_id        TEXT NOT NULL,
    fetched_at     REAL NOT NULL,
    ndwi_latest    REAL,
    ndwi_mean_30d  REAL,
    ndwi_min_30d   REAL,
    ndwi_trend_14d REAL,
    ndwi_stress    REAL,
    n_observations INTEGER,
    data           TEXT NOT NULL,   -- full JSON
    PRIMARY KEY (node_id)
);

CREATE TABLE IF NOT EXISTS qte_results (
    node_id      TEXT NOT NULL,
    computed_at  REAL NOT NULL,
    backend      TEXT,
    bridge_fired INTEGER NOT NULL DEFAULT 0,
    zz_01        REAL,
    zz_12        REAL,
    zz_23        REAL,
    zz_34        REAL,
    qte_score    REAL NOT NULL,
    data         TEXT NOT NULL,   -- full JSON
    PRIMARY KEY (node_id)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    node_id        TEXT NOT NULL,
    computed_at    REAL NOT NULL,
    final_score    REAL NOT NULL,
    tier           TEXT NOT NULL,
    ndwi_stress    REAL,
    qte_score      REAL,
    fwi_score      REAL,
    bridge_fired   INTEGER,
    eco_multiplier REAL,
    drought_days   INTEGER,
    data           TEXT NOT NULL,   -- full JSON
    PRIMARY KEY (node_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id      TEXT NOT NULL,
    node_name    TEXT NOT NULL,
    tier         TEXT NOT NULL,
    score        REAL NOT NULL,
    bridge_fired INTEGER NOT NULL DEFAULT 0,
    ndwi_latest  REAL,
    drought_days INTEGER,
    reason       TEXT,
    drone_recommended INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL DEFAULT (unixepoch()),
    run_id       TEXT    -- groups alerts from same pipeline run
);

CREATE INDEX IF NOT EXISTS idx_alerts_node    ON alerts(node_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_tier    ON alerts(tier);
"""

# =========================
# CONNECTION
# =========================
@contextmanager
def get_conn(db_path: Path = DB_PATH):
    """Context manager returning an open SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info("Database initialised: %s", db_path)


# =========================
# TTL HELPERS
# =========================
def _is_fresh(fetched_at: float | None, ttl_s: int) -> bool:
    if fetched_at is None:
        return False
    return (time.time() - fetched_at) < ttl_s


# =========================
# NADLEŚNICTWA
# =========================
def upsert_nadlesnictwo(node: dict, rdlp: str = "Wrocław",
                        woj: str = "dolnośląskie",
                        db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO nadlesnictwa (id, name, lat, lon, eco, rdlp, wojewodztwo)
            VALUES (:id, :name, :lat, :lon, :eco, :rdlp, :woj)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, lat=excluded.lat, lon=excluded.lon,
                eco=excluded.eco, rdlp=excluded.rdlp, wojewodztwo=excluded.wojewodztwo
        """, {**node, "rdlp": rdlp, "woj": woj})


def get_nadlesnictwa(rdlp: str | None = None,
                     woj:  str | None = None,
                     db_path: Path = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        if rdlp:
            rows = conn.execute(
                "SELECT * FROM nadlesnictwa WHERE rdlp=?", (rdlp,)
            ).fetchall()
        elif woj:
            rows = conn.execute(
                "SELECT * FROM nadlesnictwa WHERE wojewodztwo=?", (woj,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM nadlesnictwa").fetchall()
    return [dict(r) for r in rows]


# =========================
# WEATHER
# =========================
def get_weather(node_id: str, db_path: Path = DB_PATH) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM weather_history WHERE node_id=?", (node_id,)
        ).fetchone()
    if row and _is_fresh(row["fetched_at"], TTL_WEATHER_S):
        return json.loads(row["data"])
    return None


def save_weather(node_id: str, data: dict, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO weather_history (node_id, fetched_at, data)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                fetched_at=excluded.fetched_at, data=excluded.data
        """, (node_id, time.time(), json.dumps(data)))


# =========================
# NDWI SENTINEL
# =========================
def get_ndwi(node_id: str, db_path: Path = DB_PATH) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM ndwi_sentinel WHERE node_id=?", (node_id,)
        ).fetchone()
    if row and _is_fresh(row["fetched_at"], TTL_NDWI_S):
        return json.loads(row["data"])
    return None


def save_ndwi(node_id: str, result: dict, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO ndwi_sentinel
                (node_id, fetched_at, ndwi_latest, ndwi_mean_30d, ndwi_min_30d,
                 ndwi_trend_14d, ndwi_stress, n_observations, data)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(node_id) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                ndwi_latest=excluded.ndwi_latest,
                ndwi_mean_30d=excluded.ndwi_mean_30d,
                ndwi_min_30d=excluded.ndwi_min_30d,
                ndwi_trend_14d=excluded.ndwi_trend_14d,
                ndwi_stress=excluded.ndwi_stress,
                n_observations=excluded.n_observations,
                data=excluded.data
        """, (
            node_id, time.time(),
            result.get("ndwi_latest"),
            result.get("ndwi_mean_30d"),
            result.get("ndwi_min_30d"),
            result.get("ndwi_trend_14d"),
            result.get("ndwi_stress_latest"),
            result.get("n_observations"),
            json.dumps(result),
        ))


def ndwi_needs_refresh(node_id: str, db_path: Path = DB_PATH) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at FROM ndwi_sentinel WHERE node_id=?", (node_id,)
        ).fetchone()
    return not _is_fresh(row["fetched_at"] if row else None, TTL_NDWI_S)


# =========================
# QTE
# =========================
def get_qte(node_id: str, db_path: Path = DB_PATH) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM qte_results WHERE node_id=?", (node_id,)
        ).fetchone()
    if row and _is_fresh(row["computed_at"], TTL_QTE_S):
        return json.loads(row["data"])
    return None


def save_qte(node_id: str, result: dict, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO qte_results
                (node_id, computed_at, backend, bridge_fired,
                 zz_01, zz_12, zz_23, zz_34, qte_score, data)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(node_id) DO UPDATE SET
                computed_at=excluded.computed_at,
                backend=excluded.backend,
                bridge_fired=excluded.bridge_fired,
                zz_01=excluded.zz_01, zz_12=excluded.zz_12,
                zz_23=excluded.zz_23, zz_34=excluded.zz_34,
                qte_score=excluded.qte_score, data=excluded.data
        """, (
            node_id, time.time(),
            result.get("backend"),
            int(result.get("bridge_fired", False)),
            result.get("zz_01"), result.get("zz_12"),
            result.get("zz_23"), result.get("zz_34"),
            result.get("qte_score"),
            json.dumps(result),
        ))


# =========================
# RISK SCORES
# =========================
def get_risk(node_id: str, db_path: Path = DB_PATH) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM risk_scores WHERE node_id=?", (node_id,)
        ).fetchone()
    if row and _is_fresh(row["computed_at"], TTL_RISK_S):
        return json.loads(row["data"])
    return None


def save_risk(node_id: str, score: dict, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO risk_scores
                (node_id, computed_at, final_score, tier, ndwi_stress,
                 qte_score, fwi_score, bridge_fired, eco_multiplier,
                 drought_days, data)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(node_id) DO UPDATE SET
                computed_at=excluded.computed_at,
                final_score=excluded.final_score,
                tier=excluded.tier,
                ndwi_stress=excluded.ndwi_stress,
                qte_score=excluded.qte_score,
                fwi_score=excluded.fwi_score,
                bridge_fired=excluded.bridge_fired,
                eco_multiplier=excluded.eco_multiplier,
                drought_days=excluded.drought_days,
                data=excluded.data
        """, (
            node_id, time.time(),
            score.get("final_score"),
            score.get("tier"),
            score.get("signals", {}).get("ndwi_stress"),
            score.get("signals", {}).get("qte_score"),
            score.get("signals", {}).get("fwi_score"),
            int(score.get("signals", {}).get("bridge_fired", False)),
            score.get("modifiers", {}).get("eco_multiplier"),
            score.get("context", {}).get("drought_days"),
            json.dumps(score),
        ))


def get_all_risk_scores(rdlp: str | None = None,
                        db_path: Path = DB_PATH) -> list[dict]:
    """Return all current risk scores, optionally filtered by RDLP."""
    with get_conn(db_path) as conn:
        if rdlp:
            rows = conn.execute("""
                SELECT r.* FROM risk_scores r
                JOIN nadlesnictwa n ON r.node_id = n.id
                WHERE n.rdlp = ?
                ORDER BY r.final_score DESC
            """, (rdlp,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM risk_scores ORDER BY final_score DESC"
            ).fetchall()
    return [json.loads(r["data"]) for r in rows]


# =========================
# ALERTS (append-only log)
# =========================
def save_alert(alert: dict, run_id: str = "",
               db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO alerts
                (node_id, node_name, tier, score, bridge_fired,
                 ndwi_latest, drought_days, reason, drone_recommended, run_id)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            alert["node_id"], alert["node_name"],
            alert["tier"], alert["score"],
            int(alert.get("bridge_fired", False)),
            alert.get("ndwi_latest"),
            alert.get("drought_days"),
            alert.get("reason", ""),
            int(alert.get("drone_recommended", False)),
            run_id,
        ))


def get_alerts_history(node_id: str | None = None,
                       days: int = 30,
                       db_path: Path = DB_PATH) -> list[dict]:
    """Return alert history for last N days."""
    cutoff = time.time() - days * 86400
    with get_conn(db_path) as conn:
        if node_id:
            rows = conn.execute("""
                SELECT * FROM alerts
                WHERE node_id=? AND created_at > ?
                ORDER BY created_at DESC
            """, (node_id, cutoff)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM alerts
                WHERE created_at > ?
                ORDER BY created_at DESC
            """, (cutoff,)).fetchall()
    return [dict(r) for r in rows]


# =========================
# MIGRATION FROM JSON
# =========================
def migrate_from_json(
    topology_dir: Path,
    db_path: Path = DB_PATH,
) -> None:
    """
    One-time migration of existing JSON outputs to SQLite.
    Safe to run multiple times — uses upsert.
    """
    import time as _time

    log.info("Migrating JSON data to SQLite: %s", db_path)
    init_db(db_path)

    # ── 1. Nadleśnictwa + weather from nodes_enriched.json ───────────────
    enriched_path = topology_dir / "nodes_enriched.json"
    nodes_path    = topology_dir / "nodes.json"

    source = enriched_path if enriched_path.exists() else nodes_path
    if source.exists():
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", [])
        log.info("Migrating %d nadleśnictwa from %s", len(nodes), source.name)

        for node in nodes:
            # Static data
            upsert_nadlesnictwo({
                "id":  node["id"],
                "name": node["name"],
                "lat":  node["lat"],
                "lon":  node["lon"],
                "eco":  node["eco"],
            }, db_path=db_path)

            # Weather history
            wh = node.get("weather_history")
            if wh:
                # Preserve original fetch time or use now
                save_weather(node["id"], wh, db_path=db_path)

            # NDWI sentinel
            ns = node.get("ndwi_sentinel")
            if ns:
                save_ndwi(node["id"], ns, db_path=db_path)

        log.info("Nadleśnictwa migrated: %d", len(nodes))
    else:
        log.warning("No nodes file found at %s", topology_dir)

    # ── 2. QTE results ────────────────────────────────────────────────────
    qte_path = topology_dir / "qte_results.json"
    if qte_path.exists():
        with open(qte_path, encoding="utf-8") as f:
            qte_data = json.load(f)
        results = qte_data.get("results", [])
        for r in results:
            save_qte(r["node_id"], r, db_path=db_path)
        log.info("QTE results migrated: %d", len(results))

    # ── 3. Risk scores ────────────────────────────────────────────────────
    risk_path = topology_dir / "risk_scores.json"
    if risk_path.exists():
        with open(risk_path, encoding="utf-8") as f:
            risk_data = json.load(f)
        scores = risk_data.get("scores", [])
        for s in scores:
            save_risk(s["node_id"], s, db_path=db_path)
        log.info("Risk scores migrated: %d", len(scores))

    # ── 4. Alerts ─────────────────────────────────────────────────────────
    alerts_path = topology_dir / "alerts.json"
    if alerts_path.exists():
        with open(alerts_path, encoding="utf-8") as f:
            alerts_data = json.load(f)
        alerts = alerts_data.get("alerts", [])
        # Check if already migrated (avoid duplicates)
        with get_conn(db_path) as conn:
            existing = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        if existing == 0:
            run_id = alerts_data.get("generated_at", "migrated")[:19]
            for a in alerts:
                save_alert(a, run_id=run_id, db_path=db_path)
            log.info("Alerts migrated: %d", len(alerts))
        else:
            log.info("Alerts already in DB (%d rows) — skipping migration", existing)

    # ── Summary ───────────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        counts = {
            tbl: conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            for tbl in ["nadlesnictwa", "weather_history", "ndwi_sentinel",
                        "qte_results", "risk_scores", "alerts"]
        }
    log.info("Migration complete. DB contents: %s", counts)


# =========================
# DB INFO
# =========================
def db_summary(db_path: Path = DB_PATH) -> dict:
    """Return row counts and freshness info for all tables."""
    with get_conn(db_path) as conn:
        summary: dict[str, Any] = {}
        for tbl in ["nadlesnictwa", "weather_history", "ndwi_sentinel",
                    "qte_results", "risk_scores", "alerts"]:
            summary[tbl] = conn.execute(
                f"SELECT COUNT(*) FROM {tbl}"
            ).fetchone()[0]

        # Freshness of NDWI (most expensive — track carefully)
        oldest_ndwi = conn.execute(
            "SELECT MIN(fetched_at) FROM ndwi_sentinel"
        ).fetchone()[0]
        if oldest_ndwi:
            age_h = (time.time() - oldest_ndwi) / 3600
            summary["ndwi_oldest_h"] = round(age_h, 1)

        # Active alerts today
        today = time.time() - 86400
        summary["alerts_24h"] = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE created_at > ?", (today,)
        ).fetchone()[0]

    return summary


if __name__ == "__main__":
    # Run migration when called directly
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    topology_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("topology")
    migrate_from_json(topology_dir)
    print("\nDB summary:", db_summary())
