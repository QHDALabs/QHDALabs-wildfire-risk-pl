# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 4 — Fusion Risk Scorer & Alert Engine
# File          : qhdalabs_wildfire_fusion_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# Integrates all three signal layers into one unified wildfire risk score
# per nadleśnictwo and generates operational alerts.
#
# Signal layers
# -----------------------------------------------------------------------------
#  Layer 1 — Satellite (Sentinel-2 NDWI)           weight: 45%
#    Real physiological state of the forest canopy.
#    Source: topology/nodes_enriched.json (Step 2)
#
#  Layer 2 — Quantum Temporal Encoder (QTE)         weight: 30%
#    Sequential pattern: was dry THEN wind rose.
#    Captures what RF cannot see: order matters.
#    Source: topology/qte_results.json (Step 3)
#
#  Layer 3 — Weather proxy (Open-Meteo FWI)         weight: 15%
#    Current meteorological fire danger.
#    Source: topology/nodes_enriched.json (Step 2, weather history)
#
#  Layer 4 — NDWI drying trend                     weight: 10%
#    Velocity of vegetation stress change.
#    Negative trend = accelerating drought.
#
# Modifiers (applied after blending)
# -----------------------------------------------------------------------------
#  Ecosystem multiplier   pine: ×1.30, mixed: ×1.00, spruce: ×0.80
#    Pine forests ignite more easily; spruce wetter but catastrophic when burned
#
#  Bridge bonus           +0.08 if QTE bridge fired (sequential pattern)
#    This is the key quantum contribution — marks nodes where the
#    pre-fire sequence (dry→wind) is confirmed, not just current conditions
#
#  Network propagation    +0.05 × neighbour_stress
#    Modelling the mycelium: stressed neighbours lower your alert threshold
#
# Alert tiers
# -----------------------------------------------------------------------------
#  CRITICAL  > 0.75  — immediate drone verification recommended
#  HIGH      > 0.60  — elevated patrol priority
#  MODERATE  > 0.45  — monitoring flag
#  LOW       ≤ 0.45  — normal monitoring
#
# Outputs
# -----------------------------------------------------------------------------
#  topology/risk_scores.json     — full scored results
#  topology/alerts.json          — CRITICAL/HIGH alerts only
#  topology/final_map.html       — operational map with all layers visible
#
# Author        : Krzysztof W. Banasiewicz / QHDALabs
# License       : MIT
# =============================================================================

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from pipeline_contract import (
    StepStatus,
    atomic_write_json,
    validate_node_payload,
    write_step_manifest,
)
from qhdalabs_wildfire_ignition_v1 import (
    IgnitionScore,
    compute_fei,
    compute_qies,
    load_ignition_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = "topology"

# =========================
# FUSION CONFIG
# =========================
W_NDWI = 0.45  # satellite vegetation stress
W_QTE = 0.30  # quantum temporal encoder
W_FWI = 0.15  # weather fire index
W_TREND = 0.10  # drying velocity (negative trend = higher risk)

BRIDGE_BONUS = 0.08  # sequential pattern confirmed by QTE bridge
NETWORK_COEFF = 0.05  # neighbour propagation weight

ALERT_CRITICAL = 0.75
ALERT_HIGH = 0.60
ALERT_MODERATE = 0.45

ECO_MULTIPLIERS = {
    "pine": 1.30,
    "pine_wetland": 1.10,
    "mixed": 1.00,
    "spruce_mountain": 0.80,
}


# =========================
# FWI PROXY (from weather)
# =========================
def _fwi_from_weather(node: dict) -> float:
    """Compute FWI proxy from latest weather data."""
    latest = node.get("latest", {})
    vpd = float(
        latest.get("vpd_mean")
        or node.get("weather_history", {}).get("vpd_mean", [0.5])[-1]
        if node.get("weather_history")
        else 0.5
    )
    soil = float(latest.get("soil_min") or 0.20)
    temp = float(latest.get("temp_max") or 20.0)
    rh = float(latest.get("rh_min") or 55.0)
    wind = float(latest.get("wind_max") or 5.0)
    rain = float(latest.get("rain_total") or 0.0)
    drought = int(node.get("drought_days", 0))

    vpd_n = min(1.0, vpd / 3.5)
    soil_n = max(0.0, min(1.0, 1.0 - soil / 0.30))
    temp_n = max(0.0, min(1.0, (temp - 10.0) / 25.0))
    wind_n = min(1.0, wind / 15.0)
    rh_n = max(0.0, (72.0 - rh) / 72.0)
    drought_n = min(1.0, drought / 45.0)
    rain_pen = max(0.0, 1.0 - rain / 4.0)

    raw = (
        0.22 * vpd_n
        + 0.20 * soil_n
        + 0.14 * temp_n
        + 0.14 * wind_n
        + 0.17 * rh_n
        + 0.08 * drought_n
        + 0.05 * rain_pen
    )
    return float(np.clip(raw, 0.0, 1.0))


# =========================
# CORE FUSION
# =========================
@dataclass
class RiskScore:
    node_id: str
    node_name: str
    lat: float
    lon: float
    eco: str
    # Input signals
    ndwi_stress: float  # satellite [0,1]
    ndwi_trend_14d: float  # slope (neg = drying)
    qte_score: float  # quantum [0,1]
    bridge_fired: bool  # sequential pattern
    fwi_score: float  # weather [0,1]
    network_stress: float  # propagated [0,1]
    ignition_score: float | None
    ignition_coverage_percent: float
    ignition_valid_for_fusion: bool
    fei: float | None
    qies: float | None
    # Components
    base_score: float  # before modifiers
    eco_multiplier: float
    # Final
    unclipped_score: float
    final_score: float
    tier: str  # CRITICAL | HIGH | MODERATE | LOW
    pipeline_status: str
    operational_validity: bool
    data_quality_flags: list[str]
    calibration_flags: list[str]
    upstream_statuses: dict[str, str]
    # Context
    drought_days: int
    temp_max: float | None
    wind_max: float | None
    rh_min: float | None
    ndwi_latest: float | None
    # Explanation
    top_drivers: list[dict]


def _trend_signal(trend: float) -> float:
    """Convert 14d NDWI trend to risk signal [0,1]. Negative = drying."""
    return float(min(1.0, max(0.0, -trend * 3.0)))


def _tier(score: float) -> str:
    if score > ALERT_CRITICAL:
        return "CRITICAL"
    if score > ALERT_HIGH:
        return "HIGH"
    if score > ALERT_MODERATE:
        return "MODERATE"
    return "LOW"


def _explain(ndwi, qte, fwi, trend_s, eco_mult, bridge, net_stress) -> list[dict]:
    """Return top contributing factors sorted by impact."""
    factors = [
        {
            "factor": "ndwi_satellite",
            "contribution": round(W_NDWI * ndwi * eco_mult, 3),
            "label": f"NDWI stress {ndwi:.2f} × eco {eco_mult:.2f}",
        },
        {
            "factor": "quantum_temporal",
            "contribution": round(W_QTE * qte * eco_mult, 3),
            "label": f"QTE score {qte:.3f} × eco {eco_mult:.2f}",
        },
        {
            "factor": "weather_fwi",
            "contribution": round(W_FWI * fwi * eco_mult, 3),
            "label": f"FWI {fwi:.3f} × eco {eco_mult:.2f}",
        },
        {
            "factor": "drying_trend",
            "contribution": round(W_TREND * trend_s * eco_mult, 3),
            "label": f"14d trend signal {trend_s:.3f}",
        },
    ]
    if bridge:
        factors.append(
            {
                "factor": "quantum_bridge",
                "contribution": BRIDGE_BONUS,
                "label": "Sequential dry→wind pattern confirmed 🔥",
            }
        )
    if net_stress > 0.3:
        factors.append(
            {
                "factor": "network_propagation",
                "contribution": round(NETWORK_COEFF * net_stress, 3),
                "label": f"Neighbour stress {net_stress:.2f}",
            }
        )
    return sorted(factors, key=lambda x: x["contribution"], reverse=True)[:4]


def compute_risk_score(
    node: dict,
    qte_map: dict[str, dict],
    graph: dict,
    all_nodes: dict[str, dict],
    ignition_map: dict[str, IgnitionScore] | None = None,
) -> RiskScore:
    """Compute unified risk score for one node."""
    nid = node["id"]
    latest = node.get("latest", {})
    ndwi_s = node.get("ndwi_sentinel", {})
    eco = node.get("eco", "mixed")

    # ── Signals ──────────────────────────────────────────────────────────
    ndwi_stress = float(
        node.get("ndwi_stress_normalised") or node.get("ndwi_stress_latest", 0.0)
    )
    ndwi_trend = float(node.get("ndwi_trend_14d") or 0.0)
    trend_s = _trend_signal(ndwi_trend)

    qte_data = qte_map.get(nid, {})
    qte_score = float(qte_data.get("qte_score", 0.0))
    bridge_fired = bool(qte_data.get("bridge_fired", False))

    fwi = _fwi_from_weather(node)

    # Network propagation: average normalised NDWI of neighbours
    neighbours = graph.get(nid, [])
    if neighbours:
        nb_stresses = []
        for nb in neighbours:
            nb_node = all_nodes.get(nb["id"])
            if nb_node:
                nb_stresses.append(float(nb_node.get("ndwi_stress_normalised") or 0.0))
        net_stress = float(np.mean(nb_stresses)) if nb_stresses else 0.0
    else:
        net_stress = 0.0

    # ── Ecosystem multiplier ──────────────────────────────────────────────
    eco_mult = ECO_MULTIPLIERS.get(eco, 1.0)

    # ── Base fusion ───────────────────────────────────────────────────────
    base = W_NDWI * ndwi_stress + W_QTE * qte_score + W_FWI * fwi + W_TREND * trend_s

    # ── Apply modifiers ───────────────────────────────────────────────────
    modified = base * eco_mult
    modified += BRIDGE_BONUS if bridge_fired else 0.0
    modified += NETWORK_COEFF * net_stress

    final = float(np.clip(modified, 0.0, 1.0))

    ignition = (ignition_map or {}).get(nid)
    ignition_valid = bool(
        ignition and ignition.valid_for_fusion and ignition.ignition_score is not None
    )
    ignition_score = ignition.ignition_score if ignition_valid else None
    ignition_coverage = ignition.coverage_percent if ignition else 0.0
    fei = (
        compute_fei(final * 100.0, ignition_score)
        if ignition_score is not None
        else None
    )
    qies = (
        compute_qies(final * 100.0, ignition_score)
        if ignition_score is not None
        else None
    )

    # ── Explanability ─────────────────────────────────────────────────────
    drivers = _explain(
        ndwi_stress, qte_score, fwi, trend_s, eco_mult, bridge_fired, net_stress
    )
    data_quality_flags = []
    if not ignition_valid:
        data_quality_flags.append("ignition_invalid_or_unavailable")
    calibration_flags = ["satellite_stress_mapping_uncalibrated"]
    pipeline_status = StepStatus.SUCCESS if ignition_valid else StepStatus.DEGRADED

    return RiskScore(
        node_id=nid,
        node_name=node["name"],
        lat=node["lat"],
        lon=node["lon"],
        eco=eco,
        ndwi_stress=round(ndwi_stress, 4),
        ndwi_trend_14d=round(ndwi_trend, 5),
        qte_score=round(qte_score, 4),
        bridge_fired=bridge_fired,
        fwi_score=round(fwi, 4),
        network_stress=round(net_stress, 4),
        ignition_score=(
            round(ignition_score, 2) if ignition_score is not None else None
        ),
        ignition_coverage_percent=round(ignition_coverage, 2),
        ignition_valid_for_fusion=ignition_valid,
        fei=round(fei, 2) if fei is not None else None,
        qies=round(qies, 2) if qies is not None else None,
        base_score=round(base, 4),
        eco_multiplier=eco_mult,
        unclipped_score=round(modified, 4),
        final_score=round(final, 4),
        tier=_tier(final),
        pipeline_status=str(pipeline_status),
        operational_validity=False,
        data_quality_flags=data_quality_flags,
        calibration_flags=calibration_flags,
        upstream_statuses={
            "sentinel": "DEGRADED",
            "qte": "SUCCESS",
            "ignition": "SUCCESS" if ignition_valid else "BLOCKED",
        },
        drought_days=int(node.get("drought_days", 0)),
        temp_max=latest.get("temp_max"),
        wind_max=latest.get("wind_max"),
        rh_min=latest.get("rh_min"),
        ndwi_latest=ndwi_s.get("ndwi_latest"),
        top_drivers=drivers,
    )


# =========================
# PIPELINE
# =========================
def run_fusion_pipeline(
    enriched_json: str = os.path.join(OUTPUT_DIR, "nodes_enriched.json"),
    qte_json: str = os.path.join(OUTPUT_DIR, "qte_results.json"),
    graph_json: str = os.path.join(OUTPUT_DIR, "graph.json"),
) -> list[RiskScore]:
    log.info("=== QHDALabs Wildfire — Step 4: Fusion Risk Scorer ===")

    # ── Load data ─────────────────────────────────────────────────────────
    for path in [enriched_json, qte_json, graph_json]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    with open(enriched_json, encoding="utf-8") as f:
        enriched = json.load(f)
    with open(qte_json, encoding="utf-8") as f:
        qte_raw = json.load(f)
    with open(graph_json, encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = enriched["nodes"]
    graph = graph_data["adjacency"]
    all_nodes = {n["id"]: n for n in nodes}

    # QTE lookup: node_id -> result dict
    qte_map = {r["node_id"]: r for r in qte_raw.get("results", [])}
    ignition_map = load_ignition_map()
    valid_ignition = sum(1 for item in ignition_map.values() if item.valid_for_fusion)
    log.info(
        "Loaded %d nodes | %d QTE results | %d/%d fusion-valid ignition scores",
        len(nodes),
        len(qte_map),
        valid_ignition,
        len(ignition_map),
    )

    # ── Score all nodes ───────────────────────────────────────────────────
    scores = [
        compute_risk_score(node, qte_map, graph, all_nodes, ignition_map)
        for node in nodes
    ]
    scores.sort(key=lambda s: s.final_score, reverse=True)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(scores)

    # ── Save ──────────────────────────────────────────────────────────────
    _save_results(scores)
    _generate_final_map(scores, graph)

    return scores


# =========================
# OUTPUT
# =========================
def _print_summary(scores: list[RiskScore]) -> None:
    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for s in scores:
        tier_counts[s.tier] += 1

    log.info("\n%s", "═" * 82)
    log.info(
        "RDLP Wrocław — Unified Wildfire Risk  |  %s",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    log.info(
        "Signals: Sentinel-2 moisture (45%) + QTE quantum (30%) + "
        "FWI weather (15%) + trend (10%)"
    )
    log.info("%s", "═" * 82)
    log.info(
        "%-24s %-6s %-6s %-6s %-6s %7s  tier   top driver",
        "Nadleśnictwo",
        "NDWI",
        "QTE",
        "FWI",
        "brg",
        "SCORE",
    )
    log.info("%s", "─" * 82)

    tier_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
    for s in scores:
        bflag = "🔥" if s.bridge_fired else "  "
        icon = tier_colors[s.tier]
        driver = s.top_drivers[0]["factor"] if s.top_drivers else ""
        log.info(
            "%-24s %6.3f %6.3f %6.3f %6s %7.3f  %s %s  %s",
            s.node_name[:24],
            s.ndwi_stress,
            s.qte_score,
            s.fwi_score,
            bflag,
            s.final_score,
            icon,
            s.tier[:4],
            driver,
        )

    log.info("%s", "═" * 82)
    log.info("Tier counts: %s", "  ".join(f"{k}:{v}" for k, v in tier_counts.items()))
    bridge_n = sum(1 for s in scores if s.bridge_fired)
    log.info("Bridge fired (sequential dry→wind): %d/%d nodes", bridge_n, len(scores))
    clipped_n = sum(1 for score in scores if score.unclipped_score > 1.0)
    high_n = sum(1 for score in scores if score.tier in {"CRITICAL", "HIGH"})
    log.info("Scores clipped to 1.0: %d/%d", clipped_n, len(scores))
    log.info("HIGH/CRITICAL share: %d/%d", high_n, len(scores))


def _save_results(scores: list[RiskScore]) -> None:
    # Full results
    risk_path = os.path.join(OUTPUT_DIR, "risk_scores.json")
    total = len(scores)
    clipped_n = sum(score.unclipped_score > 1.0 for score in scores)
    elevated_n = sum(score.tier in {"CRITICAL", "HIGH"} for score in scores)
    bridge_n = sum(score.bridge_fired for score in scores)
    calibration_warnings: list[str] = []
    if total and clipped_n / total > 0.10:
        calibration_warnings.append("score_clipping_exceeds_10_percent")
    if total and elevated_n / total > 0.50:
        calibration_warnings.append("high_or_critical_exceeds_50_percent")
    if total and bridge_n / total > 0.80:
        calibration_warnings.append("bridge_activation_exceeds_80_percent")
    for score in scores:
        score.calibration_flags.extend(
            flag for flag in calibration_warnings if flag not in score.calibration_flags
        )
    pipeline_status = (
        StepStatus.DEGRADED
        if calibration_warnings
        or any(score.pipeline_status == StepStatus.DEGRADED for score in scores)
        else StepStatus.SUCCESS
    )
    payload = {
        "version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rdlp": "Wrocław",
        "pipeline_status": str(pipeline_status),
        "operational_validity": False,
        "calibration_status": "uncalibrated",
        "calibration_warnings": calibration_warnings,
        "saturation_diagnostics": {
            "clipped_to_one_count": clipped_n,
            "clipped_to_one_share": round(clipped_n / total, 4) if total else 0.0,
            "high_or_critical_count": elevated_n,
            "high_or_critical_share": (round(elevated_n / total, 4) if total else 0.0),
            "bridge_bonus_count": bridge_n,
            "bridge_bonus_share": round(bridge_n / total, 4) if total else 0.0,
            "component_distribution": {
                "base_score": [score.base_score for score in scores],
                "ecosystem_multiplier": [score.eco_multiplier for score in scores],
                "network_propagation": [
                    round(NETWORK_COEFF * score.network_stress, 4) for score in scores
                ],
            },
        },
        "fusion_weights": {
            "ndwi_satellite": W_NDWI,
            "qte_quantum": W_QTE,
            "fwi_weather": W_FWI,
            "ndwi_trend": W_TREND,
        },
        "alert_thresholds": {
            "CRITICAL": ALERT_CRITICAL,
            "HIGH": ALERT_HIGH,
            "MODERATE": ALERT_MODERATE,
        },
        "scores": [
            {
                "node_id": s.node_id,
                "node_name": s.node_name,
                "lat": s.lat,
                "lon": s.lon,
                "eco": s.eco,
                "unclipped_score": s.unclipped_score,
                "final_score": s.final_score,
                "tier": s.tier,
                "pipeline_status": s.pipeline_status,
                "operational_validity": s.operational_validity,
                "data_quality_flags": s.data_quality_flags,
                "calibration_flags": s.calibration_flags,
                "upstream_statuses": s.upstream_statuses,
                "signals": {
                    "ndwi_stress": s.ndwi_stress,
                    "ndwi_trend_14d": s.ndwi_trend_14d,
                    "ndwi_latest": s.ndwi_latest,
                    "qte_score": s.qte_score,
                    "bridge_fired": s.bridge_fired,
                    "fwi_score": s.fwi_score,
                    "network_stress": s.network_stress,
                    "ignition_score": s.ignition_score,
                    "ignition_coverage_percent": s.ignition_coverage_percent,
                    "ignition_valid_for_fusion": s.ignition_valid_for_fusion,
                    "fei": s.fei,
                    "qies": s.qies,
                },
                "modifiers": {
                    "eco_multiplier": s.eco_multiplier,
                    "base_score": s.base_score,
                    "bridge_bonus": BRIDGE_BONUS if s.bridge_fired else 0.0,
                    "network_propagation": round(
                        NETWORK_COEFF * s.network_stress,
                        4,
                    ),
                },
                "context": {
                    "drought_days": s.drought_days,
                    "temp_max": s.temp_max,
                    "wind_max": s.wind_max,
                    "rh_min": s.rh_min,
                },
                "explanation": s.top_drivers,
            }
            for s in scores
        ],
    }
    validate_node_payload(payload, "scores")
    atomic_write_json(risk_path, payload)
    log.info("Saved %s", risk_path)

    # Alerts only (CRITICAL + HIGH)
    alerts = [s for s in scores if s.tier in ("CRITICAL", "HIGH")]
    alerts_path = os.path.join(OUTPUT_DIR, "alerts.json")
    alerts_payload = {
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alert_count": len(alerts),
        "alerts": [
            {
                "priority": 1 if s.tier == "CRITICAL" else 2,
                "tier": s.tier,
                "node_id": s.node_id,
                "node_name": s.node_name,
                "lat": s.lat,
                "lon": s.lon,
                "score": s.final_score,
                "eco": s.eco,
                "bridge_fired": s.bridge_fired,
                "ndwi_latest": s.ndwi_latest,
                "drought_days": s.drought_days,
                "reason": " | ".join(d["label"] for d in s.top_drivers[:2]),
                "pipeline_status": s.pipeline_status,
                "operational_validity": s.operational_validity,
                "data_quality_flags": s.data_quality_flags,
                "calibration_flags": s.calibration_flags,
                "drone_recommended": False,
                "recommendation_status": "research_only_uncalibrated",
            }
            for s in alerts
        ],
    }
    atomic_write_json(alerts_path, alerts_payload)
    log.info("Saved %s (%d alerts)", alerts_path, len(alerts))
    write_step_manifest(
        OUTPUT_DIR,
        "fusion",
        pipeline_status,
        outputs={"risk_scores": risk_path, "alerts": alerts_path},
        coverage_percent=100.0,
        valid_for_downstream=True,
        warnings=calibration_warnings,
        operational_validity=False,
        clipped_scores=clipped_n,
        high_or_critical_share=(round(elevated_n / total, 4) if total else 0.0),
        bridge_share=round(bridge_n / total, 4) if total else 0.0,
        calibration_status="uncalibrated",
        data_version="fusion-2.0.0",
    )


def _generate_final_map(scores: list[RiskScore], graph: dict) -> None:
    node_js = []
    for s in scores:
        node_js.append(
            {
                "id": s.node_id,
                "name": s.node_name,
                "lat": s.lat,
                "lon": s.lon,
                "eco": s.eco,
                "score": s.final_score,
                "tier": s.tier,
                "ndwi": s.ndwi_stress,
                "ndwi_val": s.ndwi_latest or 0,
                "qte": s.qte_score,
                "fwi": s.fwi_score,
                "bridge": s.bridge_fired,
                "drought": s.drought_days,
                "temp": s.temp_max,
                "wind": s.wind_max,
                "rh": s.rh_min,
                "drivers": s.top_drivers,
                "neighbors": [nb["id"] for nb in graph.get(s.node_id, [])],
            }
        )

    node_data = json.dumps(node_js, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <title>QHDALabs — Wildfire Risk RDLP Wrocław</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html,body,#map{{ margin:0; height:100vh; background:#050d05; font-family:'Courier New',monospace; }}
    #panel{{
      position:absolute; top:12px; left:12px; z-index:1000;
      background:rgba(3,10,3,0.95); color:#7fff7f; padding:16px 20px;
      border:1px solid #1a4a1a; border-radius:4px; font-size:11px;
      max-width:320px; box-shadow:0 0 30px rgba(0,255,0,0.07);
    }}
    #panel h2{{ margin:0 0 3px; font-size:13px; color:#afffaf; letter-spacing:2px; text-transform:uppercase; }}
    .sub{{ color:#3a8a3a; font-size:10px; margin-bottom:12px; }}
    .row{{ display:flex; align-items:center; gap:7px; margin:4px 0; }}
    .dot{{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
    .sig{{ margin-top:10px; padding-top:8px; border-top:1px solid #1a4a1a; color:#3a8a3a; font-size:10px; line-height:1.7; }}
    table.popup{{ border-collapse:collapse; width:100%; font-size:11px; margin-top:6px; }}
    table.popup td{{ padding:2px 5px 2px 0; }}
    table.popup td:last-child{{ font-weight:bold; text-align:right; }}
  </style>
</head>
<body>
<div id="panel">
  <h2>🌲 Wildfire Risk</h2>
  <div class="sub">RDLP Wrocław · Sentinel-2 + QTE · {datetime.now(timezone.utc).strftime("%Y-%m-%d")}</div>
  <div class="row"><div class="dot" style="background:#ff0000"></div>🔴 CRITICAL (&gt;{ALERT_CRITICAL})</div>
  <div class="row"><div class="dot" style="background:#ff7700"></div>🟠 HIGH ({ALERT_HIGH}–{ALERT_CRITICAL})</div>
  <div class="row"><div class="dot" style="background:#ffcc00"></div>🟡 MODERATE ({ALERT_MODERATE}–{ALERT_HIGH})</div>
  <div class="row"><div class="dot" style="background:#22cc44"></div>🟢 LOW (&lt;{ALERT_MODERATE})</div>
  <div class="sig">
    Rozmiar = NDWI stress (satelita)<br>
    Kolor = wynik końcowy (fuzja)<br>
    🔥 = most kwantowy aktywny (sucho→wiatr)<br>
    Linie = sieć grzybni (propagacja)
  </div>
</div>
<div id="map"></div>
<script>
const NODES = {node_data};
const map = L.map('map', {{zoomControl: false}});
L.control.zoom({{position: 'bottomright'}}).addTo(map);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'© OpenStreetMap © CARTO',maxZoom:19}}).addTo(map);

const bounds = L.latLngBounds(NODES.map(n => [n.lat, n.lon]));
if (bounds.isValid()) {{
  map.fitBounds(bounds.pad(0.22));
}} else {{
  map.setView([51.0,16.5],8);
}}

function color(s){{
  if(s>{ALERT_CRITICAL}) return '#ff1111';
  if(s>{ALERT_HIGH})     return '#ff7700';
  if(s>{ALERT_MODERATE}) return '#ffcc00';
  return '#33cc55';
}}
function tierLabel(t){{
  return {{CRITICAL:'🔴 CRITICAL',HIGH:'🟠 HIGH',MODERATE:'🟡 MODERATE',LOW:'🟢 LOW'}}[t]||t;
}}

const nodeMap={{}};
NODES.forEach(n=>nodeMap[n.id]=n);

// Network edges
const drawn=new Set();
NODES.forEach(n=>{{
  (n.neighbors||[]).forEach(nbId=>{{
    const key=[n.id,nbId].sort().join('--');
    if(drawn.has(key)) return; drawn.add(key);
    const nb=nodeMap[nbId]; if(!nb) return;
    const s=(n.score+nb.score)/2;
    L.polyline([[n.lat,n.lon],[nb.lat,nb.lon]],{{
      color:color(s), weight:1.3, opacity:0.20+s*0.55
    }}).addTo(map);
  }});
}});

// Nodes
NODES.forEach(n=>{{
  const c = color(n.score);
  const r = 7 + n.ndwi * 16;
  const bf = n.bridge ? ' 🔥' : '';

  // Driver list
  var driverHtml = '';
  if(n.drivers && n.drivers.length>0){{
    driverHtml = '<div style="margin-top:6px;font-size:10px;color:#888"><b>Top drivers:</b><br>'
      + n.drivers.map(function(d){{return d.factor+': +'+d.contribution.toFixed(3);}}).join('<br>')
      + '</div>';
  }}

  const popup = `
    <div style="font-family:'Courier New',monospace;min-width:220px">
    <b style="color:${{c}};font-size:13px">${{n.name}}${{bf}}</b><br>
    <span style="color:#888;font-size:10px">${{n.eco}}</span><br>
    <b style="color:${{c}};font-size:14px">${{tierLabel(n.tier)}} — ${{n.score.toFixed(3)}}</b>
    <table class="popup" style="margin-top:8px">
      <tr><td>🛰 NDWI stres</td><td>${{(n.ndwi*100).toFixed(1)}}%</td></tr>
      <tr><td>🛰 NDWI wartość</td><td>${{n.ndwi_val.toFixed(4)}}</td></tr>
      <tr><td>⚛ QTE score</td><td>${{n.qte.toFixed(3)}}</td></tr>
      <tr><td>🌡 FWI pogoda</td><td>${{n.fwi.toFixed(3)}}</td></tr>
      <tr><td>☀ Temp max</td><td>${{n.temp!==null?n.temp+'°C':'N/A'}}</td></tr>
      <tr><td>💨 Wiatr max</td><td>${{n.wind!==null?n.wind+' m/s':'N/A'}}</td></tr>
      <tr><td>💧 RH min</td><td>${{n.rh!==null?n.rh+'%':'N/A'}}</td></tr>
      <tr><td>🌵 Susza</td><td>${{n.drought}} dni</td></tr>
    </table>
    ${{driverHtml}}
    </div>
  `;

  L.circleMarker([n.lat,n.lon],{{
    radius:      r,
    color:       c,
    fillColor:   c,
    fillOpacity: 0.60 + n.score*0.35,
    weight:      n.bridge ? 3 : 1.2,
    dashArray:   n.tier==='LOW' ? '3,3' : null,
  }}).addTo(map).bindPopup(popup,{{maxWidth:280}});
}});
</script>
</body></html>"""

    map_path = os.path.join(OUTPUT_DIR, "final_map.html")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Saved %s", map_path)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    scores = run_fusion_pipeline()

    # Print alert summary
    alerts = [s for s in scores if s.tier in ("CRITICAL", "HIGH")]
    if alerts:
        log.info("\n%s", "=" * 60)
        log.info("ACTIVE ALERTS — %d nodes require attention", len(alerts))
        log.info("%s", "=" * 60)
        for a in alerts:
            log.info(
                "[%s] %s — RESEARCH ONLY, UNCALIBRATED",
                a.tier,
                a.node_name,
            )
            log.info(
                "  Score: %.3f | NDWI: %.4f | Bridge: %s",
                a.final_score,
                a.ndwi_latest or 0,
                "🔥" if a.bridge_fired else "—",
            )
            if a.top_drivers:
                log.info("  Reason: %s", a.top_drivers[0]["label"])
        log.info("%s", "=" * 60)
    else:
        log.info("No active alerts.")

    log.info("\nOutputs: risk_scores.json  alerts.json  final_map.html")
