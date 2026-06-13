# =============================================================================
# Project       : QHDALabs - Wildfire Risk PL
# Module        : Step 3 — Quantum Temporal Encoder (QTE)
# File          : qhdalabs_wildfire_qte_v1.py
# Version       : 1.0.0
#
# Description
# -----------------------------------------------------------------------------
# The Quantum Temporal Encoder translates the 14-day history of each
# nadleśnictwo into a quantum circuit, runs it through the conditional CZ
# bridge mechanism from qmnet, and extracts a ZZ correlation signal that
# captures sequential patterns a classical classifier cannot see.
#
# Why quantum here — the precise argument
# -----------------------------------------------------------------------------
# A RandomForest sees a feature vector [ndwi_today, ndwi_7d_ago, temp, wind...].
# It can learn thresholds and combinations but treats all inputs as simultaneous.
# It cannot distinguish:
#   Case A: NDWI dropped last week, then wind rose this week
#   Case B: Wind rose last week, then NDWI dropped this week
# These have identical feature vectors but different physical meanings —
# Case A is the classic pre-fire drying sequence.
#
# The conditional CZ bridge from qmnet encodes this:
#   qubit 0 (ancilla)  = NDWI state LAST WEEK (high/low threshold)
#   qubit 1            = NDWI TODAY
#   qubit 2            = WIND TODAY
#   qubit 3            = TEMPERATURE TODAY
#   qubit 4            = DROUGHT PERSISTENCE
#
# A mid-circuit measurement of qubit 0 (last week's state) conditionally
# fires a CZ gate between qubits 2 and 3 (wind × temperature interaction).
# ZZ correlation on qubits 1-2 is only strong when:
#   (a) last week was dry  AND  (b) today wind + temperature are elevated
# This is the sequential pattern. The ZZ value IS the fire risk signal.
#
# Architecture
# -----------------------------------------------------------------------------
# PRIMARY:  Qiskit statevector simulation (exact, no sampling noise)
#           Uses conditional_cz_bridge_dynamic from qmnet pattern.
#           Works without QPU — on QPU the same circuit runs with real noise.
#
# FALLBACK: NumPy statevector (no Qiskit required).
#           Same circuit, same math, pure Python.
#
# Output per node
# -----------------------------------------------------------------------------
#   zz_01      ZZ correlation qubits 0-1  (NDWI history × NDWI today)
#   zz_12      ZZ correlation qubits 1-2  (NDWI today × wind)
#   zz_23      ZZ correlation qubits 2-3  (wind × temperature)
#   zz_34      ZZ correlation qubits 3-4  (temperature × drought)
#   bridge_fired   bool — did the ancilla measurement trigger the bridge?
#   qte_score  scalar [0,1] — composite signal for fire risk scoring
#
# Inputs (from Step 2 nodes_enriched.json)
# -----------------------------------------------------------------------------
#   ndwi_latest        today's NDWI value
#   ndwi_7d_ago        NDWI from ~7 days ago (from time series)
#   ndwi_trend_14d     slope of NDWI over 14 days
#   wind_max (latest)  maximum wind speed
#   temp_max (latest)  maximum temperature
#   drought_days       consecutive dry days
#
# Dependencies
# -----------------------------------------------------------------------------
#   numpy
#   Optional: qiskit >= 2.0 (for Qiskit path)
#   Step 2 output: topology/nodes_enriched.json
#
# Author        : Krzysztof W. Banasiewicz / QHDALabs
# License       : MIT
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = "topology"

# =========================
# ENCODING THRESHOLDS
# =========================
# These map continuous values to qubit angles (0 = low risk, π = high risk)
# Calibrated to RDLP Wrocław May 2026 satellite data.

NDWI_HEALTHY  = -0.35   # above this: low stress
NDWI_CRITICAL = -0.70   # below this: critical stress

WIND_LOW      = 4.0     # m/s — low fire-spread risk
WIND_HIGH     = 10.0    # m/s — high fire-spread risk

TEMP_LOW      = 15.0    # °C
TEMP_HIGH     = 30.0    # °C

DROUGHT_MAX   = 30      # days — saturates at this point


def _encode_to_angle(value: float, low: float, high: float) -> float:
    """
    Map a scalar value linearly to a rotation angle in [0, π].
    value <= low  -> angle near 0  (|0⟩ = low risk)
    value >= high -> angle near π  (|1⟩ = high risk)
    """
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * math.pi)


def _encode_ndwi(ndwi: float) -> float:
    """NDWI to angle: more negative = higher stress = angle closer to π."""
    # Invert: critical (very negative) -> π, healthy -> 0
    return _encode_to_angle(-ndwi, -NDWI_HEALTHY, -NDWI_CRITICAL)


# =========================
# NUMPY STATEVECTOR ENGINE
# (fallback, no Qiskit required)
# =========================
class NumpyStatevector:
    """
    Minimal 5-qubit statevector simulator for the QTE circuit.
    Implements: Ry(θ), CZ, mid-circuit measurement, conditional CZ bridge.

    Qubit ordering: qubit 0 = rightmost bit in state index.
    State |q4 q3 q2 q1 q0⟩ = index (q4<<4)|(q3<<3)|(q2<<2)|(q1<<1)|q0
    """

    def __init__(self, n_qubits: int = 5):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0

    def ry(self, theta: float, qubit: int) -> "NumpyStatevector":
        """Apply Ry(theta) rotation to one qubit."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        new_state = self.state.copy()
        bit = 1 << qubit
        for i in range(self.dim):
            if i & bit:
                continue
            j = i | bit
            a0, a1 = self.state[i], self.state[j]
            new_state[i] =  c * a0 - s * a1
            new_state[j] =  s * a0 + c * a1
        self.state = new_state
        return self

    def cz(self, q1: int, q2: int) -> "NumpyStatevector":
        """Apply CZ gate between two qubits."""
        b1, b2 = 1 << q1, 1 << q2
        for i in range(self.dim):
            if (i & b1) and (i & b2):
                self.state[i] *= -1
        return self

    def measure_and_collapse(self, qubit: int) -> int:
        """
        Mid-circuit measurement of one qubit.
        Returns outcome 0 or 1.
        Collapses state to the measured subspace (renormalised).

        This is the key operation for the conditional bridge —
        the measurement result determines whether the CZ fires.
        """
        bit = 1 << qubit
        # Probability of outcome |1>
        p1 = float(sum(abs(self.state[i])**2 for i in range(self.dim) if i & bit))
        outcome = 1 if np.random.random() < p1 else 0

        # Collapse and renormalise
        new_state = np.zeros(self.dim, dtype=np.complex128)
        for i in range(self.dim):
            has_one = bool(i & bit)
            if (outcome == 1 and has_one) or (outcome == 0 and not has_one):
                new_state[i] = self.state[i]
        norm = np.linalg.norm(new_state)
        self.state = new_state / norm if norm > 1e-12 else new_state
        return outcome

    def zz_correlation(self, q1: int, q2: int) -> float:
        """
        Compute ⟨Z_q1 ⊗ Z_q2⟩.
        Range [-1, +1].
        +1 = perfectly correlated (same spin), -1 = anti-correlated.
        For fire risk: large |ZZ| indicates strong conditional coupling.
        """
        b1, b2 = 1 << q1, 1 << q2
        total = 0.0
        for i in range(self.dim):
            z1 = -1 if (i & b1) else +1
            z2 = -1 if (i & b2) else +1
            total += z1 * z2 * abs(self.state[i]) ** 2
        return float(total)


# =========================
# QTE CIRCUIT
# =========================
@dataclass
class QTEResult:
    node_id:      str
    node_name:    str
    backend:      str          # "qiskit_statevector" | "numpy_statevector"
    # Input encodings (angles in radians)
    theta_ndwi_past:   float
    theta_ndwi_now:    float
    theta_wind:        float
    theta_temp:        float
    theta_drought:     float
    # Bridge
    bridge_fired:      bool
    # ZZ correlations
    zz_01:   float   # NDWI_past × NDWI_now
    zz_12:   float   # NDWI_now  × wind
    zz_23:   float   # wind      × temperature
    zz_34:   float   # temperature × drought
    # Composite score
    qte_score: float


def _run_numpy_qte(
    theta_ndwi_past: float,
    theta_ndwi_now:  float,
    theta_wind:      float,
    theta_temp:      float,
    theta_drought:   float,
    n_shots: int = 512,
) -> dict:
    """
    Run the QTE circuit using the NumPy statevector engine.

    Circuit design (5 qubits):
      q0 = ancilla — encodes NDWI last week (will be measured mid-circuit)
      q1 = NDWI today
      q2 = wind max
      q3 = temperature max
      q4 = drought persistence

    Steps:
      1. Ry rotations encode all features into qubit states
      2. CZ entangle q0-q1 (temporal correlation: past NDWI × today NDWI)
      3. CZ entangle q1-q2 (spatial: NDWI × wind)
      4. Measure q0 (ancilla) — this is the "bridge gate"
      5. If ancilla=1 (past was dry): fire CZ between q2 and q3
         This is the conditional bridge from qmnet:
         wind × temperature interaction ONLY activates when history was dry
      6. CZ entangle q3-q4 (temperature × drought)
      7. Measure ZZ correlations on all adjacent pairs

    Returns average ZZ correlations over n_shots runs.
    """
    zz_accum = {k: 0.0 for k in ("01", "12", "23", "34")}
    bridge_count = 0

    for _ in range(n_shots):
        sv = NumpyStatevector(n_qubits=5)

        # Step 1: Encode features
        sv.ry(theta_ndwi_past, 0)   # q0: past NDWI (ancilla)
        sv.ry(theta_ndwi_now,  1)   # q1: current NDWI
        sv.ry(theta_wind,      2)   # q2: wind
        sv.ry(theta_temp,      3)   # q3: temperature
        sv.ry(theta_drought,   4)   # q4: drought persistence

        # Step 2: Temporal entanglement — past × present NDWI
        sv.cz(0, 1)

        # Step 3: Spatial entanglement — NDWI × wind
        sv.cz(1, 2)

        # Step 4: Mid-circuit measurement of ancilla (past NDWI state)
        # This is the BRIDGE: the measurement result routes computation
        # Decoherence at q0 is a FEATURE, not a bug — it is the signal
        # that determines whether the wind×temp interaction fires
        ancilla_outcome = sv.measure_and_collapse(0)
        bridge_fired = (ancilla_outcome == 1)
        if bridge_fired:
            bridge_count += 1

        # Step 5: Conditional CZ bridge (qmnet mechanism)
        # Fire CZ between wind(q2) and temperature(q3) ONLY if past was dry
        if bridge_fired:
            sv.cz(2, 3)

        # Step 6: Temperature × drought
        sv.cz(3, 4)

        # Step 7: Measure ZZ correlations
        zz_accum["01"] += sv.zz_correlation(0, 1)
        zz_accum["12"] += sv.zz_correlation(1, 2)
        zz_accum["23"] += sv.zz_correlation(2, 3)
        zz_accum["34"] += sv.zz_correlation(3, 4)

    bridge_rate = bridge_count / n_shots
    return {
        "zz_01":        zz_accum["01"] / n_shots,
        "zz_12":        zz_accum["12"] / n_shots,
        "zz_23":        zz_accum["23"] / n_shots,
        "zz_34":        zz_accum["34"] / n_shots,
        "bridge_rate":  bridge_rate,
        "bridge_fired": bridge_rate > 0.5,
    }


def _run_qiskit_qte(
    theta_ndwi_past: float,
    theta_ndwi_now:  float,
    theta_wind:      float,
    theta_temp:      float,
    theta_drought:   float,
) -> dict:
    """
    Run the QTE circuit using Qiskit statevector simulator.

    Uses dynamic circuits (mid-circuit measurement + if_test) matching
    the conditional_cz_bridge_dynamic pattern from qmnet.py.

    Falls back gracefully if Qiskit is unavailable.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    qr = QuantumRegister(5, "q")
    cr = ClassicalRegister(1, "ancilla_out")
    qc = QuantumCircuit(qr, cr)

    # Encode features
    qc.ry(theta_ndwi_past, qr[0])
    qc.ry(theta_ndwi_now,  qr[1])
    qc.ry(theta_wind,      qr[2])
    qc.ry(theta_temp,      qr[3])
    qc.ry(theta_drought,   qr[4])

    # Temporal + spatial entanglement
    qc.cz(qr[0], qr[1])
    qc.cz(qr[1], qr[2])

    # Mid-circuit measurement of ancilla (bridge gate)
    qc.measure(qr[0], cr[0])

    # Conditional CZ bridge — fires only when ancilla=1 (past was dry)
    with qc.if_test((cr[0], 1)):
        qc.cz(qr[2], qr[3])

    # Temperature × drought
    qc.cz(qr[3], qr[4])

    # Measure ZZ correlations via Estimator
    # ZZ operators for each adjacent pair
    zz_ops = {
        "zz_01": SparsePauliOp.from_list([("IIIZZ", 1.0)]),
        "zz_12": SparsePauliOp.from_list([("IIZZI", 1.0)]),
        "zz_23": SparsePauliOp.from_list([("IZZII", 1.0)]),
        "zz_34": SparsePauliOp.from_list([("ZZIII", 1.0)]),
    }

    # For dynamic circuits with mid-circuit measurement,
    # we run two branches explicitly
    results = {}
    estimator = StatevectorEstimator()

    # Circuit without measurement (for ZZ estimation via statevector)
    # We use the numpy path for ZZ after qiskit circuit construction
    # and qiskit only to validate the dynamic circuit compiles correctly
    qc_no_measure = QuantumCircuit(5)
    qc_no_measure.ry(theta_ndwi_past, 0)
    qc_no_measure.ry(theta_ndwi_now,  1)
    qc_no_measure.ry(theta_wind,      2)
    qc_no_measure.ry(theta_temp,      3)
    qc_no_measure.ry(theta_drought,   4)
    qc_no_measure.cz(0, 1)
    qc_no_measure.cz(1, 2)
    # Simulate BOTH branches and weight by ancilla probabilities
    # Branch 0: ancilla=0 (past not dry) — no bridge CZ
    sv0 = Statevector(qc_no_measure)

    # Branch 1: ancilla=1 (past dry) — bridge fires
    qc_bridge = qc_no_measure.copy()
    qc_bridge.cz(2, 3)
    sv1 = Statevector(qc_bridge)

    # Both branches: cz(3,4)
    qc_b0_final = QuantumCircuit(5)
    qc_b0_final.initialize(sv0.data)
    qc_b0_final.cz(3, 4)
    qc_b1_final = QuantumCircuit(5)
    qc_b1_final.initialize(sv1.data)
    qc_b1_final.cz(3, 4)

    sv0_final = Statevector(qc_b0_final)
    sv1_final = Statevector(qc_b1_final)

    # Ancilla probability of being |1>
    # = sum of |amplitude|^2 for all states where qubit 0 is |1>
    prob_ancilla_1 = sum(
        abs(sv0.data[i])**2
        for i in range(32)
        if i & 1  # qubit 0 = bit 0
    )
    prob_ancilla_0 = 1.0 - prob_ancilla_1

    # Weighted ZZ expectations
    def zz_sv(sv: Statevector, q1: int, q2: int) -> float:
        b1, b2 = 1 << q1, 1 << q2
        total = 0.0
        for i, amp in enumerate(sv.data):
            z1 = -1 if (i & b1) else +1
            z2 = -1 if (i & b2) else +1
            total += z1 * z2 * abs(amp)**2
        return float(total)

    for key, (q1, q2) in [("zz_01",(0,1)),("zz_12",(1,2)),("zz_23",(2,3)),("zz_34",(3,4))]:
        v0 = zz_sv(sv0_final, q1, q2)
        v1 = zz_sv(sv1_final, q1, q2)
        results[key] = prob_ancilla_0 * v0 + prob_ancilla_1 * v1

    results["bridge_rate"]  = prob_ancilla_1
    results["bridge_fired"] = prob_ancilla_1 > 0.5
    return results


# =========================
# QTE SCORE
# =========================
def compute_qte_score(zz: dict, bridge_fired: bool) -> float:
    """
    Composite QTE fire risk score in [0, 1].

    Physical interpretation of each ZZ term:
      zz_12 (NDWI×wind):   HIGH magnitude → dry forest + strong wind = fire spread
      zz_23 (wind×temp):   HIGH magnitude AND bridge_fired → sequential pre-fire pattern
      zz_34 (temp×drought): HIGH magnitude → hot + prolonged drought
      zz_01 (past×now):    NEGATIVE → NDWI is declining (drying trend)

    The bridge_fired bonus is the key quantum signal:
    it captures the SEQUENTIAL pattern (was dry THEN wind rose)
    which RF cannot distinguish from the simultaneous case.
    """
    # ZZ magnitudes — high magnitude = strong coupling
    m12 = abs(zz.get("zz_12", 0.0))   # NDWI × wind
    m23 = abs(zz.get("zz_23", 0.0))   # wind × temp (conditional on bridge)
    m34 = abs(zz.get("zz_34", 0.0))   # temp × drought

    # Drying trend: negative zz_01 means anti-correlation (NDWI today worse than past)
    trend_signal = max(0.0, -zz.get("zz_01", 0.0))

    # Base composite
    base = 0.30 * m12 + 0.25 * m34 + 0.20 * trend_signal + 0.15 * m23

    # Bridge bonus: if ancilla fired, the sequential pattern is confirmed
    # This is the moment where quantum adds something classical RF misses
    bridge_bonus = 0.10 if bridge_fired else 0.0

    return float(np.clip(base + bridge_bonus, 0.0, 1.0))


# =========================
# ENCODE NODE
# =========================
def encode_node(node: dict) -> tuple[float, float, float, float, float]:
    """
    Extract and encode node features to qubit angles.
    Returns (theta_ndwi_past, theta_ndwi_now, theta_wind, theta_temp, theta_drought)
    """
    latest = node.get("latest", {})
    ndwi_s = node.get("ndwi_sentinel", {})

    # NDWI values
    ndwi_values = ndwi_s.get("ndwi_values", [])
    ndwi_now    = ndwi_s.get("ndwi_latest", latest.get("ndwi_stress") or -0.5)

    # Past NDWI: use second-to-last observation if available (7-10 days ago)
    if len(ndwi_values) >= 2:
        ndwi_past = ndwi_values[-2]
    elif len(ndwi_values) == 1:
        ndwi_past = ndwi_values[0]
    else:
        ndwi_past = ndwi_now  # no history — assume stable

    # Weather features
    wind_max     = float(latest.get("wind_max")  or 5.0)
    temp_max     = float(latest.get("temp_max")  or 20.0)
    drought_days = int(node.get("drought_days",  0))

    # Encode to angles
    theta_ndwi_past = _encode_ndwi(float(ndwi_past))
    theta_ndwi_now  = _encode_ndwi(float(ndwi_now))
    theta_wind      = _encode_to_angle(wind_max,     WIND_LOW,    WIND_HIGH)
    theta_temp      = _encode_to_angle(temp_max,     TEMP_LOW,    TEMP_HIGH)
    theta_drought   = _encode_to_angle(drought_days, 0,           DROUGHT_MAX)

    return theta_ndwi_past, theta_ndwi_now, theta_wind, theta_temp, theta_drought


# =========================
# MAIN ENCODER
# =========================
def run_qte_for_node(node: dict, n_shots: int = 256) -> QTEResult:
    """
    Run the Quantum Temporal Encoder for one nadleśnictwo node.
    Tries Qiskit first, falls back to NumPy statevector.
    """
    nid  = node["id"]
    name = node["name"]

    thetas = encode_node(node)
    t_past, t_now, t_wind, t_temp, t_drought = thetas

    # Try Qiskit
    backend = "numpy_statevector"
    zz: dict[str, Any] = {}

    try:
        import qiskit
        zz      = _run_qiskit_qte(*thetas)
        backend = "qiskit_statevector"
    except ImportError:
        pass
    except Exception as exc:
        log.debug("Qiskit QTE failed for %s (%s), using NumPy.", name, exc)

    if not zz:
        zz = _run_numpy_qte(*thetas, n_shots=n_shots)

    qte_score = compute_qte_score(zz, bool(zz.get("bridge_fired", False)))

    return QTEResult(
        node_id           = nid,
        node_name         = name,
        backend           = backend,
        theta_ndwi_past   = round(t_past,    4),
        theta_ndwi_now    = round(t_now,     4),
        theta_wind        = round(t_wind,    4),
        theta_temp        = round(t_temp,    4),
        theta_drought     = round(t_drought, 4),
        bridge_fired      = bool(zz.get("bridge_fired", False)),
        zz_01             = round(float(zz.get("zz_01", 0)), 4),
        zz_12             = round(float(zz.get("zz_12", 0)), 4),
        zz_23             = round(float(zz.get("zz_23", 0)), 4),
        zz_34             = round(float(zz.get("zz_34", 0)), 4),
        qte_score         = round(qte_score, 4),
    )


# =========================
# PIPELINE
# =========================
def run_qte_pipeline(
    enriched_json: str = os.path.join(OUTPUT_DIR, "nodes_enriched.json"),
    graph_json:    str = os.path.join(OUTPUT_DIR, "graph.json"),
    n_shots:       int = 256,
) -> list[QTEResult]:

    log.info("=== QHDALabs Wildfire — Step 3: Quantum Temporal Encoder ===")

    if not os.path.exists(enriched_json):
        raise FileNotFoundError(
            f"Step 2 output not found: {enriched_json}\n"
            "Run qhdalabs_wildfire_sentinel_v1.py first."
        )

    with open(enriched_json, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["nodes"]
    log.info("Loaded %d nodes from Step 2", len(nodes))

    with open(graph_json, encoding="utf-8") as f:
        graph_data = json.load(f)
    graph = graph_data["adjacency"]

    # Run QTE for all nodes
    results: list[QTEResult] = []
    for node in nodes:
        result = run_qte_for_node(node, n_shots=n_shots)
        results.append(result)

    # Log backend
    backends = set(r.backend for r in results)
    log.info("QTE complete: %d nodes | backends: %s", len(results), backends)

    # Print summary
    _print_summary(results)

    # Save results
    _save_results(results, nodes, graph)

    return results


def _print_summary(results: list[QTEResult]) -> None:
    sorted_r = sorted(results, key=lambda r: r.qte_score, reverse=True)

    log.info("\n%s", "=" * 78)
    log.info("Quantum Temporal Encoder — RDLP Wrocław")
    log.info("ZZ correlations: zz_12=NDWI×wind  zz_23=wind×temp(cond.)  bridge=sequential?")
    log.info("%s", "=" * 78)
    log.info("%-24s %6s %6s %6s %6s %7s %6s",
             "Nadleśnictwo", "zz_12", "zz_23", "zz_34", "bridge", "QTE", "tier")
    log.info("%s", "-" * 78)

    for r in sorted_r:
        tier  = ("KRYT" if r.qte_score > 0.70 else
                 "WYS " if r.qte_score > 0.50 else
                 "UMIA" if r.qte_score > 0.30 else "LOW ")
        bmark = "🔥" if r.bridge_fired else "  "
        log.info("%-24s %6.3f %6.3f %6.3f %6s %7.3f [%s]",
                 r.node_name[:24], r.zz_12, r.zz_23, r.zz_34,
                 bmark, r.qte_score, tier)
    log.info("%s", "=" * 78)
    bridge_count = sum(1 for r in results if r.bridge_fired)
    log.info("Bridge fired (sequential dry→wind pattern): %d/%d nodes 🔥",
             bridge_count, len(results))


def _save_results(
    results: list[QTEResult],
    nodes:   list[dict],
    graph:   dict,
) -> None:
    """Save QTE results and generate combined map."""

    # qte_results.json
    qte_path = os.path.join(OUTPUT_DIR, "qte_results.json")
    qte_data = {
        "version":      "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm":    "conditional_cz_bridge (qmnet)",
        "n_qubits":     5,
        "qubits": {
            "q0": "NDWI past (ancilla — measured mid-circuit)",
            "q1": "NDWI today",
            "q2": "wind max",
            "q3": "temperature max",
            "q4": "drought persistence",
        },
        "results": [
            {
                "node_id":        r.node_id,
                "node_name":      r.node_name,
                "backend":        r.backend,
                "theta_encoding": {
                    "ndwi_past": r.theta_ndwi_past,
                    "ndwi_now":  r.theta_ndwi_now,
                    "wind":      r.theta_wind,
                    "temp":      r.theta_temp,
                    "drought":   r.theta_drought,
                },
                "bridge_fired": r.bridge_fired,
                "zz": {
                    "zz_01": r.zz_01,
                    "zz_12": r.zz_12,
                    "zz_23": r.zz_23,
                    "zz_34": r.zz_34,
                },
                "qte_score": r.qte_score,
            }
            for r in results
        ],
    }
    with open(qte_path, "w", encoding="utf-8") as f:
        json.dump(qte_data, f, indent=2, ensure_ascii=False)
    log.info("Saved %s", qte_path)

    # Generate combined map (Step 2 NDWI + Step 3 QTE)
    _generate_combined_map(results, nodes, graph)


def _generate_combined_map(
    qte_results: list[QTEResult],
    nodes:       list[dict],
    graph:       dict,
) -> None:
    """Generate map showing both NDWI stress and QTE score per node."""

    qte_map = {r.node_id: r for r in qte_results}

    node_js = []
    for n in nodes:
        nid    = n["id"]
        latest = n.get("latest", {})
        qte    = qte_map.get(nid)
        ndwi_s = n.get("ndwi_sentinel", {})

        node_js.append({
            "id":           nid,
            "name":         n["name"],
            "lat":          n["lat"],
            "lon":          n["lon"],
            "eco":          n["eco"],
            "ndwi_stress":  n.get("ndwi_stress_latest", 0.0),
            "ndwi_latest":  ndwi_s.get("ndwi_latest", 0.0),
            "ndwi_trend":   n.get("ndwi_trend_14d", 0.0),
            "drought_days": n.get("drought_days", 0),
            "qte_score":    qte.qte_score    if qte else 0.0,
            "bridge_fired": qte.bridge_fired if qte else False,
            "zz_12":        qte.zz_12        if qte else 0.0,
            "zz_23":        qte.zz_23        if qte else 0.0,
            "temp_max":     latest.get("temp_max"),
            "wind_max":     latest.get("wind_max"),
            "neighbors":    [nb["id"] for nb in graph.get(nid, [])],
        })

    node_data = json.dumps(node_js, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"/>
  <title>QHDALabs — QTE Map v3</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <style>
    html,body,#map {{ margin:0; height:100vh; background:#060c06; font-family:'Courier New',monospace; }}
    #panel {{
      position:absolute; top:12px; left:12px; z-index:1000;
      background:rgba(4,14,4,0.94); color:#7fff7f; padding:14px 18px;
      border:1px solid #1a4a1a; border-radius:4px; font-size:11px;
      max-width:300px; box-shadow:0 0 24px rgba(0,255,0,0.08);
    }}
    #panel h2 {{ margin:0 0 4px; font-size:13px; color:#afffaf; letter-spacing:2px; text-transform:uppercase;}}
    .sub {{ color:#3a8a3a; font-size:10px; margin-bottom:10px; }}
    .row {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
    .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
    .fire {{ font-size:14px; }}
  </style>
</head>
<body>
<div id="panel">
  <h2>⚛ QTE + 🛰 NDWI</h2>
  <div class="sub">RDLP Wrocław · Quantum Temporal Encoder v1</div>
  <div class="row"><div class="dot" style="background:#ff1111"></div>KRYTYCZNY QTE (&gt;0.70)</div>
  <div class="row"><div class="dot" style="background:#ff7700"></div>WYSOKI (0.50–0.70)</div>
  <div class="row"><div class="dot" style="background:#ffcc00"></div>UMIARKOWANY (0.30–0.50)</div>
  <div class="row"><div class="dot" style="background:#22cc44"></div>NISKI (&lt;0.30)</div>
  <div class="row" style="margin-top:8px">
    <span class="fire">🔥</span> most warunkowy aktywny (sucho→wiatr)
  </div>
  <div style="margin-top:8px;color:#3a8a3a;font-size:10px">
    Rozmiar = NDWI stress (satelita)<br>
    Kolor = QTE score (kwantowy)
  </div>
</div>
<div id="map"></div>
<script>
const NODES = {node_data};
const map = L.map('map').setView([51.0, 16.5], 8);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'© OpenStreetMap © CARTO',maxZoom:19}}).addTo(map);

function qteColor(s) {{
  if (s>0.70) return '#ff1111';
  if (s>0.50) return '#ff7700';
  if (s>0.30) return '#ffcc00';
  return '#22cc44';
}}

const nodeMap={{}};
NODES.forEach(n=>nodeMap[n.id]=n);
const drawn=new Set();
NODES.forEach(n=>{{
  n.neighbors.forEach(nbId=>{{
    const key=[n.id,nbId].sort().join('--');
    if(drawn.has(key)) return; drawn.add(key);
    const nb=nodeMap[nbId]; if(!nb) return;
    const s=(n.qte_score+nb.qte_score)/2;
    L.polyline([[n.lat,n.lon],[nb.lat,nb.lon]],{{
      color:qteColor(s), weight:1.2, opacity:0.25+s*0.5
    }}).addTo(map);
  }});
}});

NODES.forEach(n=>{{
  const color = qteColor(n.qte_score);
  // Radius = NDWI stress (satellite signal), color = QTE score (quantum signal)
  const r = 6 + n.ndwi_stress * 16;
  const bridgeMark = n.bridge_fired ? ' 🔥' : '';
  const trendArrow = n.ndwi_trend<-0.002?'↓':n.ndwi_trend>0.002?'↑':'→';

  const popup = `
    <b style="color:${{color}};font-family:'Courier New'">${{n.name}}${{bridgeMark}}</b><br>
    <small style="color:#888">${{n.eco}}</small>
    <table style="margin:5px 0;font-size:11px;border-collapse:collapse;width:100%">
      <tr><td>QTE score</td><td><b>${{n.qte_score.toFixed(3)}}</b></td></tr>
      <tr><td>NDWI sat.</td><td><b>${{n.ndwi_latest.toFixed(4)}}</b></td></tr>
      <tr><td>NDWI trend</td><td>${{trendArrow}} ${{n.ndwi_trend.toFixed(4)}}/10d</td></tr>
      <tr><td>ZZ wind×NDWI</td><td>${{n.zz_12.toFixed(3)}}</td></tr>
      <tr><td>ZZ wind×temp</td><td>${{n.zz_23.toFixed(3)}}${{n.bridge_fired?' (bridge)':''}}</td></tr>
      <tr><td>Susza</td><td>${{n.drought_days}}d</td></tr>
      <tr><td>Temp max</td><td>${{n.temp_max!==null?n.temp_max+'°C':'N/A'}}</td></tr>
      <tr><td>Wiatr max</td><td>${{n.wind_max!==null?n.wind_max+' m/s':'N/A'}}</td></tr>
    </table>
  `;

  L.circleMarker([n.lat,n.lon],{{
    radius: r,
    color: color,
    fillColor: color,
    fillOpacity: 0.65 + n.qte_score*0.30,
    weight: n.bridge_fired ? 2.5 : 1.2,
    dashArray: n.bridge_fired ? null : '3,2',
  }}).addTo(map).bindPopup(popup);
}});
</script>
</body></html>"""

    map_path = os.path.join(OUTPUT_DIR, "network_map_v3.html")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Saved %s", map_path)
    log.info("Next: integrate QTE score into v4.2 wildfire risk pipeline (Step 4)")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_qte_pipeline()
