"""Forensic-audit regression tests for the v5 pipeline.

These tests pin the behaviour that the 2026-08-23 Jawor audit established.
They document what the implementation actually does; several of them assert
properties that are arguably defects, and are marked as such so that a future
fix fails loudly instead of silently changing alert semantics.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import qhdalabs_wildfire_fusion_v1 as fusion
import qhdalabs_wildfire_qte_v1 as qte
import qhdalabs_wildfire_sentinel_v1 as sentinel

TOPOLOGY = Path(__file__).parents[1] / "topology"


def _load(name: str) -> dict:
    with (TOPOLOGY / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _jawor() -> dict:
    scores = _load("risk_scores.json")["scores"]
    return next(s for s in scores if s["node_id"] == "jawor")


# ---------------------------------------------------------------------------
# 1. NDWI saturation
# ---------------------------------------------------------------------------
def test_stress_mapping_alone_never_saturates_for_realistic_moisture() -> None:
    """(1 - m) / 2 only reaches 1.0 at m = -1, which real canopies never hit."""
    for moisture in (-0.5, -0.0624, 0.0, 0.35):
        assert float(np.clip((1.0 - moisture) / 2.0, 0.0, 1.0)) < 1.0
    assert float(np.clip((1.0 - (-1.0)) / 2.0, 0.0, 1.0)) == 1.0


def test_minmax_normalisation_always_saturates_exactly_one_node() -> None:
    """A stress of 1.000 is a within-network rank, not an absolute extreme."""
    results = {
        "wet": {"ndwi_stress_latest": 0.30},
        "mid": {"ndwi_stress_latest": 0.40},
        "dry": {"ndwi_stress_latest": 0.53},
    }
    normalised = sentinel.normalise_stress_across_network(results)
    assert normalised["dry"] == 1.0
    assert normalised["wet"] == 0.0
    assert sum(value == 1.0 for value in normalised.values()) == 1


def test_normalisation_amplifies_a_narrow_raw_spread() -> None:
    """The observed 0.049 raw gap becomes a 0.236 normalised gap."""
    results = {
        "jawor": {"ndwi_stress_latest": 0.5312},
        "rychtal": {"ndwi_stress_latest": 0.4825},
        "szklarska": {"ndwi_stress_latest": 0.3247},
    }
    normalised = sentinel.normalise_stress_across_network(results)
    raw_gap = 0.5312 - 0.4825
    normalised_gap = normalised["jawor"] - normalised["rychtal"]
    assert raw_gap == pytest.approx(0.0487, abs=1e-4)
    assert normalised_gap > 4.0 * raw_gap


def test_normalisation_is_population_dependent_not_absolute() -> None:
    """Removing the driest node promotes the runner-up to a full 1.000."""
    full = {
        "jawor": {"ndwi_stress_latest": 0.5312},
        "rychtal": {"ndwi_stress_latest": 0.4825},
        "szklarska": {"ndwi_stress_latest": 0.3247},
    }
    reduced = {k: v for k, v in full.items() if k != "jawor"}
    assert sentinel.normalise_stress_across_network(full)["rychtal"] < 1.0
    assert sentinel.normalise_stress_across_network(reduced)["rychtal"] == 1.0


def test_zero_normalised_stress_falls_back_to_raw_absolute_value() -> None:
    """DEFECT: an `or` chain treats a legitimate 0.0 as a missing value."""
    node = {
        "id": "wettest",
        "name": "Wettest",
        "lat": 51.0,
        "lon": 16.0,
        "ndwi_stress_normalised": 0.0,
        "ndwi_stress_latest": 0.3247,
    }
    score = fusion.compute_risk_score(node, {}, {}, {"wettest": node})
    assert score.ndwi_stress == 0.3247, "0.0 should survive, but the or-chain drops it"


# ---------------------------------------------------------------------------
# 2. Cache freshness versus acquisition freshness
# ---------------------------------------------------------------------------
def test_cache_fresh_means_ttl_only_not_recent_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = {"id": "jawor", "name": "Jawor", "lat": 51.041, "lon": 16.197}
    created = datetime.now(timezone.utc) - timedelta(days=8)
    entry = {
        "schema_version": sentinel.SCHEMA_VERSION,
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(days=10)).isoformat(),
        "query_fingerprint": sentinel.sentinel_query_fingerprint(node, 30),
        "cache_status": "fresh",
        # The newest acquisition is 18 days old; the classifier never reads it.
        "result": {"dates": ["2026-07-16", "2026-08-05"], "ndwi_latest": -0.0624},
    }
    (tmp_path / "jawor.json").write_text(json.dumps(entry), encoding="utf-8")
    monkeypatch.setattr(sentinel, "CACHE_DIR", tmp_path)
    status, _ = sentinel._load_cache_entry(node, 30)
    assert status == "fresh", "TTL-based freshness ignores acquisition age"


def test_default_ttl_exceeds_the_sentinel_revisit_interval() -> None:
    """A 10-day TTL can straddle two full ~5-day revisit cycles."""
    assert sentinel.DEFAULT_CACHE_TTL_DAYS == 10
    assert sentinel.DEFAULT_CACHE_TTL_DAYS > 5


def test_published_alert_carries_no_acquisition_timestamp() -> None:
    """DEFECT: an operator cannot tell how old the satellite reading is."""
    alerts = _load("alerts.json")["alerts"]
    if not alerts:
        pytest.skip("no alerts in the current run")
    blob = json.dumps(alerts[0])
    assert "ndwi_latest" in blob
    for field in ("fetch_date", "acquisition", "dates", "n_observations"):
        assert field not in blob


# ---------------------------------------------------------------------------
# 3. Jawor forensic replay
# ---------------------------------------------------------------------------
def test_jawor_score_replays_exactly_from_published_signals() -> None:
    score = _jawor()
    signal = score["signals"]
    trend = fusion._trend_signal(signal["ndwi_trend_14d"])
    base = (
        fusion.W_NDWI * signal["ndwi_stress"]
        + fusion.W_QTE * signal["qte_score"]
        + fusion.W_FWI * signal["fwi_score"]
        + fusion.W_TREND * trend
    )
    total = base * score["modifiers"]["eco_multiplier"]
    total += fusion.BRIDGE_BONUS if signal["bridge_fired"] else 0.0
    total += fusion.NETWORK_COEFF * signal["network_stress"]
    assert base == pytest.approx(score["modifiers"]["base_score"], abs=5e-4)
    assert total == pytest.approx(score["final_score"], abs=5e-4)


def test_jawor_clears_critical_by_less_than_the_bridge_bonus() -> None:
    score = _jawor()
    margin = score["final_score"] - fusion.ALERT_CRITICAL
    assert 0.0 < margin < fusion.BRIDGE_BONUS
    assert score["final_score"] - fusion.BRIDGE_BONUS < fusion.ALERT_CRITICAL


# ---------------------------------------------------------------------------
# 4. QTE determinism and the bridge
# ---------------------------------------------------------------------------
def test_bridge_rate_is_exactly_sin_squared_of_the_past_ndwi_angle() -> None:
    """The Qiskit path is analytic: no shots, and no entanglement enters it."""
    for past in (-0.5, -0.003, 0.0, 0.2, 0.4):
        theta = qte._encode_ndwi(past)
        result = qte._run_qiskit_qte(theta, 1.0, 2.0, 0.8, 0.1)
        expected = math.sin(theta / 2) ** 2
        assert result["bridge_rate"] == pytest.approx(expected, abs=1e-12)


def test_bridge_fires_exactly_when_previous_moisture_is_negative() -> None:
    """The whole quantum bridge reduces to ndwi_values[-2] < 0."""
    for past, expected in ((-0.05, True), (-0.001, True), (0.01, False), (0.05, False)):
        theta = qte._encode_ndwi(past)
        fired = bool(qte._run_qiskit_qte(theta, 1.0, 2.0, 0.8, 0.1)["bridge_fired"])
        assert fired is expected


def test_bridge_decision_is_independent_of_wind() -> None:
    """DEFECT: the label claims a dry-then-wind sequence, but wind never gates it."""
    theta_past = qte._encode_ndwi(-0.003)
    rates = [
        float(
            qte._run_qiskit_qte(
                theta_past, 1.0, qte._encode_to_angle(wind, 4.0, 10.0), 0.8, 0.1
            )["bridge_rate"]
        )
        for wind in (0.0, 5.0, 20.3, 40.0)
    ]
    assert max(rates) - min(rates) < 1e-12


def test_numpy_backend_bridge_is_seed_dependent_near_the_threshold() -> None:
    """DEFECT: at ndwi_past = -0.003 the fallback backend flips the alert tier."""
    angles = (
        qte._encode_ndwi(-0.003),
        qte._encode_ndwi(-0.0624),
        math.pi,
        0.837758,
        0.104720,
    )
    fired = {
        qte._run_numpy_qte(*angles, n_shots=256, random_seed=seed)["bridge_fired"]
        for seed in range(8)
    }
    assert fired == {True, False}, "expected backend disagreement near p = 0.5"


def test_numpy_backend_is_reproducible_for_a_fixed_seed() -> None:
    angles = (1.5755, 1.6688, math.pi, 0.8378, 0.1047)
    first = qte._run_numpy_qte(*angles, n_shots=256, random_seed=20260601)
    second = qte._run_numpy_qte(*angles, n_shots=256, random_seed=20260601)
    assert first == second


# ---------------------------------------------------------------------------
# 5. Bridge threshold edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("past", "expected"),
    # DEFECT: the decision boundary sits exactly at ndwi_past = 0, and floating
    # point puts 0.0 itself on the firing side. A 1e-9 change flips the tier.
    [(-1e-9, True), (0.0, True), (1e-3, False)],
)
def test_bridge_threshold_is_a_hard_sign_test(past: float, expected: bool) -> None:
    theta = qte._encode_ndwi(past)
    rate = float(qte._run_qiskit_qte(theta, 1.0, 2.0, 0.8, 0.1)["bridge_rate"])
    assert (rate > 0.5) is expected


def test_bridge_bonus_alone_spans_more_than_half_a_tier() -> None:
    assert fusion.BRIDGE_BONUS > (fusion.ALERT_CRITICAL - fusion.ALERT_HIGH) / 2


# ---------------------------------------------------------------------------
# 6. Network propagation
# ---------------------------------------------------------------------------
def test_network_propagation_is_single_pass_and_cannot_feed_back() -> None:
    """Neighbour stress reads the sentinel field, never a fusion output."""
    nodes = {
        "a": {
            "id": "a", "name": "A", "lat": 51.0, "lon": 16.0,
            "ndwi_stress_normalised": 1.0,
        },
        "b": {
            "id": "b", "name": "B", "lat": 51.1, "lon": 16.1,
            "ndwi_stress_normalised": 1.0,
        },
    }
    graph = {"a": [{"id": "b"}], "b": [{"id": "a"}]}
    first = fusion.compute_risk_score(nodes["a"], {}, graph, nodes)
    second = fusion.compute_risk_score(nodes["a"], {}, graph, nodes)
    assert first.final_score == second.final_score
    assert first.network_stress == 1.0


def test_network_propagation_is_bounded_by_its_coefficient() -> None:
    nodes = {
        "a": {
            "id": "a", "name": "A", "lat": 51.0, "lon": 16.0,
            "ndwi_stress_normalised": 0.0,
        },
        "b": {
            "id": "b", "name": "B", "lat": 51.1, "lon": 16.1,
            "ndwi_stress_normalised": 1.0,
        },
    }
    score = fusion.compute_risk_score(nodes["a"], {}, {"a": [{"id": "b"}]}, nodes)
    assert score.unclipped_score - score.base_score <= fusion.NETWORK_COEFF + 1e-9


# ---------------------------------------------------------------------------
# 7. Missing ignition data
# ---------------------------------------------------------------------------
def test_missing_ignition_does_not_change_the_final_score() -> None:
    """Ignition consumes the score; it is never an input to it."""
    node = {
        "id": "a",
        "name": "A",
        "lat": 51.0,
        "lon": 16.0,
        "ndwi_stress_normalised": 0.8,
        "drought_days": 5,
    }
    score = fusion.compute_risk_score(node, {}, {}, {"a": node}, None)
    assert score.final_score > 0.0
    assert score.ignition_score is None
    assert score.fei is None
    assert "ignition_invalid_or_unavailable" in score.data_quality_flags
    assert score.pipeline_status == "DEGRADED"


def test_published_scores_confirm_ignition_is_not_a_fusion_input() -> None:
    payload = _load("risk_scores.json")
    assert "ignition" not in " ".join(payload["fusion_weights"])
    for score in payload["scores"]:
        signal = score["signals"]
        recomputed = (
            fusion.W_NDWI * signal["ndwi_stress"]
            + fusion.W_QTE * signal["qte_score"]
            + fusion.W_FWI * signal["fwi_score"]
            + fusion.W_TREND * fusion._trend_signal(signal["ndwi_trend_14d"])
        )
        assert recomputed == pytest.approx(score["modifiers"]["base_score"], abs=5e-4)


# ---------------------------------------------------------------------------
# 8. Stale and degraded upstream data
# ---------------------------------------------------------------------------
def test_sentinel_step_reports_full_coverage_with_zero_api_requests() -> None:
    """DEFECT: coverage_percent counts cache entries, not fresh acquisitions."""
    manifest = _load("pipeline_steps/sentinel.json")
    if manifest.get("api_requests") != 0:
        pytest.skip("this run actually reached the API")
    assert manifest["coverage_percent"] == 100.0
    assert manifest["status"] == "SUCCESS"
    assert manifest["cache_classification"]["fresh"] == 34


def test_trend_slope_is_per_observation_index_not_per_day() -> None:
    """DEFECT: cloud gaps inflate the trend of nodes with fewer observations."""
    dense = float(np.polyfit(np.arange(3.0), [0.20, 0.15, 0.10], 1)[0])
    sparse = float(np.polyfit(np.arange(2.0), [0.20, 0.10], 1)[0])
    assert abs(sparse) > abs(dense)
    assert fusion._trend_signal(sparse) > fusion._trend_signal(dense)


# ---------------------------------------------------------------------------
# 9. Extreme values
# ---------------------------------------------------------------------------
def test_unclipped_score_can_exceed_one_by_the_multiplier_headroom() -> None:
    node = {
        "id": "a",
        "name": "A",
        "lat": 51.0,
        "lon": 16.0,
        "eco": "pine",
        "ndwi_stress_normalised": 1.0,
        "ndwi_trend_14d": -1.0,
        "drought_days": 60,
        "latest": {
            "temp_max": 45.0, "rh_min": 5.0, "wind_max": 30.0,
            "rain_total": 0.0, "vpd_mean": 6.0, "soil_min": 0.0,
        },
    }
    qte_map = {"a": {"qte_score": 1.0, "bridge_fired": True}}
    score = fusion.compute_risk_score(node, qte_map, {}, {"a": node})
    assert score.unclipped_score > 1.0
    assert score.final_score == 1.0


def test_ecosystem_multiplier_scales_weather_and_trend_too() -> None:
    """The multiplier hits the whole blend, not just the canopy term."""
    base = {
        "id": "a", "name": "A", "lat": 51.0, "lon": 16.0,
        "ndwi_stress_normalised": 0.0, "latest": {"wind_max": 15.0},
    }
    mixed = fusion.compute_risk_score({**base, "eco": "mixed"}, {}, {}, {})
    pine = fusion.compute_risk_score({**base, "eco": "pine"}, {}, {}, {})
    assert pine.final_score == pytest.approx(mixed.final_score * 1.30, abs=1e-4)
    assert pine.base_score == mixed.base_score


def test_missing_soil_moisture_silently_defaults_to_a_dry_constant() -> None:
    """DEFECT: Open-Meteo returns no soil moisture, and nothing flags it."""
    latest = {
        "soil_min": None, "temp_max": 19.0, "rh_min": 48.0,
        "wind_max": 20.3, "rain_total": 0.1, "vpd_mean": 0.651,
    }
    node = {"id": "a", "latest": latest, "drought_days": 1}
    with_null = fusion._fwi_from_weather(node)
    explicit = {**node, "latest": {**latest, "soil_min": 0.20}}
    assert with_null == pytest.approx(fusion._fwi_from_weather(explicit))
    saturated = {**node, "latest": {**latest, "soil_min": 0.30}}
    assert fusion._fwi_from_weather(saturated) < with_null


def test_every_published_node_has_null_soil_moisture() -> None:
    nodes = _load("nodes_enriched.json")["nodes"]
    assert all(node.get("latest", {}).get("soil_min") is None for node in nodes)


def test_fwi_ignores_rain_older_than_the_latest_day() -> None:
    """DEFECT: 92 mm over a week is invisible; only today's rain counts."""
    latest = {
        "soil_min": None, "temp_max": 19.0, "rh_min": 48.0,
        "wind_max": 20.3, "rain_total": 0.1, "vpd_mean": 0.651,
    }
    drenched = {
        "id": "a",
        "latest": latest,
        "drought_days": 1,
        "weather_history": {"rain_total": [0.0] * 6 + [16.4, 2.3, 6.7, 3.2, 42.1, 20.8, 0.1]},
    }
    parched = {"id": "a", "latest": latest, "drought_days": 1,
               "weather_history": {"rain_total": [0.0] * 13 + [0.1]}}
    assert fusion._fwi_from_weather(drenched) == pytest.approx(
        fusion._fwi_from_weather(parched)
    )


# ---------------------------------------------------------------------------
# 10. Score threshold boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.4500, "LOW"),
        (0.4501, "MODERATE"),
        (0.6000, "MODERATE"),
        (0.6001, "HIGH"),
        (0.7500, "HIGH"),
        (0.7501, "CRITICAL"),
        (1.0000, "CRITICAL"),
        (0.0000, "LOW"),
    ],
)
def test_tier_boundaries_are_strictly_exclusive(score: float, expected: str) -> None:
    assert fusion._tier(score) == expected


def test_alert_bands_are_uniform_and_flagged_uncalibrated() -> None:
    assert fusion.ALERT_CRITICAL - fusion.ALERT_HIGH == pytest.approx(0.15)
    assert fusion.ALERT_HIGH - fusion.ALERT_MODERATE == pytest.approx(0.15)
    assert _load("risk_scores.json")["calibration_status"] == "uncalibrated"


def test_fusion_weights_sum_to_one() -> None:
    total = fusion.W_NDWI + fusion.W_QTE + fusion.W_FWI + fusion.W_TREND
    assert total == pytest.approx(1.0)
