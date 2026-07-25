from __future__ import annotations

from pathlib import Path

import pytest

import qhdalabs_wildfire_ignition_v1 as ignition
from ignition_data import FirmsDownloadResult
from qhdalabs_wildfire_fusion_v1 import compute_risk_score


def node() -> dict:
    return {
        "id": "test-node",
        "name": "Test Forest",
        "lat": 51.0,
        "lon": 16.0,
        "eco": "mixed",
        "latest": {},
    }


def features() -> dict[str, list[tuple[float, float]]]:
    return {
        "roads": [],
        "railways": [],
        "powerlines": [(51.0, 16.0)],
        "tourism": [],
        "agriculture": [],
    }


def score(coverage: list[str]):
    return ignition.compute_ignition_score(node(), features(), [], [], coverage)


def test_one_missing_layer_reports_coverage() -> None:
    coverage = [name for name in ignition.SUBLAYER_NAMES if name != "historical_kde"]
    result = score(coverage)
    assert result.missing_sublayers == ["historical_kde"]
    assert result.coverage_weight == pytest.approx(0.9)
    assert result.coverage_percent == pytest.approx(90.0)
    assert result.score_status == "partial"
    assert result.valid_for_fusion is True


def test_five_missing_layers_block_fusion() -> None:
    result = score(["powerlines"])
    assert len(result.missing_sublayers) == 5
    assert result.coverage_weight == pytest.approx(0.15)
    assert result.coverage_percent == pytest.approx(15.0)
    assert result.ignition_score_raw == pytest.approx(15.0)
    assert result.ignition_score_available == pytest.approx(100.0)
    assert result.ignition_score is None
    assert result.score_status == "partial"
    assert result.valid_for_fusion is False


def test_real_zero_is_distinct_from_missing_data() -> None:
    result = score(["roads"])
    assert result.sublayers.roads == 0.0
    assert result.sublayers.railways is None


def test_explicit_stub_is_never_fusion_valid() -> None:
    result = score(["stub"])
    assert result.score_status == "stub"
    assert result.coverage_weight == 0.0
    assert result.valid_for_fusion is False


def test_fusion_rejects_partial_ignition() -> None:
    partial = score(["powerlines"])
    risk = compute_risk_score(
        node(),
        qte_map={},
        graph={},
        all_nodes={"test-node": node()},
        ignition_map={"test-node": partial},
    )
    assert risk.ignition_valid_for_fusion is False
    assert risk.ignition_score is None
    assert risk.fei is None
    assert risk.qies is None


def test_fusion_accepts_complete_zero_score() -> None:
    complete_zero = ignition.compute_ignition_score(
        node(),
        {
            "roads": [],
            "railways": [],
            "powerlines": [],
            "tourism": [],
            "agriculture": [],
        },
        [],
        [],
        list(ignition.SUBLAYER_NAMES),
    )
    risk = compute_risk_score(
        node(),
        qte_map={},
        graph={},
        all_nodes={"test-node": node()},
        ignition_map={"test-node": complete_zero},
    )
    assert risk.ignition_valid_for_fusion is True
    assert risk.ignition_score == 0.0
    assert risk.fei == 0.0
    assert risk.qies == 0.0


def test_existing_valid_cache_works_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "firms.csv"
    cache.write_text(
        "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence\n"
        "51,16,2025-01-01,1200,N20,VIIRS,n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ignition, "FIRMS_CSV", cache)
    monkeypatch.setattr(ignition, "IBL_GEOJSON", tmp_path / "ibl.geojson")
    monkeypatch.setattr(ignition, "EFFIS_CSV", tmp_path / "effis.csv")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    assert ignition._download_firms_or_effis(refresh_firms=False) == cache
    assert ignition._download_firms_or_effis(refresh_firms=True) == cache


def test_partial_sp_does_not_automatically_start_nrt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_sources: list[str] = []

    def partial_download(destination: Path, **kwargs) -> FirmsDownloadResult:
        requested_sources.append(kwargs["source"])
        return FirmsDownloadResult(
            status="resumable_partial",
            source=kwargs["source"],
            completed_windows=12,
            total_windows=73,
        )

    monkeypatch.setattr(ignition, "FIRMS_CSV", tmp_path / "firms.csv")
    monkeypatch.setattr(ignition, "FIRMS_PARTS_DIR", tmp_path / "parts")
    monkeypatch.setattr(ignition, "IBL_GEOJSON", tmp_path / "ibl.geojson")
    monkeypatch.setattr(ignition, "EFFIS_CSV", tmp_path / "effis.csv")
    monkeypatch.setattr(ignition, "download_firms_year", partial_download)
    monkeypatch.setenv("FIRMS_MAP_KEY", "secret")
    monkeypatch.delenv("FIRMS_SOURCE", raising=False)

    assert ignition._download_firms_or_effis(refresh_firms=True) is None
    assert requested_sources == ["VIIRS_SNPP_SP"]
