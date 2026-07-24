from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path

import pytest

from ignition_data import (
    HTTPResponse,
    ResponseValidationError,
    atomic_write_bytes,
    build_clc_query_url,
    build_firms_area_url,
    download_firms_year,
    inspect_firms_map_key,
    mask_secret,
    merge_firms_csv_parts,
    parse_firms_csv_response,
    request_with_retry,
    split_date_windows,
    validate_firms_window,
)

HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
)


def firms_response(*rows: str) -> HTTPResponse:
    return HTTPResponse((HEADER + "".join(rows)).encode(), "text/csv; charset=utf-8")


ROW_A = "51.0,16.0,330,0.4,0.4,2025-01-02,1200,N20,VIIRS,n,2.0,300,4,D\n"
ROW_B = "50.5,15.5,331,0.4,0.4,2025-01-01,0900,N20,VIIRS,h,2.0,301,5,D\n"


def window_response(url: str, *, version: str = "2.0") -> HTTPResponse:
    window_start = url.rsplit("/", 1)[-1]
    row = f"51.0,16.0,330,0.4,0.4,{window_start},1200,N20,VIIRS,n,{version},300,4,D\n"
    return firms_response(row)


def test_build_firms_url_uses_official_segment_order() -> None:
    url = build_firms_area_url(
        "secret-key",
        "VIIRS_SNPP_SP",
        (14.6, 49.9, 17.9, 51.9),
        date(2025, 1, 1),
        5,
    )
    assert url.endswith("/secret-key/VIIRS_SNPP_SP/14.6,49.9,17.9,51.9/5/2025-01-01")


def test_full_year_is_split_into_supported_windows() -> None:
    windows = split_date_windows(date(2025, 1, 1), date(2025, 12, 31))
    assert len(windows) == 73
    assert sum(days for _, days in windows) == 365
    assert all(1 <= days <= 5 for _, days in windows)
    assert windows[-1] == (date(2025, 12, 27), 5)


def test_merge_deduplicates_and_sorts_deterministically() -> None:
    merged = merge_firms_csv_parts(
        [firms_response(ROW_A, ROW_B), firms_response(ROW_A)]
    )
    rows = list(csv.DictReader(io.StringIO(merged.decode())))
    assert len(rows) == 2
    assert [row["acq_date"] for row in rows] == ["2025-01-01", "2025-01-02"]


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<html>Bad gateway</html>", "text/html"),
        (b'{"error": "Invalid MAP_KEY"}', "application/json"),
        (b"", "text/csv"),
        (b"foo,bar\n1,2\n", "text/csv"),
    ],
)
def test_firms_validation_rejects_non_csv_and_bad_schema(
    body: bytes, content_type: str
) -> None:
    with pytest.raises(ResponseValidationError):
        parse_firms_csv_response(HTTPResponse(body, content_type))


def test_firms_window_validation_rejects_out_of_range_rows() -> None:
    with pytest.raises(ResponseValidationError, match="outside"):
        validate_firms_window(
            firms_response(ROW_A),
            window_start=date(2025, 1, 6),
            day_range=5,
        )


def test_mask_secret_removes_raw_and_encoded_key() -> None:
    key = "abc/123+secret"
    value = "https://example/abc/123+secret/abc%2F123%2Bsecret"
    masked = mask_secret(value, key)
    assert key not in masked
    assert "abc%2F123%2Bsecret" not in masked
    assert masked.count("***") == 2


def test_atomic_write_does_not_replace_destination_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.csv"
    destination.write_bytes(b"old")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("ignition_data.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".cache.csv.*.tmp")) == []


def test_download_year_uses_cache_safe_atomic_merge(tmp_path: Path) -> None:
    requested: list[str] = []

    def transport(url: str, timeout: float) -> HTTPResponse:
        requested.append(url)
        return window_response(url)

    destination = tmp_path / "firms.csv"
    result = download_firms_year(
        destination,
        map_key="not-logged",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        min_request_interval_seconds=0,
    )
    assert result.status == "complete"
    assert result.row_count == 73
    assert len(requested) == 73
    assert destination.exists()
    assert (tmp_path / "firms_parts" / "VIIRS_SNPP_SP" / "status.json").exists()


def test_resume_skips_valid_checkpoints_and_fetches_only_missing_windows(
    tmp_path: Path,
) -> None:
    first_calls: list[str] = []

    def interrupted_transport(url: str, timeout: float) -> HTTPResponse:
        first_calls.append(url)
        if len(first_calls) <= 2:
            return window_response(url)
        return HTTPResponse(b"temporary", "text/plain", status=503)

    destination = tmp_path / "firms.csv"
    first = download_firms_year(
        destination,
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=interrupted_transport,
        attempts=1,
        min_request_interval_seconds=0,
    )
    assert first.status == "resumable_partial"
    assert first.completed_windows == 2
    assert not destination.exists()

    resumed_calls: list[str] = []

    def resumed_transport(url: str, timeout: float) -> HTTPResponse:
        resumed_calls.append(url)
        return window_response(url)

    resumed = download_firms_year(
        destination,
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=resumed_transport,
        min_request_interval_seconds=0,
    )
    assert resumed.status == "complete"
    assert resumed.completed_windows == 73
    assert len(resumed_calls) == 71
    assert all("2025-01-01" not in url for url in resumed_calls)
    assert all("2025-01-06" not in url for url in resumed_calls)


def test_valid_csv_without_metadata_is_reused_and_gets_checksum(
    tmp_path: Path,
) -> None:
    product_root = tmp_path / "parts" / "VIIRS_SNPP_SP"
    product_root.mkdir(parents=True)
    part = product_root / "2025-01-01.csv"
    part.write_bytes(window_response("https://example/2025-01-01").body)
    requested: list[str] = []

    def transport(url: str, timeout: float) -> HTTPResponse:
        requested.append(url)
        return window_response(url)

    result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        parts_root=tmp_path / "parts",
        min_request_interval_seconds=0,
    )
    assert result.status == "complete"
    assert len(requested) == 72
    metadata = json.loads(
        (product_root / "2025-01-01.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["source"] == "VIIRS_SNPP_SP"
    assert len(metadata["sha256"]) == 64


def test_401_stops_without_retry_and_preserves_completed_parts(
    tmp_path: Path,
) -> None:
    area_calls: list[str] = []
    status_calls = 0

    def transport(url: str, timeout: float) -> HTTPResponse:
        nonlocal status_calls
        if "mapkey_status" in url:
            status_calls += 1
            return HTTPResponse(
                json.dumps(
                    {
                        "transaction_limit": 5000,
                        "current_transactions": 100,
                        "transaction_interval": "10 minutes",
                    }
                ).encode(),
                "application/json",
            )
        area_calls.append(url)
        if len(area_calls) == 1:
            return window_response(url)
        return HTTPResponse(b'{"error":"unauthorized"}', "application/json", 401)

    result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="super-secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        attempts=5,
        min_request_interval_seconds=0,
    )
    assert result.status == "authorization_failed"
    assert result.authorization_reason == "unknown_authorization_failure"
    assert len(area_calls) == 2
    assert status_calls == 1
    product_root = tmp_path / "firms_parts" / "VIIRS_SNPP_SP"
    assert (product_root / "2025-01-01.csv").exists()
    assert not (product_root / "2025-01-06.csv").exists()
    assert "super-secret" not in (product_root / "status.json").read_text()


def test_429_stops_and_reports_retry_after(tmp_path: Path) -> None:
    calls = 0

    def transport(url: str, timeout: float) -> HTTPResponse:
        nonlocal calls
        calls += 1
        return HTTPResponse(
            b"too many requests",
            "text/plain",
            status=429,
            headers={"Retry-After": "120"},
        )

    result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        attempts=5,
        min_request_interval_seconds=0,
    )
    assert result.status == "rate_limited"
    assert result.retry_after_seconds == 120
    assert calls == 1
    assert not (tmp_path / "firms.csv").exists()


def test_checksum_mismatch_forces_window_download(tmp_path: Path) -> None:
    product_root = tmp_path / "parts" / "VIIRS_SNPP_SP"
    product_root.mkdir(parents=True)
    (product_root / "2025-01-01.csv").write_bytes(
        window_response("https://example/2025-01-01").body
    )
    (product_root / "2025-01-01.meta.json").write_text(
        json.dumps(
            {
                "source": "VIIRS_SNPP_SP",
                "bbox": [14.6, 49.9, 17.9, 51.9],
                "window_start": "2025-01-01",
                "window_end": "2025-01-05",
                "day_range": 5,
                "byte_count": 1,
                "row_count": 1,
                "sha256": "incorrect",
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def transport(url: str, timeout: float) -> HTTPResponse:
        calls.append(url)
        return HTTPResponse(
            b"slow down",
            "text/plain",
            status=429,
            headers={"Retry-After": "1"},
        )

    result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        parts_root=tmp_path / "parts",
        min_request_interval_seconds=0,
    )
    assert result.status == "rate_limited"
    assert len(calls) == 1
    assert calls[0].endswith("/2025-01-01")


def test_sp_and_nrt_checkpoints_are_never_mixed(tmp_path: Path) -> None:
    parts_root = tmp_path / "parts"
    sp_calls = 0

    def partial_sp(url: str, timeout: float) -> HTTPResponse:
        nonlocal sp_calls
        sp_calls += 1
        if sp_calls == 1:
            return window_response(url, version="SP")
        return HTTPResponse(b"unavailable", "text/plain", 503)

    sp_result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=partial_sp,
        parts_root=parts_root,
        attempts=1,
        min_request_interval_seconds=0,
    )
    assert sp_result.completed_windows == 1

    nrt_calls: list[str] = []

    def complete_nrt(url: str, timeout: float) -> HTTPResponse:
        nrt_calls.append(url)
        return window_response(url, version="NRT")

    nrt_result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_NRT",
        transport=complete_nrt,
        parts_root=parts_root,
        min_request_interval_seconds=0,
    )
    assert nrt_result.status == "complete"
    assert len(nrt_calls) == 73
    rows = list(csv.DictReader(io.StringIO((tmp_path / "firms.csv").read_text())))
    assert {row["version"] for row in rows} == {"NRT"}
    assert (parts_root / "VIIRS_SNPP_SP" / "2025-01-01.csv").exists()
    assert (parts_root / "VIIRS_SNPP_NRT" / "2025-01-01.csv").exists()


def test_merge_is_not_published_until_every_window_is_valid(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "firms.csv"
    destination.write_bytes(b"previous-production-file")
    calls = 0

    def transport(url: str, timeout: float) -> HTTPResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return window_response(url)
        return HTTPResponse(b"gateway timeout", "text/plain", 504)

    result = download_firms_year(
        destination,
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        attempts=1,
        min_request_interval_seconds=0,
    )
    assert result.status == "resumable_partial"
    assert destination.read_bytes() == b"previous-production-file"


def test_transaction_counter_can_stop_before_area_request(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(url: str, timeout: float) -> HTTPResponse:
        calls.append(url)
        assert "mapkey_status" in url
        return HTTPResponse(
            json.dumps(
                {
                    "transaction_limit": 5000,
                    "current_transactions": 5000,
                    "transaction_interval": "10 minutes",
                }
            ).encode(),
            "application/json",
        )

    result = download_firms_year(
        tmp_path / "firms.csv",
        map_key="secret",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
        transaction_check_every=10,
        min_request_interval_seconds=0,
    )
    assert result.status == "rate_limited"
    assert result.retry_after_seconds == 600
    assert len(calls) == 1


def test_transient_timeout_retries_with_backoff_and_jitter() -> None:
    calls = 0
    delays: list[float] = []

    def transport(url: str, timeout: float) -> HTTPResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("slow")
        return firms_response(ROW_A)

    response = request_with_retry(
        "https://example.test/data",
        transport=transport,
        attempts=3,
        backoff_seconds=2,
        sleep_fn=delays.append,
        random_fn=lambda: 0.5,
    )
    assert response.status == 200
    assert delays == [2.25, 4.5]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"error": "Invalid MAP_KEY"}, "invalid_key"),
        (
            {
                "transaction_limit": 5000,
                "current_transactions": 5000,
                "transaction_interval": "10 minutes",
            },
            "exhausted_transaction_window",
        ),
    ],
)
def test_map_key_status_classification(payload: dict, expected: str) -> None:
    def transport(url: str, timeout: float) -> HTTPResponse:
        assert "secret" in url
        return HTTPResponse(json.dumps(payload).encode(), "application/json")

    status = inspect_firms_map_key("secret", transport=transport)
    assert status.state == expected


def test_clc_query_has_spatial_filter_pagination_and_agriculture_classes() -> None:
    url = build_clc_query_url((14.6, 49.9, 17.9, 51.9), offset=1000)
    assert "resultOffset=1000" in url
    assert "resultRecordCount=1000" in url
    assert "geometry=14.6%2C49.9%2C17.9%2C51.9" in url
    assert "Code_18+IN" in url
