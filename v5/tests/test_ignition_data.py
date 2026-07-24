from __future__ import annotations

import csv
import io
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
    mask_secret,
    merge_firms_csv_parts,
    parse_firms_csv_response,
    split_date_windows,
)

HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
)


def firms_response(*rows: str) -> HTTPResponse:
    return HTTPResponse((HEADER + "".join(rows)).encode(), "text/csv; charset=utf-8")


ROW_A = "51.0,16.0,330,0.4,0.4,2025-01-02,1200,N20,VIIRS,n,2.0,300,4,D\n"
ROW_B = "50.5,15.5,331,0.4,0.4,2025-01-01,0900,N20,VIIRS,h,2.0,301,5,D\n"


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
        return firms_response(ROW_A)

    destination = tmp_path / "firms.csv"
    count = download_firms_year(
        destination,
        map_key="not-logged",
        year=2025,
        bbox=(14.6, 49.9, 17.9, 51.9),
        source="VIIRS_SNPP_SP",
        transport=transport,
    )
    assert count == 1
    assert len(requested) == 73
    assert destination.exists()
    assert list(tmp_path.glob(".firms-*")) == []


def test_clc_query_has_spatial_filter_pagination_and_agriculture_classes() -> None:
    url = build_clc_query_url((14.6, 49.9, 17.9, 51.9), offset=1000)
    assert "resultOffset=1000" in url
    assert "resultRecordCount=1000" in url
    assert "geometry=14.6%2C49.9%2C17.9%2C51.9" in url
    assert "Code_18+IN" in url
