from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide a workspace-local temporary path on restricted Windows hosts."""
    root = Path(__file__).parents[1] / ".test-tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
