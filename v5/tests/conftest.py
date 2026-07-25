from __future__ import annotations

import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

V5_DIR = Path(__file__).parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))


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
