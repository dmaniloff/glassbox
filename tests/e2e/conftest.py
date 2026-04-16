"""Shared fixtures for end-to-end tests.

These tests require a GPU and download facebook/opt-125m (~250 MB).
Run with:  pytest -m e2e
Skip with: pytest -m "not e2e"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MODEL = "facebook/opt-125m"


@pytest.fixture(scope="session")
def model_name():
    return MODEL


@pytest.fixture()
def outdir(tmp_path):
    """Per-test output directory, cleaned up automatically."""
    d = tmp_path / "glassbox_e2e"
    d.mkdir()
    return d


def assert_jsonl_valid(path: Path, min_lines: int = 1) -> list[dict]:
    """Read a JSONL file and assert it has at least ``min_lines`` valid rows."""
    assert path.exists(), f"{path} does not exist"
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    assert len(rows) >= min_lines, f"Expected >= {min_lines} rows, got {len(rows)}"
    return rows
