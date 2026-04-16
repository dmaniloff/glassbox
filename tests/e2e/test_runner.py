"""End-to-end test for glassbox-run (single-prompt generation with features)."""

from __future__ import annotations

import json

import pytest

from glassbox.cli.runner import main

pytestmark = pytest.mark.e2e


def test_runner_spectral(outdir, model_name):
    """Generate text and extract spectral features, verify JSONL output."""
    output_path = outdir / "features.jsonl"

    main(
        args=[
            "--model",
            model_name,
            "--signal",
            "spectral",
            "--interval",
            "16",
            "--rank",
            "2",
            "--heads",
            "0",
            "--output",
            str(output_path),
            "--prompt",
            "The meaning of life is",
        ],
        standalone_mode=False,
    )

    assert output_path.exists()
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) > 0

    # Verify snapshot structure
    snap = rows[0]
    assert snap["signal"] == "spectral"
    assert "layer_idx" in snap
    assert "step" in snap
    assert "sv1" in snap["features"]
