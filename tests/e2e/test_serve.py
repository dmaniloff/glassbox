"""End-to-end test for vllm serve with glassbox.yaml (production path)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import pytest
import requests

pytestmark = pytest.mark.e2e

SERVE_PORT = 18199  # non-standard port to avoid conflicts
STARTUP_TIMEOUT = 120  # seconds
REQUEST_WAIT = 5  # seconds after request before checking features
BASE_URL = f"http://localhost:{SERVE_PORT}"


def test_serve_spectral(outdir, model_name):
    """Start vllm serve with glassbox.yaml, send a request, verify features."""
    features_path = outdir / "features.jsonl"

    # Write glassbox.yaml in the outdir (we'll cd there)
    config_yaml = outdir / "glassbox.yaml"
    config_yaml.write_text(
        f"""\
spectral:
  enabled: true
  interval: 1
  rank: 2
  heads: [0]
output:
  path: {features_path}
"""
    )

    # Start vllm serve as a subprocess
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            model_name,
            "--attention-backend",
            "CUSTOM",
            "--enforce-eager",
            "--port",
            str(SERVE_PORT),
        ],
        cwd=str(outdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for server to be ready
        ready = False
        for _ in range(STARTUP_TIMEOUT // 2):
            try:
                requests.get(f"{BASE_URL}/health", timeout=2)
                ready = True
                break
            except requests.ConnectionError:
                time.sleep(2)

        assert ready, "vllm serve did not become ready"

        # Send a completion request
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={"model": model_name, "prompt": "Hello world", "max_tokens": 16},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        assert result["choices"][0]["text"], "Empty generation"

        # Wait for features to be flushed
        time.sleep(REQUEST_WAIT)

        # Verify features
        assert features_path.exists(), "Features JSONL not created"
        rows = [json.loads(line) for line in features_path.read_text().splitlines() if line.strip()]
        assert len(rows) > 0, "No feature snapshots written"

        snap = rows[0]
        assert snap["signal"] == "spectral"
        assert "sv1" in snap["features"]

    finally:
        # Clean shutdown
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
