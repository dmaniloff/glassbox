"""End-to-end test for glassbox-extract (prefill-only feature extraction)."""

from __future__ import annotations

import json

import pytest

from glassbox.cli.extract import main, run_extraction

pytestmark = pytest.mark.e2e


def test_extract_spectral(outdir, model_name):
    """Extract spectral features from 2 HaluEval samples, verify JSONL output.

    Config is passed entirely via CLI args — no glassbox.yaml exists.
    This is intentional: it covers the regression from PR #20 where the
    vLLM plugin called set_config(GlassboxConfig()) in the engine core
    subprocess, overwriting the CLI config with defaults (output.path=None)
    and replacing the JsonlHandler with LoggingHandler.  Without the
    _config_set_explicitly guard, no features file would be created.
    """
    main(
        args=[
            "--signal",
            "spectral",
            "--dataset",
            "halueval_hallucination",
            "--max-samples",
            "2",
            "--model",
            model_name,
            "--outdir",
            str(outdir),
        ],
        standalone_mode=False,
    )

    features_path = outdir / "svd_features.jsonl"
    samples_path = outdir / "samples.jsonl"
    config_path = outdir / "config.json"

    # Config metadata written
    assert config_path.exists()
    meta = json.loads(config_path.read_text())
    assert meta["num_layers"] > 0
    assert "spectral" in meta["signals"]

    # Samples written (2 samples x 2 phases = 4 rows)
    sample_rows = [
        json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()
    ]
    assert len(sample_rows) == 4

    # Features written (4 requests x num_layers snapshots)
    snap_rows = [
        json.loads(line) for line in features_path.read_text().splitlines() if line.strip()
    ]
    expected = 4 * meta["num_layers"]
    assert len(snap_rows) == expected

    # Verify snapshot structure
    snap = snap_rows[0]
    assert snap["signal"] == "spectral"
    assert "sv1" in snap["features"]
    assert "sv_ratio" in snap["features"]
    assert "sv_entropy" in snap["features"]


def test_extract_nondefault_signal_and_rank(outdir, model_name):
    """Drive a NON-default signal with NON-default params end to end.

    ``routing`` is ``enabled=False`` by default; ``--svd-rank 3`` differs from
    the default rank of 4; ``--threshold 1`` differs from the default 512.  None
    of these can appear in the output unless the explicit CLI config propagated
    all the way into the *cached* diagnostics inside the vLLM worker subprocess
    (see ``set_config`` / ``_build_diagnostics``).  If the plugin had silently
    rebuilt diagnostics from defaults, we'd instead see ``spectral`` snapshots,
    or rank-4 singular values — so this pins the signal selection and numeric
    algorithm params to non-defaults.

    ``--threshold 1`` also forces routing's matrix-free path, whose SVD is
    dtype-aware; the materialized path crashes under fp16 (tracked in #57), so
    this test deliberately exercises the dtype-clean path.
    """
    main(
        args=[
            "--signal",
            "routing",
            "--svd-rank",
            "3",
            "--threshold",
            "1",
            "--dataset",
            "halueval_hallucination",
            "--max-samples",
            "2",
            "--model",
            model_name,
            "--outdir",
            str(outdir),
        ],
        standalone_mode=False,
    )

    meta = json.loads((outdir / "config.json").read_text())
    assert "routing" in meta["signals"]
    assert "spectral" not in meta["signals"]  # the default signal must not leak in

    snap_rows = [
        json.loads(line)
        for line in (outdir / "svd_features.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert snap_rows, "no routing snapshots written"

    # Right signal (not the default spectral) with routing-specific features...
    assert all(s["signal"] == "routing" for s in snap_rows)
    assert "phi_hat" in snap_rows[0]["features"]
    # ...and the non-default rank=3 reached the cached RoutingDiagnostic:
    # singular_values count == min(rank, L-1) == 3 on prefill (L >> 3).
    assert max(len(s["singular_values"]) for s in snap_rows) == 3


def test_prompt_builders_control_the_prefilled_string(outdir, model_name):
    """A custom prompt builder, not the default Q:/A: wrapper, decides what is prefilled.

    Callers whose labels come from the model's own generation must prefill the
    exact context that produced the response.  The default wrapper would emit
    ``Q: {question}\\nA: {response}``; here the builder emits a chat-style string
    that shares no prefix with it, and ``prompt_length`` in samples.jsonl pins
    down which one actually reached the engine.
    """
    samples = [
        {"idx": 0, "question": "What colour is the sky?", "response": "Blue.", "label": 0},
        {"idx": 1, "question": "What colour is grass?", "response": "Green.", "label": 0},
    ]

    def build_full(s: dict) -> str:
        return f"<|user|>\n{s['question']}<|end|>\n<|assistant|>\n{s['response']}"

    run_extraction(
        samples=samples,
        model=model_name,
        dataset_name="prompt_builder_probe",
        signals=("spectral",),
        outdir=outdir,
        phases=("full",),
        prompt_builders={"full": build_full},
    )

    rows = [
        json.loads(line)
        for line in (outdir / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2  # one phase per sample, not two

    for row, sample in zip(rows, samples):
        assert row["phase"] == "full"
        assert row["prompt_length"] == len(build_full(sample))
        # The default wrapper is a different length, so this cannot pass by luck.
        assert row["prompt_length"] != len(f"Q: {sample['question']}\nA: {sample['response']}")


def test_unknown_phase_is_rejected(outdir, model_name):
    """A phase with no builder fails fast, before an engine is ever created."""
    with pytest.raises(ValueError, match="No prompt builder for phase"):
        run_extraction(
            samples=[{"idx": 0, "question": "q", "response": "r", "label": 0}],
            model=model_name,
            dataset_name="phase_guard_probe",
            outdir=outdir,
            phases=("full", "nonexistent_phase"),
        )


def test_unknown_signal_is_rejected(outdir, model_name):
    """An unrecognised signal fails fast rather than emitting an empty features file.

    ``from_cli_args`` enables nothing for a name it does not recognise, so without
    this guard the run would complete and write zero snapshots.
    """
    with pytest.raises(ValueError, match="Unknown signal"):
        run_extraction(
            samples=[{"idx": 0, "question": "q", "response": "r", "label": 0}],
            model=model_name,
            dataset_name="signal_guard_probe",
            signals=("spectral", "not_a_signal"),
            outdir=outdir,
        )


def test_metadata_records_signal_params(outdir, model_name):
    """config.json keeps the flat keys glassbox-analyze reads, plus the full config.

    ``glassbox-analyze`` reports ``svd_interval``/``svd_rank`` straight out of this
    file, and regenerating features.parquet needs ``heads`` without an HF download.
    """
    main(
        args=[
            "--signal",
            "spectral",
            "--svd-rank",
            "3",
            "--dataset",
            "halueval_hallucination",
            "--max-samples",
            "1",
            "--model",
            model_name,
            "--outdir",
            str(outdir),
        ],
        standalone_mode=False,
    )

    meta = json.loads((outdir / "config.json").read_text())
    assert meta["svd_rank"] == 3
    assert meta["svd_interval"] == 1
    assert meta["heads"] == [0]
    assert meta["method"] == "randomized"
    assert meta["n_samples"] == 1
    # Full nested config travels alongside the flat keys for provenance.
    assert meta["config"]["spectral"]["rank"] == 3
