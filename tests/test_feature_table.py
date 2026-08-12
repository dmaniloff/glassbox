"""Tests for the JSONL -> wide feature-table pivot.

Column names are written out literally rather than recomputed from the helpers under
test, which would make the assertions tautological.

CPU only -- every input is synthetic, no engine required.
"""

from __future__ import annotations

import json

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostics import DIAGNOSTIC_REGISTRY, FEATURES_MODELS
from glassbox.feature_table import (
    build_feature_columns,
    feature_names,
    parse_snap_features,
    write_parquet,
)
from glassbox.results import SVDSnapshot

CONFIG = GlassboxConfig()


def _snapshot(name: str, *, layer_idx: int = 0, head: int = 0) -> SVDSnapshot:
    """A real snapshot for ``name``, produced by running the diagnostic."""
    torch.manual_seed(0)
    Q, K = torch.randn(24, 8), torch.randn(24, 8)
    out = DIAGNOSTIC_REGISTRY[name](CONFIG.signals[name]).reduce(Q, K, 24)
    return SVDSnapshot(
        signal=name,
        request_id=0,
        layer=f"model.layers.{layer_idx}.self_attn",
        layer_idx=layer_idx,
        head=head,
        step=1,
        L=24,
        features=out["features"],
        singular_values=out.get("singular_values", []),
    )


class TestColumnNames:
    def test_spectral(self):
        assert build_feature_columns(1, [0], ("spectral",), CONFIG) == [
            "spectral_sv1_L0_H0",
            "spectral_sv_ratio_L0_H0",
            "spectral_sv_entropy_L0_H0",
        ]

    def test_features_carry_no_inner_prefix(self):
        """The signal name is already the outer prefix.

        These columns were ``routing_hodge_G_L0_H0`` etc.  The inner ``hodge_`` came from
        a chain of ``if signal ==`` tests in cli/extract.py and duplicated information the
        outer prefix already carried.
        """
        cols = build_feature_columns(1, [0], ("routing",), CONFIG)
        assert "routing_G_L0_H0" in cols
        assert "routing_phi_hat_L0_H0" in cols
        assert not any("hodge" in c for c in cols)

    def test_bespoke_prefixes_are_gone(self):
        """tracker/selfattn/laplacian carried ``at_``/``ad_``/``lap_``; none now do."""
        for signal, sample in (
            ("tracker", "tracker_sigma2_L0_H0"),
            ("selfattn", "selfattn_attn_diag_logmean_L0_H0"),
            ("laplacian", "laplacian_eigval_0_L0_H0"),
        ):
            cols = build_feature_columns(1, [0], (signal,), CONFIG)
            assert sample in cols
            assert not any(p in c for c in cols for p in ("_at_", "_ad_", "_lap_"))

    @pytest.mark.parametrize(
        ("signal", "expected"),
        [
            ("asymmetry", ["asymmetry_G_L0_H0", "asymmetry_Gamma_L0_H0", "asymmetry_C_L0_H0"]),
            ("cyclic", ["cyclic_T_cyc_L0_H0"]),
            (
                "magnetic",
                [
                    "magnetic_frustration_L0_H0",
                    "magnetic_phase_curl_L0_H0",
                    "magnetic_phase_curl_w_L0_H0",
                ],
            ),
        ],
    )
    def test_signals_added_after_the_old_code_now_have_columns(self, signal, expected):
        """These three declared no columns at all and made ``--parquet`` fail.

        Nothing had to be added for them here: columns come from the Features model the
        diagnostic declares, so registering a signal is enough.
        """
        assert build_feature_columns(1, [0], (signal,), CONFIG) == expected

    def test_list_fields_expand_to_top_k_indexed_columns(self):
        cfg = GlassboxConfig(laplacian={"top_k": 3})
        assert feature_names("laplacian", cfg) == ["eigval_0", "eigval_1", "eigval_2"]

    def test_singular_values_are_not_columns(self):
        """The derived features are already columns and the vector's width varies with rank."""
        assert not any(
            "singular_value" in c for c in build_feature_columns(1, [0], ("routing",), CONFIG)
        )

    def test_ordering_is_signal_then_layer_then_head(self):
        cols = build_feature_columns(2, [0, 1], ("spectral",), CONFIG)
        assert len(cols) == 2 * 2 * 3
        assert cols[0].endswith("_L0_H0")
        assert cols[3].endswith("_L0_H1")
        assert cols[-1].endswith("_L1_H1")


class TestEmittedKeysMatchDeclaredColumns:
    """The invariant the old code lacked, and the reason three signals were broken.

    ``build_feature_columns`` fixes the schema before any data exists; ``parse_snap_features``
    produces the keys at write time. They used to be derived separately -- one from
    hand-written per-signal branches, the other from an if/elif prefix chain -- so nothing
    stopped them disagreeing. Both now read the same Features model.
    """

    @pytest.mark.parametrize("signal", sorted(FEATURES_MODELS))
    def test_every_signal(self, signal):
        emitted = {f"{signal}_{k}_L0_H0" for k in parse_snap_features(_snapshot(signal))}
        declared = set(build_feature_columns(1, [0], (signal,), CONFIG))
        assert emitted - declared == set(), f"{signal}: emitted but not declared (would raise)"
        assert declared - emitted == set(), f"{signal}: declared but never filled (always null)"


class TestWriteParquet:
    @staticmethod
    def _write_inputs(tmp_path, signal, n_requests=2):
        feats_path = tmp_path / "svd_features.jsonl"
        samples_path = tmp_path / "samples.jsonl"
        with open(feats_path, "w") as ff, open(samples_path, "w") as sf:
            for rid in range(n_requests):
                row = json.loads(_snapshot(signal).model_dump_json())
                row["request_id"] = rid
                ff.write(json.dumps(row) + "\n")
                sf.write(
                    json.dumps(
                        {
                            "request_id": rid,
                            "sample_id": rid,
                            "label": rid % 2,
                            "phase": "full",
                            "prompt_length": 100 + rid,
                            "dataset": "synthetic",
                        }
                    )
                    + "\n"
                )
        return feats_path, samples_path

    def test_schema_is_metadata_then_features(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        feats, samples = self._write_inputs(tmp_path, "spectral")
        out = tmp_path / "features.parquet"
        cols = build_feature_columns(1, [0], ("spectral",), CONFIG)
        assert write_parquet(feats, samples, out, cols) == 2

        table = pq.read_table(out)
        assert table.schema.names[:7] == [
            "request_id",
            "label",
            "length",
            "sample_id",
            "phase",
            "prompt_length",
            "source",
        ]
        assert table.schema.names[7:] == cols

    def test_metadata_round_trips(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        feats, samples = self._write_inputs(tmp_path, "spectral")
        out = tmp_path / "features.parquet"
        write_parquet(feats, samples, out, build_feature_columns(1, [0], ("spectral",), CONFIG))

        rows = pq.read_table(out).to_pylist()
        assert [r["request_id"] for r in rows] == [0, 1]
        assert [r["label"] for r in rows] == [0, 1]
        assert {r["source"] for r in rows} == {"synthetic"}
        assert rows[0]["length"] == 24
        assert rows[0]["spectral_sv1_L0_H0"] is not None

    @pytest.mark.parametrize("signal", ["asymmetry", "cyclic", "magnetic"])
    def test_previously_unwritable_signals_now_write(self, tmp_path, signal):
        pq = pytest.importorskip("pyarrow.parquet")
        feats, samples = self._write_inputs(tmp_path, signal)
        out = tmp_path / "features.parquet"
        cols = build_feature_columns(1, [0], (signal,), CONFIG)
        write_parquet(feats, samples, out, cols)

        rows = pq.read_table(out).to_pylist()
        assert len(rows) == 2
        assert all(rows[0][c] is not None for c in cols)

    def test_single_request_still_checks_the_schema(self, tmp_path):
        """A column absent from the schema must raise even when there is only one request.

        pyarrow drops unknown keys silently, so this is the difference between a loud
        failure and a parquet file quietly missing its features.  The guard used to run
        only when the request id *changed*, so a single-request file skipped it entirely
        and wrote a table with zero feature columns.
        """
        pytest.importorskip("pyarrow.parquet")
        feats, samples = self._write_inputs(tmp_path, "magnetic", n_requests=1)
        out = tmp_path / "features.parquet"
        with pytest.raises(ValueError, match="columns not in schema"):
            write_parquet(feats, samples, out, [])  # schema declares no feature columns
