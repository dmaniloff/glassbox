import copy
import pickle
import textwrap

import pydantic
import pytest

from glassbox.config import (
    THRESHOLD_SIGNALS,
    CausalMode,
    GlassboxConfig,
    IncrementalMode,
    SignalConfigBase,
    StreamingMode,
    SVDParams,
    ThresholdParams,
    validate_window_modes,
)
from glassbox.diagnostics import DIAGNOSTIC_REGISTRY


def test_matvec_strategy_default():
    assert GlassboxConfig().matvec_strategy == "auto"


def test_matvec_strategy_explicit():
    for s in ("loop", "batched", "triton", "auto"):
        assert GlassboxConfig(matvec_strategy=s).matvec_strategy == s


def test_matvec_strategy_invalid():
    with pytest.raises(pydantic.ValidationError):
        GlassboxConfig(matvec_strategy="gpu")


def test_resolve_matvec_strategy_passthrough():
    for s in ("loop", "batched", "triton"):
        assert GlassboxConfig.resolve_matvec_strategy(s) == s


def test_resolve_matvec_strategy_auto():
    # "auto" -> "triton" only with Triton+CUDA; here (CPU/no-triton) it must be "batched".
    assert GlassboxConfig.resolve_matvec_strategy("auto") in ("batched", "triton")


def test_defaults():
    config = GlassboxConfig()
    assert config.spectral.enabled is True
    assert config.routing.enabled is False
    assert config.selfattn.enabled is False
    assert config.spectral.interval == 32
    assert config.spectral.rank == 4
    assert config.spectral.method == "randomized"
    assert config.spectral.heads == [0]
    assert config.routing.interval == 32
    assert config.routing.threshold == 512
    assert config.routing.hodge_seed == 42
    assert config.selfattn.interval == 32
    assert config.selfattn.threshold == 512
    assert config.selfattn.heads == [0]
    assert config.output.path is None
    assert config.emit.otel is False


def test_programmatic_kwargs():
    config = GlassboxConfig(spectral={"interval": 16})
    assert config.spectral.interval == 16
    assert config.spectral.rank == 4  # default preserved


def test_programmatic_kwargs_routing():
    config = GlassboxConfig(routing={"enabled": True, "threshold": 1024})
    assert config.routing.enabled is True
    assert config.routing.threshold == 1024
    assert config.routing.rank == 4  # default preserved


def test_tracker_defaults():
    config = GlassboxConfig()
    assert config.tracker.enabled is False
    assert config.tracker.interval == 32
    assert config.tracker.rank == 4
    assert config.tracker.method == "randomized"
    assert config.tracker.heads == [0]
    assert config.tracker.threshold == 512
    assert config.tracker.block_size == 256


def test_programmatic_kwargs_tracker():
    config = GlassboxConfig(tracker={"enabled": True, "interval": 16, "threshold": 256})
    assert config.tracker.enabled is True
    assert config.tracker.interval == 16
    assert config.tracker.threshold == 256
    assert config.tracker.rank == 4  # default preserved


def test_yaml_loading(tmp_path, monkeypatch):
    yaml_content = "spectral:\n  interval: 16\n  rank: 8\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.spectral.interval == 16
    assert config.spectral.rank == 8


def test_yaml_tracker(tmp_path, monkeypatch):
    yaml_content = "tracker:\n  enabled: true\n  interval: 64\n  threshold: 1024\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.tracker.enabled is True
    assert config.tracker.interval == 64
    assert config.tracker.threshold == 1024


def test_yaml_routing(tmp_path, monkeypatch):
    yaml_content = (
        "routing:\n  enabled: true\n  interval: 64\n"
        "output:\n  path: /var/log/glassbox/signals.jsonl\n"
    )
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.routing.enabled is True
    assert config.routing.interval == 64
    assert config.output.path == "/var/log/glassbox/signals.jsonl"


def test_precedence_kwargs_beat_yaml(tmp_path, monkeypatch):
    yaml_content = "spectral:\n  interval: 16\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig(spectral={"interval": 8})
    assert config.spectral.interval == 8


def test_programmatic_kwargs_selfattn():
    config = GlassboxConfig(selfattn={"enabled": True, "interval": 16, "heads": [0, 1]})
    assert config.selfattn.enabled is True
    assert config.selfattn.interval == 16
    assert config.selfattn.heads == [0, 1]
    assert config.selfattn.threshold == 512  # default preserved


def test_yaml_selfattn(tmp_path, monkeypatch):
    yaml_content = "selfattn:\n  enabled: true\n  interval: 64\n  heads: [0, 2, 4]\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.selfattn.enabled is True
    assert config.selfattn.interval == 64
    assert config.selfattn.heads == [0, 2, 4]


def test_laplacian_defaults():
    config = GlassboxConfig()
    assert config.laplacian.enabled is False
    assert config.laplacian.interval == 32
    assert config.laplacian.heads == [0]
    assert config.laplacian.top_k == 10
    assert config.laplacian.threshold == 512
    assert config.laplacian.block_size == 256


def test_programmatic_kwargs_laplacian():
    config = GlassboxConfig(laplacian={"enabled": True, "interval": 16, "top_k": 20})
    assert config.laplacian.enabled is True
    assert config.laplacian.interval == 16
    assert config.laplacian.top_k == 20
    assert config.laplacian.threshold == 512  # default preserved


def test_yaml_laplacian(tmp_path, monkeypatch):
    yaml_content = "laplacian:\n  enabled: true\n  interval: 64\n  top_k: 25\n  heads: [0, 1, 2]\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.laplacian.enabled is True
    assert config.laplacian.interval == 64
    assert config.laplacian.top_k == 25
    assert config.laplacian.heads == [0, 1, 2]


# ── from_cli_args tests ──────────────────────────────────────────────────


def test_from_cli_args_signals_enable_disable():
    config = GlassboxConfig.from_cli_args(signals=("spectral", "routing"))
    assert config.spectral.enabled is True
    assert config.routing.enabled is True
    assert config.tracker.enabled is False
    assert config.selfattn.enabled is False
    assert config.laplacian.enabled is False


def test_from_cli_args_default_signals():
    config = GlassboxConfig.from_cli_args()
    assert config.spectral.enabled is True
    assert config.routing.enabled is False
    assert config.tracker.enabled is False
    assert config.selfattn.enabled is False
    assert config.laplacian.enabled is False


def test_from_cli_args_rank_method_heads():
    config = GlassboxConfig.from_cli_args(
        signals=("spectral",), rank=8, method="lanczos", heads=(0, 2)
    )
    assert config.spectral.rank == 8
    assert config.spectral.method == "lanczos"
    assert config.spectral.heads == [0, 2]


def test_from_cli_args_interval():
    config = GlassboxConfig.from_cli_args(signals=("spectral",), interval=1)
    assert config.spectral.interval == 1


def test_from_cli_args_threshold_block_size():
    config = GlassboxConfig.from_cli_args(signals=("routing",), threshold=1024, block_size=512)
    assert config.routing.threshold == 1024
    assert config.routing.block_size == 512


def test_from_cli_args_threshold_zero():
    """threshold=0 (always-matrix-free) must be kept, not dropped as falsy.

    from_cli_args guards with ``if threshold is not None``; a falsy check
    (``if threshold:``) would silently drop 0 and fall back to the default 512.
    """
    config = GlassboxConfig.from_cli_args(
        signals=tuple(THRESHOLD_SIGNALS),
        threshold=0,
    )
    for sig in THRESHOLD_SIGNALS:
        assert getattr(config, sig).threshold == 0


def test_from_cli_args_output_otel():
    config = GlassboxConfig.from_cli_args(output_path="/tmp/out.jsonl", otel=True)
    assert config.output.path == "/tmp/out.jsonl"
    assert config.emit.otel is True


def test_from_cli_args_yaml_auto_load(tmp_path, monkeypatch):
    (tmp_path / "glassbox.yaml").write_text(
        textwrap.dedent("""\
        routing:
          enabled: true
          hodge_seed: 7
    """)
    )
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig.from_cli_args(
        signals=("spectral", "routing"),
        rank=2,
    )
    assert config.routing.enabled is True
    assert config.routing.rank == 2  # CLI args beat YAML
    assert config.routing.hodge_seed == 7  # from YAML


def test_from_cli_args_svd_not_set_on_non_svd_signals():
    config = GlassboxConfig.from_cli_args(signals=("selfattn",), rank=8)
    # rank shouldn't propagate to selfattn (not an SVD signal)
    assert config.selfattn.enabled is True
    # selfattn has no rank field — defaults unchanged
    assert config.spectral.rank == 4  # default, not 8 (spectral disabled)


# ── window-mode <-> streaming-mode validator ─────────────────────────────


class TestWindowModeValidator:
    """Guards the mode<->windowing invariants (docs/streaming-modes.md)."""

    def test_default_config_valid(self):
        GlassboxConfig()  # sliding, unbounded, no streaming signal -> no error

    def test_tumbling_requires_finite_window(self):
        with pytest.raises(pydantic.ValidationError):
            GlassboxConfig(q_buffer_mode="tumbling")  # q_buffer_max_tokens defaults to 0
        GlassboxConfig(q_buffer_mode="tumbling", q_buffer_max_tokens=256)  # ok

    def test_streaming_requires_tumbling_window(self):
        # block-diagonal global accumulation is only unbiased over disjoint windows
        with pytest.raises(ValueError):
            validate_window_modes([("sig", True, True, False)], "sliding", 256)
        with pytest.raises(ValueError):
            validate_window_modes([("sig", True, True, False)], "tumbling", 0)
        validate_window_modes([("sig", True, True, False)], "tumbling", 256)  # ok

    def test_incremental_requires_unbounded_buffer(self):
        # exact full-operator streaming needs the full sequence; a bounded buffer breaks it
        with pytest.raises(ValueError):
            validate_window_modes([("sig", True, False, True)], "sliding", 256)
        validate_window_modes([("sig", True, False, True)], "sliding", 0)  # ok

    def test_disabled_signal_modes_not_enforced(self):
        validate_window_modes([("sig", False, True, True)], "sliding", 256)  # disabled -> skip

    def test_plain_signal_any_window(self):
        # a signal without streaming/incremental is valid under any window
        validate_window_modes([("sig", True, False, False)], "sliding", 0)
        validate_window_modes([("sig", True, False, False)], "tumbling", 256)


# ── derived registries ───────────────────────────────────────────────────


class TestDerivedRegistries:
    """The signal registry is derived from the GlassboxConfig field annotations.

    These pin the invariants that make that derivation safe: the one registry that
    *cannot* be derived (DIAGNOSTIC_REGISTRY, which lives in another module) must agree,
    and a capability's fields must travel with its mixin rather than be hand-declared.
    """

    def test_diagnostic_registry_matches_signal_names(self):
        """The only hand-maintained registry left — a new signal must appear in both."""
        assert set(DIAGNOSTIC_REGISTRY) == set(GlassboxConfig.signal_names())

    def test_signals_mapping_covers_every_signal(self):
        config = GlassboxConfig()
        assert set(config.signals) == set(GlassboxConfig.signal_names())
        assert all(isinstance(c, SignalConfigBase) for c in config.signals.values())

    def test_config_survives_process_boundaries(self):
        """vLLM pickles and deepcopies the config into its spawned subprocesses.

        Regression: caching ``signals`` as a ``MappingProxyType`` broke both, and because
        the validator reads ``signals`` the cache is populated on every single instance.
        """
        restored = pickle.loads(pickle.dumps(GlassboxConfig()))
        assert set(restored.signals) == set(GlassboxConfig.signal_names())
        assert set(copy.deepcopy(GlassboxConfig()).signals) == set(restored.signals)

    @pytest.mark.parametrize(
        ("mixin", "fields"),
        [
            (SVDParams, ("rank", "method")),
            (ThresholdParams, ("threshold", "block_size")),
            (CausalMode, ("causal",)),
            (StreamingMode, ("streaming",)),
            (IncrementalMode, ("incremental",)),
        ],
    )
    def test_capability_fields_are_never_hand_declared(self, mixin, fields):
        """A config carrying a capability's field must inherit that capability's mixin.

        Hand-declaring the field instead works fine as far as Python and pydantic are
        concerned — which is precisely the hazard, because the generic code keys off the
        *type*, not the field:

        - ``_check_window_modes`` tests ``isinstance(cfg, StreamingMode)``, so a
          hand-declared ``streaming`` is silently never guarded: ``streaming=True`` over a
          sliding window would be accepted and report a biased statistic.
        - a hand-declared ``rank`` / ``threshold`` would miss the mixin's shared bounds and
          drop out of ``SVD_SIGNALS`` / ``THRESHOLD_SIGNALS``.

        Only this direction is asserted. The converse — inherit the mixin, get its fields —
        is just how inheritance works, and testing it would say nothing about this design.
        """
        config = GlassboxConfig()
        for name in GlassboxConfig.signal_names():
            cfg_cls = type(getattr(config, name))
            for field in fields:
                if field in cfg_cls.model_fields:
                    assert issubclass(cfg_cls, mixin), (
                        f"{name} declares {field!r} without inheriting "
                        f"{mixin.__name__}; generic code tests issubclass and would "
                        f"silently skip it"
                    )

    def test_causal_mode_tracks_the_post_softmax_operator(self):
        """CausalMode is carried by exactly the post-softmax signals.

        The pre-softmax three omit it for two different reasons (docs/operator-choice.md):
        ``spectral`` reads raw S = QKᵀ, which is never causally masked — the mask is
        applied inside the softmax — so there is no mask to configure; ``cyclic`` and
        ``magnetic`` additionally *must not* be masked, since a causal tournament is
        transitive and their statistic would be identically zero.
        """
        pre_softmax = {"spectral", "cyclic", "magnetic"}
        assert (
            set(GlassboxConfig.signal_names()) - GlassboxConfig.signals_with(CausalMode)
            == pre_softmax
        )

    @pytest.mark.parametrize("sig", sorted(THRESHOLD_SIGNALS))
    @pytest.mark.parametrize(("field", "bad"), [("block_size", 0), ("threshold", -1)])
    def test_two_tier_bounds_enforced_uniformly(self, sig, field, bad):
        """Every two-tier signal rejects the same out-of-range values.

        Regression: these bounds used to be hand-declared and only routing/asymmetry had
        them — magnetic/tracker/selfattn/laplacian accepted block_size=0 (which raises in
        ``range()`` on the matrix-free path) and negative thresholds.
        """
        with pytest.raises(pydantic.ValidationError):
            GlassboxConfig.from_cli_args(signals=(sig,), **{field: bad})
