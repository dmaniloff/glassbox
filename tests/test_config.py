import textwrap

from glassbox.config import GlassboxConfig


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
    assert config.routing.hodge_confidence == 0.95
    assert config.routing.hodge_pilot_size == 100
    assert config.routing.hodge_min_samples == 200
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


def test_from_cli_args_output_otel():
    config = GlassboxConfig.from_cli_args(output_path="/tmp/out.jsonl", otel=True)
    assert config.output.path == "/tmp/out.jsonl"
    assert config.emit.otel is True


def test_from_cli_args_yaml_auto_load(tmp_path, monkeypatch):
    (tmp_path / "glassbox.yaml").write_text(
        textwrap.dedent("""\
        routing:
          enabled: true
          hodge_target_cv: 0.1
    """)
    )
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig.from_cli_args(
        signals=("spectral", "routing"),
        rank=2,
    )
    assert config.routing.enabled is True
    assert config.routing.rank == 2  # CLI args beat YAML
    assert config.routing.hodge_target_cv == 0.1  # from YAML


def test_from_cli_args_svd_not_set_on_non_svd_signals():
    config = GlassboxConfig.from_cli_args(signals=("selfattn",), rank=8)
    # rank shouldn't propagate to selfattn (not an SVD signal)
    assert config.selfattn.enabled is True
    # selfattn has no rank field — defaults unchanged
    assert config.spectral.rank == 4  # default, not 8 (spectral disabled)
