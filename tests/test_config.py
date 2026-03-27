import pytest
from pydantic import ValidationError

from glassbox.config import GlassboxConfig


def test_defaults():
    config = GlassboxConfig()
    assert config.scores_matrix.enabled is True
    assert config.degree_normalized_matrix.enabled is False
    assert config.scores_matrix.interval == 32
    assert config.scores_matrix.rank == 4
    assert config.scores_matrix.method == "randomized"
    assert config.scores_matrix.heads == [0]
    assert config.degree_normalized_matrix.interval == 32
    assert config.degree_normalized_matrix.threshold == 512
    assert config.degree_normalized_matrix.hodge_confidence == 0.95
    assert config.degree_normalized_matrix.hodge_pilot_size == 100
    assert config.degree_normalized_matrix.hodge_min_samples == 200
    assert config.output is None


def test_programmatic_kwargs():
    config = GlassboxConfig(scores_matrix={"interval": 16})
    assert config.scores_matrix.interval == 16
    assert config.scores_matrix.rank == 4  # default preserved


def test_programmatic_kwargs_degree_normalized():
    config = GlassboxConfig(
        degree_normalized_matrix={"enabled": True, "threshold": 1024}
    )
    assert config.degree_normalized_matrix.enabled is True
    assert config.degree_normalized_matrix.threshold == 1024
    assert config.degree_normalized_matrix.rank == 4  # default preserved


def test_attention_tracker_defaults():
    config = GlassboxConfig()
    assert config.attention_tracker.enabled is False
    assert config.attention_tracker.interval == 32
    assert config.attention_tracker.rank == 4
    assert config.attention_tracker.method == "randomized"
    assert config.attention_tracker.heads == [0]
    assert config.attention_tracker.threshold == 512
    assert config.attention_tracker.block_size == 256


def test_programmatic_kwargs_attention_tracker():
    config = GlassboxConfig(
        attention_tracker={"enabled": True, "interval": 16, "threshold": 256}
    )
    assert config.attention_tracker.enabled is True
    assert config.attention_tracker.interval == 16
    assert config.attention_tracker.threshold == 256
    assert config.attention_tracker.rank == 4  # default preserved


def test_yaml_loading(tmp_path, monkeypatch):
    yaml_content = "scores_matrix:\n  interval: 16\n  rank: 8\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.scores_matrix.interval == 16
    assert config.scores_matrix.rank == 8


def test_yaml_attention_tracker(tmp_path, monkeypatch):
    yaml_content = (
        "attention_tracker:\n"
        "  enabled: true\n"
        "  interval: 64\n"
        "  threshold: 1024\n"
    )
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.attention_tracker.enabled is True
    assert config.attention_tracker.interval == 64
    assert config.attention_tracker.threshold == 1024


def test_yaml_degree_normalized(tmp_path, monkeypatch):
    yaml_content = (
        "degree_normalized_matrix:\n"
        "  enabled: true\n"
        "  interval: 64\n"
        "output: /var/log/glassbox/signals.jsonl\n"
    )
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.degree_normalized_matrix.enabled is True
    assert config.degree_normalized_matrix.interval == 64
    assert config.output == "/var/log/glassbox/signals.jsonl"


def test_precedence_kwargs_beat_yaml(tmp_path, monkeypatch):
    yaml_content = "scores_matrix:\n  interval: 16\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig(scores_matrix={"interval": 8})
    assert config.scores_matrix.interval == 8


def test_frozen_nested():
    config = GlassboxConfig()
    with pytest.raises(ValidationError):
        config.scores_matrix.interval = 99


def test_frozen_root():
    config = GlassboxConfig()
    with pytest.raises(ValidationError):
        config.output = "foo.jsonl"


# --- matvec_strategy ---


def test_matvec_strategy_default():
    config = GlassboxConfig()
    assert config.matvec_strategy == "auto"


def test_matvec_strategy_explicit():
    for val in ("loop", "batched", "triton"):
        config = GlassboxConfig(matvec_strategy=val)
        assert config.matvec_strategy == val


def test_matvec_strategy_invalid():
    with pytest.raises(ValidationError):
        GlassboxConfig(matvec_strategy="invalid")


def test_matvec_strategy_yaml(tmp_path, monkeypatch):
    (tmp_path / "glassbox.yaml").write_text("matvec_strategy: batched\n")
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.matvec_strategy == "batched"


def test_resolve_matvec_strategy_passthrough():
    for val in ("loop", "batched", "triton"):
        assert GlassboxConfig.resolve_matvec_strategy(val) == val


def test_resolve_matvec_strategy_auto():
    result = GlassboxConfig.resolve_matvec_strategy("auto")
    assert result in ("batched", "triton")
