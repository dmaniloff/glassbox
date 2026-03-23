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
    assert config.degree_normalized_matrix.threshold == 2048  # deprecated but still present
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


def test_yaml_loading(tmp_path, monkeypatch):
    yaml_content = "scores_matrix:\n  interval: 16\n  rank: 8\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.scores_matrix.interval == 16
    assert config.scores_matrix.rank == 8


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
