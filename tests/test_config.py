import pytest
from pydantic import ValidationError

from glassbox.config import GlassboxConfig


def test_defaults():
    config = GlassboxConfig()
    assert config.spectral.enabled is True
    assert config.degree_normalized.enabled is False
    assert config.spectral.interval == 32
    assert config.spectral.rank == 4
    assert config.spectral.method == "randomized"
    assert config.spectral.heads == [0]
    assert config.degree_normalized.interval == 32
    assert config.degree_normalized.threshold == 2048
    assert config.output is None


def test_programmatic_kwargs():
    config = GlassboxConfig(spectral={"interval": 16})
    assert config.spectral.interval == 16
    assert config.spectral.rank == 4  # default preserved


def test_programmatic_kwargs_degree_normalized():
    config = GlassboxConfig(
        degree_normalized={"enabled": True, "hodge": True, "threshold": 1024}
    )
    assert config.degree_normalized.enabled is True
    assert config.degree_normalized.hodge is True
    assert config.degree_normalized.threshold == 1024
    assert config.degree_normalized.rank == 4  # default preserved


def test_yaml_loading(tmp_path, monkeypatch):
    yaml_content = "spectral:\n  interval: 16\n  rank: 8\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.spectral.interval == 16
    assert config.spectral.rank == 8


def test_yaml_degree_normalized(tmp_path, monkeypatch):
    yaml_content = (
        "degree_normalized:\n"
        "  enabled: true\n"
        "  interval: 64\n"
        "  hodge: true\n"
        "output: /var/log/glassbox/signals.jsonl\n"
    )
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.degree_normalized.enabled is True
    assert config.degree_normalized.interval == 64
    assert config.degree_normalized.hodge is True
    assert config.output == "/var/log/glassbox/signals.jsonl"


def test_precedence_kwargs_beat_yaml(tmp_path, monkeypatch):
    yaml_content = "spectral:\n  interval: 16\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig(spectral={"interval": 8})
    assert config.spectral.interval == 8


def test_frozen_nested():
    config = GlassboxConfig()
    with pytest.raises(ValidationError):
        config.spectral.interval = 99


def test_frozen_root():
    config = GlassboxConfig()
    with pytest.raises(ValidationError):
        config.output = "foo.jsonl"
