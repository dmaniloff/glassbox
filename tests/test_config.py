from glassbox.config import GlassboxConfig


def test_defaults():
    config = GlassboxConfig()
    assert config.scores_matrix.enabled is True
    assert config.degree_normalized_matrix.enabled is False
    assert config.attention_diagonal.enabled is False
    assert config.scores_matrix.interval == 32
    assert config.scores_matrix.rank == 4
    assert config.scores_matrix.method == "randomized"
    assert config.scores_matrix.heads == [0]
    assert config.degree_normalized_matrix.interval == 32
    assert config.degree_normalized_matrix.threshold == 512
    assert config.degree_normalized_matrix.hodge_confidence == 0.95
    assert config.degree_normalized_matrix.hodge_pilot_size == 100
    assert config.degree_normalized_matrix.hodge_min_samples == 200
    assert config.attention_diagonal.interval == 32
    assert config.attention_diagonal.threshold == 512
    assert config.attention_diagonal.heads == [0]
    assert config.output is None


def test_programmatic_kwargs():
    config = GlassboxConfig(scores_matrix={"interval": 16})
    assert config.scores_matrix.interval == 16
    assert config.scores_matrix.rank == 4  # default preserved


def test_programmatic_kwargs_degree_normalized():
    config = GlassboxConfig(degree_normalized_matrix={"enabled": True, "threshold": 1024})
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
    config = GlassboxConfig(attention_tracker={"enabled": True, "interval": 16, "threshold": 256})
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
    yaml_content = "attention_tracker:\n  enabled: true\n  interval: 64\n  threshold: 1024\n"
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


def test_programmatic_kwargs_attention_diagonal():
    config = GlassboxConfig(attention_diagonal={"enabled": True, "interval": 16, "heads": [0, 1]})
    assert config.attention_diagonal.enabled is True
    assert config.attention_diagonal.interval == 16
    assert config.attention_diagonal.heads == [0, 1]
    assert config.attention_diagonal.threshold == 512  # default preserved


def test_yaml_attention_diagonal(tmp_path, monkeypatch):
    yaml_content = "attention_diagonal:\n  enabled: true\n  interval: 64\n  heads: [0, 2, 4]\n"
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.attention_diagonal.enabled is True
    assert config.attention_diagonal.interval == 64
    assert config.attention_diagonal.heads == [0, 2, 4]


def test_laplacian_eigvals_defaults():
    config = GlassboxConfig()
    assert config.laplacian_eigvals.enabled is False
    assert config.laplacian_eigvals.interval == 32
    assert config.laplacian_eigvals.heads == [0]
    assert config.laplacian_eigvals.top_k == 10
    assert config.laplacian_eigvals.threshold == 512
    assert config.laplacian_eigvals.block_size == 256


def test_programmatic_kwargs_laplacian_eigvals():
    config = GlassboxConfig(laplacian_eigvals={"enabled": True, "interval": 16, "top_k": 20})
    assert config.laplacian_eigvals.enabled is True
    assert config.laplacian_eigvals.interval == 16
    assert config.laplacian_eigvals.top_k == 20
    assert config.laplacian_eigvals.threshold == 512  # default preserved


def test_yaml_laplacian_eigvals(tmp_path, monkeypatch):
    yaml_content = (
        "laplacian_eigvals:\n  enabled: true\n  interval: 64\n  top_k: 25\n  heads: [0, 1, 2]\n"
    )
    (tmp_path / "glassbox.yaml").write_text(yaml_content)
    monkeypatch.chdir(tmp_path)
    config = GlassboxConfig()
    assert config.laplacian_eigvals.enabled is True
    assert config.laplacian_eigvals.interval == 64
    assert config.laplacian_eigvals.top_k == 25
    assert config.laplacian_eigvals.heads == [0, 1, 2]


