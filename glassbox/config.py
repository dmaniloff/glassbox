from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical signal names (user-facing)
SIGNAL_NAMES: list[str] = ["spectral", "routing", "tracker", "selfattn", "laplacian"]

# Signals that use SVD rank/method
SVD_SIGNALS: set[str] = {"spectral", "routing", "tracker"}

# Signals that use threshold/block_size (materialized vs matrix-free two-tier)
THRESHOLD_SIGNALS: set[str] = {"routing", "tracker", "selfattn", "laplacian"}


class SpectralConfig(BaseModel):
    """SVD of pre-softmax scores matrix S = QK^T."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]


class RoutingConfig(BaseModel):
    """SVD of post-softmax degree-normalized operator M = D_Q^{-1/2} A D_K^{-1/2}."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]
    # Materialize M for L <= threshold, matrix-free above.
    # Crossover ~512 on NVIDIA A10G (bench_hodge.py, 2026-03-24, d=64, rank=4):
    #   L=256: mat 21ms vs mf 39ms (1.8x), L=512: 54ms vs 61ms (1.1x),
    #   L=1024: 174ms vs 110ms (0.6x). Materialized dominated by svdvals ~L^1.6.
    threshold: int = 512
    block_size: int = 256
    causal: bool = True
    hodge_target_cv: float = 0.05
    hodge_curl_seed: int = 42
    hodge_confidence: float = 0.95
    hodge_pilot_size: int = 100
    hodge_min_samples: int = 200


class TrackerConfig(BaseModel):
    """Features from raw post-softmax attention A (AttentionTracker, arXiv:2411.00348)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class SelfAttnConfig(BaseModel):
    """Attention diagonal features (LLM-Check, NeurIPS 2024 + LapEigvals, EMNLP 2025)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    heads: list[int] = [0]
    top_k: int = 10
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class LaplacianConfig(BaseModel):
    """Laplacian eigenvalues from attention graphs (LapEigvals, EMNLP 2025)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    heads: list[int] = [0]
    top_k: int = 10
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class GlassboxConfig(BaseSettings):
    """Root configuration for the Glassbox observability framework.

    Precedence (highest wins):
      1. Programmatic kwargs
      2. YAML config file (glassbox.yaml)
      3. Field defaults
    """

    model_config = SettingsConfigDict(
        yaml_file="glassbox.yaml",
        extra="ignore",
        frozen=True,
    )

    spectral: SpectralConfig = SpectralConfig()
    routing: RoutingConfig = RoutingConfig()
    tracker: TrackerConfig = TrackerConfig()
    selfattn: SelfAttnConfig = SelfAttnConfig()
    laplacian: LaplacianConfig = LaplacianConfig()
    output: str | None = None
    otel: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        from pydantic_settings import YamlConfigSettingsSource

        # config sources and their precedence
        return (
            init_settings,  # 1. programmatic kwargs
            YamlConfigSettingsSource(settings_cls),  # 2. glassbox.yaml
        )
