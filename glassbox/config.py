from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoresMatrixConfig(BaseModel):
    """SVD of pre-softmax scores matrix S = QK^T."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]


class DegreeNormalizedMatrixConfig(BaseModel):
    """SVD of post-softmax degree-normalized operator M = D_Q^{-1/2} A D_K^{-1/2}."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]
    threshold: int = 2048
    block_size: int = 256
    hodge_target_cv: float = 0.05
    hodge_curl_seed: int = 42


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

    scores_matrix: ScoresMatrixConfig = ScoresMatrixConfig()
    degree_normalized_matrix: DegreeNormalizedMatrixConfig = DegreeNormalizedMatrixConfig()
    output: str | None = None

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
