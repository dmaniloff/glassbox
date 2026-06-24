from __future__ import annotations

from typing import Literal

import click
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

# Canonical signal names (user-facing)
SIGNAL_NAMES: list[str] = ["spectral", "routing", "magnetic", "tracker", "selfattn", "laplacian"]

# Signals that use SVD rank/method
SVD_SIGNALS: set[str] = {"spectral", "routing", "tracker"}

# Signals that use threshold/block_size (materialized vs matrix-free two-tier)
THRESHOLD_SIGNALS: set[str] = {"routing", "magnetic", "tracker", "selfattn", "laplacian"}


def validate_window_modes(
    modes: list[tuple[str, bool, bool, bool]],
    q_buffer_mode: str,
    q_buffer_max_tokens: int,
) -> None:
    """Enforce mode <-> windowing invariants so a streaming statistic is never silently wrong.

    ``modes`` is a list of ``(signal_name, enabled, streaming, incremental)``. The soundness
    of each global streaming mode depends on the window (see docs/streaming-modes.md):

    - ``streaming=True`` (block-diagonal global accumulation) is unbiased ONLY over DISJOINT
      windows, so it requires ``q_buffer_mode="tumbling"`` with ``q_buffer_max_tokens > 0``.
      Sliding/overlapping windows double-count the overlap; an unbounded buffer is not a
      block-diagonal partition. Only additive statistics (e.g. Frobenius sums-of-squares)
      may set it at all.
    - ``incremental=True`` (exact full-operator streaming) requires the UNBOUNDED
      full-sequence buffer ``q_buffer_max_tokens == 0``; a bounded buffer trims priors and
      breaks exactness.

    Also: ``q_buffer_mode="tumbling"`` is meaningless without a finite window, so it requires
    ``q_buffer_max_tokens > 0`` regardless of signals.

    Raises ``ValueError`` on any unsound combination.
    """
    if q_buffer_mode == "tumbling" and q_buffer_max_tokens <= 0:
        raise ValueError(
            "q_buffer_mode='tumbling' requires a finite window (q_buffer_max_tokens > 0); "
            "got q_buffer_max_tokens=0 (unbounded). Tumbling = non-overlapping fixed windows."
        )
    for name, enabled, streaming, incremental in modes:
        if not enabled:
            continue
        if streaming and not (q_buffer_mode == "tumbling" and q_buffer_max_tokens > 0):
            raise ValueError(
                f"{name}.streaming=True (block-diagonal global accumulation) requires disjoint "
                f"windows: q_buffer_mode='tumbling' and q_buffer_max_tokens>0; got "
                f"q_buffer_mode={q_buffer_mode!r}, q_buffer_max_tokens={q_buffer_max_tokens}. "
                "Sliding windows double-count overlap; an unbounded buffer is not block-diagonal. "
                "See docs/streaming-modes.md."
            )
        if incremental and q_buffer_max_tokens != 0:
            raise ValueError(
                f"{name}.incremental=True (exact full-operator streaming) requires the unbounded "
                f"buffer q_buffer_max_tokens=0; got {q_buffer_max_tokens}. A bounded buffer trims "
                "priors and breaks exactness. See docs/streaming-modes.md."
            )


def parse_signal_names(ctx, param, value):
    """Click callback: parse --signal values (repeatable or comma-separated)."""
    if not value:
        return ("spectral",)
    result = []
    for v in value:
        for part in v.split(","):
            part = part.strip()
            if part not in SIGNAL_NAMES:
                raise click.BadParameter(
                    f"Unknown signal {part!r}. Choose from: {', '.join(SIGNAL_NAMES)}"
                )
            result.append(part)
    return tuple(result)


class SignalConfigBase(BaseModel):
    """Orchestration fields shared by every signal config.

    Subclasses add their algorithm-specific parameters. The backend strips
    these orchestration fields (via ``model_dump(exclude=...)``) before
    constructing the corresponding Diagnostic, so any field declared here is
    automatically kept out of the diagnostic constructor.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    heads: list[int] = [0]


class SpectralConfig(SignalConfigBase):
    """SVD of pre-softmax scores matrix S = QK^T."""

    enabled: bool = True
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"


class RoutingConfig(SignalConfigBase):
    """SVD of post-softmax degree-normalized operator M = D_Q^{-1/2} A D_K^{-1/2}."""

    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    # Materialize M for L <= threshold, matrix-free above.
    # Crossover ~512 on NVIDIA A10G (bench_hodge.py, 2026-03-24, d=64, rank=4):
    #   L=256: mat 21ms vs mf 39ms (1.8x), L=512: 54ms vs 61ms (1.1x),
    #   L=1024: 174ms vs 110ms (0.6x). Materialized dominated by svdvals ~L^1.6.
    threshold: int = 512
    block_size: int = 256
    causal: bool = True
    # Seed for the matrix-free commutator-norm Hutchinson estimator.
    hodge_seed: int = 42


class MagneticConfig(SignalConfigBase):
    """Magnetic-Laplacian frustration λ₁ of the pre-softmax tournament ω(QKᵀ).

    Operates on the UNMASKED pre-softmax scores S = QKᵀ (NOT post-softmax — a causal tournament
    is transitive ⇒ λ₁ = 0; see docs/operator-choice.md). Dense Hermitian eig for L ≤ threshold,
    complex-Hermitian Lanczos (which="smallest") above. The construction (L_φ = D − A⊙e^{iθ},
    W=(|S_ij|+|S_ji|)/2, θ=arctan((S_ij−S_ji)/(S_ij+S_ji))) is formally verified in shade-formal.
    """

    threshold: int = 512
    block_size: int = 256
    # incremental: report the streamable phase-curl frustration energy (Hodge curl of θ via the
    # row-sum identity, eigensolver-free) maintained across fires, instead of the dense λ₁.
    # Exact full-sequence frustration energy; requires the unbounded buffer. See issue #68.
    incremental: bool = False


class TrackerConfig(SignalConfigBase):
    """Features from raw post-softmax attention A (AttentionTracker, arXiv:2411.00348)."""

    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class SelfAttnConfig(SignalConfigBase):
    """Attention diagonal features (LLM-Check, NeurIPS 2024 + LapEigvals, EMNLP 2025)."""

    top_k: int = 10
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class LaplacianConfig(SignalConfigBase):
    """Laplacian eigenvalues from attention graphs (LapEigvals, EMNLP 2025)."""

    top_k: int = 10
    threshold: int = 512
    block_size: int = 256
    causal: bool = True


class OutputConfig(BaseModel):
    """Feature logging pipeline — write full snapshots for training/analysis."""

    model_config = ConfigDict(frozen=True)

    path: str | None = None


class EmitConfig(BaseModel):
    """Inference pipeline — real-time signal emission for live monitoring."""

    model_config = ConfigDict(frozen=True)

    otel: bool = False


class GlassboxConfig(BaseSettings):
    """Root configuration for the Glassbox observability framework.

    Precedence (highest → lowest):
      Programmatic kwargs > glassbox.yaml in cwd > field defaults
    """

    model_config = SettingsConfigDict(
        yaml_file="glassbox.yaml",
        extra="ignore",
        frozen=True,
    )

    spectral: SpectralConfig = SpectralConfig()
    routing: RoutingConfig = RoutingConfig()
    magnetic: MagneticConfig = MagneticConfig()
    tracker: TrackerConfig = TrackerConfig()
    selfattn: SelfAttnConfig = SelfAttnConfig()
    laplacian: LaplacianConfig = LaplacianConfig()
    output: OutputConfig = OutputConfig()
    emit: EmitConfig = EmitConfig()
    emit_witness: bool = False

    # Q-buffer windowing — bounds memory and enables streaming diagnostics.
    # 0 = unbounded (full sequence), > 0 = max tokens retained per layer.
    q_buffer_max_tokens: int = 0

    # "sliding": overlapping windows, trim oldest on every step, fire per
    #   signal interval.  Window overlap = W - interval.
    # "tumbling": non-overlapping windows — accumulate W tokens, fire all
    #   enabled signals, flush.  Window independence simplifies accumulation
    #   proofs for streaming local→global merges.
    q_buffer_mode: Literal["sliding", "tumbling"] = "sliding"

    @model_validator(mode="after")
    def _check_window_modes(self) -> GlassboxConfig:
        """Reject mode<->windowing combinations that would silently mis-report a statistic.

        Generic over SIGNAL_NAMES via getattr, so it is a no-op for signals without
        streaming modes and automatically guards any signal that adds ``streaming`` /
        ``incremental`` flags. See docs/streaming-modes.md for the sound-mode matrix.
        """
        modes = [
            (
                name,
                bool(getattr(getattr(self, name), "enabled", False)),
                bool(getattr(getattr(self, name), "streaming", False)),
                bool(getattr(getattr(self, name), "incremental", False)),
            )
            for name in SIGNAL_NAMES
        ]
        validate_window_modes(modes, self.q_buffer_mode, self.q_buffer_max_tokens)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # config sources and their precedence
        return (
            init_settings,  # 1. programmatic kwargs
            YamlConfigSettingsSource(settings_cls),  # 2. glassbox.yaml
        )

    @classmethod
    def from_cli_args(
        cls,
        *,
        signals: tuple[str, ...] = ("spectral",),
        interval: int | None = None,
        rank: int | None = None,
        method: str | None = None,
        heads: tuple[int, ...] | list[int] = (),
        threshold: int | None = None,
        block_size: int | None = None,
        output_path: str | None = None,
        otel: bool | None = None,
        q_buffer_max_tokens: int | None = None,
        q_buffer_mode: str | None = None,
    ) -> GlassboxConfig:
        """Build a GlassboxConfig from CLI-style arguments.

        Precedence (highest → lowest):
          keyword args here > glassbox.yaml in cwd > field defaults

        Signals in the *signals* tuple are enabled; all others are
        explicitly disabled.
        """
        overrides: dict = {}

        if output_path is not None:
            overrides["output"] = {"path": output_path}
        if otel is not None:
            overrides["emit"] = {"otel": otel}
        if q_buffer_max_tokens is not None:
            overrides["q_buffer_max_tokens"] = q_buffer_max_tokens
        if q_buffer_mode is not None:
            overrides["q_buffer_mode"] = q_buffer_mode

        signal_set = set(signals)

        for sig_name in SIGNAL_NAMES:
            sig_dict: dict = {"enabled": sig_name in signal_set}

            if sig_name in signal_set:
                if interval is not None:
                    sig_dict["interval"] = interval
                if heads:
                    sig_dict["heads"] = list(heads)
                if sig_name in SVD_SIGNALS:
                    if rank is not None:
                        sig_dict["rank"] = rank
                    if method is not None:
                        sig_dict["method"] = method
                if sig_name in THRESHOLD_SIGNALS:
                    if threshold is not None:
                        sig_dict["threshold"] = threshold
                    if block_size is not None:
                        sig_dict["block_size"] = block_size

            overrides[sig_name] = sig_dict

        return cls(**overrides)
