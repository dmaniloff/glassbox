from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Literal

import click
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

# SIGNAL_NAMES / SVD_SIGNALS / THRESHOLD_SIGNALS are DERIVED from the GlassboxConfig
# field annotations at the bottom of this module (see ``GlassboxConfig.signal_names``
# and ``GlassboxConfig.signals_with``).  Registering a signal is therefore two edits --
# declare its config class, add the field -- and every generic loop picks it up.


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


# --- Capability mixins -------------------------------------------------------------
#
# A signal opts into a capability by inheriting the mixin carrying its fields.  Generic
# code then tests ``issubclass``/``isinstance`` instead of consulting a hand-maintained
# set of signal names, so a capability and its parameters cannot drift apart.  Two
# families, distinguished by suffix:
#
#   ``*Params`` -- algorithm knobs.
#   ``*Mode``   -- opt-in behaviours.  StreamingMode/IncrementalMode are policed by
#                  ``validate_window_modes``; CausalMode is a plain algorithm switch.


class SVDParams(BaseModel):
    """Truncated-SVD knobs, for signals whose statistic is a singular triplet.

    ``rank`` is bounded because a negative rank silently drops singular values rather
    than failing.
    """

    rank: int = Field(4, ge=1)
    method: Literal["randomized", "lanczos"] = "randomized"


class ThresholdParams(BaseModel):
    """Two-tier crossover: materialize the L x L operator for ``L <= threshold``, use
    the blocked matrix-free path above it.

    Bounds guard against crashes / silent garbage: ``block_size=0`` raises in ``range()``
    on the matrix-free path, and a negative ``threshold`` forces the noisy path for all L.

    Crossover ~512 on NVIDIA A10G (bench_hodge.py, 2026-03-24, d=64, rank=4):
      L=256: mat 21ms vs mf 39ms (1.8x), L=512: 54ms vs 61ms (1.1x),
      L=1024: 174ms vs 110ms (0.6x). Materialized dominated by svdvals ~L^1.6.
    """

    threshold: int = Field(512, ge=0)
    block_size: int = Field(256, ge=1)


class CausalMode(BaseModel):
    """Apply the causal mask when forming the operator.

    Only for signals reading the post-softmax operator.  The orientation signals
    (``cyclic``, ``magnetic``) deliberately omit this mixin: they live on the UNMASKED
    pre-softmax scores, where causal masking would make them vacuous.  ``False`` is
    meaningful for encoder / cross-attention.  See docs/operator-choice.md.
    """

    causal: bool = True


class StreamingMode(BaseModel):
    """Opt into *block-diagonal global* accumulation (docs/streaming-modes.md).

    Per-window sufficient statistics are summed into one global number, which is
    unbiased ONLY over disjoint (tumbling) windows -- enforced by
    ``validate_window_modes``.  Sound only for ADDITIVE statistics (Frobenius
    sums-of-squares); spectral and combinatorial statistics have no valid
    block-diagonal mode at all.

    (The doc names this mode "block-diagonal global"; note that "streaming" is also its
    umbrella term for all four modes, of which this is one.)
    """

    streaming: bool = False


class IncrementalMode(BaseModel):
    """Opt into *exact-full global* streaming (docs/streaming-modes.md).

    Running state is maintained across fires and only the delta tokens are folded per
    fire (an O(delta) update), reproducing the exact full-sequence statistic.  Requires
    the unbounded buffer -- enforced by ``validate_window_modes`` -- because a bounded
    buffer trims priors and breaks exactness.
    """

    incremental: bool = False


class SpectralConfig(SVDParams, SignalConfigBase):
    """SVD of pre-softmax scores matrix S = QK^T."""

    enabled: bool = True


class RoutingConfig(SVDParams, ThresholdParams, CausalMode, SignalConfigBase):
    """SVD of post-softmax degree-normalized operator M = D_Q^{-1/2} A D_K^{-1/2}."""

    # Seed for the matrix-free commutator-norm Hutchinson estimator.
    hodge_seed: int = 42


class CyclicTrianglesConfig(IncrementalMode, SignalConfigBase):
    """Cyclic-triangle count |T_cyc| of the pre-softmax sign tournament ω(QKᵀ).

    Operates on the UNMASKED pre-softmax scores S = QKᵀ (NOT post-softmax — a causal
    post-softmax tournament is transitive ⇒ |T_cyc| = 0; see docs/operator-choice.md). The
    count is exact (Kendall identity); no threshold/estimation. Under ``incremental`` the
    out-degree vector + running count are maintained across fires and only the delta tokens
    are folded per fire (the O(ΔE) streaming update).
    """


class MagneticConfig(ThresholdParams, IncrementalMode, SignalConfigBase):
    """Magnetic-Laplacian frustration λ₁ of the pre-softmax tournament ω(QKᵀ).

    Operates on the UNMASKED pre-softmax scores S = QKᵀ (NOT post-softmax — a causal tournament
    is transitive ⇒ λ₁ = 0; see docs/operator-choice.md). Dense Hermitian eig for L ≤ threshold,
    complex-Hermitian Lanczos (which="smallest") above. The construction (L_φ = D − A⊙e^{iθ},
    W=(|S_ij|+|S_ji|)/2, θ=arctan((S_ij−S_ji)/(S_ij+S_ji))); see *directed-attention-geometry*.

    Under ``incremental`` it reports the streamable phase-curl frustration energy (Hodge curl
    of θ via the row-sum identity, eigensolver-free) maintained across fires, instead of the
    dense λ₁ — the exact full-sequence frustration energy. See issue #68.
    """


class AsymmetryConfig(
    ThresholdParams, CausalMode, StreamingMode, IncrementalMode, SignalConfigBase
):
    """Asymmetry coefficient G = ||P_asym||_F / ||P||_F of row-stochastic attention P.

    Hodge G signal.  Computed on the post-softmax attention P (NOT the degree-normalized
    M — see docs/operator-choice.md).  Matrix-free Hutchinson estimator (Route B, direct
    ||P_asym z||^2) above ``threshold``, exact materialized below.  Under ``streaming`` the
    per-window sufficient statistics (||P_asym||_F^2, ||P||_F^2) are accumulated into a
    global G.  Under ``incremental`` it is the exact full-operator G, folding only delta
    tokens per fire — O(1) scalars but an O(N) row-sum vector r per (layer, head); see
    ``_incremental_reduce``.
    """

    # n_hutchinson=0 would divide by zero.
    n_hutchinson: int = Field(32, ge=1)
    seed: int = 42


class TrackerConfig(SVDParams, ThresholdParams, CausalMode, SignalConfigBase):
    """Features from raw post-softmax attention A (AttentionTracker, arXiv:2411.00348)."""


class SelfAttnConfig(ThresholdParams, CausalMode, SignalConfigBase):
    """Attention diagonal features (LLM-Check, NeurIPS 2024 + LapEigvals, EMNLP 2025)."""

    top_k: int = 10


class LaplacianConfig(ThresholdParams, CausalMode, SignalConfigBase):
    """Laplacian eigenvalues from attention graphs (LapEigvals, EMNLP 2025)."""

    top_k: int = 10


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
    asymmetry: AsymmetryConfig = AsymmetryConfig()
    cyclic: CyclicTrianglesConfig = CyclicTrianglesConfig()
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

    @classmethod
    def signal_names(cls) -> tuple[str, ...]:
        """Names of the per-signal config fields, in declaration order.

        Derived from the field annotations, so this is the single source of truth for
        "which signals exist" -- declaring the field is enough to register one.
        """
        return tuple(
            name
            for name, f in cls.model_fields.items()
            if isinstance(f.annotation, type) and issubclass(f.annotation, SignalConfigBase)
        )

    @classmethod
    def signals_with(cls, capability: type[BaseModel]) -> frozenset[str]:
        """Names of the signals whose config inherits ``capability`` (a mixin above)."""
        return frozenset(
            name
            for name in cls.signal_names()
            if issubclass(cls.model_fields[name].annotation, capability)  # type: ignore[arg-type]
        )

    @cached_property
    def signals(self) -> Mapping[str, SignalConfigBase]:
        """Per-signal configs as a name -> config mapping.

        The typed iteration surface for "do X for every signal" -- prefer it to
        ``getattr(config, name)``.  Cached because the backend consults it per layer per
        forward step (~1us to rebuild, ~1% of decode at 32 layers); safe to cache because
        the model is frozen.

        Declared ``Mapping`` so type checkers reject mutation.  It is deliberately a plain
        dict at runtime rather than a ``MappingProxyType``: the validator populates this
        cache on every instance, and vLLM both pickles the config into its spawned
        subprocesses and deepcopies it -- neither of which accepts a ``mappingproxy``.
        """
        return {name: getattr(self, name) for name in self.signal_names()}

    @model_validator(mode="after")
    def _check_window_modes(self) -> GlassboxConfig:
        """Reject mode<->windowing combinations that would silently mis-report a statistic.

        A signal is guarded iff it inherits ``StreamingMode`` / ``IncrementalMode``, so
        opting into a mode and being policed for it are the same act -- there is no way to
        add the flag but miss the check.  See docs/streaming-modes.md for the sound matrix.
        """
        modes = [
            (
                name,
                cfg.enabled,
                isinstance(cfg, StreamingMode) and cfg.streaming,
                isinstance(cfg, IncrementalMode) and cfg.incremental,
            )
            for name, cfg in self.signals.items()
        ]
        validate_window_modes(modes, self.q_buffer_mode, self.q_buffer_max_tokens)
        return self

    # Forward-matvec strategy for the matrix-free SVD paths (routing/tracker/spectral).
    # "loop"/"batched" both use the blocked PyTorch matvecs in svd.py (already batched +
    # fused); "triton" uses the optional fused online-softmax kernel (CUDA only, never
    # materializes the L×L scores); "auto" resolves to "triton" iff available else "batched".
    matvec_strategy: Literal["loop", "batched", "triton", "auto"] = "auto"

    @classmethod
    def resolve_matvec_strategy(cls, strategy: str) -> Literal["loop", "batched", "triton"]:
        """Resolve "auto" to a concrete strategy from runtime capabilities.

        "auto" -> "triton" only when Triton is importable AND a CUDA device is present;
        otherwise "batched" (the blocked PyTorch path). Explicit strategies pass through.
        """
        if strategy != "auto":
            return strategy  # type: ignore[return-value]
        try:
            import torch

            from glassbox.triton_kernels import HAS_TRITON

            if HAS_TRITON and torch.cuda.is_available():
                return "triton"
        except ImportError:  # pragma: no cover
            pass
        return "batched"

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

        # Each flag lands only on the signals whose config actually declares that field, so
        # "which flag applies where" is answered by the config classes themselves rather
        # than by a parallel set of signal names.  Note ``is not None``: threshold=0 is
        # meaningful (forces matrix-free for every L) and must not be dropped as falsy.
        requested: dict = {
            "interval": interval,
            "rank": rank,
            "method": method,
            "threshold": threshold,
            "block_size": block_size,
            "heads": list(heads) if heads else None,
        }

        for sig_name in cls.signal_names():
            sig_dict: dict = {"enabled": sig_name in signal_set}
            if sig_name in signal_set:
                fields = cls.model_fields[sig_name].annotation.model_fields  # type: ignore[union-attr]
                sig_dict.update(
                    {k: v for k, v in requested.items() if v is not None and k in fields}
                )
            overrides[sig_name] = sig_dict

        return cls(**overrides)


# Derived registries.  These are the public constants callers import; they are computed
# from the GlassboxConfig field annotations rather than hand-listed, so they cannot drift
# out of sync with the config classes they describe.
SIGNAL_NAMES: tuple[str, ...] = GlassboxConfig.signal_names()
SVD_SIGNALS: frozenset[str] = GlassboxConfig.signals_with(SVDParams)
THRESHOLD_SIGNALS: frozenset[str] = GlassboxConfig.signals_with(ThresholdParams)
