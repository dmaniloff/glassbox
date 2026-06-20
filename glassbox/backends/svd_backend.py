"""
Custom vLLM attention backend that wraps Triton attention to perform
matrix-free SVD of attention matrices at configurable intervals.

Usage:
    1. Import this module (triggers @register_backend)
    2. Launch vLLM with attention_backend="CUSTOM", enforce_eager=True

Configuration via GlassboxConfig (see glassbox/config.py):
    - glassbox.yaml (primary)
    - Programmatic kwargs
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import torch
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from glassbox.config import SIGNAL_NAMES, GlassboxConfig, SignalConfigBase
from glassbox.diagnostics import DIAGNOSTIC_REGISTRY
from glassbox.handlers import LoggingHandler, create_handlers_from_config
from glassbox.qbuffer import QBuffer
from glassbox.results import SVDSnapshot

logger = logging.getLogger(__name__)

# Matches layer index from various model naming conventions:
#   OPT/Qwen/Llama: "model.layers.0.self_attn"
#   GPT-2:          "transformer.h.0.attn.attn"
_LAYER_IDX_RE = re.compile(r"(?:layers|\.h)\.(\d+)")


@dataclass
class PerLayerState:
    """Per-layer streaming state for a single attention layer.

    Holds the windowed Q buffer (``qbuf``), the firing-cadence counter
    (``step``), and the per-signal ``accumulate()`` state (``accum``).
    """

    qbuf: QBuffer
    step: int = 0
    accum: dict[str, dict] = field(default_factory=dict)


@dataclass
class ReqTracker:
    """Tracks request boundaries across layers during prefill.

    request_id is incremented once per new request when layer 0 sees a
    prefill (q_span > 1).  Previous versions used an in_prefill flag that
    was cleared during decode, but with max_tokens=1 there is no decode
    step so the flag was never cleared and request_id stopped incrementing.
    """

    request_id: int = -1


@register_backend(AttentionBackendEnum.CUSTOM)
class SVDTritonAttentionBackend(TritonAttentionBackend):
    """Drop-in replacement for TritonAttentionBackend that runs matrix-free
    SVD on the scores matrix during decode."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SVDTritonAttentionImpl"]:
        return SVDTritonAttentionImpl


class SVDTritonAttentionImpl(TritonAttentionImpl):
    """Wraps TritonAttentionImpl.forward() to accumulate Q and periodically
    extract K from the paged cache for matrix-free SVD."""

    # Class-level config; immutable after creation.
    # If you need to override, set it before vLLM creates the engine.
    # vLLM controls the constructor signature so we can't pass it in.
    config: GlassboxConfig = GlassboxConfig()

    # Class-level layer state; shared mutable.
    # Shared by all impl instances (one per layer or shared).
    # Keyed by layer_name (e.g. "model.layers.0.self_attn").
    state_dict: dict[str, PerLayerState] = {}

    # Class-level request tracking; shared mutable.
    req_tracker: ReqTracker = ReqTracker()

    # Class-level snapshot handlers; shared mutable.
    # Same rationale as other class-level state: vLLM may create one attention
    # impl per layer (many instances of SVDTritonAttentionImpl).
    # Keeping one handler list on the class so every layer emits to the same sinks.
    _handlers: list = [LoggingHandler()]

    # Diagnostic instances, keyed by signal name. Built lazily or via set_config().
    _diagnostics: dict = {}

    # True after an explicit set_config() call.  The vLLM plugin checks this
    # to avoid overwriting programmatic config with defaults in subprocesses.
    _config_set_explicitly: bool = False

    @classmethod
    def set_config(cls, config: GlassboxConfig) -> None:
        """Set config and (re)initialise handlers, the diagnostics cache, and
        per-layer state.

        This is the single entry point for configuration — it keeps
        ``cls.config``, ``cls._handlers``, the ``cls._diagnostics`` cache, and
        the per-layer ``cls.state_dict`` (whose QBuffers capture the windowing
        policy at construction) in sync. Always use it; never assign
        ``cls.config`` directly, or those config-derived caches would go stale.

        vLLM forks/spawns subprocesses (API server, engine core, workers) and
        loads this plugin in *every* one of them. Setting ``_config_set_explicitly``
        here is what lets the plugin tell "config was set programmatically" from
        "use defaults": after a fork the child inherits the explicit config (and
        this flag), so the plugin must NOT re-run ``set_config(GlassboxConfig())``
        and clobber it with defaults. Call this before engine creation.
        """
        cls.config = config
        cls._config_set_explicitly = True
        for h in cls._handlers:
            h.close()
        cls._handlers = create_handlers_from_config(config)
        cls._build_diagnostics()
        # Drop per-layer state so QBuffers rebuild with the new windowing policy
        # (max_tokens/mode are captured at QBuffer construction). Mirrors the
        # diagnostics rebuild above; no-op in the normal "set_config before engine
        # creation" path since no state exists yet.
        cls.state_dict = {}

    @classmethod
    def _build_diagnostics(cls) -> None:
        """Instantiate Diagnostic objects from config for each signal."""
        cls._diagnostics = {}
        for sig_name, diag_cls in DIAGNOSTIC_REGISTRY.items():
            sig_cfg = getattr(cls.config, sig_name)
            # Strip orchestration fields (enabled/interval/heads); the rest are
            # the diagnostic's algorithm params. Derived from the base so adding
            # an orchestration field there keeps it out of the constructor.
            params = sig_cfg.model_dump(exclude=set(SignalConfigBase.model_fields))
            cls._diagnostics[sig_name] = diag_cls(**params)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Run normal Triton attention (unchanged)
        result = super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        # 2. Skip during profiling
        if attn_metadata is None:
            return result

        # Skip if no signals enabled
        if not (
            self.config.spectral.enabled
            or self.config.routing.enabled
            or self.config.tracker.enabled
            or self.config.selfattn.enabled
            or self.config.laplacian.enabled
        ):
            return result

        # 3. Accumulate Q for the first sequence in the batch
        layer_name = getattr(layer, "layer_name", None)
        if layer_name is None:
            return result

        cls = type(self)
        state = cls.state_dict.get(layer_name)
        if state is None:
            state = PerLayerState(
                qbuf=QBuffer(
                    max_tokens=self.config.q_buffer_max_tokens,
                    mode=self.config.q_buffer_mode,
                )
            )
            cls.state_dict[layer_name] = state

        m = _LAYER_IDX_RE.search(layer_name)
        layer_idx: int | None = int(m.group(1)) if m else None

        # query shape: [num_tokens, num_heads, head_size]
        # query_start_loc is a cumulative offset tensor for sequences in the
        # batch.  For sequence 0 the Q rows live at
        #   query[query_start_loc[0] : query_start_loc[1]]
        # During prefill this span equals the prompt length; during decode
        # it is exactly 1.
        q_start = attn_metadata.query_start_loc[0].item()
        q_end = attn_metadata.query_start_loc[1].item()
        q_span = q_end - q_start

        if q_span > 1:  # prefill = new request
            # Increment request_id only on layer 0 to avoid double-counting
            # across the 12 layers that all see the same prefill batch.
            if layer_idx == 0:
                cls.req_tracker.request_id += 1
            state.qbuf.flush()
            state.step = 0

        state.qbuf.append(query[q_start:q_end].detach().clone())
        state.step += 1

        # 4. Determine which signals fire. QBuffer owns the windowing policy:
        # sliding auto-trims on append; tumbling reports window_complete().
        if state.qbuf.tumbling:
            # Window size IS the cadence — per-signal interval is ignored.
            if state.qbuf.window_complete():
                due_signals = {s for s in SIGNAL_NAMES if getattr(self.config, s).enabled}
            else:
                due_signals = set()
        else:
            # Sliding: per-signal interval check (buffer already trimmed on append).
            due_signals = set()
            for sig_name in SIGNAL_NAMES:
                sig_cfg = getattr(self.config, sig_name)
                if sig_cfg.enabled and state.step % sig_cfg.interval == 0:
                    due_signals.add(sig_name)

        if due_signals:
            try:
                self._run_svd(
                    layer_name,
                    layer_idx,
                    state,
                    kv_cache,
                    attn_metadata,
                    due_signals,
                )
            except Exception:
                logger.exception("[SVD] error in layer %s at step %d", layer_name, state.step)
            if state.qbuf.tumbling:
                state.qbuf.flush()

        return result

    @staticmethod
    def _extract_k_from_cache(
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_idx: int = 0,
    ) -> torch.Tensor:
        """Extract the full K matrix for one sequence from the paged cache.

        Args:
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: contains block_table and seq_lens
            seq_idx: which sequence in the batch (default 0)

        Returns:
            K tensor of shape [seq_len, num_kv_heads, head_size]
        """
        # Unbind the KV dimension to get key_cache
        key_cache, _ = kv_cache.unbind(dim=1)
        # key_cache: [num_blocks, block_size, num_kv_heads, head_size]

        block_size = key_cache.shape[1]
        seq_len = attn_metadata.seq_lens[seq_idx].item()

        # Number of blocks needed for this sequence
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        # Get physical block indices for this sequence
        block_indices = attn_metadata.block_table[seq_idx, :num_blocks_needed]

        # Gather the blocks: [num_blocks_needed, block_size, num_kv_heads, head_size]
        k_blocks = key_cache[block_indices]

        # Reshape to [num_blocks_needed * block_size, num_kv_heads, head_size]
        k_flat = k_blocks.reshape(-1, k_blocks.shape[2], k_blocks.shape[3])

        # Trim to actual sequence length
        k_flat = k_flat[:seq_len]

        # Upcast FP8 (1-byte floats) — fp16/bf16 pass through natively.
        if k_flat.dtype.is_floating_point and k_flat.element_size() == 1:
            return k_flat.half()
        return k_flat

    def _run_svd(
        self,
        layer_name: str,
        layer_idx: int | None,
        state: PerLayerState,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        due_signals: set[str],
    ) -> None:
        cls = type(self)
        if not cls._diagnostics:
            # _build_diagnostics() runs in set_config(), which every run mode
            # calls (plugin, runner, extract). Reaching extraction with an empty
            # cache means set_config() was never called in this process — fail
            # loudly rather than silently extracting with default config.
            raise RuntimeError(
                "Diagnostics not initialised — set_config() must be called "
                "before extraction (no glassbox plugin or programmatic config?)."
            )

        # Stack accumulated Q: [L_q, num_heads, head_size]
        Q_all = state.qbuf.window()
        L_q = Q_all.shape[0]

        # Extract K: [L_k, num_kv_heads, head_size]
        K_all = self._extract_k_from_cache(kv_cache, attn_metadata, seq_idx=0)
        L_k = K_all.shape[0]

        # When the Q buffer is windowed, L_q < L_k.  Both Q and K must
        # refer to the same (most recent) positions, so take the last L
        # rows from each.
        L = min(L_q, L_k)
        if L < 2:
            return
        Q_all = Q_all[-L:]
        K_all = K_all[-L:]

        # Union of active heads for due signals
        heads: set[int] = set()
        for sig_name in due_signals:
            heads.update(getattr(self.config, sig_name).heads)

        for head_idx in sorted(heads):
            if head_idx >= Q_all.shape[1]:
                continue

            kv_head_idx = head_idx // self.num_queries_per_kv
            Qh = Q_all[:, head_idx, :]
            Kh = K_all[:, kv_head_idx, :]

            for sig_name in due_signals:
                sig_cfg = getattr(self.config, sig_name)
                if head_idx not in sig_cfg.heads:
                    continue
                diag = cls._diagnostics.get(sig_name)
                if diag is None:
                    continue

                result = diag.reduce(Qh, Kh, L)
                features = result["features"]

                state.accum[sig_name] = diag.accumulate(result, state.accum.get(sig_name))

                witness = None
                if self.config.emit_witness:
                    try:
                        witness = diag.witness(Qh, Kh, L).tolist()
                    except NotImplementedError:
                        pass

                snapshot = SVDSnapshot(
                    signal=sig_name,
                    request_id=cls.req_tracker.request_id,
                    layer=layer_name,
                    layer_idx=layer_idx,
                    head=head_idx,
                    step=state.step,
                    L=L,
                    singular_values=result.get("singular_values", []),
                    tier=result.get("tier"),
                    witness=witness,
                    features=features,
                )
                self._emit_result(snapshot)

    def _emit_result(self, snapshot: SVDSnapshot) -> None:
        """Dispatch snapshot to all registered handlers."""
        for handler in type(self)._handlers:
            handler.handle(snapshot)
