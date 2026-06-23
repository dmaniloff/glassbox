"""Per-layer Q-slice accumulation with sliding / tumbling windowing.

`QBuffer` holds the buffer of accumulated query slices for one attention layer
and owns the windowing policy that bounds memory during decode. It is kept free
of any vLLM dependency so the policy can be unit-tested directly.

Two modes (see ``GlassboxConfig.q_buffer_mode`` / ``q_buffer_max_tokens``):

- **sliding** (default): overlapping windows. Keep the last ``max_tokens``
  tokens (``0`` = unbounded), trimming the oldest slices on append. The firing
  cadence is the per-signal interval, decided by the caller.
- **tumbling**: non-overlapping windows. Accumulate until ``max_tokens`` tokens
  are buffered (``window_complete()``), fire, then ``flush()``. The window size
  *is* the cadence — per-signal intervals are ignored.

Only Q is buffered here; K is read from the vLLM KV cache and sliced to match
the window at compute time, so it lives outside this class.
"""

from __future__ import annotations

import torch


class QBuffer:
    """Accumulates query slices and applies the configured windowing policy."""

    def __init__(self, max_tokens: int = 0, mode: str = "sliding") -> None:
        self.max_tokens = max_tokens
        self.mode = mode
        self._slices: list[torch.Tensor] = []

    @property
    def tumbling(self) -> bool:
        """Whether tumbling windowing is active (mode set and a finite bound)."""
        return self.mode == "tumbling" and self.max_tokens > 0

    @property
    def tokens(self) -> int:
        """Total buffered query tokens across all slices."""
        return sum(t.shape[0] for t in self._slices)

    def __len__(self) -> int:
        return len(self._slices)

    def append(self, q: torch.Tensor) -> None:
        """Append a query slice. In sliding mode, trim back to the window bound."""
        self._slices.append(q)
        if not self.tumbling:
            self._trim()

    def _trim(self) -> None:
        """Drop oldest slices until total tokens <= max_tokens (no-op if 0)."""
        if self.max_tokens <= 0:
            return
        total = self.tokens
        while total > self.max_tokens and self._slices:
            total -= self._slices.pop(0).shape[0]

    def window_complete(self) -> bool:
        """Tumbling: a full window (>= max_tokens tokens) is ready to fire."""
        return self.tumbling and self.tokens >= self.max_tokens

    def window(self) -> torch.Tensor:
        """Concatenate the buffered slices into a single ``[L, ...]`` tensor."""
        return torch.cat(self._slices, dim=0)

    def flush(self) -> None:
        """Clear the buffer (tumbling after a window fires, or on a new request)."""
        self._slices = []
