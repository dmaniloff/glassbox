"""Diagnostic protocol for streaming attention diagnostics.

Every diagnostic produces two readouts from a single streamed pass over a
window of (Q, K) data:

- **reduce** → scalar features (detect): global summary of the window.
- **witness** → per-token vector (localize): where in the window the signal
  concentrates.

A third method, **accumulate**, merges a local reduce() result into a running
global state.  The accumulation strategy is diagnostic-specific — the streaming
math proving correctness lives in the companion papers; this protocol provides
the interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Diagnostic(Protocol):
    """Interface for streaming attention diagnostics."""

    @property
    def signal_name(self) -> str:
        """Canonical signal name (e.g. 'spectral', 'routing')."""
        ...

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        """Local scalar features from a window of (Q, K).

        Args:
            Qh: Query tensor for one head, shape ``[L, d]``.
            Kh: Key tensor for one head, shape ``[L, d]``.
            L:  Sequence length of the window.
            **ctx: Signal-specific context (config, scale, etc.).

        Returns:
            Dict of scalar features (the 'detect' readout).  Must include
            a ``'features'`` key whose value is the appropriate Features
            pydantic model, and optionally ``'singular_values'`` and
            ``'tier'``.
        """
        ...

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        """Per-token localization vector from the same window.

        Args:
            Qh: Query tensor for one head, shape ``[L, d]``.
            Kh: Key tensor for one head, shape ``[L, d]``.
            L:  Sequence length of the window.
            **ctx: Signal-specific context.

        Returns:
            Tensor of shape ``[L]`` (the 'localize' readout).

        Raises:
            NotImplementedError: If witness is not supported by this diagnostic.
        """
        ...

    def accumulate(self, local: dict, state: dict | None) -> dict:
        """Merge a local reduce() result into a running global state.

        Each diagnostic defines its own accumulation strategy.  The default
        (latest-only) simply returns the local result as the new state.

        Args:
            local: Result from reduce().
            state: Previous accumulated state, or None on first call.

        Returns:
            Updated state dict.
        """
        ...
