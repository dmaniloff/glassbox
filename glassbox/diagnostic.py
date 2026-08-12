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
    """Interface for streaming attention diagnostics.

    A diagnostic declares the three things that identify a signal, so that generic
    code -- the backend building diagnostics, a schema builder asking what columns a
    signal produces -- never has to pair them up by hand:

    ``signal_name``    the canonical name, and the key it registers under.
    ``features_model`` the Features model ``reduce`` returns, declared rather than
                       left implicit in the ``{"features": XFeatures(...)}`` literal,
                       so the columns can be read without running the diagnostic.
    ``__init__``       takes the signal's own config object.

    Note ``runtime_checkable`` only checks that the members *exist*; it cannot check
    the constructor's parameter type.  ``tests/test_diagnostic.py`` covers what the
    Protocol cannot.
    """

    @property
    def signal_name(self) -> str:
        """Canonical signal name (e.g. 'spectral', 'routing')."""
        ...

    @property
    def features_model(self) -> type:
        """The Features model returned under ``reduce()['features']``."""
        ...

    def __init__(self, config: Any) -> None:
        """Build from the signal's config object (e.g. ``SpectralConfig``).

        Taking the config rather than loose kwargs keeps each parameter's default and
        bounds in one place -- the config class -- and means a field added there
        reaches the diagnostic without a matching signature edit.
        """
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
            a ``'features'`` key whose value is an instance of
            ``features_model``, and optionally ``'singular_values'`` and
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
