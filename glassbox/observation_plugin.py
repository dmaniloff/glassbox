"""vLLM ObservationPlugin that returns ABORT/CONTINUE based on attention features.

Reads verdicts from the :class:`~glassbox.verdict.VerdictStore` (written by
:class:`~glassbox.handlers.ClassifierHandler` during the attention forward
pass) and returns ``ObservationAction.ABORT`` or ``CONTINUE`` to the vLLM
engine.

This plugin does **not** use ``batch_hidden_states`` — it already has its
own attention-derived features from the custom backend.

Graceful degradation: when vLLM's observation API is not available (the
RFC PR #37002 is not merged), the module still loads but the entry point
is simply never discovered.
"""

from __future__ import annotations

import logging
from typing import Any

from glassbox.verdict import VerdictStore

logger = logging.getLogger(__name__)

# Conditional import — the RFC may not be merged yet.
try:
    from vllm.plugins.observation import (
        ObservationAction,
        ObservationPlugin,
        ObservationResult,
        RequestContext,
    )

    _HAS_OBSERVATION_API = True
except ImportError:
    _HAS_OBSERVATION_API = False

    # Stubs so the class definition works even without vLLM's observation API.
    class ObservationPlugin:  # type: ignore[no-redef]
        pass

    class ObservationAction:  # type: ignore[no-redef]
        CONTINUE = 0
        ABORT = 1
        PREEMPT = 2

    class ObservationResult:  # type: ignore[no-redef]
        def __init__(self, action=0, metadata=None):
            self.action = action
            self.metadata = metadata or {}

    class RequestContext:  # type: ignore[no-redef]
        request_id: str = ""


class GlassboxObservationPlugin(ObservationPlugin):
    """Bridge between glassbox attention features and vLLM's action system."""

    def __init__(self, vllm_config=None) -> None:
        if _HAS_OBSERVATION_API:
            super().__init__(vllm_config)
        self._next_glassbox_id = 0

    def get_observation_layers(self) -> list[int]:
        # We don't need hidden states — our features come from the
        # custom attention backend, not from layer output hooks.
        return []

    def observe_decode(self) -> bool:
        # We want verdicts during decode too (ClassifierHandler may
        # classify at any step, not just prefill).
        return True

    def on_request_start(self, request_id: str, prompt: str | None = None) -> None:
        VerdictStore.map_request_id(request_id, self._next_glassbox_id)
        self._next_glassbox_id += 1

    def on_step_batch(
        self,
        batch_hidden_states: dict[int, Any],
        request_contexts: list,
    ) -> list:
        from glassbox.verdict import VerdictStore

        results = []
        for ctx in request_contexts:
            verdict = VerdictStore.consume(ctx.request_id)
            if verdict is not None and verdict.action == "abort":
                results.append(
                    ObservationResult(
                        action=ObservationAction.ABORT,
                        metadata={
                            "glassbox.probability": verdict.probability,
                            "glassbox.source": "classifier",
                        },
                    )
                )
            elif verdict is not None and verdict.action == "preempt":
                results.append(
                    ObservationResult(
                        action=ObservationAction.PREEMPT,
                        metadata={
                            "glassbox.probability": verdict.probability,
                            "glassbox.source": "classifier",
                        },
                    )
                )
            else:
                results.append(ObservationResult(action=ObservationAction.CONTINUE))
        return results

    def on_request_complete(self, request_id: str) -> None:
        VerdictStore.clear_by_vllm_id(request_id)

    def reload_config(self, config_data: dict[str, Any]) -> None:
        # Future: hot-reload classifier threshold, action mode, etc.
        pass


def register_observation_plugin() -> str:
    """Entry point for ``vllm.observation_plugins``.

    Returns the fully-qualified class name for vLLM's plugin loader.
    """
    return "glassbox.observation_plugin:GlassboxObservationPlugin"
