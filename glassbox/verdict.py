"""Shared verdict store bridging ClassifierHandler and ObservationPlugin.

The ClassifierHandler writes verdicts during the attention forward pass.
The ObservationPlugin reads them in ``on_step_batch()`` and returns
``ABORT``/``CONTINUE`` to the vLLM engine.

Thread-safe: both sides run on the same GPU worker process but may
execute on different threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Verdict:
    """Classification verdict for a single request."""

    request_id: int
    probability: float
    action: str  # "continue" | "abort" | "preempt" | "log_only"
    timestamp: float = field(default_factory=time.monotonic)


class VerdictStore:
    """Module-level singleton for cross-component verdict passing.

    Write side: ``ClassifierHandler`` calls ``report()`` after classification.
    Read side: ``GlassboxObservationPlugin`` calls ``consume()`` in
    ``on_step_batch()``.
    """

    _lock = threading.Lock()
    _verdicts: dict[int, Verdict] = {}  # glassbox request_id → Verdict
    _id_map: dict[str, int] = {}  # vllm string id → glassbox int id
    _reverse_map: dict[int, str] = {}  # glassbox int id → vllm string id

    @classmethod
    def report(cls, request_id: int, probability: float, action: str) -> None:
        """Record a classification verdict (called by ClassifierHandler)."""
        with cls._lock:
            cls._verdicts[request_id] = Verdict(
                request_id=request_id,
                probability=probability,
                action=action,
            )

    @classmethod
    def map_request_id(cls, vllm_id: str, glassbox_id: int) -> None:
        """Establish mapping between vLLM string ID and glassbox integer ID."""
        with cls._lock:
            cls._id_map[vllm_id] = glassbox_id
            cls._reverse_map[glassbox_id] = vllm_id

    @classmethod
    def consume(cls, vllm_id: str) -> Verdict | None:
        """Read and remove a verdict by vLLM request ID.

        Returns ``None`` if no verdict exists (request hasn't been
        classified yet, or ID mapping is missing).
        """
        with cls._lock:
            glassbox_id = cls._id_map.get(vllm_id)
            if glassbox_id is None:
                return None
            return cls._verdicts.pop(glassbox_id, None)

    @classmethod
    def clear_by_vllm_id(cls, vllm_id: str) -> None:
        """Remove all state for a completed request."""
        with cls._lock:
            glassbox_id = cls._id_map.pop(vllm_id, None)
            if glassbox_id is not None:
                cls._verdicts.pop(glassbox_id, None)
                cls._reverse_map.pop(glassbox_id, None)

    @classmethod
    def reset(cls) -> None:
        """Clear all state (for testing)."""
        with cls._lock:
            cls._verdicts.clear()
            cls._id_map.clear()
            cls._reverse_map.clear()
