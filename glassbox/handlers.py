"""Pluggable snapshot emission handlers.

Glassbox extracts signals from attention internals and emits them as
``SVDSnapshot`` objects.  Handlers define *where* those snapshots go:

- ``JsonlHandler`` — append to a JSONL file (training / bulk analysis)
- ``LoggingHandler`` — write to the Python logger (development / debugging)
- ``OtelHandler`` — emit as OpenTelemetry spans (real-time detection)

Custom handlers only need to implement ``handle()`` and ``close()``.
"""

from __future__ import annotations

import json
import logging
from typing import IO, TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from glassbox.config import GlassboxConfig
    from glassbox.results import SVDSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SnapshotHandler(Protocol):
    """Interface for snapshot emission handlers."""

    def handle(self, snapshot: SVDSnapshot) -> None:
        """Process a single snapshot."""
        ...

    def close(self) -> None:
        """Release resources (file handles, connections, etc.)."""
        ...


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


class JsonlHandler:
    """Writes snapshots as newline-delimited JSON to a file.

    The file is opened lazily on the first ``handle()`` call and flushed
    after every write so downstream consumers can tail the file.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: IO | None = None

    def handle(self, snapshot: SVDSnapshot) -> None:
        if self._fh is None:
            self._fh = open(self._path, "a")  # noqa: SIM115
        self._fh.write(json.dumps(snapshot.model_dump(exclude_none=True)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class LoggingHandler:
    """Logs snapshots via Python's logging module.

    This is the default fallback when no other handler is configured.
    """

    def handle(self, snapshot: SVDSnapshot) -> None:
        if snapshot.singular_values:
            k = len(snapshot.singular_values)
            logger.info(
                "[SVD] %s head=%d step=%d L=%d top-%d singular values: %s",
                snapshot.layer,
                snapshot.head,
                snapshot.step,
                snapshot.L,
                k,
                snapshot.singular_values,
            )
        else:
            logger.info(
                "[%s] %s head=%d step=%d L=%d features=%s",
                snapshot.signal,
                snapshot.layer,
                snapshot.head,
                snapshot.step,
                snapshot.L,
                snapshot.features.model_dump(exclude_none=True),
            )

    def close(self) -> None:
        pass


class OtelHandler:
    """Emits snapshots as OpenTelemetry spans with ``glassbox.*`` attributes.

    Piggybacks on whatever ``TracerProvider`` is globally configured (e.g.
    by vLLM's ``--otlp-traces-endpoint``).  If no provider is set, the
    OpenTelemetry API returns a no-op tracer and the overhead is negligible.

    If the ``opentelemetry`` packages are not installed, the handler
    silently no-ops.

    One span is emitted per snapshot (i.e. per layer/head/step).  The
    ``heads``, ``interval``, and signal selection should be configured to
    match what your trained detection model expects.
    """

    def __init__(self) -> None:
        self._tracer = None
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("glassbox")
        except ImportError:
            logger.debug("opentelemetry not available; OtelHandler will no-op")

    def handle(self, snapshot: SVDSnapshot) -> None:
        if self._tracer is None:
            return

        with self._tracer.start_as_current_span(
            f"glassbox.{snapshot.signal}",
        ) as span:
            span.set_attribute("glassbox.signal", snapshot.signal)
            span.set_attribute("glassbox.request_id", snapshot.request_id)
            span.set_attribute("glassbox.layer", snapshot.layer)
            if snapshot.layer_idx is not None:
                span.set_attribute("glassbox.layer_idx", snapshot.layer_idx)
            span.set_attribute("glassbox.head", snapshot.head)
            span.set_attribute("glassbox.step", snapshot.step)
            span.set_attribute("glassbox.L", snapshot.L)
            if snapshot.tier is not None:
                span.set_attribute("glassbox.tier", snapshot.tier)

            # Flatten features into glassbox.* attributes
            feat_dict = snapshot.features.model_dump(exclude_none=True)
            for key, value in feat_dict.items():
                if isinstance(value, list):
                    # OTel supports homogeneous list attributes (e.g. list[float])
                    span.set_attribute(f"glassbox.{key}", value)
                elif isinstance(value, (int, float, str, bool)):
                    span.set_attribute(f"glassbox.{key}", value)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_handlers_from_config(config: GlassboxConfig) -> list[SnapshotHandler]:
    """Build handler list from config fields.

    - ``config.output.path`` set   -> ``JsonlHandler``
    - ``config.emit.otel`` True    -> ``OtelHandler``
    - neither                      -> ``LoggingHandler`` (fallback)

    Multiple handlers can be active simultaneously (e.g. JSONL + OTel).
    """
    handlers: list[SnapshotHandler] = []
    if config.output.path:
        handlers.append(JsonlHandler(config.output.path))
    if config.emit.otel:
        handlers.append(OtelHandler())
    if not handlers:
        handlers.append(LoggingHandler())
    return handlers
