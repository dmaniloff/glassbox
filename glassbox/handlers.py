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
import re
from typing import IO, TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from glassbox.verdict import VerdictStore

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

    This handler is designed to do feature logging for training a detection model.

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
        logger.info("%r", snapshot)

    def close(self) -> None:
        pass


class OtelHandler:
    """Emits snapshots as OpenTelemetry spans with ``glassbox.*`` attributes.

    This handler is designed to be used with a trained detection model.
    The ``heads``, ``interval``, and signal selection should be configured
    to match what your trained detection model expects.

    Piggybacks on whatever ``TracerProvider`` is globally configured (e.g.
    by vLLM's ``--otlp-traces-endpoint``).  If no provider is set, the
    OpenTelemetry API returns a no-op tracer and the overhead is negligible.

    If the ``opentelemetry`` packages are not installed, the handler
    silently no-ops.

    Spans are organised in a two-level hierarchy:

    - A **parent span** (``glassbox.step``) groups all snapshots for a
    given ``(request_id, step)`` pair.
    - **Child spans** (``glassbox.<signal>``) carry per-layer/head features.

    The parent span is created lazily on the first snapshot for a new
    ``(request_id, step)`` and ended when the next step (or request)
    arrives.  This is safe because vLLM processes layers sequentially
    within a single decode step — all snapshots for step *N* arrive
    before any snapshot for step *N+1*.

    In Jaeger this should show up as:

    glassbox.step (request_id=7, step=32)
    ├── glassbox.spectral (layer=0, head=0, sv_ratio=2.1)
    ├── glassbox.spectral (layer=1, head=0, sv_ratio=1.8)
    ├── glassbox.routing  (layer=0, head=0, curl_norm=0.3)
    └── ...
    """

    def __init__(self) -> None:
        self._tracer = None
        self._trace_mod = None
        self._active_key: tuple[int, int] | None = None
        self._active_span = None
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("glassbox")
            self._trace_mod = trace
        except ImportError:
            logger.debug("opentelemetry not available; OtelHandler will no-op")

    def _end_active_span(self) -> None:
        if self._active_span is not None:
            self._active_span.end()
            self._active_span = None
            self._active_key = None

    def handle(self, snapshot: SVDSnapshot) -> None:
        if self._tracer is None:
            return

        key = (snapshot.request_id, snapshot.step)

        # New (request_id, step) → close previous parent, start new one
        if self._active_key != key:
            self._end_active_span()
            self._active_span = self._tracer.start_span("glassbox.step")
            self._active_span.set_attribute("glassbox.request_id", snapshot.request_id)
            self._active_span.set_attribute("glassbox.step", snapshot.step)
            self._active_key = key

        # Child span nested under the parent
        ctx = self._trace_mod.set_span_in_context(self._active_span)
        with self._tracer.start_as_current_span(
            f"glassbox.{snapshot.signal}",
            context=ctx,
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
        self._end_active_span()


class ClassifierHandler:
    """Runs a trained probe on extracted features and reports verdicts.

    Buffers snapshots for a ``(request_id, step)`` group.  When the group
    changes (next step or next request), assembles a feature vector from
    the buffer, runs the classifier, and reports the verdict to the
    :class:`~glassbox.verdict.VerdictStore`.

    The model is loaded lazily on the first ``handle()`` call so that
    importing ``glassbox`` doesn't require ``joblib``/``sklearn``.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        signal: str = "laplacian",
        action: str = "log_only",
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._signal = signal
        self._action = action

        # Loaded lazily
        self._model_dict: dict | None = None
        self._col_index: dict[tuple[int, int, int], int] | None = None
        self._n_features: int = 0

        # Step-boundary buffer
        self._active_key: tuple[int, int] | None = None
        self._buffer: dict[tuple[int, int], list[float]] = {}

    def _load_model(self) -> None:
        try:
            import joblib
        except ImportError:
            logger.warning(
                "joblib not installed; ClassifierHandler disabled. "
                "Install with: pip install 'glassbox[train]'"
            )
            self._model_dict = {}
            return

        self._model_dict = joblib.load(self._model_path)
        feat_cols = self._model_dict["feature_columns"]
        self._n_features = len(feat_cols)

        self._col_index = {}
        pattern = re.compile(r"^\w+_\w+eigval_(\d+)_L(\d+)_H(\d+)$")
        for pos, col in enumerate(feat_cols):
            m = pattern.match(col)
            if m:
                eigval_idx, layer_idx, head = int(m.group(1)), int(m.group(2)), int(m.group(3))
                self._col_index[(layer_idx, head, eigval_idx)] = pos

    def handle(self, snapshot: SVDSnapshot) -> None:
        if snapshot.signal != self._signal:
            return

        if self._model_dict is None:
            self._load_model()
        if not self._model_dict:
            return

        key = (snapshot.request_id, snapshot.step)

        if self._active_key is not None and self._active_key != key:
            self._classify_and_emit()

        self._active_key = key

        feat = snapshot.features
        if hasattr(feat, "eigvals"):
            self._buffer[(snapshot.layer_idx, snapshot.head)] = feat.eigvals

    def _classify_and_emit(self) -> None:
        if not self._buffer or self._col_index is None:
            self._buffer = {}
            return

        vec = np.full(self._n_features, np.nan)
        for (layer_idx, head), eigvals in self._buffer.items():
            for ei, val in enumerate(eigvals):
                pos = self._col_index.get((layer_idx, head, ei))
                if pos is not None:
                    vec[pos] = val

        self._buffer = {}

        nan_frac = np.isnan(vec).mean()
        if nan_frac > 0.5:
            logger.debug(
                "ClassifierHandler: skipping, %.0f%% features missing",
                nan_frac * 100,
            )
            return

        vec = np.nan_to_num(vec, nan=0.0)

        model_dict = self._model_dict
        X = vec.reshape(1, -1)

        if model_dict.get("pca") is not None:
            X = model_dict["pca"].transform(X)

        proba = model_dict["model"].predict_proba(X)[0, 1]

        req_id, step = self._active_key

        verdict_action = self._action if proba >= self._threshold else "continue"
        VerdictStore.report(req_id, proba, verdict_action)

        if proba >= self._threshold:
            logger.warning(
                "Hallucination detected: p=%.3f (threshold=%.3f) request_id=%d step=%d action=%s",
                proba,
                self._threshold,
                req_id,
                step,
                self._action,
            )

    def close(self) -> None:
        if self._buffer:
            self._classify_and_emit()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_handlers_from_config(config: GlassboxConfig) -> list[SnapshotHandler]:
    """Build handler list from config fields.

    - ``config.output.path`` set   -> ``JsonlHandler``
    - ``config.emit.otel`` True    -> ``OtelHandler``
    - ``config.classifier.enabled`` -> ``ClassifierHandler``
    - neither                      -> ``LoggingHandler`` (fallback)

    Multiple handlers can be active simultaneously (e.g. JSONL + OTel).
    """
    handlers: list[SnapshotHandler] = []
    if config.output.path:
        handlers.append(JsonlHandler(config.output.path))
    if config.emit.otel:
        handlers.append(OtelHandler())
    if config.classifier.enabled and config.classifier.model_path:
        handlers.append(
            ClassifierHandler(
                model_path=config.classifier.model_path,
                threshold=config.classifier.threshold,
                signal=config.classifier.signal,
                action=config.classifier.action,
            )
        )
    if not handlers:
        handlers.append(LoggingHandler())
    return handlers
