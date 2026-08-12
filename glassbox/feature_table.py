"""Pivot snapshot JSONL into a wide feature table (one row per request).

A signal's columns are derived from the Features model it declares -- see
``Diagnostic.features_model`` and ``glassbox.diagnostics.FEATURES_MODELS`` -- rather
than from per-signal branches naming the models by hand.  That pairing used to be
written out here, and the three signals added after it was written (asymmetry, cyclic,
magnetic) were never added to it, so they produced no columns at all and made
``--parquet`` fail outright.

The two halves of the job read the same model:

``feature_names``      what a signal's columns are *called*, needed before any data
                       exists so the parquet schema can be fixed up front.
``parse_snap_features`` what a snapshot actually *contains*.

Deriving both from ``features_model`` is what keeps them in agreement;
``test_emitted_keys_match_declared_columns`` pins it for every signal.

Column layout::

    {signal}_{feature}_L{layer}_H{head}      e.g. routing_G_L0_H0

The signal name is already the outer prefix, so features carry no second, inner one.
Before this module they did -- ``hodge_``/``at_``/``ad_``/``lap_``, chosen by a chain of
``if signal ==`` tests ending in a catch-all that labelled anything unrecognised as a
Hodge quantity, so a cyclic-triangle count came out as ``cyclic_hodge_T_cyc``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from glassbox.config import GlassboxConfig, TopKParams
from glassbox.diagnostics import FEATURES_MODELS
from glassbox.results import SVDSnapshot

if TYPE_CHECKING:
    from pydantic import BaseModel

# Raw singular values are kept on the snapshot but not turned into columns: the
# features derived from them (sv1, sv_ratio, sv_entropy) are already columns, and the
# vector's length varies with rank, so it has no fixed schema.
_SKIP_FEATURE_FIELDS = frozenset({"singular_values"})

# Metadata carried alongside the features, one entry per column. Kept as (name, arrow
# type) pairs so the schema and the pivot cannot disagree about which columns exist.
_META_FIELDS: tuple[tuple[str, str], ...] = (
    ("request_id", "int64"),
    ("label", "int64"),
    ("length", "int64"),
    ("sample_id", "int64"),
    ("phase", "string"),
    ("prompt_length", "int64"),
    ("source", "string"),
)


def _is_list_field(model: type[BaseModel], name: str) -> bool:
    """True if the field is list-typed, and so expands into indexed columns."""
    annotation = model.model_fields[name].annotation
    return getattr(annotation, "__origin__", None) is list


def _split_fields(model: type[BaseModel]) -> tuple[list[str], list[str]]:
    """(scalar fields, list fields) of a Features model, in declaration order."""
    scalars, lists = [], []
    for name in model.model_fields:
        if name in _SKIP_FEATURE_FIELDS:
            continue
        (lists if _is_list_field(model, name) else scalars).append(name)
    return scalars, lists


def _indexed(field: str, i: int) -> str:
    """Column name for element *i* of a list-valued field: ``eigvals`` -> ``eigval_3``."""
    return f"{field.removesuffix('s')}_{i}"


def feature_names(signal: str, config: GlassboxConfig) -> list[str]:
    """Column names a signal contributes, before the ``_L{layer}_H{head}`` suffix.

    List-valued fields expand to ``top_k`` indexed columns; ``top_k`` comes from the
    signal's config, which declares it by inheriting :class:`~glassbox.config.TopKParams`.
    A signal with a list field but no ``top_k`` has no fixed width, so it is skipped
    rather than guessed at.
    """
    model = FEATURES_MODELS[signal]
    scalars, lists = _split_fields(model)
    names = list(scalars)
    sig_cfg = config.signals[signal]
    if lists and isinstance(sig_cfg, TopKParams):
        for field in lists:
            names += [_indexed(field, i) for i in range(sig_cfg.top_k)]
    return names


def build_feature_columns(
    num_layers: int,
    heads: list[int] | tuple[int, ...],
    signals: tuple[str, ...],
    config: GlassboxConfig,
) -> list[str]:
    """Full column list for the wide table, ordered signal -> layer -> head -> feature."""
    return [
        f"{signal}_{feat}_L{li}_H{hi}"
        for signal in signals
        for li in range(num_layers)
        for hi in heads
        for feat in feature_names(signal, config)
    ]


def parse_snap_features(snap: SVDSnapshot) -> dict[str, float]:
    """Scalar features of one snapshot, keyed exactly as :func:`feature_names` names them.

    Raises ``TypeError`` on a field that is neither scalar nor a list of scalars, rather
    than letting an unexpected type reach the parquet writer as a null.
    """
    result: dict[str, float] = {}
    for k, v in snap.features.model_dump(exclude_none=True).items():
        if k in _SKIP_FEATURE_FIELDS:
            continue
        if isinstance(v, list):
            for i, x in enumerate(v):
                result[_indexed(k, i)] = x
        elif isinstance(v, (int, float)):
            result[k] = v
        else:
            raise TypeError(f"Unexpected feature type {k!r}: {type(v).__name__} = {v!r}")
    return result


def write_parquet(
    svd_features_path: Path,
    samples_path: Path,
    out_path: Path,
    feature_columns: list[str],
) -> int:
    """Pivot the snapshot JSONL into a wide parquet table, streaming by request.

    The schema is fixed up front from ``feature_columns`` so a value missing from one
    request (a failed solve, say) lands as a null rather than changing the schema
    mid-file.  Rows are written in batches to bound memory.

    One row per request (i.e. per phase), with the metadata columns then the features.
    Returns the number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    BATCH_SIZE = 500

    arrow = {"int64": pa.int64(), "string": pa.string()}
    schema = pa.schema(
        [pa.field(name, arrow[kind]) for name, kind in _META_FIELDS]
        + [pa.field(col, pa.float64()) for col in feature_columns]
    )

    sample_meta: dict[int, dict] = {}
    with open(samples_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sample_meta[row["request_id"]] = row

    def _pivot_request(buf: list[tuple[str, int, int, int, dict]], rid: int) -> dict:
        wide: dict = {"request_id": rid}
        length = None
        for sig, li, hi, seq_len, feats in buf:
            if length is None:
                length = seq_len
            for k, v in feats.items():
                wide[f"{sig}_{k}_L{li}_H{hi}"] = v

        if rid not in sample_meta:
            raise KeyError(f"request_id {rid} not found in samples.jsonl")
        meta = sample_meta[rid]
        for required in ("label", "sample_id", "phase"):
            if required not in meta:
                raise KeyError(
                    f"request_id {rid} missing required field {required!r} in samples.jsonl"
                )
        wide["label"] = meta["label"]
        wide["length"] = length
        wide["sample_id"] = meta["sample_id"]
        wide["phase"] = meta["phase"]
        if "prompt_length" in meta:
            wide["prompt_length"] = meta["prompt_length"]
        if "dataset" in meta:
            wide["source"] = meta["dataset"]
        return wide

    schema_cols = set(schema.names)
    checked_first_row = False
    total_rows = 0
    wide_rows: list[dict] = []
    current_rid: int | None = None
    buf: list[tuple[str, int, int, int, dict]] = []
    n_expected = len(sample_meta)

    def _flush_batch(writer, rows: list[dict]) -> int:
        if not rows:
            return 0
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        return len(rows)

    pbar = tqdm(total=n_expected, desc="Pivoting to parquet", unit="rows")

    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        with open(svd_features_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snap = SVDSnapshot.from_jsonl_row(json.loads(line))
                feats = parse_snap_features(snap)

                if current_rid is not None and snap.request_id != current_rid:
                    row = _pivot_request(buf, current_rid)
                    if not checked_first_row:
                        extra = set(row.keys()) - schema_cols
                        if extra:
                            raise ValueError(
                                f"Pivoted row has columns not in schema"
                                f" (would be silently dropped): {extra}"
                            )
                        checked_first_row = True
                    wide_rows.append(row)
                    pbar.update(1)
                    buf = []
                    if len(wide_rows) >= BATCH_SIZE:
                        total_rows += _flush_batch(writer, wide_rows)
                        wide_rows = []

                current_rid = snap.request_id
                buf.append((snap.signal, snap.layer_idx, snap.head, snap.L, feats))

        if buf and current_rid is not None:
            row = _pivot_request(buf, current_rid)
            extra = set(row.keys()) - schema_cols
            if extra:
                raise ValueError(
                    f"Pivoted row has columns not in schema (would be silently dropped): {extra}"
                )
            wide_rows.append(row)
            pbar.update(1)

        total_rows += _flush_batch(writer, wide_rows)

    pbar.close()
    return total_rows
