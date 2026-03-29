"""Prefill-only spectral feature extraction on labeled datasets.

For each sample, runs two prefill phases through the model:
  1. "question" phase: prefill with just the prompt
  2. "full" phase: prefill with prompt + known response
SVD features are extracted from attention internals during each prefill.
No text is generated — max_tokens=1 is a vLLM requirement.

Datasets are loaded from pre-split HuggingFace repos that contain the
exact 30% test split produced by shade-train's HashBasedSplitter
(hash_fields=["prompt"], 70/0/30 ratio).  This ensures glassbox
experiments run on the same samples as shade without a shade-train
dependency.  See ``scripts/upload_test_splits.py`` for how these
datasets were created.

Usage:
    python experiments/extract.py --dataset halueval_hallucination --scores-matrix
    python experiments/extract.py --dataset all --degree-normalized
    python experiments/extract.py --dataset halueval_hallucination --max-samples 50
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from glassbox.results import (
    SPECTRAL_FEATURE_NAMES,
    AttentionDiagonalFeatures,
    DegreeNormalizedFeatures,
    SVDSnapshot,
)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_HF_ORG = "dmaniloff"


def log(msg: str) -> None:
    click.echo(f"[spectral] {msg}")


# ── Dataset loading ───────────────────────────────────────────────────────
# Each dataset is a pre-split HuggingFace dataset with columns:
#   prompt, response, label (0=ok, 1=bad), unique_id, failure_mode


DATASET_REGISTRY = {
    "deepset_injection": {
        "hf_repo": "glassbox_deepset_injection_test",
        "failure_mode": "injection",
    },
    "protectai_injection": {
        "hf_repo": "glassbox_protectai_injection_test",
        "failure_mode": "injection",
    },
    "halueval_hallucination": {
        "hf_repo": "glassbox_halueval_hallucination_test",
        "failure_mode": "hallucination",
    },
    "truthfulqa_hallucination": {
        "hf_repo": "glassbox_truthfulqa_hallucination_test",
        "failure_mode": "hallucination",
    },
    "medhallu_hallucination": {
        "hf_repo": "glassbox_medhallu_hallucination_test",
        "failure_mode": "hallucination",
    },
    "ragtruth_hallucination": {
        "hf_repo": "glassbox_ragtruth_hallucination_test",
        "failure_mode": "hallucination",
    },
    "halubench_hallucination": {
        "hf_repo": "glassbox_halubench_hallucination_test",
        "failure_mode": "hallucination",
    },
    "felm_hallucination": {
        "hf_repo": "glassbox_felm_hallucination_test",
        "failure_mode": "hallucination",
    },
}


def load_dataset_samples(
    dataset_name: str,
    max_samples: int | None = None,
    hf_org: str = DEFAULT_HF_ORG,
) -> list[dict]:
    """Load a pre-split test dataset from HuggingFace.

    Returns list of dicts with keys: idx, question, response, label, unique_id.
    """
    from datasets import load_dataset

    info = DATASET_REGISTRY[dataset_name]
    repo_id = f"{hf_org}/{info['hf_repo']}"
    log(f"Loading {dataset_name} from {repo_id}...")

    ds = load_dataset(repo_id, split="test")
    samples = []
    for i, row in enumerate(ds):
        if max_samples is not None and len(samples) >= max_samples:
            break
        samples.append(
            {
                "idx": i,
                "question": row["prompt"],
                "response": row["response"],
                "label": int(row["label"]),
                "unique_id": row.get("unique_id", ""),
            }
        )

    n_pos = sum(s["label"] for s in samples)
    log(f"Loaded {len(samples)} samples ({n_pos} positive / {len(samples) - n_pos} negative)")
    return samples


_SKIP_FEATURE_FIELDS = {"singular_values"}  # redundant copy of raw SVs; skip in wide parquet

# Hodge feature names derived from DegreeNormalizedFeatures model
_HODGE_FEATURE_NAMES = [
    f"hodge_{f}"
    for f in DegreeNormalizedFeatures.model_fields
    if f not in _SKIP_FEATURE_FIELDS and f not in SPECTRAL_FEATURE_NAMES
]

# AttentionDiagonal scalar feature names (eigvals handled separately as indexed columns)
_AD_FEATURE_NAMES = [
    f"ad_{f}"
    for f in AttentionDiagonalFeatures.model_fields
    if f not in _SKIP_FEATURE_FIELDS and f != "eigvals"
]


_META_COLUMNS = ["request_id", "label", "length", "sample_id", "phase", "prompt_length", "source"]


def _parse_snap_features(snap: SVDSnapshot) -> dict[str, float]:
    """Extract scalar features from a snapshot, raising on unexpected types.

    List-valued features (e.g. eigvals) are expanded into indexed columns:
    ``eigvals: [0.9, 0.7]`` → ``{prefix}eigval_0: 0.9, {prefix}eigval_1: 0.7``.
    """
    result: dict[str, float] = {}
    feat_dict = snap.features.model_dump(exclude_none=True)
    # Prefix depends on signal type
    if snap.feature_group == "attention_diagonal":
        non_spectral_prefix = "ad_"
    elif snap.feature_group == "attention_tracker":
        non_spectral_prefix = "at_"
    elif snap.feature_group == "laplacian_eigvals":
        non_spectral_prefix = "lap_"
    else:
        non_spectral_prefix = "hodge_"
    for k, v in feat_dict.items():
        if k in _SKIP_FEATURE_FIELDS:
            continue
        if isinstance(v, list):
            # Expand list into indexed columns (eigvals → eigval_0, eigval_1, ...)
            stem = k.rstrip("s")  # eigvals → eigval
            for i, x in enumerate(v):
                result[f"{non_spectral_prefix}{stem}_{i}"] = x
        elif isinstance(v, (int, float)):
            if k in SPECTRAL_FEATURE_NAMES:
                result[k] = v
            else:
                result[f"{non_spectral_prefix}{k}"] = v
        else:
            raise TypeError(f"Unexpected feature type {k!r}: {type(v).__name__} = {v!r}")
    return result


def _build_feature_columns(
    num_layers: int,
    heads: list[int] | tuple[int, ...],
    scores_matrix: bool,
    degree_normalized: bool,
    attention_diagonal: bool = False,
    laplacian_eigvals: bool = False,
    ad_top_k: int = 0,
    lap_top_k: int = 10,
) -> list[str]:
    """Pre-compute all feature column names from model architecture.

    Column names follow the pattern: {signal_prefix}{feature}_L{layer}_H{head}
    where signal_prefix is only added when multiple signals are enabled.
    """
    signals: list[tuple[str, list[str]]] = []
    if scores_matrix:
        signals.append(("scores_matrix", list(SPECTRAL_FEATURE_NAMES)))
    if degree_normalized:
        signals.append(
            ("degree_normalized_matrix", list(SPECTRAL_FEATURE_NAMES) + _HODGE_FEATURE_NAMES)
        )
    if attention_diagonal:
        ad_cols = list(_AD_FEATURE_NAMES)
        if ad_top_k > 0:
            ad_cols += [f"ad_eigval_{i}" for i in range(ad_top_k)]
        signals.append(("attention_diagonal", ad_cols))
    if laplacian_eigvals:
        lap_cols = [f"lap_eigval_{i}" for i in range(lap_top_k)]
        signals.append(("laplacian_eigvals", lap_cols))

    use_signal_prefix = len(signals) > 1
    columns: list[str] = []
    for signal_name, feature_names in signals:
        prefix = f"{signal_name}_" if use_signal_prefix else ""
        for li in range(num_layers):
            for hi in heads:
                for feat in feature_names:
                    columns.append(f"{prefix}{feat}_L{li}_H{hi}")
    return columns


def _write_parquet(
    svd_features_path: Path,
    samples_path: Path,
    out_path: Path,
    feature_columns: list[str],
) -> None:
    """Pivot JSONL results into wide Parquet using batched writes.

    Schema is pre-computed from model architecture (via feature_columns),
    so missing values from cuSOLVER failures are null, not schema errors.

    Streams the SVD JSONL by request_id, pivots each request into a wide
    dict, and writes in batches via ParquetWriter to bound memory usage.

    Output has one row per request (i.e. per phase) with columns:
        {signal}_{feature}_L{layer}_H{head}  (e.g. scores_matrix_sv_ratio_L0_H0)
        label, source, length, phase, sample_id
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    BATCH_SIZE = 500

    # Build schema: metadata columns + pre-computed feature columns
    fields = [
        pa.field("request_id", pa.int64()),
        pa.field("label", pa.int64()),
        pa.field("length", pa.int64()),
        pa.field("sample_id", pa.int64()),
        pa.field("phase", pa.string()),
        pa.field("prompt_length", pa.int64()),
        pa.field("source", pa.string()),
    ]
    for col in feature_columns:
        fields.append(pa.field(col, pa.float64()))
    schema = pa.schema(fields)

    # Load sample metadata into a dict keyed by request_id (small)
    sample_meta: dict[int, dict] = {}
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                sample_meta[row["request_id"]] = row

    # Signal prefixes are present when multiple signals are enabled
    use_signal_prefix = any(
        col.startswith("scores_matrix_")
        or col.startswith("degree_normalized_matrix_")
        or col.startswith("attention_tracker_")
        or col.startswith("attention_diagonal_")
        for col in feature_columns[:1]
    )

    def _pivot_request(buf: list[tuple[str, int, int, int, dict]], rid: int) -> dict:
        """Pivot one request_id's SVD rows into a single wide dict."""
        wide: dict = {"request_id": rid}
        length = None
        for sig, li, hi, seq_len, feats in buf:
            prefix = f"{sig}_" if use_signal_prefix else ""
            if length is None:
                length = seq_len
            for k, v in feats.items():
                wide[f"{prefix}{k}_L{li}_H{hi}"] = v

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

    # Stream JSONL, pivot per request_id, write in batches
    schema_cols = set(schema.names)
    checked_first_row = False
    total_rows = 0
    wide_rows: list[dict] = []
    current_rid: int | None = None
    buf: list[tuple[str, int, int, int, dict]] = []
    n_expected = len(sample_meta)  # one wide row per request_id

    def _flush_batch(writer, rows: list[dict]) -> int:
        if not rows:
            return 0
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)
        return len(rows)

    pbar = tqdm(total=n_expected, desc="Pivoting to parquet", unit="rows")

    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        with open(svd_features_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snap = SVDSnapshot.from_jsonl_row(json.loads(line))
                feats = _parse_snap_features(snap)

                if current_rid is not None and snap.request_id != current_rid:
                    row = _pivot_request(buf, current_rid)
                    # Check first row for columns that would be silently dropped
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
                buf.append((snap.feature_group, snap.layer_idx, snap.head, snap.L, feats))

        # Flush last request + remaining batch
        if buf and current_rid is not None:
            wide_rows.append(_pivot_request(buf, current_rid))
            pbar.update(1)
        total_rows += _flush_batch(writer, wide_rows)

    pbar.close()

    if total_rows == 0:
        log(f"No data written to {out_path}")
        return

    log(f"Parquet saved: {out_path} ({total_rows} rows, {len(feature_columns)} feature columns)")


# ── CLI ────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name.")
@click.option(
    "--dataset",
    "dataset_name",
    default="halueval_hallucination",
    show_default=True,
    type=click.Choice(list(DATASET_REGISTRY.keys()) + ["all"]),
    help="Dataset to use.",
)
@click.option(
    "--hf-org",
    "hf_org",
    default=DEFAULT_HF_ORG,
    show_default=True,
    help="HuggingFace org hosting the pre-split datasets.",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    show_default=True,
    help="Max samples to process (default: all).",
)
@click.option("--svd-rank", default=4, show_default=True, help="SVD rank (k).")
@click.option(
    "--method",
    type=click.Choice(["randomized", "lanczos"]),
    default=None,
    help="SVD algorithm. [default: randomized]",
)
@click.option(
    "--heads",
    type=str,
    default="0",
    callback=lambda ctx, param, value: tuple(int(x.strip()) for x in value.split(",")),
    show_default=True,
    help="Comma-separated head indices to analyze.",
)
@click.option(
    "--scores-matrix",
    "scores_matrix",
    is_flag=True,
    default=False,
    help="Compute scores-matrix SVD features.",
)
@click.option(
    "--degree-normalized",
    "degree_normalized",
    is_flag=True,
    default=False,
    help="Compute degree-normalized matrix features.",
)
@click.option(
    "--attention-diagonal",
    "attention_diagonal",
    is_flag=True,
    default=False,
    help="Compute attention diagonal features (LLM-Check).",
)
@click.option(
    "--laplacian-eigvals",
    "laplacian_eigvals",
    is_flag=True,
    default=False,
    help="Compute Laplacian eigenvalue features (LapEigvals).",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Seq length threshold for materialized vs matrix-free. [default: 2048]",
)
@click.option(
    "--parquet",
    "parquet",
    is_flag=True,
    default=False,
    help="Also save results as wide Parquet (shade-compatible format).",
)
def main(
    model: str,
    dataset_name: str,
    hf_org: str,
    max_samples: int | None,
    svd_rank: int,
    method: str | None,
    heads: tuple[int, ...],
    scores_matrix: bool,
    degree_normalized: bool,
    attention_diagonal: bool,
    laplacian_eigvals: bool,
    threshold: int | None,
    parquet: bool,
) -> None:
    """Run prefill-only spectral feature extraction on a labeled dataset."""
    import vllm

    import glassbox.backends.svd_backend as svd_mod
    from glassbox.config import GlassboxConfig

    # Load dataset(s)
    if dataset_name == "all":
        all_samples: list[dict] = []
        for name in DATASET_REGISTRY:
            all_samples.extend(load_dataset_samples(name, max_samples, hf_org=hf_org))
        samples = all_samples
    else:
        samples = load_dataset_samples(dataset_name, max_samples, hf_org=hf_org)

    # Prefill-only: SVD fires every token, max_tokens=1 (vLLM minimum)
    max_tokens = 1

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("experiments/results") / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    # Write config metadata (num_layers added after LLM creation below)
    config = {
        "model": model,
        "dataset": dataset_name,
        "max_samples": max_samples,
        "svd_interval": 1,
        "svd_rank": svd_rank,
        "method": method or "randomized",
        "heads": list(heads) if heads else [0],
        "scores_matrix": scores_matrix,
        "degree_normalized": degree_normalized,
        "attention_diagonal": attention_diagonal,
        "laplacian_eigvals": laplacian_eigvals,
        "max_tokens": max_tokens,
    }

    svd_features_path = outdir / "svd_features.jsonl"

    log(f"Results directory: {outdir}")
    log(f"Model: {model}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"SVD: rank={svd_rank}, method={method or 'randomized'}")
    if heads:
        log(f"Heads: {list(heads)}")
    if degree_normalized:
        log(f"Degree-normalized: enabled (threshold={threshold or 2048})")
    if attention_diagonal:
        log(f"Attention diagonal: enabled (threshold={threshold or 512})")
    if laplacian_eigvals:
        log(f"Laplacian eigvals: enabled (threshold={threshold or 512})")

    any_enabled = scores_matrix or degree_normalized or attention_diagonal or laplacian_eigvals
    if not any_enabled:
        raise click.UsageError(
            "At least one of --scores-matrix, --degree-normalized,"
            " --attention-diagonal, or --laplacian-eigvals must be enabled."
        )

    # Configure glassbox backend
    gb_kwargs: dict = {"output": str(svd_features_path)}

    if scores_matrix:
        scores_cfg: dict = {"interval": 1, "rank": svd_rank}
        if method is not None:
            scores_cfg["method"] = method
        if heads:
            scores_cfg["heads"] = list(heads)
        gb_kwargs["scores_matrix"] = scores_cfg

    if degree_normalized:
        dn_cfg: dict = {"enabled": True, "interval": 1, "rank": svd_rank}
        if method is not None:
            dn_cfg["method"] = method
        if heads:
            dn_cfg["heads"] = list(heads)
        if threshold is not None:
            dn_cfg["threshold"] = threshold
        gb_kwargs["degree_normalized_matrix"] = dn_cfg

    if attention_diagonal:
        ad_cfg: dict = {"enabled": True, "interval": 1}
        if heads:
            ad_cfg["heads"] = list(heads)
        if threshold is not None:
            ad_cfg["threshold"] = threshold
        gb_kwargs["attention_diagonal"] = ad_cfg

    if laplacian_eigvals:
        lap_cfg: dict = {"enabled": True, "interval": 1}
        if heads:
            lap_cfg["heads"] = list(heads)
        if threshold is not None:
            lap_cfg["threshold"] = threshold
        gb_kwargs["laplacian_eigvals"] = lap_cfg

    gb_config = GlassboxConfig(**gb_kwargs)
    svd_mod.SVDTritonAttentionImpl.config = gb_config

    # Create vLLM engine
    log("Creating vLLM engine with CUSTOM attention backend")
    # Chunked prefill and prefix caching both cause the SVD backend to
    # see partial Q tensors instead of the full sequence:
    # - Chunked prefill splits long sequences into multiple forward passes
    # - Prefix caching skips cached prefix tokens, only forwarding the
    #   uncached suffix (e.g., in evaluate mode the full phase shares the
    #   question prefix, so only the response tokens are forwarded)
    # Disable both until the backend can reconstruct full Q from partial views.
    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    # Save num_layers from model config so parquet can be regenerated without HF download
    num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    config["num_layers"] = num_layers
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    samples_path = outdir / "samples.jsonl"
    samples_f = open(samples_path, "w")

    request_counter = 0
    for i, sample in enumerate(samples):
        try:
            # Two-phase prefill: question-only baseline, then full (prompt + response)
            prompt_q = f"Q: {sample['question']}\nA:"
            prompt_full = f"Q: {sample['question']}\nA: {sample['response']}"

            for phase, prompt in [
                ("question", prompt_q),
                ("full", prompt_full),
            ]:
                outputs = llm.generate(
                    [prompt],
                    vllm.SamplingParams(max_tokens=max_tokens),
                )
                sample_row = {
                    "request_id": request_counter,
                    "sample_id": sample["idx"],
                    "phase": phase,
                    "dataset": dataset_name,
                    **sample,
                    "prompt_length": len(prompt),
                    "generated": outputs[0].outputs[0].text,
                }
                samples_f.write(json.dumps(sample_row) + "\n")
                samples_f.flush()
                request_counter += 1

        except Exception as e:
            log(f"  [{i + 1}/{len(samples)}] ERROR: {e}")
            continue

        label_str = "HALL" if sample["label"] == 1 else "OK"
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  [{i + 1}/{len(samples)}] {label_str}")

    samples_f.close()
    log(f"Done! {len(samples)} samples, {request_counter} requests")
    log(f"  samples:      {samples_path}")
    log(f"  svd features: {svd_features_path}")

    if parquet:
        feature_columns = _build_feature_columns(
            num_layers,
            heads,
            scores_matrix,
            degree_normalized,
            attention_diagonal,
            laplacian_eigvals,
            ad_top_k=gb_config.attention_diagonal.top_k,
            lap_top_k=gb_config.laplacian_eigvals.top_k,
        )
        parquet_path = outdir / "features.parquet"
        _write_parquet(svd_features_path, samples_path, parquet_path, feature_columns)


if __name__ == "__main__":
    main()
