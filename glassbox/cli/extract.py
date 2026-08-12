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
    glassbox-extract --signal spectral --dataset halueval_hallucination
    glassbox-extract --signal spectral,routing --dataset all
    glassbox-extract --signal selfattn --dataset halueval_hallucination --max-samples 50
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import vllm

import glassbox.backends.svd_backend as svd_mod
from glassbox.config import SIGNAL_NAMES, GlassboxConfig, parse_signal_names
from glassbox.results import (
    SPECTRAL_FEATURE_NAMES,
    RoutingFeatures,
    SelfAttnFeatures,
    SVDSnapshot,
    TrackerFeatures,
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


def _is_list_field(model: type, name: str) -> bool:
    """True if the pydantic model field is list-typed (expanded into indexed columns)."""
    annotation = model.model_fields[name].annotation
    return getattr(annotation, "__origin__", None) is list


# Hodge feature names derived from RoutingFeatures model
_HODGE_FEATURE_NAMES = [
    f"hodge_{f}"
    for f in RoutingFeatures.model_fields
    if f not in _SKIP_FEATURE_FIELDS and f not in SPECTRAL_FEATURE_NAMES
]

# Tracker feature names derived from TrackerFeatures model
_AT_FEATURE_NAMES = [
    f"at_{f}"
    for f in TrackerFeatures.model_fields
    if f not in _SKIP_FEATURE_FIELDS and f not in SPECTRAL_FEATURE_NAMES
]

# SelfAttn scalar feature names (list fields expanded separately as indexed columns)
_AD_FEATURE_NAMES = [
    f"ad_{f}"
    for f in SelfAttnFeatures.model_fields
    if f not in _SKIP_FEATURE_FIELDS and not _is_list_field(SelfAttnFeatures, f)
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
    if snap.signal == "selfattn":
        non_spectral_prefix = "ad_"
    elif snap.signal == "tracker":
        non_spectral_prefix = "at_"
    elif snap.signal == "laplacian":
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
    signals: tuple[str, ...],
    ad_top_k: int = 0,
    lap_top_k: int = 10,
) -> list[str]:
    """Pre-compute all feature column names from model architecture.

    Column names follow the pattern: {signal_prefix}{feature}_L{layer}_H{head}
    where signal_prefix is only added when multiple signals are enabled.
    """
    signal_entries: list[tuple[str, list[str]]] = []
    if "spectral" in signals:
        signal_entries.append(("spectral", list(SPECTRAL_FEATURE_NAMES)))
    if "routing" in signals:
        signal_entries.append(("routing", list(SPECTRAL_FEATURE_NAMES) + _HODGE_FEATURE_NAMES))
    if "tracker" in signals:
        signal_entries.append(("tracker", list(SPECTRAL_FEATURE_NAMES) + _AT_FEATURE_NAMES))
    if "selfattn" in signals:
        ad_cols = list(_AD_FEATURE_NAMES)
        if ad_top_k > 0:
            ad_cols += [f"ad_eigval_{i}" for i in range(ad_top_k)]
        signal_entries.append(("selfattn", ad_cols))
    if "laplacian" in signals:
        lap_cols = [f"lap_eigval_{i}" for i in range(lap_top_k)]
        signal_entries.append(("laplacian", lap_cols))

    columns: list[str] = []
    for signal_name, feature_names in signal_entries:
        prefix = f"{signal_name}_"
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
        {signal}_{feature}_L{layer}_H{head}  (e.g. spectral_sv_ratio_L0_H0)
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

    def _pivot_request(buf: list[tuple[str, int, int, int, dict]], rid: int) -> dict:
        """Pivot one request_id's SVD rows into a single wide dict."""
        wide: dict = {"request_id": rid}
        length = None
        for sig, li, hi, seq_len, feats in buf:
            prefix = f"{sig}_"
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
                buf.append((snap.signal, snap.layer_idx, snap.head, snap.L, feats))

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
    "--signal",
    "signals",
    multiple=True,
    default=None,
    callback=parse_signal_names,
    help=(f"Signals to enable. Repeatable or comma-separated. Choices: {', '.join(SIGNAL_NAMES)}."),
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Seq length threshold for materialized vs matrix-free. [default: 512]",
)
@click.option(
    "--parquet",
    "parquet",
    is_flag=True,
    default=False,
    help="Also save results as wide Parquet (shade-compatible format).",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory for results. [default: experiments/results/{timestamp}]",
)
@click.option(
    "--otel/--no-otel",
    default=False,
    help=(
        "Also emit snapshots as OTel spans "
        "(for debugging; typical extract use is JSONL). [default: False]"
    ),
)
def main(
    model: str,
    dataset_name: str,
    hf_org: str,
    max_samples: int | None,
    svd_rank: int,
    method: str | None,
    heads: tuple[int, ...],
    signals: tuple[str, ...],
    threshold: int | None,
    parquet: bool,
    outdir: str | None,
    otel: bool,
) -> None:
    """Run prefill-only spectral feature extraction on a labeled dataset."""
    if not signals:
        raise click.UsageError(
            "At least one signal must be specified. "
            f"Use --signal with one or more of: {', '.join(SIGNAL_NAMES)}"
        )

    # Load dataset(s)
    if dataset_name == "all":
        all_samples: list[dict] = []
        for name in DATASET_REGISTRY:
            all_samples.extend(load_dataset_samples(name, max_samples, hf_org=hf_org))
        samples = all_samples
    else:
        samples = load_dataset_samples(dataset_name, max_samples, hf_org=hf_org)

    # Configure glassbox backend (interval=1 for prefill-only extraction)
    gb_config = GlassboxConfig.from_cli_args(
        signals=signals,
        interval=1,
        rank=svd_rank,
        method=method,
        heads=heads,
        threshold=threshold,
        otel=True if otel else None,
    )

    run_extraction(
        samples=samples,
        model=model,
        config=gb_config,
        outdir=outdir,
        dataset_name=dataset_name,
        parquet=parquet,
    )


def run_extraction(
    samples: list[dict],
    model: str,
    config: GlassboxConfig,
    outdir: str | Path | None = None,
    *,
    dataset_name: str = "unknown",
    parquet: bool = False,
    phases: tuple[str, ...] = ("question", "full"),
    max_model_len: int | None = None,
) -> Path:
    """Run prefill-only feature extraction on a list of samples.

    Parameters
    ----------
    samples
        List of dicts, each with ``question``, ``response``, ``label``, ``idx``.
    model
        HuggingFace model name.
    config
        Glassbox configuration (signals, ranks, heads, thresholds, etc.).
        If ``config.output.path`` is *None*, it is set to ``outdir/svd_features.jsonl``.
    outdir
        Output directory. Auto-generated under ``experiments/results/`` if *None*.
    dataset_name
        Label stored in output metadata and per-sample JSONL rows.
    parquet
        If *True*, also write a wide ``features.parquet`` file.
    phases
        Which prefill phases to run.  Default ``("question", "full")``.
        Use ``("full",)`` to skip the question-only baseline.
    max_model_len
        Cap vLLM's max sequence length (useful on smaller GPUs).

    Returns
    -------
    Path
        The output directory containing ``svd_features.jsonl``,
        ``samples.jsonl``, ``config.json``, and optionally ``features.parquet``.
    """
    # Prefill-only: we want the model to process the prompt without generating.
    # vLLM requires at least one token of generation, so we set max_tokens=1.
    prefill_params = vllm.SamplingParams(max_tokens=1)

    # Set up output directory
    if outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir_path = Path("experiments/results") / timestamp
    else:
        outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    svd_features_path = outdir_path / "svd_features.jsonl"

    # Set output path on config if not already set
    if config.output.path is None:
        config = config.model_copy(update={"output": {"path": str(svd_features_path)}})
    else:
        svd_features_path = Path(config.output.path)

    # Determine enabled signals from config
    enabled_signals = [s for s in SIGNAL_NAMES if getattr(config, s).enabled]

    log(f"Results directory: {outdir_path}")
    log(f"Model: {model}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"Signals: {', '.join(enabled_signals)}")

    # Save extraction metadata
    extract_metadata = {
        "model": model,
        "dataset": dataset_name,
        "signals": enabled_signals,
        "phases": list(phases),
    }

    svd_mod.SVDTritonAttentionImpl.set_config(config)

    # Create vLLM engine
    log("Creating vLLM engine with CUSTOM attention backend")
    # Chunked prefill and prefix caching both cause the SVD backend to
    # see partial Q tensors instead of the full sequence:
    # - Chunked prefill splits long sequences into multiple forward passes
    # - Prefix caching skips cached prefix tokens, only forwarding the
    #   uncached suffix (e.g., in evaluate mode the full phase shares the
    #   question prefix, so only the response tokens are forwarded)
    # Disable both until the backend can reconstruct full Q from partial views.
    vllm_kwargs: dict = dict(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    llm = vllm.LLM(**vllm_kwargs)

    # Save num_layers from model config so parquet can be regenerated without HF download
    num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    extract_metadata["num_layers"] = num_layers
    (outdir_path / "config.json").write_text(json.dumps(extract_metadata, indent=2))

    # Build phase prompts
    phase_prompts = {
        "question": lambda s: f"Q: {s['question']}\nA:",
        "full": lambda s: f"Q: {s['question']}\nA: {s['response']}",
    }

    samples_path = outdir_path / "samples.jsonl"
    samples_f = open(samples_path, "w")

    request_counter = 0
    for i, sample in enumerate(samples):
        for phase in phases:
            prompt = phase_prompts[phase](sample)
            outputs = llm.generate(
                [prompt],
                prefill_params,
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

        label_str = "HALL" if sample["label"] == 1 else "OK"
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  [{i + 1}/{len(samples)}] {label_str}")

    samples_f.close()
    log(f"Done! {len(samples)} samples, {request_counter} requests")
    log(f"  samples:      {samples_path}")
    log(f"  svd features: {svd_features_path}")

    if parquet:
        heads = config.spectral.heads
        feature_columns = _build_feature_columns(
            num_layers,
            heads,
            tuple(enabled_signals),
            ad_top_k=config.selfattn.top_k,
            lap_top_k=config.laplacian.top_k,
        )
        parquet_path = outdir_path / "features.parquet"
        _write_parquet(svd_features_path, samples_path, parquet_path, feature_columns)

    return outdir_path


if __name__ == "__main__":
    main()
