"""Prefill-only spectral feature extraction on labeled datasets.

For each sample, runs two prefill phases through the model:
  1. "question" phase: prefill with just the prompt
  2. "full" phase: prefill with prompt + known response
SVD features are extracted from attention internals during each prefill.
No text is generated — max_tokens=1 is a vLLM requirement.

Programmatic callers use :func:`run_extraction`, which takes the same loop with
the phase list and the sample-to-prompt mapping as parameters.

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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import click

from glassbox.config import SIGNAL_NAMES, GlassboxConfig, parse_signal_names
from glassbox.feature_table import build_feature_columns, write_parquet

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_HF_ORG = "dmaniloff"

# How a sample dict becomes the string the model prefills on, per phase.
#
# This is the convention for the labeled datasets in DATASET_REGISTRY: each row
# arrives as a (prompt, response, label) triple where the response is canned and
# the label ships with the dataset, so every sample goes through the identical
# wrapper and nothing depends on a string the model generated itself.
#
# Callers whose labels come from the model's *own* generation must override this
# via ``run_extraction(prompt_builders=...)`` so that the prefill reproduces the
# exact context that produced the response — otherwise the features describe a
# forward pass that never happened.  See the ``prompt_builders`` docstring.
_DEFAULT_PHASE_PROMPTS: dict[str, Callable[[dict], str]] = {
    "question": lambda s: f"Q: {s['question']}\nA:",
    "full": lambda s: f"Q: {s['question']}\nA: {s['response']}",
}


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

    run_extraction(
        samples=samples,
        model=model,
        signals=signals,
        rank=svd_rank,
        method=method,
        heads=heads,
        threshold=threshold,
        otel=True if otel else None,
        outdir=outdir,
        dataset_name=dataset_name,
        parquet=parquet,
    )


def run_extraction(
    *,
    samples: list[dict],
    model: str,
    dataset_name: str,
    signals: tuple[str, ...] = ("spectral",),
    rank: int | None = None,
    method: str | None = None,
    heads: tuple[int, ...] | list[int] = (),
    threshold: int | None = None,
    block_size: int | None = None,
    otel: bool | None = None,
    outdir: str | Path | None = None,
    parquet: bool = False,
    phases: tuple[str, ...] = ("question", "full"),
    prompt_builders: dict[str, Callable[[dict], str]] | None = None,
) -> Path:
    """Run prefill-only feature extraction on a list of samples.

    Takes the same flat knobs the CLI exposes and builds the ``GlassboxConfig``
    internally via :meth:`GlassboxConfig.from_cli_args`, which applies them
    uniformly to every enabled signal.  That is the only shape the extraction
    path supports — ``_build_feature_columns`` takes a single head list for all
    signals — so the signature states it rather than accepting a richer config
    it would silently flatten.  ``glassbox.yaml`` still applies underneath
    (precedence: these arguments > yaml > field defaults).

    Parameters
    ----------
    samples
        List of dicts, each with ``question``, ``response``, ``label``, ``idx``.
    model
        HuggingFace model name.
    dataset_name
        Provenance label stored in ``config.json`` and on every ``samples.jsonl``
        row, and surfaced as the parquet ``source`` column.  Required: a result
        directory that does not record what it was run on cannot be traced back.
    signals
        Signal names to enable; all others are explicitly disabled.
    rank, method, heads, threshold, block_size, otel
        Passed through to :meth:`GlassboxConfig.from_cli_args`.  *None* means
        "leave at the configured default".  Note there is no ``interval``
        knob — see the ``from_cli_args`` call below for why it is pinned to 1.
    outdir
        Output directory. Auto-generated under ``experiments/results/`` if *None*.
    parquet
        If *True*, also write a wide ``features.parquet`` file.
    phases
        Which prefill phases to run.  Default ``("question", "full")``.
        Use ``("full",)`` to skip the question-only baseline.
    prompt_builders
        Per-phase overrides for turning a sample into the string the model
        prefills on, merged over :data:`_DEFAULT_PHASE_PROMPTS`.

        Override this whenever the label is derived from the model's own
        generation rather than shipped with the dataset.  Features are only
        comparable to such a label if the prefill reproduces the context that
        produced the response — same instructions, same chat template, no
        re-wrapping of text that already carries its own formatting::

            run_extraction(
                samples=samples,
                model=model,
                phases=("full",),
                prompt_builders={"full": lambda s: s["chat_prompt"] + s["response"]},
            )

        The default wrapper would instead prefill ``Q: {question}\\nA: {response}``,
        which for an already-formatted sample can share no prefix at all with what
        the model actually saw.

    Returns
    -------
    Path
        The output directory containing ``svd_features.jsonl``,
        ``samples.jsonl``, ``config.json``, and optionally ``features.parquet``.
    """
    # Imported lazily: vLLM costs ~10s to import, which the Click CLI would
    # otherwise pay just to print --help.
    import vllm

    import glassbox.backends.svd_backend as svd_mod

    phase_prompts = {**_DEFAULT_PHASE_PROMPTS, **(prompt_builders or {})}
    missing = [p for p in phases if p not in phase_prompts]
    if missing:
        raise ValueError(
            f"No prompt builder for phase(s) {missing}. "
            f"Known phases: {sorted(phase_prompts)}. "
            "Pass prompt_builders={'<phase>': lambda sample: ...} to add one."
        )

    # from_cli_args silently enables nothing for an unrecognised name, which
    # would run the whole extraction and emit an empty features file.
    choices = f"Choose from: {', '.join(SIGNAL_NAMES)}."
    if not signals:
        raise ValueError(f"At least one signal is required. {choices}")
    unknown = [s for s in signals if s not in SIGNAL_NAMES]
    if unknown:
        raise ValueError(f"Unknown signal(s) {unknown}. {choices}")

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

    # Built here rather than by the caller so the output path goes in at
    # construction time — the config is frozen, and patching it afterwards with
    # model_copy(update={"output": {...}}) assigns the dict verbatim without
    # validation, leaving config.output a plain dict that breaks attribute
    # access downstream.
    config = GlassboxConfig.from_cli_args(
        signals=signals,
        # Pinned, not a parameter. Extraction is prefill-only, so each request is
        # a single forward pass and the backend gates on `state.step % interval`
        # with step already incremented to 1. Only interval=1 satisfies that;
        # any other value emits nothing and the run "succeeds" with no features
        # file at all.
        interval=1,
        rank=rank,
        method=method,
        heads=heads,
        threshold=threshold,
        block_size=block_size,
        output_path=str(svd_features_path),
        otel=otel,
    )

    log(f"Results directory: {outdir_path}")
    log(f"Model: {model}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"Signals: {', '.join(signals)}")
    log(f"Params: rank={rank}, method={method}, heads={list(heads) or [0]}")

    # Save extraction metadata.  These are the arguments as given, not values
    # recovered from the config — `glassbox-analyze` reads svd_interval/svd_rank
    # for its config banner and regenerating features.parquet needs heads.  The
    # resolved nested config rides along as the authoritative record.
    extract_metadata = {
        "model": model,
        "dataset": dataset_name,
        "signals": list(signals),
        "phases": list(phases),
        "n_samples": len(samples),
        "svd_interval": 1,
        "heads": list(heads) or [0],
        "svd_rank": rank,
        "method": method or "randomized",
        "config": config.model_dump(mode="json"),
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
    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    # Save num_layers from model config so parquet can be regenerated without HF download
    num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    extract_metadata["num_layers"] = num_layers
    (outdir_path / "config.json").write_text(json.dumps(extract_metadata, indent=2))

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
        feature_columns = build_feature_columns(
            num_layers, list(heads) or [0], tuple(signals), config
        )
        parquet_path = outdir_path / "features.parquet"
        n_rows = write_parquet(svd_features_path, samples_path, parquet_path, feature_columns)
        log(f"Parquet saved: {parquet_path} ({n_rows} rows, {len(feature_columns)} columns)")

    return outdir_path


if __name__ == "__main__":
    main()
