"""
Entry-point script that launches vLLM with the custom SVD attention backend.

Usage:
    python -m glassbox.backends.runner [OPTIONS]
    python -m glassbox.backends.runner --interval 16 --rank 2 --heads 0,1,2
    python -m glassbox.backends.runner --model facebook/opt-350m --method lanczos
    python -m glassbox.backends.runner --config glassbox.yaml

Options can also be set via glassbox.yaml or legacy GLASSBOX_SVD_* env vars.
CLI args take highest precedence.
"""

from __future__ import annotations

import logging

import click
import vllm

# Import triggers @register_backend(AttentionBackendEnum.CUSTOM)
import glassbox.backends.svd_backend as svd_mod
from glassbox.config import GlassboxConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    default="facebook/opt-125m",
    show_default=True,
    help="HuggingFace model name.",
)
@click.option(
    "--interval",
    type=int,
    default=None,
    help="Run SVD every N decode steps. [default: from config (32)]",
)
@click.option(
    "--rank",
    type=int,
    default=None,
    help="Number of singular values to compute. [default: from config (4)]",
)
@click.option(
    "--method",
    type=click.Choice(["randomized", "lanczos"]),
    default=None,
    help="SVD algorithm. [default: from config (randomized)]",
)
@click.option(
    "--heads",
    type=str,
    default=None,
    callback=lambda ctx, param, value: tuple(int(x.strip()) for x in value.split(",")) if value else (),
    help="Comma-separated head indices to analyze. [default: from config ([0])]",
)
@click.option(
    "--operator",
    type=click.Choice(["S", "M"]),
    default=None,
    help="Operator to SVD: S=scores, M=degree-normalized. [default: from config (S)]",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Materialize M for L <= threshold, matrix-free above. [default: from config (2048)]",
)
@click.option(
    "--block-size",
    type=int,
    default=None,
    help="Block size for blocked-streaming matvecs. [default: from config (256)]",
)
@click.option(
    "--hodge/--no-hodge",
    default=None,
    help="Compute Hodge decomposition features. [default: from config (False)]",
)
@click.option(
    "--hodge-target-cv",
    type=float,
    default=None,
    help="Target CV for adaptive curl sampling. [default: from config (0.05)]",
)
@click.option(
    "--hodge-curl-seed",
    type=int,
    default=None,
    help="Seed for curl triangle sampling. [default: from config (42)]",
)
@click.option(
    "--hodge-confidence",
    type=float,
    default=None,
    help="Bernstein bound confidence level. [default: from config (0.95)]",
)
@click.option(
    "--hodge-pilot-size",
    type=int,
    default=None,
    help="Pilot triangle samples for kurtosis estimation. [default: from config (100)]",
)
@click.option(
    "--hodge-min-samples",
    type=int,
    default=None,
    help="Minimum triangle sample count. [default: from config (200)]",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="JSONL output file path. [default: from config (log to stderr)]",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file. [default: glassbox.yaml if present]",
)
@click.option(
    "--max-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Maximum tokens to generate.",
)
@click.option(
    "--prompt",
    default="The future of artificial intelligence is",
    show_default=True,
    help="Input prompt.",
)
def main(
    model: str,
    interval: int | None,
    rank: int | None,
    method: str | None,
    heads: tuple[int, ...],
    output: str | None,
    config_file: str | None,
    operator: str | None,
    threshold: int | None,
    block_size: int | None,
    hodge: bool | None,
    hodge_target_cv: float | None,
    hodge_curl_seed: int | None,
    hodge_confidence: float | None,
    hodge_pilot_size: int | None,
    hodge_min_samples: int | None,
    max_tokens: int,
    prompt: str,
) -> None:
    """Launch vLLM with the custom SVD attention backend."""

    # Build nested config overrides from CLI args
    overrides: dict = {}
    scores_matrix: dict = {}
    degree_normalized_matrix: dict = {}

    if interval is not None:
        scores_matrix["interval"] = interval
    if rank is not None:
        scores_matrix["rank"] = rank
    if method is not None:
        scores_matrix["method"] = method
    if heads:
        scores_matrix["heads"] = list(heads)
    if output is not None:
        overrides["output"] = output

    # Handle --operator for backward compat
    if operator == "M":
        scores_matrix["enabled"] = False
        degree_normalized_matrix["enabled"] = True
        if interval is not None:
            degree_normalized_matrix["interval"] = interval
        if rank is not None:
            degree_normalized_matrix["rank"] = rank
        if method is not None:
            degree_normalized_matrix["method"] = method
        if heads:
            degree_normalized_matrix["heads"] = list(heads)

    # M-specific params
    if threshold is not None:
        degree_normalized_matrix["threshold"] = threshold
    if block_size is not None:
        degree_normalized_matrix["block_size"] = block_size
    if hodge is not None:
        degree_normalized_matrix["hodge"] = hodge
    if hodge_target_cv is not None:
        degree_normalized_matrix["hodge_target_cv"] = hodge_target_cv
    if hodge_curl_seed is not None:
        degree_normalized_matrix["hodge_curl_seed"] = hodge_curl_seed
    if hodge_confidence is not None:
        degree_normalized_matrix["hodge_confidence"] = hodge_confidence
    if hodge_pilot_size is not None:
        degree_normalized_matrix["hodge_pilot_size"] = hodge_pilot_size
    if hodge_min_samples is not None:
        degree_normalized_matrix["hodge_min_samples"] = hodge_min_samples

    if scores_matrix:
        overrides["scores_matrix"] = scores_matrix
    if degree_normalized_matrix:
        overrides["degree_normalized_matrix"] = degree_normalized_matrix

    # Handle --config YAML file: read it and merge (CLI overrides beat YAML)
    if config_file:
        import yaml

        with open(config_file) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key, val in yaml_data.items():
            if key not in overrides:
                overrides[key] = val
            elif isinstance(overrides[key], dict) and isinstance(val, dict):
                overrides[key] = {**val, **overrides[key]}

    # vLLM calls impl_cls(). There doesn't seem to be a way to inject extra
    # args through the vLLM call path. So we set the config as a class
    # variable on SVDTritonAttentionImpl before vLLM creates the engine.
    config = GlassboxConfig(**overrides)
    svd_mod.SVDTritonAttentionImpl.config = config

    logger.info("Creating vLLM engine with CUSTOM attention backend")
    logger.info("Model: %s", model)
    logger.info(
        "Config: scores_matrix=%s degree_normalized_matrix=%s",
        "enabled" if config.scores_matrix.enabled else "disabled",
        "enabled" if config.degree_normalized_matrix.enabled else "disabled",
    )
    if config.scores_matrix.enabled:
        logger.info(
            "Scores matrix: interval=%s rank=%s method=%s heads=%s",
            config.scores_matrix.interval,
            config.scores_matrix.rank,
            config.scores_matrix.method,
            config.scores_matrix.heads,
        )
    if config.degree_normalized_matrix.enabled:
        logger.info(
            "Degree-normalized matrix: interval=%s rank=%s method=%s heads=%s",
            config.degree_normalized_matrix.interval,
            config.degree_normalized_matrix.rank,
            config.degree_normalized_matrix.method,
            config.degree_normalized_matrix.heads,
        )

    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
    )

    logger.info("Starting generation...")
    outputs = llm.generate(
        [prompt],
        vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens),
    )

    for output in outputs:
        logger.info("Prompt: %s", output.prompt)
        logger.info("Generated: %s", output.outputs[0].text)


if __name__ == "__main__":
    main()
