"""
Entry-point script that launches vLLM with the custom SVD attention backend.

Usage:
    glassbox-run [OPTIONS]
    glassbox-run --signal spectral,routing --rank 2 --heads 0,1,2
    glassbox-run --model facebook/opt-350m --method lanczos

Options can also be set via glassbox.yaml in the working directory.
CLI args take highest precedence.
"""

from __future__ import annotations

import logging

import click
import vllm

# Import triggers @register_backend(AttentionBackendEnum.CUSTOM)
import glassbox.backends.svd_backend as svd_mod
from glassbox.config import SIGNAL_NAMES, GlassboxConfig, parse_signal_names

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
    "--signal",
    "signals",
    multiple=True,
    default=None,
    callback=parse_signal_names,
    help=(
        "Signals to enable. Repeatable or comma-separated. "
        f"Choices: {', '.join(SIGNAL_NAMES)}. "
        "Routing Hodge parameters (hodge_target_cv, etc.) are configurable via YAML only."
    ),
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
    callback=lambda ctx, param, value: tuple(int(x.strip()) for x in value.split(","))
    if value
    else (),
    help="Comma-separated head indices to analyze. [default: from config ([0])]",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Sequence length threshold for materialized vs matrix-free. [default: from config (512)]",
)
@click.option(
    "--block-size",
    type=int,
    default=None,
    help="Block size for blocked-streaming matvecs. [default: from config (256)]",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help=(
        "JSONL output file path "
        "(for debugging/archival; typical runner use is --otel). "
        "[default: log to stderr]"
    ),
)
@click.option(
    "--otel/--no-otel",
    default=None,
    help="Emit snapshots as OpenTelemetry spans. [default: from config (False)]",
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
    signals: tuple[str, ...],
    interval: int | None,
    rank: int | None,
    method: str | None,
    heads: tuple[int, ...],
    output: str | None,
    otel: bool | None,
    threshold: int | None,
    block_size: int | None,
    max_tokens: int,
    prompt: str,
) -> None:
    """Launch vLLM with the custom SVD attention backend."""

    # vLLM calls impl_cls(). There doesn't seem to be a way to inject extra
    # args through the vLLM call path. So we set the config as a class
    # variable on SVDTritonAttentionImpl before vLLM creates the engine.
    config = GlassboxConfig.from_cli_args(
        signals=signals,
        interval=interval,
        rank=rank,
        method=method,
        heads=heads,
        threshold=threshold,
        block_size=block_size,
        output_path=output,
        otel=otel,
    )
    svd_mod.SVDTritonAttentionImpl.set_config(config)

    logger.info("Creating vLLM engine with CUSTOM attention backend")
    logger.info("Model: %s", model)
    logger.info(
        "Signals: spectral=%s routing=%s tracker=%s selfattn=%s laplacian=%s",
        "enabled" if config.spectral.enabled else "disabled",
        "enabled" if config.routing.enabled else "disabled",
        "enabled" if config.tracker.enabled else "disabled",
        "enabled" if config.selfattn.enabled else "disabled",
        "enabled" if config.laplacian.enabled else "disabled",
    )
    if config.spectral.enabled:
        logger.info(
            "Spectral: interval=%s rank=%s method=%s heads=%s",
            config.spectral.interval,
            config.spectral.rank,
            config.spectral.method,
            config.spectral.heads,
        )
    if config.routing.enabled:
        logger.info(
            "Routing: interval=%s rank=%s method=%s heads=%s",
            config.routing.interval,
            config.routing.rank,
            config.routing.method,
            config.routing.heads,
        )
    if config.tracker.enabled:
        logger.info(
            "Tracker: interval=%s rank=%s method=%s heads=%s",
            config.tracker.interval,
            config.tracker.rank,
            config.tracker.method,
            config.tracker.heads,
        )
    if config.selfattn.enabled:
        logger.info(
            "SelfAttn: interval=%s heads=%s",
            config.selfattn.interval,
            config.selfattn.heads,
        )
    if config.laplacian.enabled:
        logger.info(
            "Laplacian: interval=%s heads=%s top_k=%s",
            config.laplacian.interval,
            config.laplacian.heads,
            config.laplacian.top_k,
        )
    if config.emit.otel:
        logger.info("OTel emission: enabled")

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
