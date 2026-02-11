"""
Entry-point script that launches vLLM with the custom SVD attention backend.

Usage:
    GLASSBOX_SVD_INTERVAL=16 GLASSBOX_SVD_RANK=2 python -m glassbox.svd_backend_runner

Environment variables (see glassbox.backends.svd_backend.py):
    GLASSBOX_SVD_INTERVAL  - run SVD every N decode steps (default: 32)
    GLASSBOX_SVD_RANK      - number of singular values (default: 4)
    GLASSBOX_SVD_METHOD    - "randomized" or "lanczos" (default: "randomized")
    GLASSBOX_SVD_HEADS     - comma-separated head indices (default: "0")
    GLASSBOX_MODEL         - HuggingFace model name (default: "facebook/opt-125m")
"""

from __future__ import annotations

import logging
import os

import vllm

# Import triggers @register_backend(AttentionBackendEnum.CUSTOM)
import glassbox.backends.svd_backend  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = os.environ.get("GLASSBOX_MODEL", "facebook/opt-125m")


def main() -> None:
    logger.info("Creating vLLM engine with CUSTOM attention backend")
    logger.info("Model: %s", MODEL)
    logger.info(
        "SVD config: interval=%s rank=%s method=%s heads=%s",
        glassbox.backends.svd_backend.SVD_INTERVAL,
        glassbox.backends.svd_backend.SVD_RANK,
        glassbox.backends.svd_backend.SVD_METHOD,
        glassbox.backends.svd_backend.SVD_HEADS,
    )

    llm = vllm.LLM(
        model=MODEL,
        attention_backend="CUSTOM",
        enforce_eager=True,
    )

    prompts = [
        "The future of artificial intelligence is",
    ]

    logger.info("Starting generation...")
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64),
    )

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        logger.info("Prompt: %s", prompt)
        logger.info("Generated: %s", generated)


if __name__ == "__main__":
    main()
