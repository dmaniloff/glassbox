"""Extract Q/K tensors from GPT-2 small for Cheeger benchmarks.

Usage:
    pip install transformers
    python benchmarks/extract_gpt2_qk.py [--output-dir benchmarks/fixtures/gpt2_qk]

Captures Q, K at each layer/head for a set of diverse prompts.
Saves as .pt files for reuse by bench_cheeger.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

PROMPTS = [
    # Wikipedia-style factual
    (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
        "company designed and built the tower from 1887 to 1889 as the centerpiece "
        "of the 1889 World's Fair. Although initially criticized by some of France's "
        "leading artists and intellectuals for its design, it has since become a "
        "global cultural icon of France and one of the most recognizable structures "
        "in the world."
    ),
    # Code-like
    (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
        "# Test the function\nfor i in range(10):\n    print(f'F({i}) = {fibonacci(i)}')\n"
    ),
    # Dialogue
    (
        "Alice: I've been thinking about switching to a plant-based diet.\n"
        "Bob: That's interesting! What motivated you?\n"
        "Alice: Mostly environmental concerns. I read that livestock farming "
        "accounts for a significant portion of greenhouse gas emissions.\n"
        "Bob: Have you considered the nutritional challenges?\n"
        "Alice: Yes, I've been researching protein sources and supplements.\n"
    ),
    # Technical reasoning
    (
        "The singular value decomposition (SVD) of a matrix M factors it as "
        "M = U Sigma V^T, where U and V are orthogonal and Sigma is diagonal "
        "with non-negative entries. The spectral gap, defined as sigma_1 - sigma_2, "
        "measures how well the rank-1 approximation captures the matrix. In the "
        "context of attention matrices, a large spectral gap indicates that the "
        "attention pattern is dominated by a single mode."
    ),
    # Narrative
    (
        "The old lighthouse keeper climbed the spiral staircase for the last time. "
        "After forty years of service, the automated system would take over tomorrow. "
        "He paused at each landing, running his weathered hand along the cold stone "
        "walls. The beam above still turned faithfully, cutting through the fog as "
        "it had for over a century. He wondered if the ships at sea would notice "
        "any difference."
    ),
]


def extract_qk(output_dir: str = "benchmarks/fixtures/gpt2_qk") -> None:
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    captures: dict[str, dict[str, torch.Tensor]] = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = input[0]
            B, L, _ = hidden.shape

            attn = model.transformer.h[layer_idx].attn
            qkv = attn.c_attn(hidden)
            q, k, _ = qkv.split(model.config.n_embd, dim=2)

            q = q.view(B, L, n_heads, head_dim).squeeze(0)  # [L, n_heads, head_dim]
            k = k.view(B, L, n_heads, head_dim).squeeze(0)

            for h in range(n_heads):
                key = f"layer{layer_idx}_head{h}"
                captures[key] = {"Q": q[:, h, :].detach().clone(), "K": k[:, h, :].detach().clone()}

        return hook_fn

    hooks = []
    for i in range(n_layers):
        h = model.transformer.h[i].attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    saved_count = 0
    for prompt_idx, prompt in enumerate(PROMPTS):
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        L = input_ids.shape[1]
        print(f"  Prompt {prompt_idx}: L={L} tokens")

        captures.clear()
        with torch.no_grad():
            model(input_ids)

        for key, tensors in captures.items():
            fname = f"prompt{prompt_idx}_{key}_L{L}.pt"
            torch.save(tensors, out / fname)
            saved_count += 1

    for h in hooks:
        h.remove()

    print(f"Saved {saved_count} Q/K pairs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GPT-2 Q/K fixtures")
    parser.add_argument(
        "--output-dir", default="benchmarks/fixtures/gpt2_qk",
        help="Directory for .pt output files",
    )
    args = parser.parse_args()
    extract_qk(args.output_dir)
