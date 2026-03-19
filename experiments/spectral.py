"""Spectral feature extraction and hallucination correlation analysis.

Usage:
    python experiments/spectral.py run                          # default: HaluEval, OPT-125m
    python experiments/spectral.py run --mode evaluate          # Mode A: prefill evaluation
    python experiments/spectral.py run --model Qwen/Qwen2-7B-Instruct --request-type chat_completions
    python experiments/spectral.py run --max-samples 50         # quick test

    python experiments/spectral.py analyze experiments/results/<timestamp>
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import click

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "facebook/opt-125m"

SPECTRAL_FEATURES = ["sv_ratio", "sv1", "sv_entropy"]

LABEL_COLORS = {0: "#1565C0", 1: "#C62828"}
LABEL_NAMES = {0: "Correct", 1: "Hallucinated"}


# ── Shared helpers ─────────────────────────────────────────────────────────


def log(msg: str) -> None:
    click.echo(f"[spectral] {msg}")


def plot_violin_pointrange(
    df,
    features: list[str],
    feat_labels: list[str],
    layer_ids: list[int],
    title: str,
    out_path: str,
    show_zero_line: bool = False,
) -> None:
    """Half-violin + pointrange + strip plot, split by label.

    Expects *df* to have columns: sample_idx, layer_idx, label, L, and each
    feature in *features*.  Dot size encodes sequence length (L).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_feats = len(features)
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(max(12, len(layer_ids) * 1.0), 4.5 * n_feats),
    )
    if n_feats == 1:
        axes = [axes]

    # Sequence-length → dot size mapping
    L_vals = df["L"].dropna()
    L_min, L_max = (L_vals.min(), L_vals.max()) if len(L_vals) else (1, 1)
    s_min, s_max = 15, 120

    def _size(L):
        t = (L - L_min) / max(L_max - L_min, 1)
        return s_min + t * (s_max - s_min)

    for ax_i, (feat, feat_label) in enumerate(zip(features, feat_labels)):
        ax = axes[ax_i]

        for label_val, side in [(0, "left"), (1, "right")]:
            color = LABEL_COLORS[label_val]
            sub = df[df["label"] == label_val]

            # Collect per-layer arrays
            layer_data, layer_pos, layer_pts = [], [], []
            for li_i, li in enumerate(layer_ids):
                chunk = sub[sub["layer_idx"] == li]
                vals = chunk[feat].dropna().values
                if len(vals) == 0:
                    continue
                layer_data.append(vals)
                layer_pos.append(li_i)
                Ls = chunk.loc[chunk[feat].notna(), "L"].values
                layer_pts.append(list(zip(vals, Ls)))

            if not layer_data:
                continue

            # ── Half violin ───────────────────────────────────────────
            parts = ax.violinplot(
                layer_data, positions=layer_pos,
                widths=0.8, showextrema=False, showmedians=False,
            )
            for pc in parts["bodies"]:
                verts = pc.get_paths()[0].vertices
                m = np.mean(verts[:, 0])
                if side == "left":
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], m, np.inf)
                pc.set_facecolor(color)
                pc.set_alpha(0.2)
                pc.set_edgecolor(color)
                pc.set_linewidth(0.8)

            # ── Pointrange (thin whisker, thick IQR, median dot) ──────
            offset = -0.13 if side == "left" else 0.13
            for li_i, vals in zip(layer_pos, layer_data):
                if len(vals) < 2:
                    ax.plot(li_i + offset, vals[0], "o", color=color, ms=5)
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3 - q1
                lo = max(vals.min(), q1 - 1.5 * iqr)
                hi = min(vals.max(), q3 + 1.5 * iqr)
                x = li_i + offset
                ax.plot([x, x], [lo, hi], color=color, lw=1,
                        solid_capstyle="round", zorder=5)
                ax.plot([x, x], [q1, q3], color=color, lw=4.5,
                        solid_capstyle="round", alpha=0.7, zorder=6)
                ax.plot(x, med, "o", color="white", ms=4.5,
                        mec=color, mew=1.2, zorder=7)

            # ── Strip dots (size = seq length) ────────────────────────
            jitter_base = -0.28 if side == "left" else 0.22
            rng = np.random.RandomState(42 + label_val)
            for li_i, pts in zip(layer_pos, layer_pts):
                for val, L in pts:
                    jx = jitter_base + rng.uniform(-0.08, 0.08)
                    ax.scatter(
                        li_i + jx, val, c=color, s=_size(L), alpha=0.5,
                        edgecolors="white", linewidths=0.3, zorder=4,
                    )

            # Legend entry (first panel only)
            if ax_i == 0:
                ax.plot([], [], color=color, lw=6, alpha=0.5,
                        label=LABEL_NAMES[label_val])

        if show_zero_line:
            ax.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)

        ax.set_ylabel(feat_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(labelsize=10)
        ax.set_xticks(range(len(layer_ids)))
        ax.set_xticklabels(
            [f"L{li}" for li in layer_ids] if ax_i == n_feats - 1 else [],
            fontsize=10,
        )

        if ax_i == 0:
            # Size legend
            for L_ex, lbl in [
                (int(L_min), f"L={int(L_min)}"),
                (int((L_min + L_max) // 2), f"L={int((L_min + L_max) // 2)}"),
                (int(L_max), f"L={int(L_max)}"),
            ]:
                ax.scatter([], [], c="gray", s=_size(L_ex), alpha=0.6,
                           edgecolors="white", linewidths=0.3, label=lbl)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
                      ncol=2, columnspacing=0.8, handletextpad=0.3)

    axes[-1].set_xlabel("Layer", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Distributions saved to {out_path}")


# ── Dataset loading ───────────────────────────────────────────────────────


def load_halueval(max_samples: int) -> list[dict]:
    from datasets import load_dataset

    log("Loading HaluEval qa...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    indices = list(range(len(ds)))
    random.Random(42).shuffle(indices)
    samples = []
    for i in indices:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        question = row["question"]
        # Each row produces 2 samples: right_answer (label=0) + hallucinated_answer (label=1)
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["right_answer"],
                "label": 0,
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["hallucinated_answer"],
                "label": 1,
            }
        )
    log(
        f"Loaded {len(samples)} samples ({sum(s['label'] for s in samples)} hallucinated)"
    )
    return samples


def load_truthfulqa(max_samples: int) -> list[dict]:
    from datasets import load_dataset

    log("Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    samples = []
    for i in indices:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        question = row["question"]
        incorrect = row["incorrect_answers"]
        if not incorrect:
            continue
        # Pair best_answer (label=0) with a random incorrect_answer (label=1)
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["best_answer"],
                "label": 0,
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": rng.choice(incorrect),
                "label": 1,
            }
        )
    log(
        f"Loaded {len(samples)} samples ({sum(s['label'] for s in samples)} hallucinated)"
    )
    return samples


DATASET_LOADERS = {
    "halueval": load_halueval,
    "truthfulqa": load_truthfulqa,
}


# ── CLI ────────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Spectral feature extraction and hallucination correlation analysis."""


@cli.command()
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name."
)
@click.option(
    "--dataset",
    "dataset_name",
    default="halueval",
    show_default=True,
    type=click.Choice(list(DATASET_LOADERS.keys())),
    help="Dataset to use.",
)
@click.option(
    "--max-samples", default=200, show_default=True, help="Max samples to process."
)
@click.option(
    "--svd-interval", default=16, show_default=True, help="SVD snapshot interval."
)
@click.option("--svd-rank", default=4, show_default=True, help="SVD rank (k).")
@click.option(
    "--request-type",
    "request_type",
    default="text_completions",
    show_default=True,
    type=click.Choice(["text_completions", "chat_completions"]),
    help="Request type (use chat_completions for instruct models).",
)
@click.option(
    "--max-tokens", default=128, show_default=True, help="Max tokens per completion."
)
@click.option(
    "--mode",
    default="generate",
    show_default=True,
    type=click.Choice(["generate", "evaluate"]),
    help="Mode: generate (Mode B) or evaluate (Mode A, prefill-only).",
)
def run(
    model: str,
    dataset_name: str,
    max_samples: int,
    svd_interval: int,
    svd_rank: int,
    request_type: str,
    max_tokens: int,
    mode: str,
) -> None:
    """Run spectral feature extraction on a labeled dataset."""
    import vllm

    import glassbox.backends.svd_backend as svd_mod
    from glassbox.config import GlassboxConfig

    # Load dataset
    samples = DATASET_LOADERS[dataset_name](max_samples)

    if mode == "evaluate":
        # Mode A: prefill-only, SVD fires every token
        svd_interval = 1
        max_tokens = 1

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("experiments/results") / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    # Write config metadata
    config = {
        "model": model,
        "dataset": dataset_name,
        "max_samples": max_samples,
        "svd_interval": svd_interval,
        "svd_rank": svd_rank,
        "request_type": request_type,
        "max_tokens": max_tokens,
        "mode": mode,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    svd_features_path = outdir / "svd_features.jsonl"

    log(f"Results directory: {outdir}")
    log(f"Model: {model}")
    log(f"Mode: {mode}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"SVD: interval={svd_interval}, rank={svd_rank}")

    # Configure glassbox backend
    gb_config = GlassboxConfig(
        spectral={"interval": svd_interval, "rank": svd_rank},
        output=str(svd_features_path),
    )
    svd_mod.SVDTritonAttentionImpl.config = gb_config

    # Create vLLM engine
    log("Creating vLLM engine with CUSTOM attention backend")
    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
    )

    samples_path = outdir / "samples.jsonl"
    samples_f = open(samples_path, "w")

    request_counter = 0
    for i, sample in enumerate(samples):
        try:
            if mode == "evaluate":
                # Two-phase prefill: question-only baseline, then full
                prompt_q = f"Q: {sample['question']}\nA:"
                prompt_full = f"Q: {sample['question']}\nA: {sample['response']}"

                for phase, prompt in [
                    ("question", prompt_q),
                    ("full", prompt_full),
                ]:
                    outputs = llm.generate(
                        [prompt],
                        vllm.SamplingParams(max_tokens=1),
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

            else:
                sampling_params = vllm.SamplingParams(max_tokens=max_tokens)
                if request_type == "chat_completions":
                    outputs = llm.chat(
                        messages=[{"role": "user", "content": sample["question"]}],
                        sampling_params=sampling_params,
                    )
                    generated = outputs[0].outputs[0].text
                    prompt = sample["question"]
                else:
                    prompt = f"Q: {sample['question']}\nA:"
                    outputs = llm.generate(
                        [prompt], sampling_params,
                    )
                    generated = outputs[0].outputs[0].text

                sample_row = {
                    "request_id": request_counter,
                    "sample_id": sample["idx"],
                    "phase": "generate",
                    "dataset": dataset_name,
                    **sample,
                    "prompt_length": len(prompt),
                    "generated": generated,
                    "response_length": len(sample.get("response", "")),
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


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, help="Override plot output directory.")
def analyze(results_dir: str, output_dir: str | None) -> None:
    """Analyze spectral features and correlate with hallucination labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import pointbiserialr
    from sklearn.metrics import roc_auc_score

    base = Path(results_dir)
    plot_dir = Path(output_dir) if output_dir else base

    # ── Load data ─────────────────────────────────────────────────────────
    svd_path = base / "svd_features.jsonl"
    legacy_path = base / "features.jsonl"

    if svd_path.exists():
        log("Loading svd_features.jsonl (structured output from backend)")
        svd_rows = []
        with open(svd_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                svs = row["singular_values"]
                # Compute derived features from raw singular values
                row["sv_ratio"] = (
                    svs[0] / svs[1] if len(svs) >= 2 and svs[1] > 0 else None
                )
                row["sv1"] = svs[0]
                total = sum(svs)
                if total > 0:
                    ps = [s / total for s in svs]
                    row["sv_entropy"] = -sum(p * math.log(p + 1e-12) for p in ps)
                else:
                    row["sv_entropy"] = None
                del row["singular_values"]
                svd_rows.append(row)

        df_svd = pd.DataFrame(svd_rows)

        # Load sample metadata and join on request_id
        samples_path = base / "samples.jsonl"
        if not samples_path.exists():
            click.echo(f"No samples.jsonl found in {base}")
            sys.exit(1)

        sample_rows = []
        with open(samples_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_rows.append(json.loads(line))
        df_samples = pd.DataFrame(sample_rows)

        # Join: always label, plus phase/sample_id/prompt_length if present
        join_cols = ["request_id", "label"]
        for col in ["sample_id", "phase", "prompt_length"]:
            if col in df_samples.columns:
                join_cols.append(col)

        df_all = df_svd.merge(df_samples[join_cols], on="request_id", how="left")

        has_phases = "phase" in df_all.columns and {"question", "full"} <= set(
            df_all["phase"].unique()
        )

        if has_phases:
            # Main analysis uses "full" phase (comparable to single-phase)
            df = df_all[df_all["phase"] == "full"].copy()
            df = df.rename(columns={"sample_id": "sample_idx"})
        elif "sample_id" in df_all.columns:
            df = df_all.rename(columns={"sample_id": "sample_idx"})
        else:
            df = df_all.rename(columns={"request_id": "sample_idx"})

    elif legacy_path.exists():  # TODO: remove legacy path
        # Backward compat: load old log-parsed features.jsonl
        log("Loading features.jsonl (legacy log-parsed output)")
        rows = []
        with open(legacy_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.pop("singular_values", None)
                    rows.append(row)
        df = pd.DataFrame(rows)
        has_phases = False
    else:
        click.echo(f"No svd_features.jsonl or features.jsonl found in {base}")
        sys.exit(1)

    # ── Basic stats ───────────────────────────────────────────────────────
    n_samples = df["sample_idx"].nunique()
    n_layers = df["layer_idx"].nunique()
    log(f"Loaded {len(df)} snapshots from {n_samples} samples across {n_layers} layers")

    labels = df.groupby("sample_idx")["label"].first()
    n_hall = int(labels.sum())
    n_ok = len(labels) - n_hall
    log(f"Label distribution: {n_hall} hallucinated, {n_ok} correct")

    if has_phases:
        log("Two-phase data detected (question + full); main tables use 'full' phase")

    if n_hall < 3 or n_ok < 3:
        click.echo("Too few samples in one class for meaningful analysis.")
        sys.exit(1)

    # Print config if available
    config_path = base / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        log(
            f"Config: model={config.get('model')}, dataset={config.get('dataset')}, "
            f"interval={config.get('svd_interval')}, rank={config.get('svd_rank')}, "
            f"mode={config.get('mode', 'generate')}"
        )

    layer_ids = sorted(df["layer_idx"].unique())

    # ── Per-layer correlation table ───────────────────────────────────────
    # Aggregate: per (sample, layer), take mean of each feature.
    # Then correlate per-sample means with label.
    agg = (
        df.groupby(["sample_idx", "layer_idx"])[SPECTRAL_FEATURES].mean().reset_index()
    )
    agg = agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    click.echo("=" * 72)
    click.echo("Point-Biserial Correlations by Layer (sample-level means vs label)")
    click.echo("=" * 72)

    # Header
    feat_hdrs = "".join(f" | {f:>14s}" for f in SPECTRAL_FEATURES)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 17 * len(SPECTRAL_FEATURES)))

    corr_matrix = {}  # (layer_idx, feature) -> (r, p)
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in SPECTRAL_FEATURES:
            vals = layer_agg[feat].dropna()
            lbls = layer_agg.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>14s}"
                continue
            r, p = pointbiserialr(lbls, vals)
            corr_matrix[(layer_idx, feat)] = (r, p)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row_str += f" | {r:>+7.4f} {sig:<2s} {p:>4.3f}"
        click.echo(row_str)

    # ── Global (all-layer) correlation ────────────────────────────────────
    global_agg = df.groupby("sample_idx")[SPECTRAL_FEATURES].mean().reset_index()
    global_agg = global_agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    row_str = f"{'ALL':>6s}"
    for feat in SPECTRAL_FEATURES:
        vals = global_agg[feat].dropna()
        lbls = global_agg.loc[vals.index, "label"]
        if len(vals) < 10 or lbls.nunique() < 2:
            row_str += f" | {'n/a':>14s}"
            continue
        r, p = pointbiserialr(lbls, vals)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        row_str += f" | {r:>+7.4f} {sig:<2s} {p:>4.3f}"
    click.echo(row_str)
    click.echo("")

    # ── AUROC evaluation ─────────────────────────────────────────────────
    def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
        """Compute AUROC with bootstrap 95% CI."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return None, None, None
        auc = roc_auc_score(y_true, y_score)
        rng = np.random.RandomState(seed)
        aucs = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        if not aucs:
            return auc, None, None
        lo = np.percentile(aucs, 2.5)
        hi = np.percentile(aucs, 97.5)
        return auc, lo, hi

    click.echo("=" * 72)
    click.echo("AUROC by Layer (sample-level means)")
    click.echo("=" * 72)

    feat_hdrs = "".join(f" | {f:>20s}" for f in SPECTRAL_FEATURES)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 23 * len(SPECTRAL_FEATURES)))

    auroc_matrix = {}  # (layer_idx, feature) -> (auc, lo, hi)
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in SPECTRAL_FEATURES:
            vals = layer_agg[feat].dropna()
            lbls = layer_agg.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            auc, lo, hi = bootstrap_auroc(lbls.values, vals.values)
            if auc is not None:
                auroc_matrix[(layer_idx, feat)] = (auc, lo, hi)
                if lo is not None:
                    row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
                else:
                    row_str += f" | {auc:.3f} [n/a]"
            else:
                row_str += f" | {'n/a':>20s}"
        click.echo(row_str)

    # ── Aggregated AUROC (mean/max across layers per sample) ─────────────
    click.echo("")
    sample_layer_agg = agg.pivot_table(
        index="sample_idx",
        columns="layer_idx",
        values=SPECTRAL_FEATURES,
        aggfunc="mean",
    )
    sample_labels = labels.loc[sample_layer_agg.index].values

    for agg_name, agg_fn in [("mean", np.nanmean), ("max", np.nanmax)]:
        row_str = f"{agg_name:>6s}"
        for feat in SPECTRAL_FEATURES:
            feat_cols = [
                (feat, li) for li in layer_ids if (feat, li) in sample_layer_agg.columns
            ]
            if not feat_cols:
                row_str += f" | {'n/a':>20s}"
                continue
            vals_matrix = sample_layer_agg[feat_cols].values
            agg_vals = agg_fn(vals_matrix, axis=1)
            mask = ~np.isnan(agg_vals)
            if mask.sum() < 10 or len(np.unique(sample_labels[mask])) < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            auc, lo, hi = bootstrap_auroc(sample_labels[mask], agg_vals[mask])
            if auc is not None and lo is not None:
                row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
            elif auc is not None:
                row_str += f" | {auc:.3f} [n/a]"
            else:
                row_str += f" | {'n/a':>20s}"
        click.echo(row_str)
    click.echo("")

    # ── Plot 1: Correlation heatmap ───────────────────────────────────────
    sns.set_theme(style="whitegrid")

    corr_data = pd.DataFrame(index=layer_ids, columns=SPECTRAL_FEATURES, dtype=float)
    for (li, feat), (r, _) in corr_matrix.items():
        corr_data.loc[li, feat] = r

    fig, ax = plt.subplots(figsize=(6, max(4, len(layer_ids) * 0.5)))
    sns.heatmap(
        corr_data.astype(float),
        annot=True,
        fmt="+.3f",
        center=0,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_ylabel("Layer")
    ax.set_xlabel("Feature")
    ax.set_title("Point-Biserial Correlation with Hallucination Label")
    plt.tight_layout()

    heatmap_path = str(plot_dir / "spectral_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Heatmap saved to {heatmap_path}")

    # ── Plot 2: AUROC heatmap ────────────────────────────────────────────
    if auroc_matrix:
        auroc_data = pd.DataFrame(
            index=layer_ids, columns=SPECTRAL_FEATURES, dtype=float
        )
        for (li, feat), (auc, _, _) in auroc_matrix.items():
            auroc_data.loc[li, feat] = auc

        fig, ax = plt.subplots(figsize=(6, max(4, len(layer_ids) * 0.5)))
        sns.heatmap(
            auroc_data.astype(float),
            annot=True,
            fmt=".3f",
            center=0.5,
            cmap="RdYlGn",
            vmin=0.3,
            vmax=0.7,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_ylabel("Layer")
        ax.set_xlabel("Feature")
        ax.set_title("AUROC by Layer and Feature")
        plt.tight_layout()

        auroc_heatmap_path = str(plot_dir / "spectral_auroc_heatmap.png")
        plt.savefig(auroc_heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"AUROC heatmap saved to {auroc_heatmap_path}")

    # ── Plot 3: Per-layer distributions (half-violin + pointrange) ────────
    # Need L column for dot sizing; use max L per sample if multiple snapshots
    if "L" not in df.columns:
        df["L"] = 0
    dist_df = df[["sample_idx", "layer_idx", "label", "L"] + SPECTRAL_FEATURES].copy()

    dist_path = str(plot_dir / "spectral_distributions.png")
    plot_violin_pointrange(
        dist_df,
        features=SPECTRAL_FEATURES,
        feat_labels=["σ₁/σ₂ Ratio", "σ₁ (Leading SV)", "SV Entropy"],
        layer_ids=layer_ids,
        title="Spectral Feature Distributions (Full Phase) by Layer",
        out_path=dist_path,
    )

    # ── Prompt length analysis ─────────────────────────────────────────────
    if "L" in df.columns:
        sample_length = df.groupby("sample_idx")["L"].max().reset_index()
        sample_length = sample_length.merge(labels.reset_index(), on="sample_idx")
        sample_length["label_str"] = sample_length["label"].map(
            {0: "Correct", 1: "Hallucinated"}
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        for lv, ls, c in [(0, "Correct", "steelblue"), (1, "Hallucinated", "tomato")]:
            subset = sample_length[sample_length["label"] == lv]
            ax.hist(subset["L"], bins=30, alpha=0.5, label=ls, color=c)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Count")
        ax.set_title("Sequence Length Distribution by Label")
        ax.legend()
        plt.tight_layout()
        len_dist_path = str(plot_dir / "seq_length_dist.png")
        plt.savefig(len_dist_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Sequence length distribution saved to {len_dist_path}")

        # Feature vs length scatter for the layer with strongest correlation
        if corr_matrix:
            best_layer = max(corr_matrix, key=lambda k: abs(corr_matrix[k][0]))[0]
        else:
            best_layer = layer_ids[len(layer_ids) // 2]

        layer_df = df[df["layer_idx"] == best_layer].copy()
        layer_df["label_str"] = layer_df["label"].map({0: "Correct", 1: "Hallucinated"})
        n_feats = len(SPECTRAL_FEATURES)
        fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4))
        if n_feats == 1:
            axes = [axes]
        for ax, feat in zip(axes, SPECTRAL_FEATURES):
            sns.scatterplot(
                data=layer_df,
                x="L",
                y=feat,
                hue="label_str",
                alpha=0.4,
                s=15,
                ax=ax,
            )
            ax.set_title(f"Layer {best_layer}")
            ax.set_xlabel("Sequence Length (tokens)")
        plt.suptitle(
            "Spectral Features vs Sequence Length", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        feat_len_path = str(plot_dir / f"feature_vs_length_layer{best_layer}.png")
        plt.savefig(feat_len_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Feature vs length scatter saved to {feat_len_path}")

    # ── Two-phase analysis (evaluate mode) ─────────────────────────────────
    if has_phases:
        click.echo("=" * 72)
        click.echo("Two-Phase Analysis: Question vs Full (evaluate mode)")
        click.echo("=" * 72)

        df_q = df_all[df_all["phase"] == "question"].copy()
        df_q = df_q.rename(columns={"sample_id": "sample_idx"})
        df_f = df  # already filtered to "full" and renamed

        agg_q = (
            df_q.groupby(["sample_idx", "layer_idx"])[SPECTRAL_FEATURES]
            .mean()
            .reset_index()
        )
        agg_f = (
            df_f.groupby(["sample_idx", "layer_idx"])[SPECTRAL_FEATURES]
            .mean()
            .reset_index()
        )

        merged_phases = agg_q.merge(
            agg_f, on=["sample_idx", "layer_idx"], suffixes=("_q", "_f")
        )
        delta_features = [f"{f}_delta" for f in SPECTRAL_FEATURES]
        for feat in SPECTRAL_FEATURES:
            merged_phases[f"{feat}_delta"] = (
                merged_phases[f"{feat}_f"] - merged_phases[f"{feat}_q"]
            )

        merged_phases = merged_phases.merge(labels.reset_index(), on="sample_idx")

        # Delta correlation table
        click.echo("")
        click.echo("Delta Correlations (full - question) vs label:")
        feat_hdrs = "".join(f" | {f:>20s}" for f in delta_features)
        click.echo(f"{'Layer':>6s}{feat_hdrs}")
        click.echo("-" * (8 + 23 * len(delta_features)))

        delta_corr = {}
        for layer_idx in layer_ids:
            lm = merged_phases[merged_phases["layer_idx"] == layer_idx]
            row_str = f"{layer_idx:>6d}"
            for feat in delta_features:
                vals = lm[feat].dropna()
                lbls = lm.loc[vals.index, "label"]
                if len(vals) < 10 or lbls.nunique() < 2:
                    row_str += f" | {'n/a':>20s}"
                    continue
                r, p = pointbiserialr(lbls, vals)
                delta_corr[(layer_idx, feat)] = (r, p)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                row_str += f" | {r:>+7.4f} {sig:<2s}  {p:>5.3f}"
            click.echo(row_str)

        # Global delta row
        click.echo("")
        global_delta = (
            merged_phases.groupby("sample_idx")[delta_features].mean().reset_index()
        )
        global_delta = global_delta.merge(labels.reset_index(), on="sample_idx")
        row_str = f"{'ALL':>6s}"
        for feat in delta_features:
            vals = global_delta[feat].dropna()
            lbls = global_delta.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            r, p = pointbiserialr(lbls, vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row_str += f" | {r:>+7.4f} {sig:<2s}  {p:>5.3f}"
        click.echo(row_str)

        # Delta AUROC
        click.echo("")
        click.echo("Delta AUROC (full - question) by Layer:")
        feat_hdrs = "".join(f" | {f:>20s}" for f in delta_features)
        click.echo(f"{'Layer':>6s}{feat_hdrs}")
        click.echo("-" * (8 + 23 * len(delta_features)))

        for layer_idx in layer_ids:
            lm = merged_phases[merged_phases["layer_idx"] == layer_idx]
            row_str = f"{layer_idx:>6d}"
            for feat in delta_features:
                vals = lm[feat].dropna()
                lbls = lm.loc[vals.index, "label"]
                if len(vals) < 10 or lbls.nunique() < 2:
                    row_str += f" | {'n/a':>20s}"
                    continue
                auc, lo, hi = bootstrap_auroc(lbls.values, vals.values)
                if auc is not None and lo is not None:
                    row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
                elif auc is not None:
                    row_str += f" | {auc:.3f} [n/a]"
                else:
                    row_str += f" | {'n/a':>20s}"
            click.echo(row_str)
        click.echo("")

        # Plot: Delta correlation heatmap
        if delta_corr:
            delta_corr_data = pd.DataFrame(
                index=layer_ids, columns=delta_features, dtype=float
            )
            for (li, feat), (r, _) in delta_corr.items():
                delta_corr_data.loc[li, feat] = r

            fig, ax = plt.subplots(figsize=(8, max(4, len(layer_ids) * 0.5)))
            sns.heatmap(
                delta_corr_data.astype(float),
                annot=True,
                fmt="+.3f",
                center=0,
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_ylabel("Layer")
            ax.set_xlabel("Feature (full - question)")
            ax.set_title("Delta Correlation with Hallucination Label")
            plt.tight_layout()
            delta_hm_path = str(plot_dir / "delta_heatmap.png")
            plt.savefig(delta_hm_path, dpi=150, bbox_inches="tight")
            plt.close()
            log(f"Delta heatmap saved to {delta_hm_path}")

        # Plot: Delta distributions (half-violin + pointrange)
        # Attach L from the full phase for dot sizing
        L_by_sample = (
            df_f.groupby("sample_idx")["L"].max().reset_index()
            if "L" in df_f.columns
            else pd.DataFrame({"sample_idx": merged_phases["sample_idx"].unique(), "L": 0})
        )
        delta_dist_df = merged_phases[
            ["sample_idx", "layer_idx", "label"] + delta_features
        ].merge(L_by_sample, on="sample_idx", how="left")

        delta_dist_path = str(plot_dir / "delta_distributions.png")
        plot_violin_pointrange(
            delta_dist_df,
            features=delta_features,
            feat_labels=["Δ σ₁/σ₂ Ratio", "Δ σ₁", "Δ SV Entropy"],
            layer_ids=layer_ids,
            title="Delta (Full − Question) Distributions by Layer",
            out_path=delta_dist_path,
            show_zero_line=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    if corr_matrix:
        best_key = max(corr_matrix, key=lambda k: abs(corr_matrix[k][0]))
        r_val, p_val = corr_matrix[best_key]
        log(
            f"Strongest per-layer correlation: layer {best_key[0]} / {best_key[1]} "
            f"(r={r_val:+.4f}, p={p_val:.4f})"
        )
    if auroc_matrix:
        best_auc_key = max(auroc_matrix, key=lambda k: abs(auroc_matrix[k][0] - 0.5))
        auc_val, auc_lo, auc_hi = auroc_matrix[best_auc_key]
        ci_str = f" [{auc_lo:.3f}-{auc_hi:.3f}]" if auc_lo is not None else ""
        log(
            f"Best per-layer AUROC: layer {best_auc_key[0]} / {best_auc_key[1]} "
            f"(AUROC={auc_val:.3f}{ci_str})"
        )
    click.echo("")


if __name__ == "__main__":
    cli()
