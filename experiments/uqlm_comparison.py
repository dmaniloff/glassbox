#!/usr/bin/env python
"""Compare Glassbox spectral features vs UQLM white-box UQ for hallucination detection.

Both methods analyse the same model on the same dataset in a single generation pass:
- UQLM white-box: fixed formulas on token log-probabilities (no training)
- Glassbox: logistic regression on attention spectral features (needs a train split)

Usage:
    python experiments/uqlm_comparison.py --model microsoft/Phi-4-mini-instruct
    python experiments/uqlm_comparison.py --dataset gsm8k --n-samples 20 --no-plot
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import vllm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from uqlm.utils.dataloader import load_example_dataset
from uqlm.utils.postprocessors import math_postprocessor
from uqlm.white_box.single_logprobs import SingleLogprobsScorer
from uqlm.white_box.top_logprobs import TopLogprobsScorer

from glassbox.cli.extract import run_extraction
from glassbox.config import GlassboxConfig
from glassbox.results import SPECTRAL_FEATURE_NAMES

PROMPT_INSTRUCTIONS = {
    "gsm8k": (
        "When you solve this math problem only return the answer"
        " with no additional text.\n"
    ),
    "svamp": (
        "When you solve this math problem only return the answer"
        " with no additional text.\n"
    ),
    "csqa": (
        "You will be given a multiple choice question. Return only the letter"
        " of the response with no additional text or explanation.\n"
    ),
    "ai2_arc": (
        "You will be given a multiple choice question. Return only the letter"
        " of the response with no additional text or explanation.\n"
    ),
    "popqa": (
        "You will be given a question. Return only the answer as concisely"
        " as possible without providing an explanation.\n"
    ),
    "nq_open": (
        "You will be given a question. Return only the answer as concisely"
        " as possible without providing an explanation.\n"
    ),
}

def _math_grader(r, a):
    return math_postprocessor(r) == str(a)


def _mc_grader(r, a):
    return r.strip().upper()[:1] == str(a).strip().upper()[:1]


def _substr_grader(r, a):
    answers = [a] if isinstance(a, str) else a
    return any(ans.lower() in r.lower() for ans in answers)


GRADERS = {
    "gsm8k": _math_grader,
    "svamp": _math_grader,
    "csqa": _mc_grader,
    "ai2_arc": _mc_grader,
    "popqa": _substr_grader,
    "nq_open": _substr_grader,
}

TOP_K_LOGPROBS = 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--dataset", default="csqa", choices=list(PROMPT_INSTRUCTIONS))
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--test-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--heads", type=int, nargs="+", default=[0])
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ── Pipeline steps ────────────────────────────────────────────────────────


def load_data(dataset_name: str, n_samples: int, test_ratio: float, seed: int):
    print(f"Loading dataset: {dataset_name} (n={n_samples})")
    df = load_example_dataset(dataset_name, n=n_samples).reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_ratio, random_state=seed,
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    return df, train_idx, test_idx


def generate_with_logprobs(model: str, prompts: list[str], max_tokens: int):
    """Generate responses and extract per-token logprobs for UQLM scoring."""
    llm = vllm.LLM(model=model, enforce_eager=True, max_model_len=2048)
    params = vllm.SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=TOP_K_LOGPROBS)
    outputs = llm.generate(prompts, params)

    responses, uqlm_logprobs = [], []
    for output in outputs:
        comp = output.outputs[0]
        responses.append(comp.text)

        token_dicts = []
        if comp.logprobs:
            for i, lp_dict in enumerate(comp.logprobs):
                token_id = comp.token_ids[i]
                if token_id in lp_dict:
                    chosen_lp = lp_dict[token_id].logprob
                else:
                    chosen_lp = next(iter(lp_dict.values())).logprob
                top_lps = [{"logprob": lp.logprob} for lp in lp_dict.values()]
                token_dicts.append({"logprob": chosen_lp, "top_logprobs": top_lps})
        uqlm_logprobs.append(token_dicts)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return responses, uqlm_logprobs


def grade_and_score_uqlm(df: pd.DataFrame, uqlm_logprobs: list, dataset_name: str):
    """Grade responses against ground truth and compute UQLM white-box scores."""
    grader = GRADERS[dataset_name]
    df["correct"] = [grader(r, a) for r, a in zip(df["generated_response"], df["answer"])]
    n_correct = df["correct"].sum()
    print(f"Correct: {n_correct}/{len(df)} ({n_correct / len(df):.1%})")

    single_scores = SingleLogprobsScorer(
        scorers=["min_probability", "sequence_probability"], length_normalize=True,
    ).evaluate(uqlm_logprobs)
    top_scores = TopLogprobsScorer(
        scorers=["mean_token_negentropy", "min_token_negentropy", "probability_margin"],
        top_k_logprobs=TOP_K_LOGPROBS,
    ).evaluate(uqlm_logprobs)

    uqlm_cols = {}
    for name, values in {**single_scores, **top_scores}.items():
        col = f"uqlm_{name}"
        df[col] = values
        uqlm_cols[name] = col

    print(f"UQLM scorers: {list(uqlm_cols.keys())}")
    return uqlm_cols


def extract_glassbox(df: pd.DataFrame, args: argparse.Namespace):
    """Run Glassbox spectral extraction and return wide feature matrix."""
    samples = [
        {"idx": i, "question": q, "response": r, "label": int(c)}
        for i, (q, r, c) in enumerate(zip(df["question"], df["generated_response"], df["correct"]))
    ]
    gb_config = GlassboxConfig(
        spectral={"enabled": True, "interval": 1, "rank": args.rank, "heads": args.heads},
        routing={"enabled": False},
        tracker={"enabled": False},
        selfattn={"enabled": False},
        laplacian={"enabled": False},
    )
    outdir = run_extraction(
        samples=samples, model=args.model, config=gb_config,
        phases=("full",), max_model_len=2048, dataset_name=args.dataset,
    )
    print(f"Glassbox output: {outdir}")

    config = json.loads((outdir / "config.json").read_text())
    num_layers = config["num_layers"]
    snapshots = [json.loads(line) for line in open(outdir / "svd_features.jsonl") if line.strip()]
    print(f"  {len(snapshots)} snapshots ({num_layers} layers)")

    feature_names = list(SPECTRAL_FEATURE_NAMES)
    columns = [f"{feat}_L{li}_H0" for li in range(num_layers) for feat in feature_names]

    wide = {}
    for snap in snapshots:
        rid = snap["request_id"]
        if rid not in wide:
            wide[rid] = {}
        li = snap["layer_idx"]
        feats = snap.get("features", {})
        for feat in feature_names:
            wide[rid][f"{feat}_L{li}_H0"] = feats.get(feat)

    features_df = pd.DataFrame.from_dict(wide, orient="index")
    features_df = features_df.reindex(columns=columns).sort_index()
    return features_df, columns


def train_and_evaluate(
    features_df: pd.DataFrame,
    columns: list[str],
    df: pd.DataFrame,
    uqlm_cols: dict[str, str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
):
    labels = df["correct"].astype(int).values
    X = np.nan_to_num(features_df[columns].values, nan=0.0)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    n_ok = y_train.sum()
    print(f"Train: {len(y_train)} ({n_ok} correct, {len(y_train) - n_ok} incorrect)")
    n_ok = y_test.sum()
    print(f"Test:  {len(y_test)} ({n_ok} correct, {len(y_test) - n_ok} incorrect)")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("\n*** Only one class present — AUROC is undefined. ***")
        print("Try a larger model or a dataset where the model has mixed accuracy.")
        return None, None, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_s, y_train)
    gb_proba = clf.predict_proba(X_test_s)[:, 1]

    results = {"Glassbox (logreg)": roc_auc_score(y_test, gb_proba)}
    for scorer_name, col in uqlm_cols.items():
        scores_test = np.nan_to_num(df[col].values[test_idx], nan=0.0)
        if not np.all(np.isnan(scores_test)):
            results[f"UQLM {scorer_name}"] = roc_auc_score(y_test, scores_test)

    print("\n" + "=" * 55)
    print("  AUROC — Hallucination Detection (test set)")
    print("=" * 55)
    for name, auroc in results.items():
        print(f"  {name:<35s} {auroc:.4f}")
    print("=" * 55)

    scorer_data = [("Glassbox (logreg)", gb_proba)]
    for scorer_name, col in uqlm_cols.items():
        label = f"UQLM {scorer_name}"
        if label in results:
            scorer_data.append((label, np.nan_to_num(df[col].values[test_idx], nan=0.0)))

    return results, scorer_data, clf


def save_plots(
    results: dict,
    scorer_data: list,
    clf: LogisticRegression,
    columns: list[str],
    y_test: np.ndarray,
    model: str,
    dataset_name: str,
    n_samples: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    correct_mask = y_test.astype(bool)
    cmap = plt.cm.tab10

    # ── ROC curves + AUROC bar chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for idx, (label, scores) in enumerate(scorer_data):
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, color=cmap(idx), label=f"{label} ({results[label]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    names = list(results.keys())
    values = list(results.values())
    bars = ax.barh(names, values, color=[cmap(i) for i in range(len(names))])
    ax.set_xlabel("AUROC")
    ax.set_xlim(0, 1)
    ax.set_title("AUROC Comparison")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
    ax.invert_yaxis()

    plt.suptitle(f"{model} on {dataset_name} (n={n_samples})", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Score distributions ───────────────────────────────────────────
    n_scorers = len(scorer_data)
    fig, axes = plt.subplots(1, n_scorers, figsize=(4 * n_scorers, 4))
    if n_scorers == 1:
        axes = [axes]
    for ax, (label, scores) in zip(axes, scorer_data):
        ax.hist(scores[correct_mask], bins=20, alpha=0.6, label="Correct", color="green")
        ax.hist(scores[~correct_mask], bins=20, alpha=0.6, label="Incorrect", color="red")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.suptitle("Score Distributions: Correct vs Incorrect", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Feature importance ────────────────────────────────────────────
    coefs = pd.Series(clf.coef_[0], index=columns).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    top_n = min(20, len(coefs))
    top = coefs.head(top_n)
    colors = ["steelblue" if v > 0 else "coral" for v in top.values]
    top.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(f"Top {top_n} Glassbox Features by |coefficient|")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    instruction = PROMPT_INSTRUCTIONS[args.dataset]

    df, train_idx, test_idx = load_data(args.dataset, args.n_samples, args.test_ratio, args.seed)

    prompts = [instruction + q for q in df["question"]]
    print(f"\nGenerating responses with {args.model} (max_tokens={args.max_tokens})...")
    responses, uqlm_logprobs = generate_with_logprobs(args.model, prompts, args.max_tokens)
    df["generated_response"] = responses

    print("\nGrading + computing UQLM scores...")
    uqlm_cols = grade_and_score_uqlm(df, uqlm_logprobs, args.dataset)

    print(f"\nExtracting Glassbox features (rank={args.rank}, heads={args.heads})...")
    features_df, columns = extract_glassbox(df, args)

    print("\nTraining classifier + evaluating...")
    results, scorer_data, clf = train_and_evaluate(
        features_df, columns, df, uqlm_cols, train_idx, test_idx, args.seed,
    )

    if results and not args.no_plot:
        output_dir = args.output_dir
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("experiments/results") / f"uqlm_comparison_{timestamp}"
        y_test = df["correct"].astype(int).values[test_idx]
        save_plots(
            results, scorer_data, clf, columns, y_test,
            args.model, args.dataset, args.n_samples, output_dir,
        )


if __name__ == "__main__":
    main()
