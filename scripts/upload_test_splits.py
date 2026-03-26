"""Create and upload hash-split test datasets to HuggingFace.

Reproduces the exact 30% test split used by shade-experiments
(HashBasedSplitter with hash_fields=["prompt"], 70/0/30 split)
and uploads each as a standalone HuggingFace dataset.

This ensures glassbox experiments run on the same samples as shade
without depending on shade-train at runtime.  The hash-based splitting
logic and dataset expansion are inlined from shade-train to avoid
pulling in pytorch_lightning, sklearn, etc.

Prerequisites:
    pip install datasets huggingface_hub

Usage:
    # Dry run (local parquet only, no upload):
    python scripts/upload_test_splits.py --dry-run

    # Upload to a specific HF user/org:
    python scripts/upload_test_splits.py --hf-org dmaniloff

    # Single dataset:
    python scripts/upload_test_splits.py --dataset halueval_hallucination --dry-run
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import click
import datasets as hf_datasets

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Hash-based splitter (inlined from shade_train.data.splitter)
# =============================================================================


def _hash_split_test(
    samples: list[dict[str, Any]],
    hash_fields: list[str] = ("prompt",),
    train_ratio: float = 0.7,
) -> list[dict[str, Any]]:
    """Return the test portion of a hash-based split.

    Reproduces shade_train.data.splitter.HashBasedSplitter exactly:
      1. SHA256 of "|".join(hash_field values)
      2. First 16 hex chars → int → prob in [0, 1)
      3. prob < train_ratio → train, else test  (val_ratio=0)
    """
    test = []
    for s in samples:
        parts = [str(s.get(f, "")) for f in hash_fields]
        h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        prob = (int(h[:16], 16) % (2**64)) / (2**64)
        if prob >= train_ratio:  # test bucket (val_ratio=0)
            test.append(s)
    return test


def _balance(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downsample the majority class to match the minority class."""
    pos = [s for s in samples if s["label"] == 1]
    neg = [s for s in samples if s["label"] == 0]
    if len(pos) == len(neg):
        return samples
    if len(neg) > len(pos):
        neg = neg[: len(pos)]
    else:
        pos = pos[: len(neg)]
    return neg + pos


# =============================================================================
# Dataset loaders — replicate shade-train dataset class expansion logic
# =============================================================================


def _load_deepset() -> tuple[list[dict], str, bool]:
    """deepset/prompt-injections: full dataset, no split."""
    logger.info("Loading deepset/prompt-injections...")
    ds = hf_datasets.load_dataset("deepset/prompt-injections", split="train")
    samples = []
    for i, row in enumerate(ds):
        label_raw = row.get("label", 0)
        # Normalize label: shade-train maps string labels too
        if isinstance(label_raw, str):
            label_raw = 1 if label_raw.lower() in ("1", "true", "injection") else 0
        samples.append(
            {
                "prompt": "",
                "response": row.get("text", ""),
                "label": int(label_raw),
                "unique_id": f"deepset/prompt-injections|{i}",
            }
        )
    logger.info(f"  deepset: {len(samples)} samples (no split)")
    return samples, "injection", False


def _load_protectai() -> tuple[list[dict], str, bool]:
    """protectai/prompt-injection-validation: 30% test split."""
    logger.info("Loading protectai/prompt-injection-validation...")
    # This dataset has per-source splits, not a single "train" split.
    # shade-train uses split="all" which concatenates all splits.
    ds_dict = hf_datasets.load_dataset("protectai/prompt-injection-validation")
    all_rows = hf_datasets.concatenate_datasets(list(ds_dict.values()))
    samples = []
    for i, row in enumerate(all_rows):
        label_raw = row.get("label", 0)
        if isinstance(label_raw, str):
            label_raw = 1 if label_raw.lower() in ("1", "true", "injection") else 0
        samples.append(
            {
                "prompt": "",
                "response": row.get("text", ""),
                "label": int(label_raw),
                "unique_id": f"protectai/prompt-injection-validation|{i}",
            }
        )
    return samples, "injection", True


def _load_halueval() -> tuple[list[dict], str, bool]:
    """HaluEval QA: 30% test split. Expands each row → 2 samples."""
    logger.info("Loading HaluEval (qa)...")
    ds = hf_datasets.load_dataset("pminervini/HaluEval", "qa", split="data")
    samples = []
    for row in ds:
        question = row.get("question", "")
        right = row.get("right_answer", "")
        halluc = row.get("hallucinated_answer", "")
        uid_base = f"{row.get('knowledge', '')[:200]}|{question[:200]}"
        if right:
            samples.append(
                {
                    "prompt": question,
                    "response": right,
                    "label": 0,
                    "unique_id": f"{uid_base}|factual",
                }
            )
        if halluc:
            samples.append(
                {
                    "prompt": question,
                    "response": halluc,
                    "label": 1,
                    "unique_id": f"{uid_base}|hallucinated",
                }
            )
    # HaluEvalDataset default: balance_classes=True
    samples = _balance(samples)
    return samples, "hallucination", True


def _load_truthfulqa() -> tuple[list[dict], str, bool]:
    """TruthfulQA generation: 30% test split. Expands all answers."""
    logger.info("Loading TruthfulQA (generation)...")
    ds = hf_datasets.load_dataset(
        "truthfulqa/truthful_qa", "generation", split="validation"
    )
    samples = []
    for row in ds:
        question = row.get("question", "")
        category = row.get("category", "")
        prompt_id = f"{question[:300]}|{category}"

        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])
        best = row.get("best_answer", "")

        # Add best_answer as correct if not already in correct_answers
        all_correct = list(correct_answers)
        if best and best not in all_correct:
            all_correct.insert(0, best)

        for ci, ans in enumerate(all_correct):
            if ans:
                samples.append(
                    {
                        "prompt": question,
                        "response": ans,
                        "label": 0,
                        "unique_id": f"{prompt_id}|correct_{ci}",
                    }
                )
        for ii, ans in enumerate(incorrect_answers):
            if ans:
                samples.append(
                    {
                        "prompt": question,
                        "response": ans,
                        "label": 1,
                        "unique_id": f"{prompt_id}|incorrect_{ii}",
                    }
                )
    # TruthfulQADataset default: balance_classes=False
    return samples, "hallucination", True


def _load_medhallu() -> tuple[list[dict], str, bool]:
    """MedHallu pqa_labeled: 30% test split. Expands each row → 2 samples."""
    logger.info("Loading MedHallu (pqa_labeled)...")
    ds = hf_datasets.load_dataset(
        "UTAustin-AIHealth/MedHallu", "pqa_labeled", split="train"
    )
    samples = []
    for row in ds:
        question = row.get("Question", "")
        ground_truth = row.get("Ground Truth", "")
        hallucinated = row.get("Hallucinated Answer", "")
        knowledge = row.get("knowledge", "")
        if isinstance(knowledge, list):
            knowledge = " ".join(str(k) for k in knowledge)
        uid_base = f"{question[:200]}|{str(knowledge)[:200]}"

        if ground_truth:
            samples.append(
                {
                    "prompt": question,
                    "response": ground_truth,
                    "label": 0,
                    "unique_id": f"{uid_base}|ground_truth",
                }
            )
        if hallucinated:
            samples.append(
                {
                    "prompt": question,
                    "response": hallucinated,
                    "label": 1,
                    "unique_id": f"{uid_base}|hallucinated",
                }
            )
    # MedHalluDataset default: balance_classes=True
    samples = _balance(samples)
    return samples, "hallucination", True


def _load_ragtruth() -> tuple[list[dict], str, bool]:
    """RAGTruth: 30% test split."""
    logger.info("Loading RAGTruth...")
    ds = hf_datasets.load_dataset("wandb/RAGTruth-processed", split="test")
    samples = []
    for row in ds:
        context = row.get("context", "")
        query = row.get("query", "")
        prompt = f"{context}\n\n{query}" if context else query
        response = row.get("output", "")
        # Labels are nested in hallucination_labels_processed dict
        hall_labels = row.get("hallucination_labels_processed", {}) or {}
        evident = hall_labels.get("evident_conflict", 0) or 0
        baseless = hall_labels.get("baseless_info", 0) or 0
        label = 1 if (evident > 0 or baseless > 0) else 0
        model_name = row.get("model", "")
        uid = f"ragtruth|{row.get('id', '')}|{model_name}"
        samples.append(
            {
                "prompt": prompt,
                "response": response,
                "label": label,
                "unique_id": uid,
            }
        )
    # RAGTruthDataset default: balance_classes=True
    samples = _balance(samples)
    return samples, "hallucination", True


def _load_halubench() -> tuple[list[dict], str, bool]:
    """HaluBench: 30% test split (excluding halueval source)."""
    logger.info("Loading HaluBench...")
    ds = hf_datasets.load_dataset("PatronusAI/HaluBench", split="test")
    samples = []
    for row in ds:
        source = row.get("source", "")
        # shade-experiments excludes halueval to avoid overlap
        if source.lower() == "halueval":
            continue
        passage = row.get("passage", "")
        question = row.get("question", "")
        prompt = f"{passage}\n\nQuestion: {question}" if passage else question
        answer = row.get("answer", "")
        label_str = str(row.get("label", "")).upper()
        label = 0 if label_str == "PASS" else 1
        uid = f"{row.get('id', '')}|{source}"
        samples.append(
            {
                "prompt": prompt,
                "response": answer,
                "label": label,
                "unique_id": uid,
            }
        )
    # HaluBenchDataset default: balance_classes=True
    samples = _balance(samples)
    return samples, "hallucination", True


def _load_felm() -> tuple[list[dict], str, bool]:
    """FELM: 30% test split. Manual JSONL from HuggingFace Hub."""
    logger.info("Loading FELM...")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("hkust-nlp/felm", "all.jsonl", repo_type="dataset")
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line.strip()))

    samples = []
    for item in raw:
        prompt = str(item.get("prompt", "") or "")
        response = str(item.get("response", "") or "")
        if not response or response == "nan":
            continue
        domain = item.get("domain", "")
        labels = item.get("labels", [])
        label = (1 if any(not lbl for lbl in labels) else 0) if labels else 0
        uid = f"felm|{item.get('index', '')}|{domain}"
        samples.append(
            {
                "prompt": prompt,
                "response": response,
                "label": label,
                "unique_id": uid,
            }
        )
    # FELMDataset default: balance_classes=True
    samples = _balance(samples)
    return samples, "hallucination", True


DATASET_REGISTRY: dict[str, callable] = {
    "deepset_injection": _load_deepset,
    "protectai_injection": _load_protectai,
    "halueval_hallucination": _load_halueval,
    "truthfulqa_hallucination": _load_truthfulqa,
    "medhallu_hallucination": _load_medhallu,
    "ragtruth_hallucination": _load_ragtruth,
    "halubench_hallucination": _load_halubench,
    "felm_hallucination": _load_felm,
}


# =============================================================================
# Upload logic
# =============================================================================


def _to_hf_dataset(samples: list[dict], failure_mode: str) -> hf_datasets.Dataset:
    """Convert sample dicts to a HuggingFace Dataset."""
    rows: dict[str, list] = {
        "prompt": [],
        "response": [],
        "label": [],
        "unique_id": [],
        "failure_mode": [],
    }
    for s in samples:
        rows["prompt"].append(s.get("prompt", ""))
        rows["response"].append(s.get("response", ""))
        rows["label"].append(int(s.get("label", 0)))
        rows["unique_id"].append(s.get("unique_id", ""))
        rows["failure_mode"].append(failure_mode)
    return hf_datasets.Dataset.from_dict(rows)


def process_dataset(
    name: str,
    hf_org: str,
    output_dir: Path,
    dry_run: bool = False,
    private: bool = False,
) -> None:
    """Load, split, and upload one dataset."""
    loader = DATASET_REGISTRY[name]
    all_samples, failure_mode, needs_split = loader()

    if needs_split:
        test_samples = _hash_split_test(all_samples, hash_fields=["prompt"])
        logger.info(f"  {name}: {len(test_samples)}/{len(all_samples)} in test split")
    else:
        test_samples = all_samples
        logger.info(f"  {name}: {len(test_samples)} samples (no split)")

    n_pos = sum(1 for s in test_samples if s.get("label") == 1)
    n_neg = len(test_samples) - n_pos
    logger.info(f"  Class balance: {n_neg} neg (0) / {n_pos} pos (1)")

    ds = _to_hf_dataset(test_samples, failure_mode)

    # Save locally
    local_path = output_dir / name
    local_path.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(local_path / "test.parquet")
    logger.info(f"  Saved locally: {local_path / 'test.parquet'}")

    if not dry_run:
        repo_id = f"{hf_org}/glassbox_{name}_test"
        ds.push_to_hub(repo_id, split="test", private=private)
        logger.info(f"  Uploaded to: {repo_id}")


@click.command(help="Create and upload hash-split test datasets for glassbox")
@click.option(
    "--hf-org",
    default="dmaniloff",
    show_default=True,
    help="HuggingFace user/org to upload to.",
)
@click.option(
    "--dataset",
    type=click.Choice(list(DATASET_REGISTRY) + ["all"], case_sensitive=True),
    default="all",
    show_default=True,
    help="Dataset to process.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("scripts/test_splits"),
    show_default=True,
    help="Local output directory for parquet files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only save locally, do not upload to HuggingFace.",
)
@click.option(
    "--private",
    is_flag=True,
    help="Create private HuggingFace repos.",
)
def main(
    hf_org: str,
    dataset: str,
    output: Path,
    dry_run: bool,
    private: bool,
) -> None:
    output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(DATASET_REGISTRY) if dataset == "all" else [dataset]

    for name in names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Processing: {name}")
        logger.info(f"{'=' * 60}")
        process_dataset(
            name,
            hf_org=hf_org,
            output_dir=output_dir,
            dry_run=dry_run,
            private=private,
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
