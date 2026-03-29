#!/usr/bin/env bash
# Full Pythia sweep: all datasets × all models × all heads
# Outer loop: datasets (halueval first, then ascending by size)
# Inner loop: models (smallest → largest)
# This way all models finish halueval before moving to the next dataset.
set -euo pipefail
cd /home/ubuntu/src/glassbox

EXTRACT="glassbox-extract"
LOGDIR=experiments/sweep_logs_full

mkdir -p "$LOGDIR"

# model:num_heads
CONFIGS=(
  "EleutherAI/pythia-70m:8"
  "EleutherAI/pythia-160m:12"
  "EleutherAI/pythia-410m:16"
  "EleutherAI/pythia-1b:8"
  "EleutherAI/pythia-1.4b:16"
  # "EleutherAI/pythia-2.8b:32"
  # "EleutherAI/pythia-6.9b:32"
)

# Datasets ordered: halueval first, then smaller datasets
DATASETS=(
  "halueval_hallucination"    # 5904 (done for all models)
  "medhallu_hallucination"    #  618
  "truthfulqa_hallucination"  # 1951
  # "felm_hallucination"        #  172
  # "ragtruth_hallucination"    #  541
  # "deepset_injection"         #  546
  # "protectai_injection"       # 3227
  # "halubench_hallucination"   # 4368
)

for DATASET in "${DATASETS[@]}"; do
  for ENTRY in "${CONFIGS[@]}"; do
    MODEL="${ENTRY%%:*}"
    NHEADS="${ENTRY##*:}"
    SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')

    # Build comma-separated head list: 0,1,2,...,N-1
    HEAD_LIST=$(seq -s, 0 $((NHEADS - 1)))

    LOG="$LOGDIR/${SAFE_MODEL}_${DATASET}.log"

    # Resume support: skip if log already shows completion
    if [[ -f "$LOG" ]] && grep -q "Done!" "$LOG"; then
      echo "[sweep] SKIP (already done): $MODEL × $DATASET"
      continue
    fi

    echo "=========================================="
    echo "[sweep] $MODEL × $DATASET ($NHEADS heads) ($(date))"
    echo "=========================================="

    $EXTRACT \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --scores-matrix \
      --degree-normalized \
      --heads "$HEAD_LIST" \
      2>&1 | tee "$LOG"

    echo "[sweep] Done: $MODEL × $DATASET ($(date))"
    echo ""
  done

  echo "[sweep] Completed all models for $DATASET ($(date))"
  echo ""
done

echo "[sweep] All datasets × all Pythia models complete! ($(date))"
