#!/usr/bin/env bash
# R22-H189 mmBERT LAMBDA-0 ABLATION - train, then the arena read.
#
# One variable against R18-H150 draw 1: LAMBDA_MAX 0.02 -> 0.0.  Registered in
# docs/experiments/semantic-grounding-experiments.md, block "R22-H189 mmBERT
# LAMBDA-0 ABLATION", author-ordered 2026-08-17 ~23:15.
#
# The census stage already verified the two fingerprints against the matched
# control before this script existed; the train stage re-checks them on the
# result file and VOIDS the run on any drift.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is already on disk, so a relaunch after a restart is the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
# .to("cuda") under WSL2 on this box.
#
# Usage (detached):
#   GPU=0 nohup setsid bash experiments/grounding-semantic/R22-H189_campaign.sh \
#       > logs/R22-H189_lambda0_gpu0.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (0, 1 or 2)}"

E=experiments/grounding-semantic
RUN="$E/R22-H189_lambda0_run.py"

echo "=== R22-H189 LAMBDA-0 ABLATION  GPU${CUDA_VISIBLE_DEVICES}  $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

stage_unless() {
  local art="$1" label="$2"; shift 2
  if [ -s "$art" ]; then
    echo ""
    echo "--- SKIP (on disk: $art)  $label ---"
    return 0
  fi
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

stage_unless "$E/R22-H189_lambda0_result.json" \
  "train + in-domain suite (adversary OFF)" \
  uv run python "$RUN" --stage train

stage_unless "$E/R22-H189_lambda0_windowed_result.json" \
  "windowed arena read (REPORTED, not a gate)" \
  uv run python "$RUN" --stage windowed

echo ""
echo "=== R22-H189 COMPLETE  $(date '+%F %T') ==="
echo "adjudication is the coordinator's: PRIMARY ragtruth_nonen LOAD-BEARING at or"
echo "below 0.83366, else BOUNDED SMALL; SECONDARY gold_full at 0.84013;"
echo "arena REPORTED vs 0.71218 with a 0.02355 floor and NO promotion path at k=1"
