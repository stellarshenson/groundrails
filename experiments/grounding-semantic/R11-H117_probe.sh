#!/usr/bin/env bash
# R11-H117 kill-gate 2 probe - 3 arms sequentially on GPU1, draw seed 1117.
# Idempotent: an arm whose train json exists is skipped.
set -u
cd /home/lab/workspace/private/ai-assistants/groundrails
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
STEPS=3125
for LAM in 0 0.1 0.3; do
  J="experiments/grounding-semantic/R11-H117_probe_lam${LAM}_train.json"
  if [ -f "$J" ]; then echo "SKIP lam=$LAM (already done)"; continue; fi
  echo "=== ARM lambda_margin=$LAM start $(date -Is) ==="
  uv run python experiments/grounding-semantic/R11-H117_probe_trainer.py \
      --draw 1 --lambda-margin "$LAM" --max-steps "$STEPS"
  echo "=== ARM lambda_margin=$LAM exit=$? $(date -Is) ==="
done
echo "=== H117 PROBE CAMPAIGN DONE $(date -Is) ==="
