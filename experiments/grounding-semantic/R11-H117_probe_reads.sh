#!/usr/bin/env bash
# R11-H117 probe reads - gold_full + (ragtruth_en, held-out pair acc) per arm.
# Idempotent: an existing output json is skipped.
set -u
cd /home/lab/workspace/private/ai-assistants/groundrails
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
for LAM in 0 0.1 0.3; do
  M="models/H117-probe-lam${LAM}"
  G="experiments/grounding-semantic/R11-H117_probe_lam${LAM}_goldfull.json"
  R="experiments/grounding-semantic/R11-H117_probe_lam${LAM}_read.json"
  [ -f "$M/dann_student.pt" ] || { echo "MISSING $M checkpoint - skip"; continue; }
  if [ -f "$G" ]; then echo "SKIP goldfull lam=$LAM"; else
    uv run python experiments/grounding-semantic/R11-H118_goldfull_read.py \
        --model "$M" --out "R11-H117_probe_lam${LAM}_goldfull.json"
  fi
  if [ -f "$R" ]; then echo "SKIP read lam=$LAM"; else
    uv run python experiments/grounding-semantic/R11-H117_probe_read.py \
        --model "$M" --out "R11-H117_probe_lam${LAM}_read.json"
  fi
done
echo "=== H117 PROBE READS DONE $(date -Is) ==="
