#!/usr/bin/env bash
# R14-H131 Stage 1 on GPU2 (RTX 5000 Ada): frozen-weights windowed read at
# max_length=1024 on four checkpoints. Idempotent - each stage skips if its
# result JSON exists.
#
# Launch detached:
#   nohup setsid bash experiments/grounding-semantic/R14_H131_reads.sh \
#       >> logs/R14_gates_gpu.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
R=$E/R14_H131_read1024.py

run() {
  local label="$1" model="$2" out="$3"
  if [ -s "$E/$out" ]; then
    echo "--- $label: already done ($out), skipping ---"
    return 0
  fi
  echo ""
  echo "--- H131 $label  $(date '+%F %T') ---"
  if ! uv run python "$R" --model "$model" --out "$out"; then
    echo "=== FAILED: H131 $label ==="
    exit 1
  fi
}

echo "=== R14-H131 STAGE 1 READS (max_length=1024) $(date '+%F %T') ==="

run h105d1 models/R9-H105-mmbert-dann-clean  R14_H131_h105d1_1024.json
run h105d2 models/R9-H105-draw2              R14_H131_h105d2_1024.json
run h108d1 models/R10-H108-lane-draw1        R14_H131_h108d1_1024.json
run h108d2 models/R10-H108-lane-draw2        R14_H131_h108d2_1024.json

echo ""
echo "=== R14-H131 STAGE 1 READS COMPLETE $(date '+%F %T') ==="
