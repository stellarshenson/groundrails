#!/usr/bin/env bash
# R14 gate chain on GPU0 (RTX PRO 4000). Runs after the H130 dumps.
# Idempotent - each stage skips if its result JSON exists.
#
#   nohup setsid bash experiments/grounding-semantic/R14_gpu0_chain.sh \
#       >> logs/R14_gates_gpu.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic

stage() {
  local label="$1" out="$2" script="$3"
  if [ -s "$E/$out" ]; then
    echo "--- $label: already done ($out), skipping ---"
    return 0
  fi
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! uv run python "$E/$script"; then
    echo "=== FAILED: $label ==="
    return 1
  fi
}

echo "=== R14 GPU0 CHAIN $(date '+%F %T') ==="

stage H133 R14_gate_H133_probe.json     R14_H133_probe.py
stage H134 R14_gate_H134_partialr.json  R14_H134_partialr.py
stage H132 R14_gate_H132_layerprobe.json R14_H132_layerprobe.py

echo ""
echo "=== R14 GPU0 CHAIN COMPLETE $(date '+%F %T') ==="
