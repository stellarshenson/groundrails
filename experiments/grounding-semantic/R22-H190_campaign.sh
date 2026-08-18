#!/usr/bin/env bash
# R22-H190 ARITHMETIC-CAPABILITY PROBE SUITE - one lane of conditions on one card.
#
# Registered in docs/experiments/semantic-grounding-experiments.md, block
# "R22-H190 ARITHMETIC-CAPABILITY PROBE SUITE", author-ordered 2026-08-18 ~02:45.
# Entirely off-arena: diagnostic only, no arena read, no promotion path.
#
# Idempotent: every condition skips when its result JSON is on disk, so a
# relaunch after a container restart is the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
# .to("cuda") under WSL2 on this box.
#
# Usage (detached):
#   GPU=2 CONDS="P1 P5 P4" nohup setsid \
#     bash experiments/grounding-semantic/R22-H190_campaign.sh \
#     > logs/R22-H190_laneA_gpu2.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (0, 1 or 2)}"

E=experiments/grounding-semantic
CONDS="${CONDS:?set CONDS, e.g. \"P1 P5 P4\"}"

echo "=== R22-H190 PROBE SUITE  GPU${CUDA_VISIBLE_DEVICES}  conditions: $CONDS  $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

for c in $CONDS; do
  art="$E/R22-H190_${c}_result.json"
  if [ -s "$art" ]; then
    echo ""; echo "--- SKIP (on disk: $art) ---"; continue
  fi
  echo ""; echo "--- condition $c  $(date '+%F %T') ---"
  if ! uv run python "$E/R22-H190_arith_probe.py" --condition "$c"; then
    echo "=== FAILED: $c ==="; exit 1
  fi
done

echo ""
echo "=== R22-H190 LANE COMPLETE  $(date '+%F %T') ==="
echo "adjudication is the coordinator's: LEARNABLE >= 0.70, NOT LEARNABLE < 0.55;"
echo "P5 NOT LEARNABLE fires the author's rule and derivation is abandoned"
