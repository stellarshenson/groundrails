#!/usr/bin/env bash
# R19-H161 lane chain - waits on the A0 substrate dump, then runs the lanes that
# read it, then the GPU geometry lane.
#
# Written by the coordinator after the fanout's five executors were killed by a
# session limit at 17:08 on 2026-08-14. Their scripts survived on disk; only the
# wait-and-run wrappers were missing, so this replaces them. L2 (failure-class
# autopsy) is NOT chained here - its script was never written and is respawned
# as its own executor.
#
# Order: L1 and L4 are CPU-only and run first, sequentially, so a failure in one
# is legible in the log rather than interleaved. L3 is a GPU lane and runs last,
# on GPU0, after the dump has released the card.
#
# Usage:
#   nohup setsid bash experiments/grounding-semantic/R19-H161_chain.sh \
#     >> logs/R19-H161_chain.log 2>&1 &

set -u
cd /home/lab/workspace/private/ai-assistants/groundrails || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DUMP=logs/R19-H161_dump.log

echo "=== H161 lane chain armed $(date '+%F %T') - waiting on the A0 dump ==="

while true; do
  if grep -q "=== H161 DUMP FAILED:" "$DUMP" 2>/dev/null; then
    echo "=== H161 CHAIN ABORT: the dump reported FAILED $(date '+%F %T') ==="
    grep "=== H161 DUMP FAILED:" "$DUMP" | tail -3
    exit 1
  fi
  if grep -q "=== H161 DUMP ALL COMPLETE ===" "$DUMP" 2>/dev/null; then
    echo "=== dump complete $(date '+%F %T') - running lanes ==="
    break
  fi
  sleep 60
done

run_lane() {
  local label="$1"
  shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if "$@"; then
    echo "--- $label OK  $(date '+%F %T') ---"
  else
    echo "=== H161 LANE FAILED: $label ==="
  fi
}

run_lane "L1 overlap-prior sensitivity" \
  uv run python "$E/R19-H161_L1.py"

run_lane "L4 window strata + argmax drift" \
  uv run python "$E/R19-H161_L4.py"

CUDA_VISIBLE_DEVICES=0 run_lane "L3 adversarial group geometry (GPU0)" \
  uv run python "$E/R19-H161_L3_geometry.py" --stage all

echo ""
echo "=== H161 LANE CHAIN COMPLETE $(date '+%F %T') ==="
ls -la "$E"/R19-H161_L*_result.json 2>/dev/null || echo "(no lane results on disk)"
