#!/usr/bin/env bash
# R19-H161 lane L2 chain - waits on the A0 substrate dump, then runs the failure-
# class autopsy. Structure copied from R19-H161_chain.sh (the coordinator's wrapper
# for L1/L3/L4), which is left untouched; L2 was respawned as its own executor and
# so carries its own wrapper.
#
# CPU only - the lane forces CUDA_VISIBLE_DEVICES empty itself.
#
# Usage:
#   nohup setsid bash experiments/grounding-semantic/R19-H161_L2_chain.sh \
#     >> logs/R19-H161_L2.log 2>&1 &

set -u
cd /home/lab/workspace/private/ai-assistants/groundrails || exit 1

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DUMP=logs/R19-H161_dump.log

echo "=== H161 L2 chain armed $(date '+%F %T') - waiting on the A0 dump ==="

while true; do
  if grep -q "=== H161 DUMP FAILED:" "$DUMP" 2>/dev/null; then
    echo "=== H161 L2 ABORT: the dump reported FAILED $(date '+%F %T') ==="
    grep "=== H161 DUMP FAILED:" "$DUMP" | tail -3
    exit 1
  fi
  if grep -q "=== H161 DUMP ALL COMPLETE ===" "$DUMP" 2>/dev/null; then
    echo "=== dump complete $(date '+%F %T') - running L2 ==="
    break
  fi
  sleep 60
done

echo ""
echo "--- L2 failure-class autopsy  $(date '+%F %T') ---"
if uv run python "$E/R19-H161_L2.py"; then
  echo "--- L2 OK  $(date '+%F %T') ---"
else
  echo "=== H161 LANE FAILED: L2 ==="
fi

echo ""
echo "=== H161 L2 CHAIN COMPLETE $(date '+%F %T') ==="
ls -la "$E"/R19-H161_L2_result.json 2>/dev/null || echo "(no L2 result on disk)"
