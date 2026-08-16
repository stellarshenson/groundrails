#!/usr/bin/env bash
# R19-H168 EuroBERT-210m trunk swap - draw 1 train, then the blind windowed read.
#
# Waits for the R19-H165 length ladder to clear GPU1 before starting. The
# H158/H159 record has a co-location stall precedent, so the training run takes
# the 96 GB card exclusively rather than sharing it with the ladder's long cells.
#
# Restart-resilient: if the container dies, re-running this script is safe. The
# train stage is skipped when its result JSON already exists, and the read stage
# is skipped when its own output exists.
#
# Usage:
#   nohup setsid bash experiments/grounding-semantic/R19-H168_campaign.sh \
#     >> logs/R19-H168_campaign_d1.log 2>&1 &

set -u
cd /home/lab/workspace/private/ai-assistants/groundrails || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
R="$E/R19-H168_arm_run.py"
LADDER=logs/R19-H165_ladder.log
TRAIN_JSON="$E/R19-H168_arm_draw1_result.json"
READ_JSON="$E/R19-H168_arm_draw1_windowed_result.json"

echo "=== R19-H168 campaign armed $(date '+%F %T') ==="

# --- wait for the ladder to release GPU1 -------------------------------------
if ! grep -q 'H165 LADDER COMPLETE' "$LADDER" 2>/dev/null; then
  echo "waiting for the R19-H165 ladder to finish before taking GPU1..."
  while ! grep -q 'H165 LADDER COMPLETE' "$LADDER" 2>/dev/null; do
    sleep 60
  done
fi
echo "=== ladder clear $(date '+%F %T') - GPU1 is ours ==="

# --- draw 1 -------------------------------------------------------------------
if [ -s "$TRAIN_JSON" ]; then
  echo "SKIP train (on disk: $(basename "$TRAIN_JSON"))"
else
  echo ""
  echo "--- H168 draw 1 TRAIN on GPU1  $(date '+%F %T') ---"
  CUDA_VISIBLE_DEVICES=1 uv run python "$R" --stage train
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$TRAIN_JSON" ]; then
    echo "=== H168 TRAIN FAILED (rc=$rc) - nothing further is spent ==="
    exit 1
  fi
fi

# --- the blind windowed read (PRIMARY) ----------------------------------------
if [ -s "$READ_JSON" ]; then
  echo "SKIP read (on disk: $(basename "$READ_JSON"))"
else
  echo ""
  echo "--- H168 draw 1 WINDOWED READ on GPU1  $(date '+%F %T') ---"
  CUDA_VISIBLE_DEVICES=1 uv run python "$R" --stage windowed
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$READ_JSON" ]; then
    echo "=== H168 READ FAILED (rc=$rc) ==="
    exit 1
  fi
fi

echo ""
echo "=== H168 DRAW 1 COMPLETE $(date '+%F %T') ==="
python3 - "$READ_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
per = d.get("per_subset") or d.get("subsets") or {}
mean = d.get("mean") or d.get("blind_mean")
print(f"  H168 EuroBERT draw 1 windowed mean {mean}")
print("  flagship draw 1 windowed mean 0.71436 | flagship 2-draw 0.71549")
print("  pilot KILL bar 0.71049 | PRIMARY 2-draw bar 0.72049")
if isinstance(per, dict):
    for k in sorted(per):
        print(f"    {k:<12} {per[k]}")
PY
