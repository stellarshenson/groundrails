#!/usr/bin/env bash
# R22-H188 POST-TRAINING MEASUREMENTS - the stages R22-H188_campaign.sh does NOT run.
#
# campaign.sh chains train -> windowed arena read and stops there. The arm's
# PRIMARY gate is the FinDVer read, and the finqa detail is owed by the
# registration; neither is wired into that script. The executor that built the
# campaign held them in its own watcher, and that watcher died with it.
#
# This script is that watcher, detached, so it survives any agent or session.
# It waits on ARTIFACTS, not on processes - a process watch cannot tell a
# finished run from a killed one, and pgrep self-matches its own command line.
#
# Idempotent: every stage is skipped when its artifact is already on disk, so a
# relaunch after a container restart is the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
# .to("cuda") under WSL2 on this box.
#
# Usage (detached):
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R22-H188_measure.sh \
#       > logs/R22-H188_measure.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (0, 1 or 2)}"

E=experiments/grounding-semantic
D1="$E/R22-H188_arm_draw1_windowed_result.json"
D2="$E/R22-H188_arm_draw2_windowed_result.json"

echo "=== R22-H188 MEASUREMENTS  GPU${CUDA_VISIBLE_DEVICES}  $(date '+%F %T') ==="
echo "waiting on both draws' arena reads:"
echo "  $D1"
echo "  $D2"

until [ -s "$D1" ] && [ -s "$D2" ]; do
  sleep 120
done

echo "both draws landed  $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

run_unless() {
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

# PRIMARY - the arm's decision. Off-arena.
run_unless "$E/R22-H188_findver_read.json" \
  "FinDVer read, both draws (PRIMARY)" \
  uv run python "$E/R22-H188_findver_read.py"

# finqa detail - owed by the registration, reported against the 9-of-20 ceiling.
for d in 1 2; do
  run_unless "$E/R22-H188_finqa_scores_draw${d}.npz" \
    "finqa detail score, draw ${d}" \
    uv run python "$E/R22-H188_finqa_detail.py" --stage score --draw "$d"
done

run_unless "$E/R22-H188_finqa_detail.json" \
  "finqa detail analyse" \
  uv run python "$E/R22-H188_finqa_detail.py" --stage analyse

echo ""
echo "=== R22-H188 MEASUREMENTS COMPLETE  $(date '+%F %T') ==="
echo "adjudication is the coordinator's: PRIMARY FinDVer-numeric 2-draw mean >= 0.55;"
echo "CONTROL ie/knowledge within 0.02 of 0.66095/0.58380; KILL numeric < 0.52"
