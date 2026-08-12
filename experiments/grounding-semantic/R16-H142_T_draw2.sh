#!/usr/bin/env bash
# R16-H142-T draw 2 - the confirming twin draw registered for the twin promotion.
#
# The twin leg of R16-H142_G1_campaign.sh and nothing else: same 685,670-row
# clean mix, evidence UNTRUNCATED and windowed 1500/750, MIL max-over-window
# BCE, 12-group DANN, full trunk at lr 1e-5, adapter FROZEN at its zero init.
# The one difference from draw 1 is the seed: 2142, against draw 1's 1142.
#
# Stages, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#   2  blind windowed decomposed-min arena read (PRIMARY)
#   3  anti-gaming stage, prefix R16-H142-T-d2
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer continues from models/R16-H142-T-draw2/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# NOT LAUNCHED BY THE HOLDS PASS - the launch decision is the coordinator's.
#
# Launch detached, GPU by env (never GPU1 while the H146 arm trains there):
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R16-H142_T_draw2.sh \
#       >> logs/R16-H142_T_draw2.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R16-H142_T_draw2.sh --census

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments is NOT set: it breaks .to("cuda")
# on this WSL2 host and cost the campaign four relaunches.

E=experiments/grounding-semantic
CKPT=models/R16-H142-T-draw2
AG_PREFIX=R16-H142-T-d2
AG_LINK=models/${AG_PREFIX}-arm-draw2

if [ "${1:-}" = "--census" ]; then
  echo "=== H142-T draw 2 CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R16-H142_T_draw2_run.py" --stage census
  exit $?
fi

stage() {
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

# stage_unless <artifact> <label> <cmd...>
stage_unless() {
  local art="$1" label="$2"; shift 2
  if [ -s "$art" ]; then
    echo ""
    echo "--- SKIP (already on disk: $art)  $label ---"
    return 0
  fi
  stage "$label" "$@"
}

echo "=== H142-T DRAW 2 $(date '+%F %T') ==="
echo "twin protocol, seed 2142 (draw 1: 1142), adapter frozen at zero"
echo "checkpoint $CKPT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/R16-H142_T_draw2_result.json" \
  "H142-T draw 2 train + in-domain suite" \
  uv run python "$E/R16-H142_T_draw2_run.py" --stage train

stage_unless "$E/R16-H142_T_draw2_windowed_result.json" \
  "H142-T draw 2 blind windowed arena read (PRIMARY)" \
  uv run python "$E/R16-H142_T_draw2_run.py" --stage windowed

# R14-H133_antigaming.py resolves its checkpoint as models/<arm>-arm-draw<N>,
# which no --arm prefix can make name models/R16-H142-T-draw2 - hence the link.
ln -sfn "$(basename $CKPT)" "$AG_LINK"

stage_unless "$E/${AG_PREFIX}_antigaming_draw2_result.json" \
  "H142-T draw 2 anti-gaming stage" \
  uv run python "$E/R14-H133_antigaming.py" --draw 2 --arm "$AG_PREFIX"

echo ""
echo "=== H142-T DRAW 2 COMPLETE $(date '+%F %T') ==="
echo "adjudicate against the R16-H142-T promotion registration: 2-draw mean,"
echo "per-subset floors, gold_full >= 0.84, anti-gaming vs the A1 re-priced band"
