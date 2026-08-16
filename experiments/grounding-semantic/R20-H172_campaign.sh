#!/usr/bin/env bash
# R20-H172 FLAGSHIP VARIANCE DRAWS - draws 5 and 6 of the R18-H150 recipe,
# seeds 5150/6150, executing author ruling 2 of 2026-08-16 (k=6 draws).
#
# Recipe: the R18-H150 flagship VERBATIM through the banked R19-H160 wrapper +
# cotangent split executor; see R20-H172_flagship_run.py for the injection.
# No promotion bar - the deliverable is the k=6 mean and its SE (0.00485).
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda") under WSL2 on this box.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk; the trainer resumes from models/R20-H172-arm-draw<N>/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# Usage:
#   CPU census, no GPU touched:
#     DRAW=5 bash experiments/grounding-semantic/R20-H172_campaign.sh --census
#   a draw, detached:
#     GPU=1 DRAW=5 nohup setsid bash experiments/grounding-semantic/R20-H172_campaign.sh \
#         >> logs/R20-H172_campaign_d5.log 2>&1 &
#     GPU=2 DRAW=6 nohup setsid bash experiments/grounding-semantic/R20-H172_campaign.sh \
#         >> logs/R20-H172_campaign_d6.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW="${DRAW:?set DRAW explicitly (5 or 6)}"
ARM=R20-H172

if [ "${1:-}" = "--census" ]; then
  echo "=== H172 draw ${DRAW} CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R20-H172_flagship_run.py" \
    --stage census --draw "$DRAW"
  exit $?
fi

export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (0, 1 or 2)}"

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

echo "=== H172 FLAGSHIP DRAW ${DRAW} $(date '+%F %T') ==="
echo "R18-H150 recipe verbatim via R19-H160 wrapper + split executor; seed $((DRAW * 1000 + 150))"
echo "no promotion bar - variance draw under the adopted protocol (ruling 1-2 of 2026-08-16)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H172 draw${DRAW} train + in-domain suite (split executor)" \
  uv run python "$E/R20-H172_flagship_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H172 draw${DRAW} blind windowed decomposed-min arena read" \
  uv run python "$E/R20-H172_flagship_run.py" --stage windowed --draw "$DRAW"

echo ""
echo "=== H172 DRAW ${DRAW} COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudication: pool with draws 1-4 (0.71436 / 0.71661 / 0.70870 / 0.72365)"
echo "into the k=6 mean; SE 0.00485; pre-stated readings in the registration block"
