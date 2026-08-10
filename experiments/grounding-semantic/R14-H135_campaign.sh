#!/usr/bin/env bash
# R14-H135 campaign: one minimal-pair co-location draw (or its seeded control)
# on the admitted H108 lane, plus both arena reads.
#
# Resumable across container restarts: the trainer continues from
# models/R14-H135-<arm|ctrl>-draw<N>/resume.pt automatically, replaying the SAME
# persisted permutation. Relaunch = same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R14-H135_campaign.sh <draw> [colocate|control] \
#       >> logs/R14-H135_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"
arm="${2:-colocate}"

case "$arm" in
  colocate) tag=arm ;;
  control)  tag=ctrl ;;
  *) echo "arm must be colocate|control"; exit 1 ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
ckpt="models/R14-H135-${tag}-draw${draw}"

stage() {
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== H135 ${tag^^} CAMPAIGN draw${draw} $(date '+%F %T') ==="

stage "H135 ${tag} draw${draw} train + in-domain holds" \
  uv run python "$E/R14-H135_trainer.py" --draw "$draw" --arm "$arm"
stage "H135 ${tag} draw${draw} truncated read" uv run python "$E/R8_decomposed_read.py" \
  --model "$ckpt" --tag "R14-H135-${tag}-draw${draw}"
stage "H135 ${tag} draw${draw} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
  --model "$ckpt" --out "R14-H135_${tag}_draw${draw}_windowed_result.json"

echo ""
echo "=== H135 ${tag^^} D${draw} CAMPAIGN COMPLETE $(date '+%F %T') ==="
