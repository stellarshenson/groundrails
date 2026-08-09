#!/usr/bin/env bash
# R12-H122 campaign: one DANN-group-collapse training draw + both arena reads.
#
# Resumable across container restarts: the trainer continues from
# models/R12-H122-draw<N>/resume.pt automatically. Relaunch = same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R12-H122_campaign.sh <draw> \
#       >> logs/R12-H122_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"
shift

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
ckpt="models/R12-H122-draw${draw}"

stage() {
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== H122 CAMPAIGN draw${draw} $(date '+%F %T') ==="

stage "H122 draw${draw} train" uv run python "$E/R12-H122_trainer.py" --draw "$draw" "$@"
stage "H122 draw${draw} truncated read" uv run python "$E/R8_decomposed_read.py" \
  --model "$ckpt" --tag "R12-H122-draw${draw}"
stage "H122 draw${draw} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
  --model "$ckpt" --out "R12-H122_draw${draw}_windowed_result.json"

echo ""
echo "=== H122 D${draw} CAMPAIGN COMPLETE $(date '+%F %T') ==="
