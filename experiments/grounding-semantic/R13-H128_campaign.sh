#!/usr/bin/env bash
# R13-H128 campaign: one WiCE-attributed-support-lane training draw + both arena reads.
#
# Resumable across container restarts: the trainer continues from
# models/R13-H128-draw<N>/resume.pt automatically. Relaunch = same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R13-H128_campaign.sh <draw> \
#       >> logs/R13-H128_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"
shift

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
ckpt="models/R13-H128-draw${draw}"

stage() {
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== H128 CAMPAIGN draw${draw} $(date '+%F %T') ==="

stage "H128 draw${draw} train + in-domain holds" \
  uv run python "$E/R13-H128_trainer.py" --draw "$draw" "$@"
stage "H128 draw${draw} truncated read" uv run python "$E/R8_decomposed_read.py" \
  --model "$ckpt" --tag "R13-H128-draw${draw}"
stage "H128 draw${draw} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
  --model "$ckpt" --out "R13-H128_draw${draw}_windowed_result.json"

echo ""
echo "=== H128 D${draw} CAMPAIGN COMPLETE $(date '+%F %T') ==="
