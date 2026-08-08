#!/usr/bin/env bash
# DR lane campaign: one training draw (control or margin arm) + both arena reads.
#
# Resumable across container restarts: the trainer continues from
# models/DR-lane-draw<N>-<arm>/resume.pt automatically; finished stages no-op
# via their result artifacts being consumed downstream. Relaunch = same command.
#
# Extra arguments after [lambda_margin] are passed through to the trainer (e.g.
# --ema, the R12-H120 buffer, when this draw hosts the H120 read).
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/DR_campaign.sh <draw> <arm> [lambda_margin] [trainer args...] \
#       >> logs/DR_campaign_d<draw>_<arm>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"
arm="${2:?arm (control|margin)}"
lm="${3:-0.1}"
shift $(( $# > 3 ? 3 : $# ))  # remaining args pass through to the trainer (e.g. --ema)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
ckpt="models/DR-lane-draw${draw}-${arm}"

stage() {
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== DR CAMPAIGN draw${draw} ${arm} $(date '+%F %T') ==="

stage "DR draw${draw} ${arm} train" uv run python "$E/DR_lane_trainer.py" \
  --draw "$draw" --arm "$arm" --lambda-margin "$lm" "$@"
stage "DR draw${draw} ${arm} truncated read" uv run python "$E/R8_decomposed_read.py" \
  --model "$ckpt" --tag "DR-lane-draw${draw}-${arm}"
stage "DR draw${draw} ${arm} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
  --model "$ckpt" --out "DR_lane_draw${draw}_${arm}_windowed_result.json"

echo ""
echo "=== DR D${draw} ${arm^^} CAMPAIGN COMPLETE $(date '+%F %T') ==="
