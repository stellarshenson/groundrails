#!/usr/bin/env bash
# R10 lane campaign - H107 then H108, two draws each, both blind reads per draw.
#
# Every stage is gated on the previous one: a non-zero exit prints a FAILED
# marker and aborts the campaign, so a crashed draw never leaves a later stage
# scoring a stale or missing checkpoint. Each draw checkpoints its model and
# writes its result JSON before the next stage starts, so the campaign is
# resumable from disk.
#
# Launch detached:
#   nohup setsid bash experiments/grounding-semantic/R10_lane_campaign.sh \
#       > logs/R10_lane_campaign.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic

stage() {  # stage <label> <cmd...>
  local label="$1"; shift
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

draw() {  # draw <lane> <n>
  local lane="$1" n="$2"
  local ckpt="models/${lane}-lane-draw${n}"
  stage "${lane} draw${n} train" uv run python "$E/${lane}_lane.py" --draw "$n"
  stage "${lane} draw${n} truncated read" uv run python "$E/R8_decomposed_read.py" \
    --model "$ckpt" --tag "${lane}-lane-draw${n}"
  stage "${lane} draw${n} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
    --model "$ckpt" --out "${lane}_lane_draw${n}_windowed_result.json"
  stage "${lane} draw${n} collect" uv run python "$E/R10_lane_collect.py" \
    --lane "$lane" --draw "$n"
}

echo "=== R10 LANE CAMPAIGN START $(date '+%F %T') ==="

draw R10-H107 1
draw R10-H107 2
echo ""
echo "=== H107 LANE DONE ==="

draw R10-H108 1
draw R10-H108 2
echo ""
echo "=== H108 LANE DONE ==="

echo "=== R10 LANE CAMPAIGN COMPLETE $(date '+%F %T') ==="
