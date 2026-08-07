#!/usr/bin/env bash
# Resume tail of the R10 lane campaign: H108 draw 2 only.
#
# The container was restarted mid-draw and killed the chain; the draw resumes
# from models/R10-H108-lane-draw2/resume.pt (step 6000). Everything before this
# draw is already recorded on disk.
#
# Launch detached:
#   nohup setsid bash experiments/grounding-semantic/R10_lane_campaign_h108d2.sh \
#       >> logs/R10_lane_campaign.log 2>&1 &

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

echo "=== R10 LANE CAMPAIGN RESUME (H108 draw 2) $(date '+%F %T') ==="

lane=R10-H108
n=2
ckpt="models/${lane}-lane-draw${n}"

stage "${lane} draw${n} train" uv run python "$E/${lane}_lane.py" --draw "$n"
stage "${lane} draw${n} truncated read" uv run python "$E/R8_decomposed_read.py" \
  --model "$ckpt" --tag "${lane}-lane-draw${n}"
stage "${lane} draw${n} windowed read (PRIMARY)" uv run python "$E/R8-H101_windowed_read.py" \
  --model "$ckpt" --out "${lane}_lane_draw${n}_windowed_result.json"
stage "${lane} draw${n} collect" uv run python "$E/R10_lane_collect.py" \
  --lane "$lane" --draw "$n"

echo ""
echo "=== H108 LANE DONE ==="
echo "=== R10 LANE CAMPAIGN COMPLETE $(date '+%F %T') ==="
