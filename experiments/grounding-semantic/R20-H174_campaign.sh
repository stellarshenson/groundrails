#!/usr/bin/env bash
# R20-H174 HAGRID/EMANUAL PORTFOLIO ARM - one draw of the H150 flagship recipe
# plus the L1/L2/L4 stage-0 lanes (760,618 rows, 17 DANN groups), through the
# banked R19-H160 wrapper + cotangent split executor. See R20-H174_arm_run.py.
#
# The window-census control is REPOINTED, not weakened: R20-H174_window_census.py
# recomputes the combined census from the built mix, asserts the registered
# 760,618 / 1.5977 / 0.2094 geometry, and banks R20-H174_window_census.json;
# census_crosscheck reads that file. A drifting mix still aborts before a card.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda") under WSL2 on this box.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk; the trainer resumes from models/R20-H174-arm-draw<N>/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# Usage:
#   CPU census, no GPU touched:
#     DRAW=1 bash experiments/grounding-semantic/R20-H174_campaign.sh --census
#   a draw, detached:
#     GPU=1 DRAW=1 nohup setsid bash experiments/grounding-semantic/R20-H174_campaign.sh \
#         >> logs/R20-H174_campaign_d1.log 2>&1 &
#     GPU=0 DRAW=2 nohup setsid bash experiments/grounding-semantic/R20-H174_campaign.sh \
#         >> logs/R20-H174_campaign_d2.log 2>&1 &
#
# Draws 3 and 4 are declared by the registration at k=4 but are launched only on
# the coordinator's word after draw 1 clears its gate (KILL: draw 1 blind arena
# mean < 0.695, or a table-guard breach on the 2-draw mean).

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW="${DRAW:?set DRAW explicitly (1-4)}"
ARM=R20-H174

if [ "${1:-}" = "--census" ]; then
  echo "=== H174 draw ${DRAW} CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R20-H174_arm_run.py" \
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

echo "=== H174 PORTFOLIO DRAW ${DRAW} $(date '+%F %T') ==="
echo "H150 flagship recipe verbatim + L1 frame_reject 8,000 + L2 attr_pool 21,408"
echo "+ L4 path_bind 10,000 = 760,618 rows, 17 DANN groups; seed $((DRAW * 1000 + 174))"
echo "census rebound to R20-H174_window_census.json (760,618 / 1.5977 / 0.2094)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H174 draw${DRAW} train + in-domain suite (split executor)" \
  uv run python "$E/R20-H174_arm_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H174 draw${DRAW} blind windowed decomposed-min arena read" \
  uv run python "$E/R20-H174_arm_run.py" --stage windowed --draw "$DRAW"

echo ""
echo "=== H174 DRAW ${DRAW} COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudication is the coordinator's: PRIMARY k=4 mean >= k=6 flagship mean + 0.01407;"
echo "mechanism gates hagrid >= 0.680 / frame-only misrank < 5% / k-doc slope >= 0;"
echo "table guard finqa 0.062, tatqa 0.025, delucionqa 0.012; KILL draw 1 < 0.695"
