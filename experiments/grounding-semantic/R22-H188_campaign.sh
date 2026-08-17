#!/usr/bin/env bash
# R22-H188 DERIVATION-ENHANCED MIX - one draw of the H150 flagship recipe plus
# the R22-H187 num_derive lane (751,210 rows, 15 DANN groups), through the
# banked R19-H160 wrapper + cotangent split executor. See R22-H188_arm_run.py.
#
# The window-census control is REPOINTED, not weakened: R22-H188_window_census.py
# recomputes the combined census from the built mix, asserts the derived
# 751,210 / 1.4628 / 0.1832 geometry, and banks R22-H188_window_census.json;
# census_crosscheck reads that file. A drifting mix still aborts before a card.
#
# The permutation guard runs BEFORE this script, as its own stage:
#   uv run python .../R22-H188_arm_run.py --stage permguard --draw <N>
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda") under WSL2 on this box.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk; the trainer resumes from models/R22-H188-arm-draw<N>/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# Usage (detached):
#   GPU=1 DRAW=1 nohup setsid bash experiments/grounding-semantic/R22-H188_campaign.sh \
#       >> logs/R22-H188_draw1_gpu1.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW="${DRAW:?set DRAW explicitly (1 or 2)}"
ARM=R22-H188
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

echo "=== H188 DERIVATION DRAW ${DRAW} $(date '+%F %T') ==="
echo "H150 flagship recipe verbatim + R22-H187 num_derive 30,000"
echo "= 751,210 rows, 15 DANN groups; seed $((DRAW * 1000 + 188))"
echo "census rebound to R22-H188_window_census.json (751,210 / 1.4628 / 0.1832)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H188 draw${DRAW} train + in-domain suite (split executor)" \
  uv run python "$E/R22-H188_arm_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H188 draw${DRAW} blind windowed decomposed-min arena read" \
  uv run python "$E/R22-H188_arm_run.py" --stage windowed --draw "$DRAW"

echo ""
echo "=== H188 DRAW ${DRAW} COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudication is the coordinator's: PRIMARY FinDVer-numeric 2-draw mean >= 0.55;"
echo "CONTROL ie/knowledge within 0.02 of 0.66095/0.58380; KILL numeric < 0.52;"
echo "arena measured not tuned - difference floor 0.01780, table guard G3, gold_full >= 0.84"
