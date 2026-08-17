#!/usr/bin/env bash
# R20-H175b QUESTION CONDITIONING (MEASUREMENT ONLY) - one draw of the H150
# flagship recipe plus an optional question prefix on the claim side and the
# stage-0 `qrel_contrast` contrast lane (739,182 rows, 15 DANN groups), through
# the banked R19-H160 wrapper + cotangent split executor. See R20-H175b_arm_run.py.
#
# MEASUREMENT ONLY: no promotion route to the shipped ground()/ground_batch()
# API. Nothing under src/groundrails/ is touched by this campaign.
#
# The window-census control is REPOINTED, not weakened: R20-H175b_window_census.py
# recomputes the combined census from the built mix, asserts the flagship sub-mix
# against the banked R18-H150 census and the contrast lane against its own
# manifest, and banks R20-H175b_window_census.json; census_crosscheck reads that
# file. A drifting mix still aborts before a card.
#
# The MANDATORY loader assertion runs inside build_mix on every stage that
# assembles the mix: a question composed for all 17,972 lane rows, all 8,986
# pairs differing as composed strings, 400 sampled pairs differing as TOKENIZED
# inputs. Any failure aborts before a card is touched.
#
# GPU2 ONLY. GPUs 0 and 1 carry the R20-H174 draws and are not touched, queried
# for allocation, or waited on.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda") under WSL2 on this box.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk; the trainer resumes from models/R20-H175b-arm-draw<N>/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# Usage:
#   CPU census, no GPU touched:
#     DRAW=1 bash experiments/grounding-semantic/R20-H175b_campaign.sh --census
#   the draw, detached:
#     GPU=2 DRAW=1 nohup setsid bash \
#       experiments/grounding-semantic/R20-H175b_campaign.sh \
#       2>&1 | tee -a logs/R20-H175b_campaign_d1.log &
#
# DRAW 1 ONLY is committed (QUEUE AMENDMENT Q1). Further draws exist only if the
# coordinator adjudicates the mechanism gate a pass.

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW="${DRAW:?set DRAW explicitly (1)}"
ARM=R20-H175b

if [ "${1:-}" = "--census" ]; then
  echo "=== H175b draw ${DRAW} CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R20-H175b_arm_run.py" \
    --stage census --draw "$DRAW"
  exit $?
fi

export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (2)}"
if [ "$CUDA_VISIBLE_DEVICES" != "2" ]; then
  echo "=== ABORT: this campaign is GPU2 ONLY (GPUs 0 and 1 carry R20-H174) ==="
  exit 1
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

echo "=== H175b QUESTION CONDITIONING DRAW ${DRAW} $(date '+%F %T') ==="
echo "MEASUREMENT ONLY - no promotion route to the shipped API"
echo "H150 flagship recipe verbatim + optional question prefix on the claim side"
echo "+ qrel_contrast 17,972 = 739,182 rows, 15 DANN groups; seed 1175"
echo "census rebound to R20-H175b_window_census.json (739,182 / 1.4731 / 0.1880)"
nvidia-smi --id=2 --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_intervention_proof.json" \
  "H175b intervention proof (question coverage + loader assertion)" \
  uv run python "$E/R20-H175b_arm_run.py" --stage qproof --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H175b draw${DRAW} train + in-domain suite (split executor)" \
  uv run python "$E/R20-H175b_arm_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H175b draw${DRAW} blind windowed arena read (= the empty-question hold)" \
  uv run python "$E/R20-H175b_arm_run.py" --stage windowed --draw "$DRAW"

stage_unless "$E/${ARM}_qeval_read.json" \
  "H175b draw${DRAW} PRIMARY question-relevance mechanism read" \
  uv run python "$E/R20-H175b_qeval_read.py" --draw "$DRAW"

echo ""
echo "=== H175b DRAW ${DRAW} COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudication is the coordinator's. PRIMARY: qlane_eval AUROC >= 0.80 against"
echo "the higher of the banked floors 0.5000 and 0.5816. SECONDARY (report-bearing,"
echo "not promotion): hagrid vs 0.6393, emanual vs 0.6787. GUARDS (Amendment G1):"
echo "finqa within 0.1000 of 0.6619, tatqa within 0.1147 of 0.7787, delucionqa"
echo "within 0.1202 of 0.8267; gold_full >= 0.84; non-EN >= 0.82. Empty-question"
echo "hold: the arena read within 0.0218 of the k=6 flagship mean 0.71218."
