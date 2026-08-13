#!/usr/bin/env bash
# R18-H155 INIT-VS-ORDER ATTRIBUTION PAIR - the decomposition of the twin
# pair's 0.0243 two-seed arena-mean spread into its order component: the twin
# protocol verbatim (clean 685,670-row mix via the H108 public_train lineage,
# evidence UNTRUNCATED, 1,500/750 windowed presentation, MIL max-over-windows
# BCE, 12-group DANN, full trunk lr 1e-5 OneCycleLR 1 epoch, adapter frozen at
# its zero init, NO H152 regularizers) with the init DECOUPLED from the data
# order:
#
#   init seed  5155  SHARED by both draws (H126 facility: manual_seed before
#                    construction, re-issued after) - identical init weights
#                    AND identical dropout stream; the ONLY difference between
#                    the draws is the flat-shuffle permutation
#   perm seed  51551 draw 1 (registered label 5155a)
#   perm seed  51552 draw 2 (registered label 5155b)
#
# Registered in docs/experiments/semantic-grounding-experiments.md, block
# "R18-H155 INIT-VS-ORDER ATTRIBUTION PAIR" (2026-08-12 ~18:40). PRIMARY =
# the init-paired two-draw arena-mean spread vs the twin's 0.0243 (order share
# = init-paired spread / 0.0243). NO accuracy bars (attribution measurement,
# not a candidate); the holds are read for context only.
#
# Draws, sequential:
#   draw 1  (5155a)  models/R18-H155-initpair-draw1
#   draw 2  (5155b)  models/R18-H155-initpair-draw2
#
# Stages per draw, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#      through the BANKED G1 trainer dispatched byte-identical by the wrapper
#   2  blind windowed decomposed-min arena read (PRIMARY)
#   3  anti-gaming stage, prefix R18-H155-d<N> - R14-H133_antigaming.py
#      resolves its checkpoint as models/<arm>-arm-draw<N>, hence the symlink
#      (the R16-H142_T_draw2.sh pattern)
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the banked trainer resumes from
# models/R18-H155-initpair-draw<N>/resume.pt replaying the SAME decoupled
# permutation (the rebind is process-level and applies identically on resume).
# Relaunch = the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda")/.cuda() under WSL2 on this box
# (confirmed in R17-H144, R17-H145 and R16-H142).
#
# Launch detached, GPU by env (GPU1 after the H152 pair completes, per the
# registration; GPU0/GPU2 only on a coordinator ruling):
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R18-H155_campaign.sh \
#       >> logs/R18-H155_campaign.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R18-H155_campaign.sh --census \
#       2>&1 | tee logs/R18-H155_census.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic

if [ "${1:-}" = "--census" ]; then
  echo "=== H155 init-vs-order attribution pair CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R18-H155_arm_run.py" --stage census
  exit $?
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

# run_draw <draw> <perm_seed>
run_draw() {
  local draw="$1" perm_seed="$2"
  local ckpt="models/R18-H155-initpair-draw${draw}"
  local prefix="R18-H155-d${draw}"
  local link="models/${prefix}-arm-draw${draw}"

  stage_unless "$E/R18-H155_twin_draw${draw}_result.json" \
    "H155 draw ${draw} train + in-domain suite (init 5155 shared, perm ${perm_seed})" \
    uv run python "$E/R18-H155_arm_run.py" --stage train --draw "$draw"

  stage_unless "$E/R18-H155_twin_draw${draw}_windowed_result.json" \
    "H155 draw ${draw} blind windowed arena read (PRIMARY)" \
    uv run python "$E/R18-H155_arm_run.py" --stage windowed --draw "$draw"

  ln -sfn "$(basename "$ckpt")" "$link"

  stage_unless "$E/${prefix}_antigaming_draw${draw}_result.json" \
    "H155 draw ${draw} anti-gaming stage (prefix ${prefix}, context only)" \
    uv run python "$E/R14-H133_antigaming.py" --draw "$draw" --arm "$prefix"
}

echo "=== H155 INIT-VS-ORDER ATTRIBUTION PAIR CAMPAIGN $(date '+%F %T') ==="
echo "twin protocol verbatim, NO regularizers; init seed 5155 SHARED by both"
echo "draws (weights + dropout stream identical), only the data order differs"
echo "draw 1 (5155a) perm seed 51551 -> models/R18-H155-initpair-draw1"
echo "draw 2 (5155b) perm seed 51552 -> models/R18-H155-initpair-draw2"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

run_draw 1 51551
run_draw 2 51552

echo ""
echo "=== H155 PAIR COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against the R18-H155 registration: PRIMARY = the init-paired"
echo "two-draw arena-mean spread vs the twin's 0.0243 (order share = spread /"
echo "0.0243); no accuracy bars; holds read for context only"
