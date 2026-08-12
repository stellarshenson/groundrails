#!/usr/bin/env bash
# R18-H152 REGULARIZED TWIN PAIR - the training-side front of the seed-variance
# attack: the twin protocol verbatim (clean 685,670-row mix via the H108
# public_train lineage, evidence UNTRUNCATED, 1,500/750 windowed presentation,
# MIL max-over-windows BCE, 12-group DANN, full trunk lr 1e-5 OneCycleLR 1
# epoch, adapter frozen at its zero init) PLUS the two registered regularizers:
#
#   EMA             decay 0.999 over trunk + task_head, updated after each
#                   optimizer step; the SERVED checkpoint is the EMA copy
#                   (raw weights live in resume.pt while the run is in flight)
#   WINDOW DROPOUT  training only: each non-argmax window of a multi-window
#                   bag dropped from the MIL max with p=0.1, at least one
#                   window always kept; serving reads all windows
#
# Registered in docs/experiments/semantic-grounding-experiments.md, block
# "R18-H152 REGULARIZED TWIN PAIR" (2026-08-12 ~17:10). Bars: two-seed
# arena-mean spread <= 0.010 with 2-draw mean >= 0.70996; per-subset 2-draw
# floors at flagship - 0.02; holds gold_full >= 0.84 AND non-EN >= 0.82 AND
# anti-gaming >= 0.7438 on BOTH draws; KILL on mean < 0.70, spread >= 0.020,
# or any hold breach.
#
# Draws, sequential (a fresh seed pair - the variance claim needs two draws):
#   draw 1  seed 3151  models/R18-H152-ema-draw1
#   draw 2  seed 3152  models/R18-H152-ema-draw2
#
# Stages per draw, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#   2  blind windowed decomposed-min arena read (PRIMARY)
#   3  anti-gaming stage, prefix R18-H152-d<N> - R14-H133_antigaming.py
#      resolves its checkpoint as models/<arm>-arm-draw<N>, which no --arm
#      prefix can make name models/R18-H152-ema-draw<N>, hence the symlink
#      (the R16-H142_T_draw2.sh pattern)
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer resumes from models/R18-H152-ema-draw<N>/resume.pt
# - which persists BOTH the raw and the EMA weights plus optimizer, scheduler,
# step, permutation fingerprint, torch RNG states and the drop counters -
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda")/.cuda() under WSL2 on this box
# (confirmed in R17-H144, R17-H145 and R16-H142).
#
# Launch detached, GPU by env (GPU1 only once the H150 campaign has released
# it; GPU0 and GPU2 are never touched):
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R18-H152_campaign.sh \
#       >> logs/R18-H152_campaign.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R18-H152_campaign.sh --census \
#       2>&1 | tee logs/R18-H152_census.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic

if [ "${1:-}" = "--census" ]; then
  echo "=== H152 regularized twin pair CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R18-H152_arm_run.py" --stage census
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

# run_draw <draw> <seed>
run_draw() {
  local draw="$1" seed="$2"
  local ckpt="models/R18-H152-ema-draw${draw}"
  local prefix="R18-H152-d${draw}"
  local link="models/${prefix}-arm-draw${draw}"

  stage_unless "$E/R18-H152_arm_draw${draw}_result.json" \
    "H152 draw ${draw} train + in-domain suite (seed ${seed}, EMA + window dropout)" \
    uv run python "$E/R18-H152_arm_run.py" --stage train --draw "$draw"

  stage_unless "$E/R18-H152_arm_draw${draw}_windowed_result.json" \
    "H152 draw ${draw} blind windowed arena read (PRIMARY)" \
    uv run python "$E/R18-H152_arm_run.py" --stage windowed --draw "$draw"

  ln -sfn "$(basename "$ckpt")" "$link"

  stage_unless "$E/${prefix}_antigaming_draw${draw}_result.json" \
    "H152 draw ${draw} anti-gaming stage (prefix ${prefix})" \
    uv run python "$E/R14-H133_antigaming.py" --draw "$draw" --arm "$prefix"
}

echo "=== H152 REGULARIZED TWIN PAIR CAMPAIGN $(date '+%F %T') ==="
echo "twin protocol verbatim + EMA 0.999 (served checkpoint is the EMA copy)"
echo "+ window dropout p=0.1 (non-argmax windows, training only)"
echo "draw 1 seed 3151 -> models/R18-H152-ema-draw1"
echo "draw 2 seed 3152 -> models/R18-H152-ema-draw2"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

run_draw 1 3151
run_draw 2 3152

echo ""
echo "=== H152 PAIR COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against the R18-H152 registration: |draw1 - draw2| arena mean"
echo "spread <= 0.010, 2-draw mean >= 0.70996, per-subset floors at flagship"
echo "- 0.02, holds (gold_full >= 0.84, non-EN >= 0.82, anti-gaming >= 0.7438)"
echo "on both draws"
