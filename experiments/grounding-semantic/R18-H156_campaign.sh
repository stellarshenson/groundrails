#!/usr/bin/env bash
# R18-H156 LEARNED WINDOW-AGGREGATOR TWIN draw 1 - the H150 recipe carrying a
# learned window aggregator, trained through the cotangent split executor.
#
# Mix (H150 verbatim): clean public 685,670 + H146 misbind lane 30,000 (group
# `quant_misbind`) + H150 unit_swap lane 5,540 (group `quant_scale_unit`) =
# 721,210 rows, 14 DANN groups, seed 1156. Protocol = the twin recipe verbatim
# (evidence UNTRUNCATED, 1,500/750 windowed presentation, full trunk lr 1e-5
# OneCycleLR 1 epoch, DANN lambda 0.02 Ganin ramp, adapter frozen at zero, NO
# H152 regularizers) with the ONE registered difference: task loss
# BCE(s_agg, y), s_agg = alpha*max + (1-alpha)*softmax-weighted sum over the
# item's window logits, scorer Linear(768,64)->ReLU->Linear(64,1) zero-init
# output, alpha = sigmoid(beta) init ~0.95. The aggregator head lands in the
# agg_head.pt SIDECAR; every banked read path sees a byte-compatible
# checkpoint.
#
# Stages, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#      through the cotangent split executor (the registered geometry does not
#      fit the 32 GB card monolithically; equivalence proof
#      R18-H156_exec_equivalence.json gates this launch)
#   2  blind windowed arena read through the LEARNED AGGREGATOR (PRIMARY)
#   3  blind windowed arena read through the HARD MAX on the same checkpoint
#      (registered secondary - separates the serving-read effect from the
#      training-gradient effect), the banked reader unchanged
#   4  anti-gaming held-out near-miss eval, untraced banked set (prefix
#      R18-H156 - the checkpoint name models/R18-H156-arm-draw1 resolves
#      directly under the <arm>-arm-draw<N> convention, the H150 pattern, no
#      symlink)
#   5  probe bank - bind_col / bind_row / compare / scale-unit / P1 / triples
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer resumes from models/R18-H156-arm-draw1/resume.pt
# - which persists model + aggregator + optimizer + scheduler + step +
# permutation fingerprint + torch RNG states - replaying the SAME persisted
# permutation. Relaunch = the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda")/.cuda() under WSL2 on this box
# (confirmed in R17-H144, R17-H145 and R16-H142).
#
# Launch detached, GPU by env (GPU2 per the registration; the card carries an
# 832 MiB foreign stub belonging to another project - never touch it; the
# split executor's ~11 GB reserved budget leaves it room):
#   GPU=2 nohup setsid bash experiments/grounding-semantic/R18-H156_campaign.sh \
#       >> logs/R18-H156_campaign.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R18-H156_campaign.sh --census \
#       2>&1 | tee logs/R18-H156_census.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW=1
ARM=R18-H156
CKPT=models/${ARM}-arm-draw${DRAW}

if [ "${1:-}" = "--census" ]; then
  echo "=== H156 aggregator twin CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R18-H156_arm_run.py" --stage census
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

echo "=== H156 LEARNED WINDOW-AGGREGATOR TWIN CAMPAIGN draw${DRAW} $(date '+%F %T') ==="
echo "mix: clean 685,670 + misbind 30,000 + unit_swap 5,540 = 721,210 rows, 14 DANN groups"
echo "seed 1156; H150 recipe verbatim (NO EMA, NO window dropout) + aggregator s_agg BCE"
echo "executor: cotangent split (proof R18-H156_exec_equivalence.json); checkpoint $CKPT"
echo "PRIMARY: aggregator windowed arena mean; secondary: hard-max windowed read"
echo "holds: gold_full >= 0.84, non-EN >= 0.82, anti-gaming >= 0.7438"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H156 draw ${DRAW} train + in-domain suite (seed 1156, cotangent split executor)" \
  uv run python "$E/R18-H156_arm_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_agg_result.json" \
  "H156 draw ${DRAW} blind windowed arena read through the learned aggregator (PRIMARY)" \
  uv run python "$E/R18-H156_arm_run.py" --stage windowed_agg --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H156 draw ${DRAW} blind windowed arena read, hard max (registered secondary)" \
  uv run python "$E/R18-H156_arm_run.py" --stage windowed --draw "$DRAW"

stage_unless "$E/${ARM}_antigaming_draw${DRAW}_result.json" \
  "H156 draw ${DRAW} ANTI-GAMING held-out near-miss eval, untraced banked set" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$DRAW" --arm "$ARM"

stage_unless "$E/${ARM}_probes_draw${DRAW}_result.json" \
  "H156 draw ${DRAW} probe bank (bind_col / bind_row / compare / scale-unit)" \
  uv run python "$E/R14-H133_probes.py" --draw "$DRAW" --arm "$ARM"

echo ""
echo "=== H156 DRAW 1 COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against the R18-H156 registration: PRIMARY 2-draw blind mean vs"
echo "flagship 0.71549 (GRADUATE >= 0.72049 with all ten subset floors and holds"
echo "green both draws; KILL < 0.71049 or any hold breach); the hard-max read is"
echo "the registered secondary separating serving-read from training-gradient"
