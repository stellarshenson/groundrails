#!/usr/bin/env bash
# R19-H159 ENRICHED-MIX ARM draw 1 (amendment A1) - the flagship recipe carrying
# the five admitted R19 supply lanes as new DANN groups.
#
# Mix: flagship 721,210 (clean public 685,670 + misbind 30,000 + unit_swap
# 5,540) + FAVA 30,073 + AttributionBench 16,426 + MiniCheck 14,356 + PubHealth
# 12,251 + FinDVer 2,400 = 796,716 rows, 19 DANN groups, seed 1159. Protocol =
# the flagship recipe verbatim (evidence UNTRUNCATED, 1,500/750 windowed
# presentation, MIL max-over-windows BCE, full trunk lr 1e-5 OneCycleLR 1 epoch
# 10% warmup, clip 1.0, DANN lambda 0.02 Ganin ramp, 48 sets / 96 pairs per
# batch, adapter frozen at zero, NO EMA, NO window dropout).
#
# Amendment A1 (coordinator ruling 2026-08-14, after the first census fired the
# banked BATCH-CAP ABORT): FActScore withdrawn - its whole-biography evidence
# unit needs a lane build, not a presentation tweak; the five remaining lanes
# admit UNTRUNCATED per the twin protocol (the 1,500-char lane cap was priced
# and rejected - it trades train/serve presentation fidelity for arithmetic
# convenience); the 18 AttributionBench rows over the 96-pair cap are dropped at
# the trainer's own guard threshold and the count is recorded in both the census
# JSON and the result JSON.
#
# Stages, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#   2  blind windowed arena read (the BANKED reader, unchanged) - PRIMARY
#   3  anti-gaming held-out near-miss eval, untraced banked set (prefix
#      R19-H159 - the checkpoint name models/R19-H159-arm-draw1 resolves
#      directly under the <arm>-arm-draw<N> convention, no symlink)
#   4  probe bank - bind_col / bind_row / compare / scale-unit
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer resumes from models/R19-H159-arm-draw1/resume.pt,
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda")/.cuda() under WSL2 on this box
# (confirmed in R17-H144, R17-H145 and R16-H142).
#
# Launch detached, GPU by env (GPU1 per the placement call - 96 GB free, so the
# banked unsplit path is used; the split executor is not needed):
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R19-H159_campaign.sh \
#       >> logs/R19-H159_campaign.log 2>&1 &
#
# CPU dry run, no GPU touched (re-writes R19-H159_window_census.json):
#   bash experiments/grounding-semantic/R19-H159_campaign.sh --census \
#       2>&1 | tee logs/R19-H159_census.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW="${DRAW:-1}"
ARM=R19-H159
CKPT=models/${ARM}-arm-draw${DRAW}

if [ "${1:-}" = "--census" ]; then
  echo "=== H159 enriched-mix CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R19-H159_arm_run.py" --stage census --draw "$DRAW"
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

echo "=== H159 ENRICHED-MIX ARM CAMPAIGN draw${DRAW} $(date '+%F %T') ==="
echo "mix: flagship 721,210 + FAVA 30,073 + AttributionBench 16,426 + MiniCheck 14,356"
echo "     + PubHealth 12,251 + FinDVer 2,400 = 796,716 rows, 19 groups (A1: FActScore withdrawn)"
echo "seed $([ "$DRAW" = 1 ] && echo 1159 || echo 2159); flagship recipe verbatim (NO EMA, NO window dropout)"
echo "executor: banked unsplit path on the 96 GB card; checkpoint $CKPT"
echo "PRIMARY: blind windowed arena mean vs the flagship 0.71549"
echo "holds: gold_full >= 0.84, non-EN >= 0.82, anti-gaming >= 0.7438"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H159 draw ${DRAW} train + in-domain suite" \
  uv run python "$E/R19-H159_arm_run.py" --stage train --draw "$DRAW"

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H159 draw ${DRAW} blind windowed arena read (PRIMARY, banked reader)" \
  uv run python "$E/R19-H159_arm_run.py" --stage windowed --draw "$DRAW"

stage_unless "$E/${ARM}_antigaming_draw${DRAW}_result.json" \
  "H159 draw ${DRAW} ANTI-GAMING held-out near-miss eval, untraced banked set" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$DRAW" --arm "$ARM"

stage_unless "$E/${ARM}_probes_draw${DRAW}_result.json" \
  "H159 draw ${DRAW} probe bank (bind_col / bind_row / compare / scale-unit)" \
  uv run python "$E/R14-H133_probes.py" --draw "$DRAW" --arm "$ARM"

echo ""
echo "=== H159 DRAW ${DRAW} COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against the R19-H159 registration: PRIMARY 2-draw blind mean vs"
echo "flagship 0.71549 (GRADUATE >= 0.72049 with all ten variance-aware subset"
echo "floors and holds green both draws; KILL < 0.71049 or any hold breach);"
echo "registered watch cells finqa / pubmedqa / hagrid / gold_full"
