#!/usr/bin/env bash
# R18-H150 CONVERGENCE ARM draw 2 - amendment A3's confirming draw, seed 2150.
#
# Identical config to draw 1: clean public 685,670 + H146 misbind lane 30,000
# (group `quant_misbind`) + H150 unit_swap lane 5,540 (group `quant_scale_unit`)
# = 721,210 rows, 14 DANN groups; the twin recipe verbatim - evidence
# UNTRUNCATED, 1,500/750 windowed presentation, MIL max-over-windows BCE, full
# trunk at lr 1e-5 OneCycleLR 1 epoch, adapter frozen at its zero init. The one
# difference from draw 1 is the seed: 2150, against draw 1's 1150. Bars are
# identical to draw 1 (mean >= 0.7079, KILL < 0.6979, skills, holds, emanual
# clause) PLUS the author's official-record doctrine: both draws inside bars.
#
# Stages, in order (mirrors the draw-1 campaign):
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#   2  blind windowed decomposed-min arena read (PRIMARY)
#   3  anti-gaming held-out near-miss eval, untraced banked set (prefix R18-H150-d2)
#   4  probe bank - bind_col / bind_row / compare / scale-unit / P1 / triples
#
# The anti-gaming and probe scripts resolve their checkpoint as
# models/<arm>-arm-draw<N> and write <arm>_antigaming_set.parquet /
# <arm>_probe_scores.parquet with no draw in those names - running them at
# prefix R18-H150 would clobber draw 1's banked parquets. The -d2 prefix
# namespaces draw 2's eval artifacts (the R16-H142_T_draw2.sh pattern), and the
# trainer checkpoint models/R18-H150-arm-draw2 is linked to
# models/R18-H150-d2-arm-draw2 so both readers find it with no other change.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer continues from models/R18-H150-arm-draw2/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this
# campaign - `expandable_segments` is unusable under WSL2 on this box (it kills
# `.to("cuda")`; confirmed in R17-H144, R17-H145 and R16-H142).
#
# Launch detached, GPU by env:
#   GPU=1 nohup setsid bash experiments/grounding-semantic/R18-H150_campaign_d2.sh \
#       >> logs/R18-H150_campaign_d2.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R18-H150_campaign_d2.sh --census \
#       2>&1 | tee logs/R18-H150_census_d2.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW=2
ARM=R18-H150
AG_PREFIX=R18-H150-d2
CKPT=models/${ARM}-arm-draw${DRAW}
AG_LINK=models/${AG_PREFIX}-arm-draw${DRAW}

if [ "${1:-}" = "--census" ]; then
  echo "=== H150 arm draw 2 CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R18-H150_arm_draw2_run.py" --stage census
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

echo "=== H150 ARM DRAW 2 $(date '+%F %T') ==="
echo "amendment A3 confirming draw; mix: clean 685,670 + misbind 30,000 + unit_swap 5,540 = 721,210 rows, 14 DANN groups"
echo "seed 2150 (draw 1: 1150); twin windowed-MIL protocol verbatim, adapter frozen at zero"
echo "checkpoint $CKPT"
echo "bars identical to draw 1 (mean >= 0.7079, KILL < 0.6979, skills, holds) + both draws inside bars"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} train + in-domain suite" \
  uv run python "$E/R18-H150_arm_draw2_run.py" --stage train

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H150 arm draw${DRAW} blind windowed arena read (PRIMARY)" \
  uv run python "$E/R18-H150_arm_draw2_run.py" --stage windowed

# R14-H133_antigaming.py / R14-H133_probes.py resolve their checkpoint as
# models/<arm>-arm-draw<N>; the -d2 prefix keeps draw 2's set/scores parquets
# off draw 1's banked ones - hence the link (the R16-H142_T_draw2.sh pattern).
ln -sfn "$(basename $CKPT)" "$AG_LINK"

stage_unless "$E/${AG_PREFIX}_antigaming_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} ANTI-GAMING held-out near-miss eval, untraced banked set" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$DRAW" --arm "$AG_PREFIX"

stage_unless "$E/${AG_PREFIX}_probes_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} probe bank (bind_col / bind_row / compare / scale-unit)" \
  uv run python "$E/R14-H133_probes.py" --draw "$DRAW" --arm "$AG_PREFIX"

echo ""
echo "=== H150 ARM DRAW 2 COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against amendment A3: bars identical to draw 1 (mean >= 0.7079,"
echo "KILL < 0.6979, skills, holds, emanual clause) PLUS both draws inside bars;"
echo "on a 2-draw pass the misbind family + unit_swap lane + windowed-MIL protocol"
echo "graduate to the standing recipe and the H150 pair becomes the promotion candidate"
