#!/usr/bin/env bash
# R18-H150 CONVERGENCE ARM draw 1 - the twin windowed-MIL protocol carrying both
# lanes, plus every read the registration binds.
#
# Mix: clean public 685,670 + H146 misbind lane 30,000 (group `quant_misbind`)
# + H150 unit_swap lane 5,540 (group `quant_scale_unit`) = 721,210 rows,
# 14 DANN groups, seed 1150. Protocol = the twin recipe verbatim: evidence
# UNTRUNCATED, 1,500/750 windowed presentation, MIL max-over-windows BCE, full
# trunk at lr 1e-5 OneCycleLR 1 epoch, adapter frozen at its zero init.
#
# Stages, in order:
#   1  train + in-domain suite (gold, gold_full, RAGTruth EN + 7 translations)
#   2  blind windowed decomposed-min arena read (PRIMARY)
#   3  anti-gaming held-out near-miss eval, untraced banked set (prefix R18-H150)
#   4  probe bank - bind_col / bind_row / compare / scale-unit / P1 / triples
#
# The checkpoint is named models/R18-H150-arm-draw1 precisely so that
# R14-H133_antigaming.py and R14-H133_probes.py, which resolve their checkpoint
# as models/<arm>-arm-draw<N>, find it with no indirection. The symlink
# workaround R16-H142_T_draw2.sh needs is therefore NOT required here.
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer continues from models/R18-H150-arm-draw1/resume.pt
# replaying the SAME persisted permutation. Relaunch = the same command.
#
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this
# campaign - `expandable_segments` is unusable under WSL2 on this box (it kills
# `.to("cuda")`; confirmed in R17-H144, R17-H145 and R16-H142).
#
# NOT LAUNCHED BY THE PREP PASS - the launch decision is the coordinator's, and
# gate (a) requires the twin draw-2 verdict first.
#
# Launch detached, GPU by env (never GPU1 while the twin draw 2 trains there):
#   GPU=0 nohup setsid bash experiments/grounding-semantic/R18-H150_campaign.sh \
#       >> logs/R18-H150_campaign_d1.log 2>&1 &
#
# CPU dry run, no GPU touched:
#   bash experiments/grounding-semantic/R18-H150_campaign.sh --census \
#       2>&1 | tee logs/R18-H150_census.log

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
DRAW=1
ARM=R18-H150
CKPT=models/${ARM}-arm-draw${DRAW}

if [ "${1:-}" = "--census" ]; then
  echo "=== H150 arm CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/R18-H150_arm_run.py" --stage census
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

echo "=== H150 ARM CAMPAIGN draw${DRAW} $(date '+%F %T') ==="
echo "mix: clean 685,670 + misbind 30,000 + unit_swap 5,540 = 721,210 rows, 14 DANN groups"
echo "seed 1150; twin windowed-MIL protocol verbatim, adapter frozen at zero"
echo "checkpoint $CKPT"
echo "PRIMARY: blind windowed arena mean vs the twin adjudicated mean"
echo "co-primaries: bind_col >= 0.80 AND bind_row >= 0.95 (probe bank)"
echo "scale_unit: REPORTED SECONDARY, no bar (amendment A1)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} train + in-domain suite" \
  uv run python "$E/R18-H150_arm_run.py" --stage train

stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
  "H150 arm draw${DRAW} blind windowed arena read (PRIMARY)" \
  uv run python "$E/R18-H150_arm_run.py" --stage windowed

stage_unless "$E/${ARM}_antigaming_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} ANTI-GAMING held-out near-miss eval, untraced banked set" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$DRAW" --arm "$ARM"

stage_unless "$E/${ARM}_probes_draw${DRAW}_result.json" \
  "H150 arm draw${DRAW} probe bank (bind_col / bind_row / compare / scale-unit)" \
  uv run python "$E/R14-H133_probes.py" --draw "$DRAW" --arm "$ARM"

echo ""
echo "=== H150 ARM CAMPAIGN COMPLETE ==="
echo "finished $(date '+%F %T')"
echo "adjudicate against the R18-H150 registration as amended (A1/A2) and the"
echo "author's pre-registered attribution labels: DAMAGE-ABSORBED / MIXED /"
echo "DAMAGE-PERSISTS on hotpotqa, emanual, expertqa vs the twin adjudicated subset"
