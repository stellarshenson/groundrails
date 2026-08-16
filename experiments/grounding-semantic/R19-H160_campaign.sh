#!/usr/bin/env bash
# R19-H160 SEED-DIVERSE WEIGHT AVERAGING - the two fresh flagship draws and the
# unseen confirmatory soup.
#
# Registered in docs/experiments/semantic-grounding-experiments.md, block
# "R19-H160 SEED-DIVERSE WEIGHT AVERAGING AS THE SERVED ARTIFACT". The arm tests
# a PROTOCOL amendment - redefine the served artifact from "one draw" to the
# uniform elementwise average of the k draws the 2-draw doctrine already
# requires - at zero marginal training cost.
#
# Recipe: the R18-H150 flagship VERBATIM (clean public 685,670 + misbind 30,000
# + unit_swap 5,540 = 721,210 rows, 14 DANN groups; evidence UNTRUNCATED,
# 1,500/750 windowed presentation, MIL max-over-windows BCE, full trunk at
# lr 1e-5 OneCycleLR 1 epoch, adapter frozen at zero; NO EMA, NO window
# dropout). Seeds 3150 (draw 3) and 4150 (draw 4) - the seed governs task-head
# init, data order and the dropout stream jointly.
#
# EXECUTOR: the cotangent split executor R19-H160_split_exec.py. The registered
# batch geometry does not fit either free card monolithically (banked H152 vram
# probe: 36.96 GB allocated / 56.56 GB reserved against GPU0's 24 GB and GPU2's
# 32 GB); `--probe` re-measures it on the card in use, and `--equiv` proves the
# split executor step-equivalent to the monolithic reference before any draw.
#
# GPU PLACEMENT: GPU0 (RTX PRO 4000, 24 GB) and GPU2 (RTX 5000 Ada, 32 GB) ONLY.
# GPU1 is reserved exclusively for the R19-H159 training - a prior arm's reads
# sharing GPU1 slowed H159 from 0.915 to 4.84 s/step. Every H160 entry point
# aborts if CUDA_VISIBLE_DEVICES is unset or equals 1.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this campaign:
# expandable_segments kills .to("cuda")/.cuda() under WSL2 on this box
# (confirmed in R17-H144, R17-H145 and R16-H142).
#
# Idempotent across container restarts: each stage is skipped when its artifact
# is on disk, and the trainer resumes from models/R19-H160-arm-draw<N>/resume.pt
# replaying the SAME persisted permutation with the torch RNG states restored.
# Relaunch = the same command.
#
# Usage:
#   CPU census, no GPU touched:
#     DRAW=3 bash experiments/grounding-semantic/R19-H160_campaign.sh --census
#   monolithic vram probe on the target card (the placement measurement):
#     GPU=2 DRAW=3 bash experiments/grounding-semantic/R19-H160_campaign.sh --probe
#   split-vs-monolithic step equivalence proof (gates the draws):
#     GPU=2 bash experiments/grounding-semantic/R19-H160_campaign.sh --equiv
#   a draw, detached:
#     GPU=0 DRAW=3 nohup setsid bash experiments/grounding-semantic/R19-H160_campaign.sh \
#         >> logs/R19-H160_campaign_d3.log 2>&1 &
#     GPU=2 DRAW=4 nohup setsid bash experiments/grounding-semantic/R19-H160_campaign.sh \
#         >> logs/R19-H160_campaign_d4.log 2>&1 &
#   the soups, their reads, the anti-gaming eval and the probe bank:
#     GPU=2 nohup setsid bash experiments/grounding-semantic/R19-H160_campaign.sh --soup \
#         >> logs/R19-H160_soup.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
ARM=R19-H160
MODE="${1:-draw}"

if [ "$MODE" != "--census" ]; then
  # GPU1's reservation LIFTED 2026-08-14 16:38: R19-H159 was KILLED at draw 1
  # (blind 0.68941 vs the 0.71049 kill bar) and released the card. GPU1 is now a
  # legal placement for this arm. The unset check stays - the banked trainer
  # defaults CUDA_VISIBLE_DEVICES to "1" at import, so an unset variable must
  # never resolve silently.
  export CUDA_VISIBLE_DEVICES="${GPU:?set GPU explicitly (0, 1 or 2)}"
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

case "$MODE" in
--census)
  DRAW="${DRAW:?set DRAW (3 or 4)}"
  echo "=== H160 draw ${DRAW} CPU census (dry run, no GPU) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="" uv run python "$E/${ARM}_arm_run.py" --stage census --draw "$DRAW"
  exit $?
  ;;

--probe)
  DRAW="${DRAW:-3}"
  echo "=== H160 MONOLITHIC VRAM PROBE draw ${DRAW} $(date '+%F %T') ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  uv run python "$E/${ARM}_arm_run.py" --stage vram_probe --draw "$DRAW"
  exit $?
  ;;

--equiv)
  echo "=== H160 SPLIT-vs-MONOLITHIC EQUIVALENCE PROOF $(date '+%F %T') ==="
  for m in reference refnoise split report; do
    stage "equivalence mode $m" uv run python "$E/${ARM}_exec_equivalence.py" --mode "$m"
  done
  echo "=== H160 EQUIVALENCE PROOF COMPLETE $(date '+%F %T') ==="
  exit 0
  ;;

--soup)
  echo "=== H160 SOUPS + READS $(date '+%F %T') ==="
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  stage "H160 soup build + blind windowed arena reads + gold_full" \
    uv run python "$E/${ARM}_soup.py" --stage run

  # R14-H133_antigaming.py / R14-H133_probes.py resolve their checkpoint as
  # models/<arm>-arm-draw<N> and write <arm>_*.parquet with no draw in those
  # names; the -soupB prefix namespaces the soup's eval artifacts and the link
  # makes the soup checkpoint resolvable (the R16-H142_T_draw2.sh pattern).
  ln -sfn "${ARM}-soup-B" "models/${ARM}-soupB-arm-draw1"

  stage_unless "$E/${ARM}-soupB_antigaming_draw1_result.json" \
    "H160 soup B ANTI-GAMING held-out near-miss eval, untraced banked set" \
    uv run python "$E/R14-H133_antigaming.py" --draw 1 --arm "${ARM}-soupB"

  stage_unless "$E/${ARM}-soupB_probes_draw1_result.json" \
    "H160 soup B probe bank (bind_col / bind_row / compare / scale-unit)" \
    uv run python "$E/R14-H133_probes.py" --draw 1 --arm "${ARM}-soupB"

  stage "H160 bars report (floors, holds, k-sweep)" \
    uv run python "$E/${ARM}_report.py"
  echo ""
  echo "=== H160 SOUPS COMPLETE $(date '+%F %T') ==="
  exit 0
  ;;

draw)
  DRAW="${DRAW:?set DRAW (3 or 4)}"
  CKPT=models/${ARM}-arm-draw${DRAW}
  echo "=== H160 DRAW ${DRAW} $(date '+%F %T') ==="
  echo "recipe: R18-H150 flagship VERBATIM - clean 685,670 + misbind 30,000 + unit_swap 5,540 = 721,210 rows, 14 DANN groups"
  echo "seed: draw 3 -> 3150, draw 4 -> 4150; adapter frozen at zero; no EMA, no window dropout"
  echo "executor: R19-H160_split_exec.py (cotangent window-chunked, pass A 32 / pass B 8)"
  echo "checkpoint $CKPT"
  echo "draw bar: blind windowed decomposed-min mean >= 0.7079 (the H150 registered draw bar)"
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  df -h /dev/shm | tail -1

  stage_unless "$E/${ARM}_arm_draw${DRAW}_result.json" \
    "H160 draw${DRAW} train + in-domain suite" \
    uv run python "$E/${ARM}_arm_run.py" --stage train --draw "$DRAW"

  stage_unless "$E/${ARM}_arm_draw${DRAW}_windowed_result.json" \
    "H160 draw${DRAW} blind windowed arena read" \
    uv run python "$E/${ARM}_arm_run.py" --stage windowed --draw "$DRAW"

  echo ""
  echo "=== H160 DRAW ${DRAW} COMPLETE $(date '+%F %T') ==="
  echo "finished $(date '+%F %T')"
  exit 0
  ;;

*)
  echo "unknown mode: $MODE (expected --census | --probe | --equiv | --soup | no argument)"
  exit 1
  ;;
esac
