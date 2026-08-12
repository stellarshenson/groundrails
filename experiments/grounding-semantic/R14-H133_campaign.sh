#!/usr/bin/env bash
# R14-H133 (A4) DERIVATION-PARITY LANE campaign: one arm draw plus every read
# the registration binds - in-domain suite, blind arena (truncated + windowed
# PRIMARY), the anti-gaming held-out near-miss eval, and the probe bank.
#
# Idempotent across container restarts: every stage is skipped when its artifact
# is already on disk, and the trainer itself continues from
# models/R14-H133-arm-draw<N>/resume.pt replaying the SAME persisted
# permutation. Relaunch = the same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R14-H133_campaign.sh <draw> \
#       >> logs/R14-H133_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
ckpt="models/R14-H133-arm-draw${draw}"

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

echo "=== H133 ARM CAMPAIGN draw${draw} $(date '+%F %T') ==="
echo "mix: clean public 685,670 + H133 lane 50,000 = 735,670 rows, 14 DANN groups"
echo "control: BANKED clean pair (R9-H105 draws), finqa pair 0.6333, draw-1 finqa 0.6489"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/R14-H133_arm_draw${draw}_result.json" \
  "H133 arm draw${draw} train + in-domain holds" \
  uv run python "$E/R14-H133_trainer.py" --draw "$draw"

stage_unless "$E/R14-H133_arm_draw${draw}_truncated_marker" \
  "H133 arm draw${draw} truncated read" \
  uv run python "$E/R8_decomposed_read.py" \
    --model "$ckpt" --tag "R14-H133-arm-draw${draw}"
touch "$E/R14-H133_arm_draw${draw}_truncated_marker"

stage_unless "$E/R14-H133_arm_draw${draw}_windowed_result.json" \
  "H133 arm draw${draw} windowed read (PRIMARY)" \
  uv run python "$E/R8-H101_windowed_read.py" \
    --model "$ckpt" --out "R14-H133_arm_draw${draw}_windowed_result.json"

stage_unless "$E/R14-H133_antigaming_draw${draw}_result.json" \
  "H133 arm draw${draw} ANTI-GAMING held-out near-miss eval (arm vs banked clean draw 1)" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$draw"

stage_unless "$E/R14-H133_probes_draw${draw}_result.json" \
  "H133 arm draw${draw} probe bank (report-only)" \
  uv run python "$E/R14-H133_probes.py" --draw "$draw"

echo ""
echo "=== H133 ARM D${draw} CAMPAIGN COMPLETE $(date '+%F %T') ==="
