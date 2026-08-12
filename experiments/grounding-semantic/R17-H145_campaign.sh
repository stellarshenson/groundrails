#!/usr/bin/env bash
# R17-H145 RELATIONAL + SCALE/UNIT LANE campaign: one arm draw plus every read
# the registration binds - in-domain suite (inside the trainer), blind arena
# (truncated + windowed), the anti-gaming held-out near-miss eval on the
# UNTRACED banked set, and the probe bank.
#
# Idempotent across container restarts: every stage is skipped when its artifact
# is already on disk, and the trainer itself continues from
# models/R17-H145-arm-draw<N>/resume.pt replaying the SAME persisted
# permutation. Relaunch = the same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R17-H145_campaign.sh <draw> \
#       >> logs/R17-H145_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
ckpt="models/R17-H145-arm-draw${draw}"

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

echo "=== H145 ARM CAMPAIGN draw${draw} $(date '+%F %T') ==="
echo "mix: clean public 685,670 + relational 7,500 + scale/unit 5,672 = 698,842 rows, 14 DANN groups"
echo "seed 1145; control: BANKED clean pair (R9-H105 draws)"
echo "co-primaries: bind_col >= 0.80, bind_row >= 0.95, scale/unit >= 0.92 (probe bank)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/R17-H145_arm_draw${draw}_result.json" \
  "H145 arm draw${draw} train + in-domain holds" \
  uv run python "$E/R17-H145_trainer.py" --draw "$draw"

stage_unless "$E/R17-H145_arm_draw${draw}_truncated_marker" \
  "H145 arm draw${draw} truncated read" \
  uv run python "$E/R8_decomposed_read.py" \
    --model "$ckpt" --tag "R17-H145-arm-draw${draw}"
touch "$E/R17-H145_arm_draw${draw}_truncated_marker"

stage_unless "$E/R17-H145_arm_draw${draw}_windowed_result.json" \
  "H145 arm draw${draw} windowed arena read" \
  uv run python "$E/R8-H101_windowed_read.py" \
    --model "$ckpt" --out "R17-H145_arm_draw${draw}_windowed_result.json"

stage_unless "$E/R17-H145_antigaming_draw${draw}_result.json" \
  "H145 arm draw${draw} ANTI-GAMING held-out near-miss eval, untraced banked set (arm vs banked clean draw 1)" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$draw" --arm R17-H145

stage_unless "$E/R17-H145_probes_draw${draw}_result.json" \
  "H145 arm draw${draw} probe bank (report-only)" \
  uv run python "$E/R14-H133_probes.py" --draw "$draw" --arm R17-H145

echo ""
echo "=== H145 ARM CAMPAIGN COMPLETE $(date '+%F %T') ==="
