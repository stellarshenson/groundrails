#!/usr/bin/env bash
# R17-H146 MISBIND-VERIFICATION LANE campaign: one arm draw plus every read
# the registration binds - in-domain suite (inside the trainer), blind arena
# (truncated + windowed), the anti-gaming held-out near-miss eval on the
# UNTRACED banked set, and the probe bank.
#
# Idempotent across container restarts: every stage is skipped when its artifact
# is already on disk, and the trainer itself continues from
# models/R17-H146-arm-draw<N>/resume.pt replaying the SAME persisted
# permutation. Relaunch = the same command.
#
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set anywhere in this
# campaign - `expandable_segments` is unusable under WSL2 on this box (it kills
# `.to("cuda")`; double-confirmed in R17-H144 and R17-H145).
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R17-H146_campaign.sh <draw> \
#       >> logs/R17-H146_campaign_d<draw>.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

draw="${1:?draw (1|2)}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
ckpt="models/R17-H146-arm-draw${draw}"

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

echo "=== H146 ARM CAMPAIGN draw${draw} $(date '+%F %T') ==="
echo "mix: clean public 685,670 + misbind lane 30,000 = 715,670 rows, 13 DANN groups"
echo "seed 1146; control: BANKED clean pair (R9-H105 draws)"
echo "primary: bind_col >= 0.80 AND bind_row >= 0.95 (probe bank)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

stage_unless "$E/R17-H146_arm_draw${draw}_result.json" \
  "H146 arm draw${draw} train + in-domain holds" \
  uv run python "$E/R17-H146_trainer.py" --draw "$draw"

stage_unless "$E/R17-H146_arm_draw${draw}_truncated_marker" \
  "H146 arm draw${draw} truncated read" \
  uv run python "$E/R8_decomposed_read.py" \
    --model "$ckpt" --tag "R17-H146-arm-draw${draw}"
touch "$E/R17-H146_arm_draw${draw}_truncated_marker"

stage_unless "$E/R17-H146_arm_draw${draw}_windowed_result.json" \
  "H146 arm draw${draw} windowed arena read" \
  uv run python "$E/R8-H101_windowed_read.py" \
    --model "$ckpt" --out "R17-H146_arm_draw${draw}_windowed_result.json"

stage_unless "$E/R17-H146_antigaming_draw${draw}_result.json" \
  "H146 arm draw${draw} ANTI-GAMING held-out near-miss eval, untraced banked set (arm vs banked clean draw 1)" \
  uv run python "$E/R14-H133_antigaming.py" --draw "$draw" --arm R17-H146

stage_unless "$E/R17-H146_probes_draw${draw}_result.json" \
  "H146 arm draw${draw} probe bank" \
  uv run python "$E/R14-H133_probes.py" --draw "$draw" --arm R17-H146

echo ""
echo "=== H146 ARM CAMPAIGN COMPLETE $(date '+%F %T') ==="
