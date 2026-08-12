#!/usr/bin/env bash
# R16-H142 G1 (amendment A1) ABLATION PAIR campaign: the twin (no adapter) and
# the arm (adapter), one shared presentation, plus every read the registration
# binds - in-domain suite, blind truncated arena read, blind windowed arena read
# (PRIMARY) - and the pair delta.
#
# The TWIN runs FIRST. It is the control: if the windowed/MIL presentation
# itself collapses, that is learned before the arm spends a second card-hour.
#
# Idempotent across container restarts: every stage is skipped when its artifact
# is already on disk, and each trainer continues from its own
# models/R16-H142-G1-<run>/resume.pt replaying the SAME persisted permutation.
# Relaunch = the same command.
#
# Launch detached (GPU1):
#   nohup setsid bash experiments/grounding-semantic/R16-H142_G1_campaign.sh \
#       >> logs/R16-H142_G1_campaign.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic

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

# run_all <twin|arm> - train, in-domain suite, then both blind arena reads
run_all() {
  local run="$1"
  stage_unless "$E/R16-H142_G1_${run}_result.json" \
    "H142 G1 ${run} train + in-domain suite" \
    uv run python "$E/R16-H142_G1_arm.py" --run "$run"

  stage_unless "$E/R16-H142_G1_${run}_truncated_result.json" \
    "H142 G1 ${run} blind truncated arena read" \
    uv run python "$E/R16-H142_G1_reads.py" --run "$run" --mode truncated

  stage_unless "$E/R16-H142_G1_${run}_windowed_result.json" \
    "H142 G1 ${run} blind windowed arena read (PRIMARY)" \
    uv run python "$E/R16-H142_G1_reads.py" --run "$run" --mode windowed
}

echo "=== H142 G1 PAIR CAMPAIGN $(date '+%F %T') ==="
echo "amendment A1: the adapter carries its own ablation (review constraint F3)."
echo "shared presentation - 685,670-row clean mix, 12 DANN groups, evidence"
echo "  UNTRUNCATED and windowed 1500/750, MIL max-BCE over the window set,"
echo "  seed 1142, full trunk at lr 1e-5"
echo "TWIN = no adapter (control, runs first).  ARM = + zero-init adapter at lr 1e-3."
echo "arm minus twin attributes to the adapter and to nothing else."
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
df -h /dev/shm | tail -1

run_all twin
echo ""
echo "=== H142 G1 TWIN COMPLETE $(date '+%F %T') ==="

run_all arm

stage_unless "$E/R16-H142_G1_pair_result.json" \
  "H142 G1 pair delta (arm minus twin, flags only)" \
  uv run python "$E/R16-H142_G1_pair.py"

echo ""
echo "=== H142 G1 PAIR COMPLETE $(date '+%F %T') ==="
