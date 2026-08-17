#!/usr/bin/env bash
# R21-H179 / R21-H179Q - ONE CARD LANE of the arena scoring pass.
#
# The pass answers three questions and needs 12 GPU arena reads: the six banked
# flagship draws under TRUE evidence (Q1 contamination re-read + Q3 autopsy) and
# the same six under ABLATED evidence (Q2 evidence-use). This script is one
# lane: it binds ONE card and works a DISJOINT list of draws, so two lanes can
# run on two cards without either scoring a checkpoint the other scored.
#
# Disjointness of ARTIFACTS, not just of work:
#   score      R21-H179_arena_scores_<ckpt>.npz          per checkpoint
#              R21-H179_arena_scores_fidelity_<tag>.json per checkpoint SHARD
#              (the canonical fidelity JSON is the merge, written once by the
#              consensus stage - no two processes write it)
#   ablate     R21-H179Q_ablated_scores_<ckpt>.npz       per checkpoint
#              R21-H179Q_ablated_<tag>.json              per checkpoint
# The CPU stages (consensus, Q1, Q2 roll-up, Q3 exposure column) are run ONCE,
# after both lanes are done, and are not part of a lane.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
# .to("cuda") under WSL2 on this box.
#
# Usage (detached):
#   GPU=2 LANE=A DRAWS="d1 d2 d3" nohup setsid \
#     bash experiments/grounding-semantic/R21-H179_lane.sh \
#     2>&1 | tee logs/R21-H179_laneA_gpu2.log &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
GPU="${GPU:?set GPU}"
LANE="${LANE:-?}"
DRAWS="${DRAWS:?set DRAWS, e.g. \"d1 d2 d3\"}"
MEM_FLOOR=2000  # MiB - a training draw on this box holds ~9.5 GB
ABLATE="$E/R21-H179Q_probes.py"

gpu_mem() {
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

echo "=== R21-H179 LANE $LANE  GPU$GPU  draws: $DRAWS  $(date '+%F %T') ==="

# The card must be genuinely idle. `nvidia-smi --query-compute-apps` is NOT
# usable under WSL2 - it lists every compute process against every GPU and
# cannot attribute a process to a card - so used memory is the instrument.
echo ""
echo "--- waiting for GPU${GPU} to be idle (< ${MEM_FLOOR} MiB)  $(date '+%F %T') ---"
until [ "$(gpu_mem)" -lt "$MEM_FLOOR" ]; do
  sleep 60
done
echo "GPU${GPU} idle at $(gpu_mem) MiB  $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

for d in $DRAWS; do
  echo ""
  echo "--- TRUE evidence score, draw $d, GPU${GPU}  $(date '+%F %T') ---"
  CUDA_VISIBLE_DEVICES="$GPU" uv run python "$E/R21-H179_arena_scores.py" \
      --stage score --draw "$d" || exit 1
done

echo ""
echo "--- lane $LANE true-evidence reads DONE  $(date '+%F %T') ---"

# The ablation arm is written while the true reads run; wait for it rather than
# dying, so the lane never has to be relaunched by hand.
echo "--- waiting for $ABLATE  $(date '+%F %T') ---"
until [ -s "$ABLATE" ]; do
  sleep 60
done

for d in $DRAWS; do
  echo ""
  echo "--- ABLATED evidence score, draw $d, GPU${GPU}  $(date '+%F %T') ---"
  CUDA_VISIBLE_DEVICES="$GPU" uv run python "$ABLATE" \
      --stage ablate --draw "$d" || exit 1
done

echo ""
echo "=== R21-H179 LANE $LANE COMPLETE  $(date '+%F %T') ==="
