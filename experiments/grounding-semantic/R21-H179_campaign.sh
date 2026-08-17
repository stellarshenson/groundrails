#!/usr/bin/env bash
# R21-H179 STAGE 1 - blind-arena per-item scoring over the six flagship draws.
# See R21-H179_arena_scores.py for the design and the fidelity control.
#
# Three stages, each idempotent against on-disk artifacts:
#   items      CPU - R21-H179_arena_items.parquet (own process: R12 sets
#              CUDA_VISIBLE_DEVICES="" at import)
#   score      GPU - one npz per checkpoint + the fidelity record
#   consensus  CPU - consensus errors and rank-loss mass
#
# The score stage WAITS for GPU0 to be released by the R20-H174 draw-2 job
# before it binds a card. Three conditions, all required: the draw-2 campaign
# log shows DRAW 2 COMPLETE, no R20-H174 draw-2 process is alive, and GPU0's
# used memory has fallen below MEM_FLOOR. `nvidia-smi --query-compute-apps` is
# NOT usable here - under WSL2 it lists every compute process against every GPU,
# so it cannot attribute a process to a card. GPUs 1 and 2 carry the other two
# training draws and are never touched.
#
# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
# .to("cuda") under WSL2 on this box.
#
# Usage (detached):
#   GPU=0 nohup setsid bash experiments/grounding-semantic/R21-H179_campaign.sh \
#       2>&1 | tee logs/R21-H179_arena_scores.log &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

E=experiments/grounding-semantic
GPU="${GPU:-0}"
WAIT_LOG=logs/R20-H174_campaign_d2.log
WAIT_MARK="=== H174 DRAW 2 COMPLETE ==="
WAIT_PROC="R20-H174_arm_run.py.*--draw 2"
MEM_FLOOR=2000  # MiB; the draw-2 training job holds ~9.5 GB on GPU0

gpu_mem() {
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

echo "=== R21-H179 STAGE 1 - arena per-item scoring  $(date '+%F %T') ==="

echo ""
echo "--- items (CPU)  $(date '+%F %T') ---"
CUDA_VISIBLE_DEVICES="" uv run python "$E/R21-H179_arena_scores.py" --stage items || exit 1

echo ""
echo "--- waiting for GPU${GPU}  $(date '+%F %T') ---"
until grep -qF "$WAIT_MARK" "$WAIT_LOG" 2>/dev/null \
      && ! pgrep -f "$WAIT_PROC" >/dev/null \
      && [ "$(gpu_mem)" -lt "$MEM_FLOOR" ]; do
  sleep 60
done
echo "GPU${GPU} released  $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

echo ""
echo "--- score (GPU${GPU})  $(date '+%F %T') ---"
CUDA_VISIBLE_DEVICES="$GPU" uv run python "$E/R21-H179_arena_scores.py" --stage score || exit 1

echo ""
echo "--- consensus (CPU)  $(date '+%F %T') ---"
CUDA_VISIBLE_DEVICES="" uv run python "$E/R21-H179_arena_scores.py" --stage consensus || exit 1

echo ""
echo "=== R21-H179 STAGE 1 COMPLETE  $(date '+%F %T') ==="
