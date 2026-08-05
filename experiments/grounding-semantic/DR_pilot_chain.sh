#!/usr/bin/env bash
# DR-2 pilot chain: DR-H116 splice-only sub-gate, then pilot-scale generation.
# Detached launch:
#   nohup setsid bash experiments/grounding-semantic/DR_pilot_chain.sh \
#         > logs/DR_pilot_gen.log 2>&1 &
set -euo pipefail

cd /home/lab/workspace/private/ai-assistants/groundrails
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== DR PILOT CHAIN START $(date -Is) ==="

uv run python experiments/grounding-semantic/DR_H116_subgate.py
echo "=== DR-H116 SUBGATE DONE ==="

uv run python experiments/grounding-semantic/DR_pilot_gen.py
echo "=== DR PILOT GEN DONE ==="
