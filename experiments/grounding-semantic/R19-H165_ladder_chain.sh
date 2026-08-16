#!/usr/bin/env bash
# R19-H165 context-length ladder - gold held-out only, never the arena.
#
# L0 is the positive control and GATES the rest: it rebuilds the banked
# per-chunk presentation through this ladder's own code path and must reproduce
# the checkpoint's banked gold_full AUROC to <= 1e-3. A miss voids the ladder,
# so nothing else is spent.
#
# Placement: the small cells share GPU0 (24 GB) sequentially, L2 takes GPU2
# (32 GB), and the two long cells take GPU1 (96 GB) where the activation
# footprint at 4,096 and 8,192 tokens actually fits.
#
# Usage:
#   nohup setsid bash experiments/grounding-semantic/R19-H165_ladder_chain.sh \
#     >> logs/R19-H165_ladder.log 2>&1 &

set -u
cd /home/lab/workspace/private/ai-assistants/groundrails || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
S="$E/R19-H165_length_ladder.py"
CKPT=R18-H150-arm-draw1

run_cell() {
  local cell="$1" gpu="$2"
  echo ""
  echo "--- cell $cell on GPU$gpu  $(date '+%F %T') ---"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python "$S" --cell "$cell" --ckpt "$CKPT" \
    || echo "=== H165 CELL FAILED: $cell ==="
}

echo "=== R19-H165 length ladder armed $(date '+%F %T') - control cell first ==="

run_cell L0 0

CTRL="$E/R19-H165_ladder_L0_${CKPT}.json"
if ! grep -q '"pass": true' "$CTRL" 2>/dev/null; then
  echo "=== H165 LADDER VOID: L0 positive control did not reproduce the banked"
  echo "    gold_full AUROC within 1e-3. Nothing further is spent. ==="
  grep -o '"positive_control":.*' "$CTRL" 2>/dev/null | head -1
  exit 1
fi
echo "=== control PASS - fanning the remaining cells out $(date '+%F %T') ==="

# concatenation-alone control and the short length steps share the small card
( run_cell C0 0; run_cell L1 0 ) &
P0=$!
( run_cell L2 2 ) &
P2=$!
( run_cell L3 1; run_cell L4 1 ) &
P1=$!

wait $P0 $P2 $P1

echo ""
echo "=== H165 LADDER COMPLETE $(date '+%F %T') ==="
for f in "$E"/R19-H165_ladder_*_${CKPT}.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
p=d["presentation"]; g=d["gold_full"]; c=d["cost"]
print(f'  {d["cell"]:<3} concat={str(p["pool_concatenated"]):<5} '
      f'WIN={p["WIN"]:>6} MAX_LEN={p["MAX_LEN"]:>5}  '
      f'gold_full {g["auc"]:.4f}  '
      f'{c["windows_per_item"]:>6.2f} win/item  {c["n_pairs"]:>7} pairs  '
      f'{c["seconds"]:>6.0f}s  {c["peak_alloc_gb"]:>5.1f} GB')
PY
done
