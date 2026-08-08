#!/usr/bin/env bash
# R11-H118 licensed read chain on GPU0: soup(H105) blind -> soup(H108) gold_full
# -> soup(H108) blind. Idempotent: each stage skips if its result JSON exists.
#
# Launch detached:
#   nohup setsid bash experiments/grounding-semantic/R11-H118_reads.sh \
#       >> logs/R11-H118_reads.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic

stage() {
  local label="$1" outfile="$2"; shift 2
  if [ -s "$E/$outfile" ]; then
    echo "--- $label: already done ($outfile exists), skipping ---"
    return 0
  fi
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== H118 READS $(date '+%F %T') ==="

stage "soup(H105) blind windowed" "R11-H118_soup_h105_windowed_result.json" \
  uv run python "$E/R8-H101_windowed_read.py" --model models/R11-H118-soup-h105 \
  --out R11-H118_soup_h105_windowed_result.json
stage "soup(H108) gold_full" "R11-H118_soup_h108_goldfull.json" \
  uv run python "$E/R11-H118_goldfull_read.py" --model models/R11-H118-soup-h108 \
  --out R11-H118_soup_h108_goldfull.json
stage "soup(H108) blind windowed" "R11-H118_soup_h108_windowed_result.json" \
  uv run python "$E/R8-H101_windowed_read.py" --model models/R11-H118-soup-h108 \
  --out R11-H118_soup_h108_windowed_result.json

echo ""
echo "=== H118 READS COMPLETE $(date '+%F %T') ==="
