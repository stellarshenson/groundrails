#!/usr/bin/env bash
# R19-H160 soup chain.
#
# Waits for BOTH training draws to bank their completion markers, then launches
# the arm's --soup mode (soup B build, blind windowed arena read, gold_full,
# anti-gaming, probe bank, k-sweep soups).
#
# Detached and restart-safe by construction: it owns no state, it only polls two
# log files. Relaunching it after a container restart re-waits, and the --soup
# mode skips every stage whose artifact is already on disk, so a partially
# completed soup pass resumes rather than repeats.
#
# Launch:
#   nohup setsid bash experiments/grounding-semantic/R19-H160_soup_chain.sh \
#       >> logs/R19-H160_soup_chain.log 2>&1 &
set -u
cd /home/lab/workspace/private/ai-assistants/groundrails || exit 1

D3=logs/R19-H160_campaign_d3.log
D4=logs/R19-H160_campaign_d4.log

echo "=== H160 soup chain armed $(date '+%F %T') - waiting on draws 3 and 4 ==="

while true; do
  if grep -q "=== FAILED:" "$D3" 2>/dev/null || grep -q "=== FAILED:" "$D4" 2>/dev/null; then
    echo "=== H160 CHAIN ABORT: a draw reported FAILED $(date '+%F %T') ==="
    exit 1
  fi

  if grep -q "=== H160 DRAW 3 COMPLETE" "$D3" 2>/dev/null &&
    grep -q "=== H160 DRAW 4 COMPLETE" "$D4" 2>/dev/null; then
    echo "=== both draws complete $(date '+%F %T') - launching soup stage on GPU2 ==="
    GPU=2 bash experiments/grounding-semantic/R19-H160_campaign.sh --soup
    exit $?
  fi

  sleep 120
done
