#!/usr/bin/env bash
# R15 gate wave - the pre-arm measurements the R15 synthesis registered.
#
# Two chains, one per free card. Card 1 is carrying H127 training and is NEVER
# touched. Every step is idempotent: a step whose result JSON already exists is
# skipped, so a re-run after any death resumes where the wave stopped.
#
#   card 0 (24 GB, sm_120)  B1 -> B4 -> B6 -> B3
#   card 2 (32 GB, sm_89)   B2 -> arm8 build -> arm8 judge -> arm8 read -> B5
#
# Usage (detached, one per card):
#   nohup setsid bash experiments/grounding-semantic/R15_gate_wave.sh 0 &
#   nohup setsid bash experiments/grounding-semantic/R15_gate_wave.sh 2 &

set -u
CARD="${1:?usage: R15_gate_wave.sh <0|2>}"
if [ "$CARD" = "1" ]; then
  echo "REFUSED: card 1 is carrying H127 training" >&2
  exit 2
fi

ROOT=/home/lab/workspace/private/ai-assistants/groundrails
EXP="$ROOT/experiments/grounding-semantic"
LOG="$ROOT/logs/R15_gate_wave.log"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CARD"
export HF_HUB_OFFLINE=1
cd "$ROOT" || exit 1

step() {                       # step <id> <result-file> <runner> <script>
  local id="$1" out="$2" runner="$3" script="$4"
  if [ -s "$EXP/$out" ]; then
    echo "=== GATE $id SKIP (result exists) ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== GATE $id START card $CARD $(date -Is) ===" | tee -a "$LOG"
  if [ "$runner" = "vllm" ]; then
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 /home/lab/venvs/vllm/bin/python "$EXP/$script" 2>&1 | tee -a "$LOG"
  else
    uv run python "$EXP/$script" 2>&1 | tee -a "$LOG"
  fi
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "=== GATE $id DONE ===" | tee -a "$LOG"
  else
    echo "=== GATE $id FAILED rc=$rc ===" | tee -a "$LOG"
  fi
  return 0
}

case "$CARD" in
  0)
    step B1 R15_gate_B1_result.json uv R15_gate_B1.py
    step B4 R15_gate_B4_result.json uv R15_gate_B4.py
    step B6 R15_gate_B6_result.json uv R15_gate_B6.py
    step B3 R15_gate_B3_result.json uv R15_gate_B3.py
    ;;
  2)
    step B2          R15_gate_B2_result.json         uv   R15_gate_B2.py
    step B5arm8build R15_gate_B5arm8_candidates.parquet uv R15_gate_B5arm8_build.py
    step B5arm8judge R15_gate_B5arm8_judged.parquet  vllm R15_gate_B5arm8_judge.py
    step B5arm8      R15_gate_B5arm8_result.json     uv   R15_gate_B5arm8.py
    step B5          R15_gate_B5_result.json         uv   R15_gate_B5.py
    ;;
  *)
    echo "unknown card $CARD" >&2; exit 2 ;;
esac

echo "=== WAVE CHAIN card $CARD COMPLETE $(date -Is) ===" | tee -a "$LOG"
