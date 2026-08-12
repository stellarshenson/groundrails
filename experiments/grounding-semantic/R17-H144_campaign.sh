#!/usr/bin/env bash
# R17-H144 Stage B driver: wait out Job 1 (teacher traces, GPU1), then run Job 2
# (student SFT, GPU0) and Job 3 (eval on the banked H143 bar set, GPU0).
#
# Idempotent against on-disk state - every stage no-ops when its artifact exists,
# so a relaunch after a container restart is the same command:
#   nohup setsid experiments/grounding-semantic/R17-H144_campaign.sh \
#       >> logs/R17-H144_campaign.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

E=experiments/grounding-semantic
TRACES=$E/R17-H144_traces.parquet
RESULT=$E/R17-H144_result.json
N_ROWS=30000

echo "=== R17-H144 CAMPAIGN START $(date -Is) ==="

# --- Job 1: wait for the detached teacher trace generation ------------------ #
while true; do
    n=$(python -c "
import polars as pl, pathlib
p = pathlib.Path('$TRACES')
print(pl.read_parquet(p).height if p.exists() else 0)" 2>/dev/null || echo 0)
    [ "$n" -ge "$N_ROWS" ] && break
    pgrep -f R17-H144_tracegen.py >/dev/null || {
        echo "=== FAILED: tracegen not running and only $n/$N_ROWS rows on disk ==="
        exit 1
    }
    sleep 120
done
echo "=== JOB1 TRACES COMPLETE ($n rows) $(date -Is) ==="

# --- release GPU1 ----------------------------------------------------------- #
for _ in $(seq 30); do pgrep -f R17-H144_tracegen.py >/dev/null || break; sleep 10; done
pkill -f R17-H144_tracegen.py 2>/dev/null
sleep 15
# only reap an EngineCore that is still holding card 1 - other cards' jobs are
# not this driver's to kill
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
if [ "${used:-0}" -gt 1000 ]; then
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1); do
        ps -p "$p" -o args= | grep -q "R17-H144_tracegen\|VLLM::EngineCore" && kill -9 "$p"
    done
    sleep 10
fi
echo "[gpu] card1 after tracegen: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"

python - <<'PY'
import polars as pl
d = pl.read_parquet("experiments/grounding-semantic/R17-H144_traces.parquet")
r = float(d["accepted"].mean())
print(f"[gate] acceptance {r:.4f} over {d.height} rows", flush=True)
raise SystemExit(0 if r >= 0.50 else 1)
PY
[ $? -ne 0 ] && { echo "=== FAILED: acceptance below 0.50 - distillation corpus unusable ==="; exit 1; }

# --- Job 2: student SFT on GPU0 --------------------------------------------- #
if [ -d models/R17-H144-student/epoch1 ] && grep -q '"stopped"' $E/R17-H144_sft_stats.json 2>/dev/null \
   && python -c "
import json,sys
s=json.load(open('$E/R17-H144_sft_stats.json'))
sys.exit(0 if s.get('best_epoch') is not None else 1)" 2>/dev/null; then
    echo "=== JOB2 SFT already complete - skipping ==="
else
    echo "=== JOB2 SFT START $(date -Is) ==="
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python $E/R17-H144_sft.py >> logs/R17-H144_sft.log 2>&1
    rc=$?
    echo "=== JOB2 SFT EXIT rc=$rc $(date -Is) ==="
    [ $rc -ne 0 ] && { echo "=== FAILED: SFT rc=$rc ==="; exit 1; }
fi

# --- Job 3: eval on the banked bar set, GPU0 -------------------------------- #
if [ -f "$RESULT" ]; then
    echo "=== JOB3 EVAL already complete - skipping ==="
else
    echo "=== JOB3 EVAL START $(date -Is) ==="
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python $E/R17-H144_eval.py >> logs/R17-H144_eval.log 2>&1
    rc=$?
    echo "=== JOB3 EVAL EXIT rc=$rc $(date -Is) ==="
    [ $rc -ne 0 ] && { echo "=== FAILED: eval rc=$rc ==="; exit 1; }
fi

echo "=== R17-H144 CAMPAIGN COMPLETE $(date -Is) ==="
