#!/bin/bash
# R19 supply wave - fetch -> gate -> lane for the six new corpora.
# CPU + network only; NEVER touches the GPU.  Idempotent: completed corpora
# checkpoint to disk (archive / tree _counts.json / gate JSON / lane parquet)
# and re-runs no-op over them.  Resume after any interruption by re-running
# this exact script.  Registered: docs/experiments/semantic-dataset-enhancements.md
# section "R19 supply wave".
set -uo pipefail
cd "$(dirname "$0")/../.."

uv run --with gdown python scripts/fetch_grounding_datasets.py \
    fava pubhealth minicheck factscore findver attributionbench
uv run python experiments/grounding-semantic/R19_supply_gates.py
uv run python experiments/grounding-semantic/R19_supply_lanes.py
