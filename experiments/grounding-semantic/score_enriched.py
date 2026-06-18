"""Score the gold v3 cascade signals over the whole golden + synthetic augmentation.

Runs the OpenVINO int8 cascade (bge-m3 pre-filter -> bge-reranker + mDeBERTa-NLI) per
claim against its chunked source, producing the max-over-chunks signals the joint head
consumes. Output is a self-describing parquet keyed by `uid` (joins back to gold v3).

Source chunked with the same `chunk_text(1100/200)` the original cached scores used, so the
new scores are comparable to the v1 cache. Torch-free (OV int8 on CPU); the int8 IRs come
from the HF cache (`groundrails download`). Resumable - re-running skips uids already in the
output parquet and flushes every CHECKPOINT rows so a crash never loses more than that.

Run:  python experiments/grounding-semantic/score_enriched.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from grounding_models import chunk_text  # noqa: E402
import pandas as pd  # noqa: E402

from groundrails.semantic_ov import SemanticCascade, install_hint, is_available  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "processed" / "golden_v3.parquet"
AUG = ROOT / "data" / "processed" / "golden_v3_synth_aug.parquet"
OUT = ROOT / "data" / "processed" / "golden_v3_cascade_scores.parquet"

KEEP = ["uid", "role", "group_id", "lang_norm", "label", "claim", "source_text"]
COLS = ["uid", "role", "group_id", "lang_norm", "label", "cos_max", "rr_max",
        "nli_ent", "nli_contra", "ran_rr", "ran_nli"]
CHECKPOINT = int(os.environ.get("CHECKPOINT_EVERY", "500"))


def main() -> None:
    if not is_available():
        raise SystemExit(install_hint())
    df = pd.concat(
        [pd.read_parquet(GOLDEN)[KEEP], pd.read_parquet(AUG)[KEEP]], ignore_index=True
    )

    # Optional sharding: SHARD_COUNT workers split the undone rows by uid hash, each
    # writing its own SHARD_OUT; merge afterwards. No env -> single worker over OUT.
    shard_cnt = int(os.environ.get("SHARD_COUNT", "0"))
    shard_idx = int(os.environ.get("SHARD_INDEX", "0"))
    out_path = Path(os.environ.get("SHARD_OUT", OUT))

    done: set = set()
    if OUT.exists():  # never redo what the main checkpoint already holds
        done |= set(pd.read_parquet(OUT)["uid"])
    rows: list = []
    if out_path != OUT and out_path.exists():  # resume this shard from its own file
        prev = pd.read_parquet(out_path)
        rows = prev[COLS].values.tolist()
        done |= set(prev["uid"])
    elif out_path == OUT and OUT.exists():  # resume in place
        rows = pd.read_parquet(OUT)[COLS].values.tolist()
    print(f"resume: {len(done)} already scored", flush=True)

    todo = df[~df["uid"].isin(done)]
    if shard_cnt > 0:
        sel = todo["uid"].map(
            lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % shard_cnt == shard_idx
        )
        todo = todo[sel.to_numpy()]

    eng = SemanticCascade()
    n = len(todo)
    print(f"scoring {n} of {len(df)} pairs, shard {shard_idx}/{shard_cnt} "
          "(cold-start compiles the 3 int8 IRs)", flush=True)
    for i, r in enumerate(todo.itertuples(index=False)):
        s = eng.score(r.claim, chunk_text(r.source_text))
        rows.append([r.uid, r.role, r.group_id, r.lang_norm, int(r.label),
                     s.cos_max, s.rr_max, s.nli_ent, s.nli_contra, s.ran_rr, s.ran_nli])
        if (i + 1) % CHECKPOINT == 0:
            pd.DataFrame(rows, columns=COLS).to_parquet(out_path, index=False)
            print(f"  scored {i + 1}/{n} (checkpoint)", flush=True)

    out = pd.DataFrame(rows, columns=COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)", flush=True)
    print(f"  cross-encoder fire rate: ran_rr {out['ran_rr'].mean():.1%}  "
          f"ran_nli {out['ran_nli'].mean():.1%}", flush=True)
    print(f"  by role: {out.groupby('role').size().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
