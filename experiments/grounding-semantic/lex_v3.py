"""Lexical manifold (effort=high) pass over gold v3 - lex_p per claim, keyed by uid.

The cross-lingual hypotheses fuse the lexical verdict with the cascade signals. gold v3's
eval rows are not in the old v1 lexical cache (v2 shares only ~1.5k claims), so the
lexical P is computed fresh here over the whole golden + synthetic augmentation. Mirrors
`joint_wirings.lexical_pass` (same chunking, same blocked/fired/contra semantics) but over
gold v3 and keyed by `uid` so it joins the cascade scores.

Cross-lingual claims with no installed argos model raise UnsupportedLanguageError and are
recorded as blocked (lex_p=0, lex_blocked=1) - the tail the cascade exists to catch.

Torch-free. Resumable - skips uids already in the output and checkpoints every CHECKPOINT
rows. Output: data/processed/golden_v3_lex.parquet.

Run:  python experiments/grounding-semantic/lex_v3.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from grounding_models import chunk_text  # noqa: E402
import pandas as pd  # noqa: E402

from groundrails.grounding import UnsupportedLanguageError, ground  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "processed" / "golden_v3.parquet"
AUG = ROOT / "data" / "processed" / "golden_v3_synth_aug.parquet"
OUT = ROOT / "data" / "processed" / "golden_v3_lex.parquet"

KEEP = ["uid", "role", "group_id", "lang_norm", "label", "claim", "source_text"]
COLS = ["uid", "role", "group_id", "lang_norm", "label", "lex_p", "lex_blocked",
        "lex_fired", "lex_contra"]
CHECKPOINT = 500


def main() -> None:
    df = pd.concat(
        [pd.read_parquet(GOLDEN)[KEEP], pd.read_parquet(AUG)[KEEP]], ignore_index=True
    )
    rows, done = [], set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev[COLS].values.tolist()
        done = set(prev["uid"])
        print(f"resume: {len(done)} already passed", flush=True)

    todo = df[~df["uid"].isin(done)]
    n = len(todo)
    print(f"lexical pass over {n} of {len(df)} claims (effort=high)", flush=True)
    for i, r in enumerate(todo.itertuples(index=False)):
        srcs = [(f"s{j}", c) for j, c in enumerate(chunk_text(r.source_text))]
        try:
            m = ground(r.claim, srcs)
            lex_p = float(m.verdict_probability)
            blocked = False
            fired = m.match_type in ("exact", "fuzzy", "bm25")
            contra = bool(m.numeric_mismatches or m.entity_mismatches)
        except UnsupportedLanguageError:
            lex_p, blocked, fired, contra = 0.0, True, False, False
        rows.append([r.uid, r.role, r.group_id, r.lang_norm, int(r.label),
                     lex_p, blocked, fired, contra])
        if (i + 1) % CHECKPOINT == 0:
            pd.DataFrame(rows, columns=COLS).to_parquet(OUT, index=False)
            print(f"  passed {i + 1}/{n} (checkpoint)", flush=True)

    out = pd.DataFrame(rows, columns=COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out)} rows)", flush=True)
    print(f"  blocked (cross-lingual, no argos): {int(out['lex_blocked'].sum())}", flush=True)
    print(f"  lexical fired: {out['lex_fired'].mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
