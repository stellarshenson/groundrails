"""Build gold v4 = gold v3 + the VitaminC contrastive dev set, marked clearly.

gold v3 is the cross-lingual golden (verified eval + synthetic-MT augmentation). v4 folds
in the 800-row VitaminC dev sample (HF `tals/vitaminc`, SUPPORTS/REFUTES balanced, already
cached locally in `grounding_combined.parquet`) so the joint grounder can be scored on a
contrastive single-token-edit corpus - the regime where the lexical tier collapses and the
cascade's NLI-contradiction signal should earn its keep.

The VitaminC rows are marked unmistakably in BOTH the dataset-type and origin columns:
  role   = "eval_vitaminc"   (distinct from gold v3's "eval" so it never silently pools
                              into the cross-lingual eval macro)
  origin = "vitaminc"

Schema matches gold v3 eval exactly (11 columns), so v4 is a drop-in for the lex/cascade
producers. group_id is recomputed as sha1(source_text)[:12] - VitaminC's native group_id is
a constant - so GroupKFold leave-one-source-out groups contrastive pairs by their shared
source. label polarity already aligns: VitaminC SUPPORTS=1 / REFUTES=0 == gold supported=1 /
hallucination=0.

Outputs (gitignored, private):
  data/processed/golden_v4.parquet            - gold v3 eval (5,857) + VitaminC (800)
  data/processed/golden_v4_synth_aug.parquet  - gold v3 synthetic augmentation, carried forward

Run:  python experiments/grounding-semantic/build_golden_v4.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "data" / "processed" / "golden_v3.parquet"
V3_AUG = ROOT / "data" / "processed" / "golden_v3_synth_aug.parquet"
COMBINED = ROOT / "data" / "processed" / "grounding_combined.parquet"
OUT = ROOT / "data" / "processed" / "golden_v4.parquet"
OUT_AUG = ROOT / "data" / "processed" / "golden_v4_synth_aug.parquet"

V3_COLS = ["uid", "row_id", "role", "claim", "source_text", "label",
           "lang", "lang_norm", "origin", "trace_id", "group_id"]


def sha(s: str) -> str:
    return hashlib.sha1(str(s).encode()).hexdigest()


def main() -> None:
    v3 = pd.read_parquet(V3)
    assert list(v3.columns) == V3_COLS or set(V3_COLS).issubset(v3.columns), v3.columns

    vit = pd.read_parquet(COMBINED).query("corpus == 'vitaminc'").reset_index(drop=True)
    base = int(v3["row_id"].max()) + 1
    rows = pd.DataFrame({
        "uid": [f"vitc_{i:06d}" for i in range(len(vit))],
        "row_id": range(base, base + len(vit)),
        "role": "eval_vitaminc",                                  # dataset type - marked
        "claim": vit["claim"].astype(str),
        "source_text": vit["source_text"].astype(str),
        "label": vit["label"].astype(int),                        # SUPPORTS=1 / REFUTES=0
        "lang": "en",
        "lang_norm": "en",
        "origin": "vitaminc",                                     # origin - marked
        "trace_id": vit["source_text"].map(lambda s: "vitc_" + sha(s)[:12]),
        "group_id": vit["source_text"].map(lambda s: sha(s)[:12]),  # leave-one-source-out
    })[V3_COLS]

    out = pd.concat([v3[V3_COLS], rows], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    pd.read_parquet(V3_AUG).to_parquet(OUT_AUG, index=False)

    print(f"wrote {OUT.relative_to(ROOT)} ({len(out)} rows)")
    print("  role  :", out["role"].value_counts().to_dict())
    print("  origin:", out["origin"].value_counts().to_dict())
    print(f"  VitaminC: {len(rows)} rows, {rows['group_id'].nunique()} source groups, "
          f"label {rows['label'].value_counts().to_dict()}")
    print(f"wrote {OUT_AUG.relative_to(ROOT)} ({len(pd.read_parquet(OUT_AUG))} aug rows, carried forward)")


if __name__ == "__main__":
    main()
