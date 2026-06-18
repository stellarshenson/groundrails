"""Build gold v3 - the golden dataset, lineage-correct, in two linked files.

There is ONE golden dataset (verified ground truth). The synthetic translations are NOT
golden - they are a derived augmentation of it - so they live in a sidecar that points
back at the gold, never merged in as equal rows.

  golden_v3.parquet            role=eval        the verified ground truth (from gold_v2,
                                                639 source docs, native + non-English).
                                                THE golden dataset. Headline metrics here.
  golden_v3_synth_aug.parquet  role=augmentation  synthetic cross-lingual negatives,
                                                machine-translated from base sources and
                                                verified by claude_p_equivalence. Training
                                                augmentation + an offline TNR probe only.

Lineage / leakage safety: every row carries a `group_id` = a stable hash of its evidence
`source_text`. A synthetic row reuses its base source's text verbatim, so it inherits the
SAME group_id as the base claims on that source. GroupKFold on `group_id` therefore keeps
a base source and all its translations in one fold - a claim can never train on its own
translation (the leak that a flat concat would have caused). This is the leave-one-source-
out discipline the lexical track used (R9-R12).

Output under data/processed/ (gitignored - private text; synced to S3, never committed).

Run:  python experiments/build_gold_v3.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "experiments" / "grounding-lexical" / "private-rag-forensics" / "gold"
EVAL_SRC = GOLD_DIR / "golden_grounding_evidence_v2.parquet"  # verified base (v2 re-judge)
SYNTH_SRC = GOLD_DIR / "synthetic_mt.parquet"  # derived translations

OUT_EVAL = ROOT / "data" / "processed" / "golden_v3.parquet"
OUT_AUG = ROOT / "data" / "processed" / "golden_v3_synth_aug.parquet"

AUG_META = ["source_corpus", "source_lang", "target_lang", "translator_model",
            "verifier_model", "verified", "verify_method"]


def _lang_norm(s: pd.Series) -> pd.Series:
    """Region-stripped lowercase language (fr-FR -> fr); bare codes untouched."""
    return s.str.lower().str.split("-").str[0]


def _group_id(source_text: pd.Series) -> pd.Series:
    """Stable evidence-blob id. Shared by a base claim and any translation derived from
    the same source, so GroupKFold keeps the whole lineage in one fold."""
    return source_text.map(lambda t: hashlib.sha1(str(t).encode("utf-8")).hexdigest()[:12])


def build():
    base = pd.read_parquet(EVAL_SRC).copy()
    synth = pd.read_parquet(SYNTH_SRC).copy()

    base["role"] = "eval"
    base["lang_norm"] = _lang_norm(base["lang"])
    base["group_id"] = _group_id(base["source_text"])
    base = base.reset_index(drop=True)
    base["row_id"] = range(len(base))
    base["uid"] = base["row_id"].map(lambda i: f"eval_{i:06d}")
    eval_cols = ["uid", "row_id", "role", "claim", "source_text", "label", "lang",
                 "lang_norm", "origin", "trace_id", "group_id"]
    golden = base[eval_cols]

    synth["role"] = "augmentation"
    synth["origin"] = "synthetic_mt"
    synth["lang_norm"] = _lang_norm(synth["lang"])
    synth["group_id"] = _group_id(synth["source_text"])  # == base group (same source_text)
    synth["parent_sid"] = synth["source_sid"]
    synth = synth.reset_index(drop=True)
    synth["row_id"] = range(len(synth))
    synth["uid"] = synth["row_id"].map(lambda i: f"aug_{i:06d}")
    aug_cols = ["uid", "row_id", "role", "claim", "source_text", "label", "lang",
                "lang_norm", "origin", "parent_sid", "group_id", *AUG_META]
    aug = synth[aug_cols]
    return golden, aug


def main() -> None:
    golden, aug = build()
    OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
    golden.to_parquet(OUT_EVAL, index=False)
    aug.to_parquet(OUT_AUG, index=False)

    eg = set(golden["group_id"])
    linked = aug["group_id"].isin(eg).sum()
    gtop = golden["lang_norm"].value_counts()

    print(f"wrote {OUT_EVAL.relative_to(ROOT)}  (golden / verified, role=eval)")
    print(f"  rows      : {len(golden)}")
    print(f"  label     : {dict(golden['label'].value_counts().sort_index())}  (0=hallucination, 1=supported)")
    print(f"  origin    : {dict(golden['origin'].value_counts())}")
    print(f"  groups    : {golden['group_id'].nunique()} distinct source blobs (GroupKFold unit)")
    print(f"  lang_norm (>=40): {{ {', '.join(f'{k}:{v}' for k, v in gtop[gtop >= 40].items())} }}")
    print()
    print(f"wrote {OUT_AUG.relative_to(ROOT)}  (synthetic augmentation, role=augmentation)")
    print(f"  rows      : {len(aug)}  (all label 0; {aug['parent_sid'].nunique()} base source sentences x ~9 langs)")
    print(f"  target lang: {{ {', '.join(f'{k}:{v}' for k, v in aug['lang_norm'].value_counts().items())} }}")
    print(f"  lineage   : {linked}/{len(aug)} aug rows link to a golden source group "
          f"({aug['group_id'].nunique()} distinct source blobs, all present in golden: "
          f"{aug['group_id'].nunique() == aug['group_id'].isin(eg).pipe(lambda m: aug.loc[m, 'group_id']).nunique()})")
    print(f"  overlap with golden groups: {aug['group_id'].isin(eg).all()} (every aug source is a golden source)")


if __name__ == "__main__":
    main()
