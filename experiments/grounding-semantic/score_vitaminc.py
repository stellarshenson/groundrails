"""Run the combined (lexical + OV cascade) grounder on the VitaminC slice of gold v4.

VitaminC is the contrastive single-token-edit corpus where the lexical tier collapses to
near coin-flip - a hallucination is one flipped number, date, or entity in otherwise-matching
evidence, which token-overlap recall cannot see. This scores the 800 VitaminC rows through
BOTH tiers (the lexical manifold via `ground()` effort=high, and the OV int8 cascade
bge-m3 -> bge-reranker + mDeBERTa-NLI), fuses them with the frozen v1 joint head, and reports
whether the semantic contradiction signal closes the gap the lexical tier provably cannot.

Two passes per row, cached to data/processed/golden_v4_vitc_signals.parquet (resumable,
checkpointed). Evaluation: lexical-only vs combined macro-F1 (combined at both its in-sample
optimum AND the gold v3 operating threshold transferred over), plus per-signal AUC so the
contribution of nli_contra / nli_ent is explicit. Writes reports/grounding_vitaminc_combined.md.

Run:  uv run python experiments/grounding-semantic/score_vitaminc.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from grounding_models import chunk_text  # noqa: E402

from groundrails.grounding import UnsupportedLanguageError, ground  # noqa: E402
from groundrails.semantic_ov import SemanticCascade, install_hint, is_available  # noqa: E402

from joint_xlingual import (  # noqa: E402
    JF,
    best_threshold,
    lexical_flag,
    load,
    slice_metrics,
    v1_head_proba,
)

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data" / "processed" / "golden_v4.parquet"
SIG = ROOT / "data" / "processed" / "golden_v4_vitc_signals.parquet"
REPORT = ROOT / "reports" / "grounding_vitaminc_combined.md"

COLS = ["uid", "label", "lex_p", "lex_blocked", "lex_fired", "lex_contra",
        "cos_max", "rr_max", "nli_ent", "nli_contra", "ran_rr", "ran_nli"]
CHECKPOINT = 100


def score() -> pd.DataFrame:
    vit = pd.read_parquet(V4).query("role == 'eval_vitaminc'").reset_index(drop=True)
    rows, done = [], set()
    if SIG.exists():
        prev = pd.read_parquet(SIG)
        rows = prev[COLS].values.tolist()
        done = set(prev["uid"])
        print(f"resume: {len(done)} already scored", flush=True)
    todo = vit[~vit["uid"].isin(done)]
    if len(todo):
        if not is_available():
            raise SystemExit(install_hint())
        eng = SemanticCascade()
        n = len(todo)
        print(f"scoring {n} VitaminC rows through lexical + cascade", flush=True)
        for i, r in enumerate(todo.itertuples(index=False)):
            chunks = chunk_text(r.source_text)
            try:
                m = ground(r.claim, [(f"s{j}", c) for j, c in enumerate(chunks)])
                lex_p = float(m.verdict_probability)
                blocked, fired = False, m.match_type in ("exact", "fuzzy", "bm25")
                contra = bool(m.numeric_mismatches or m.entity_mismatches)
            except UnsupportedLanguageError:
                lex_p, blocked, fired, contra = 0.0, True, False, False
            s = eng.score(r.claim, chunks)
            rows.append([r.uid, int(r.label), lex_p, blocked, fired, contra,
                         s.cos_max, s.rr_max, s.nli_ent, s.nli_contra, s.ran_rr, s.ran_nli])
            if (i + 1) % CHECKPOINT == 0:
                pd.DataFrame(rows, columns=COLS).to_parquet(SIG, index=False)
                print(f"  scored {i + 1}/{n} (checkpoint)", flush=True)
        pd.DataFrame(rows, columns=COLS).to_parquet(SIG, index=False)
    out = pd.DataFrame(rows, columns=COLS)
    print(f"signals: {len(out)} rows, fire rate ran_rr {out['ran_rr'].mean():.1%}", flush=True)
    return out


def evaluate(df: pd.DataFrame) -> None:
    df = df.copy()
    df["lex_p"] = df["lex_p"].where(~df["lex_blocked"].astype(bool), 0.0)
    for c in ("lex_blocked", "lex_contra"):
        df[c] = df[c].astype(float)
    y = df["label"].to_numpy(int)

    yh_lex = (~lexical_flag(df)).astype(int)
    lex_macro = f1_score(y, yh_lex, average="macro")

    pv1 = v1_head_proba(df)
    best_macro, T_best = best_threshold(y, pv1)

    # transfer: the gold v3 operating point applied unchanged to VitaminC
    v3 = load()
    ev = v3["role"].eq("eval")
    p_v3 = v1_head_proba(v3)
    _, T_v3 = best_threshold(v3.loc[ev, "label"].to_numpy(int), p_v3[ev.to_numpy()])
    yh_tr = (pv1 >= T_v3).astype(int)
    transfer_macro = f1_score(y, yh_tr, average="macro")
    tr = slice_metrics(df, pv1, T_v3, pd.Series(np.ones(len(df), bool)))

    aucs = {
        "lex_p (lexical verdict)": roc_auc_score(y, df["lex_p"]),
        "cos_max (bi-encoder)": roc_auc_score(y, df["cos_max"]),
        "rr_max (reranker)": roc_auc_score(y, df["rr_max"]),
        "nli_ent (entailment)": roc_auc_score(y, df["nli_ent"]),
        "nli_contra (contradiction, inverted)": roc_auc_score(y, -df["nli_contra"]),
        "v1-head (fused)": roc_auc_score(y, pv1),
    }

    L = [
        "# Combined grounder on VitaminC (gold v4)",
        "",
        f"VitaminC dev, {len(df)} rows (SUPPORTS {int((y == 1).sum())} / REFUTES "
        f"{int((y == 0).sum())}), English single-sentence contrastive evidence. The combined "
        "grounder fuses the lexical manifold (effort=high) with the OV int8 cascade "
        "(bge-m3 -> bge-reranker + mDeBERTa-NLI) via the frozen v1 joint head. This is the "
        "regime where the lexical tier collapses: a negative is one edited token in matching "
        "text.",
        "",
        "## Macro-F1",
        "",
        "| configuration | macro-F1 | note |",
        "|---|---|---|",
        f"| lexical-only (high, shipped verdict) | {lex_macro:.3f} | token-overlap baseline |",
        f"| combined v1-head, gold v3 operating cut (T={T_v3:.2f}) | {transfer_macro:.3f} | "
        "frozen head + transferred threshold (honest) |",
        f"| combined v1-head, VitaminC-optimal cut (T={T_best:.2f}) | {best_macro:.3f} | "
        "in-sample threshold ceiling |",
        "",
        f"At the transferred operating point: support-recall {tr.get('sup_recall', float('nan')):.3f}, "
        f"TNR {tr.get('tnr', float('nan')):.3f}.",
        "",
        "## Per-signal separation (AUC on SUPPORTS vs REFUTES)",
        "",
        "| signal | AUC |",
        "|---|---|",
    ]
    for k, v in aucs.items():
        L.append(f"| {k} | {v:.3f} |")
    L += [
        "",
        f"Lift combined over lexical-only: {transfer_macro - lex_macro:+.3f} (transferred cut), "
        f"{best_macro - lex_macro:+.3f} (in-sample ceiling).",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


def main() -> None:
    evaluate(score())


if __name__ == "__main__":
    main()
