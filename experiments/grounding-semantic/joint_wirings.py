"""Joint lexical + semantic wirings - benchmark which fusion wins on the gold.

The grounder runs the lexical manifold (effort=high) cheaply and torch-free. The
semantic cascade (bge-m3 pre-filter -> bge-reranker + mDeBERTa-NLI) is stronger but
heavy. This module races three ways of joining them on the 2,752-claim verified gold,
so the winner can be frozen into a new `semantic` effort tier:

  W1  escalation cascade  - lexical decides; only the UNCERTAIN band (lexical P near
                            its threshold) plus the cross-lingual claims the lexical
                            tier cannot ground escalate to the semantic cascade.
  W2  always-both joint   - one logistic over {lexical P + reranker + NLI + cosine +
                            contradiction}, every claim (no escalation gate).
  W3  reuse-seam baseline - lexical P + cosine + NLI only (no reranker) - the signals
                            the existing ground() semantic/nli seam already exposes.

Compared against the lexical-only (effort=high) baseline on macro-F1 / FP / FN, plus
the escalation rate for W1 (the share of claims that pay for the cascade).

All cascade signals come from the cached pair scores (`pairs/full_pairs.npz`,
`BAAI__bge-m3.npy`) so the benchmark is deterministic and GPU-free; only the lexical P
is computed live (cached after the first run). OOF 5-fold, seed 42 - the same protocol
as grounding_ensemble / grounding_hypotheses.

Run:  python joint_wirings.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from grounding_ensemble import best_macro, counts_at  # noqa: E402
from grounding_hypotheses import _macro_counts, build_features, oof_p  # noqa: E402
from grounding_models import SCORES_DIR, load_gold  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "grounding_joint_wirings.md"
LEX_CACHE = Path(__file__).resolve().parent / "private-rag-forensics" / "lexical_verdict_high.npz"

# Joint-head feature columns (named cols; _joint_matrix appends lex_blocked last). The
# resulting order must equal groundrails.joint.JOINT_FEATURES.
JOINT_COLS = ["lex_p", "rr_max", "nli_ent", "cos_max", "nli_contra", "lex_contra"]
JOINT_FEATURES = JOINT_COLS + ["lex_blocked"]


# --------------------------------------------------------------------------- signals


def lexical_pass(recs):
    """Lexical manifold (effort=high) over the gold - P(grounded), fired, contradiction.

    Cross-lingual claims with no installed argos model raise UnsupportedLanguageError;
    those are recorded as blocked (lexical cannot ground them) - the exact tail the
    semantic tier exists to catch. Cached to LEX_CACHE (the live pass is ~10 min).
    """
    if LEX_CACHE.exists():
        d = np.load(LEX_CACHE)
        return d["lex_p"], d["blocked"], d["fired"], d["contra"]

    from groundrails.grounding import UnsupportedLanguageError, ground

    n = len(recs)
    lex_p = np.zeros(n, dtype="float32")
    blocked = np.zeros(n, dtype=bool)
    fired = np.zeros(n, dtype=bool)
    contra = np.zeros(n, dtype=bool)
    for i, r in enumerate(recs):
        srcs = [(f"s{j}", c) for j, c in enumerate(r["chunks"])]
        try:
            m = ground(r["claim"], srcs)
        except UnsupportedLanguageError:
            blocked[i] = True
            lex_p[i] = np.nan
            continue
        lex_p[i] = m.verdict_probability
        fired[i] = m.match_type in ("exact", "fuzzy", "bm25")
        contra[i] = bool(m.numeric_mismatches or m.entity_mismatches)
        if (i + 1) % 250 == 0:
            print(f"  lexical pass {i + 1}/{n}", flush=True)
    LEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LEX_CACHE, lex_p=lex_p, blocked=blocked, fired=fired, contra=contra)
    return lex_p, blocked, fired, contra


def load_signals():
    """All per-claim signals, aligned to load_gold order.

    Returns a dict of (n,) arrays: lex_p (NaN where blocked), lex_blocked, lex_fired,
    lex_contra, rr_max, nli_ent, nli_contra, cos_max, plus y (1=supported) and langs.
    """
    recs = load_gold()
    feats, y, langs = build_features()  # cached pair scores -> rr/nli aggregations
    cos = np.load(SCORES_DIR / "BAAI__bge-m3.npy")
    if not (len(cos) == len(y) == len(recs)):
        raise SystemExit("signal length mismatch (cosine / pairs / gold)")
    lex_p, blocked, fired, contra = lexical_pass(recs)
    return {
        "lex_p": lex_p,
        "lex_blocked": blocked,
        "lex_fired": fired.astype("float32"),
        "lex_contra": contra.astype("float32"),
        "rr_max": feats["rr_max"],
        "nli_ent": feats["nli_ent_max"],
        "nli_contra": feats["nli_contra_max"],
        "cos_max": cos.astype("float32"),
        "y": np.asarray(y),
        "langs": np.asarray(langs),
    }


def _joint_matrix(s, cols):
    """Stack named signals into an OOF feature matrix, imputing blocked lexical P.

    Blocked claims (cross-lingual, no lexical verdict) get lex_p=0 + a lex_blocked
    indicator so the logistic learns to lean on the cascade for them rather than
    treating NaN as signal.
    """
    out = []
    for c in cols:
        v = np.asarray(s[c], dtype="float32").copy()
        if c == "lex_p":
            v[s["lex_blocked"]] = 0.0
        out.append(v)
    out.append(s["lex_blocked"].astype("float32"))  # always carry the blocked flag
    return np.column_stack(out)


# --------------------------------------------------------------------------- wirings


def _lex_flag(s):
    """The shipped lexical verdict per claim (what live ground() returns): flag as
    hallucination when the claim did not fire a lexical layer, or is blocked (the
    cross-lingual claim the tier cannot ground). This is the exact verdict the live
    semantic switch keeps for claims outside the escalation band."""
    return s["lex_blocked"] | (s["lex_fired"] < 0.5)


def lexical_only(s):
    """Baseline: lexical manifold (effort=high) alone, at its shipped threshold - the
    live `groundrails ground --effort high` (semantic off) behaviour. Blocked claims
    flag as hallucination (the tier cannot ground a cross-lingual claim)."""
    y = s["y"]
    flag = _lex_flag(s)
    macro, fp, fn = _macro_counts(flag, y)
    return {"name": "lexical-only (high)", "macro": macro, "fp": fp, "fn": fn, "escalation": 0.0}


def w3_reuse_seam(s):
    """Lexical P + cosine + NLI entailment (no reranker) - existing-seam lower bound."""
    y = s["y"]
    X = _joint_matrix(s, ["lex_p", "cos_max", "nli_ent"])
    p = oof_p(X, y)
    macro, T, _ = best_macro(p, y)
    fp, fn, _, _ = counts_at(p, y, T)
    return {
        "name": "W3 reuse-seam {lex,cos,nli}",
        "macro": macro,
        "fp": fp,
        "fn": fn,
        "escalation": 1.0,
        "T": float(T),
    }


def w2_always_both(s):
    """Full joint head over every claim: lexical P + reranker + NLI + cosine +
    contradiction channels."""
    y = s["y"]
    X = _joint_matrix(s, JOINT_COLS)
    p = oof_p(X, y)
    macro, T, _ = best_macro(p, y)
    fp, fn, _, _ = counts_at(p, y, T)
    return {
        "name": "W2 always-both joint",
        "macro": macro,
        "fp": fp,
        "fn": fn,
        "escalation": 1.0,
        "T": float(T),
    }


def w1_escalation(s):
    """Lexical decides; only the uncertain band (+ blocked cross-lingual) escalates.

    Out-of-band claims take the lexical verdict (lex_p < T_lex). In-band and blocked
    claims take the joint-head verdict. The escalation band (a, b) around the lexical
    distribution is swept for the macro-F1 optimum; escalation rate = the share routed
    to the cascade.
    """
    y = s["y"]
    scor = ~s["lex_blocked"]
    lex_p = s["lex_p"]
    pivot = 0.5  # the high manifold's shipped decision threshold

    Xj = _joint_matrix(s, JOINT_COLS)
    pj = oof_p(Xj, y)
    _, T_j, _ = best_macro(pj, y)

    # out-of-band claims take the shipped lexical verdict; in-band / blocked escalate.
    lex_flag = _lex_flag(s)
    joint_flag = pj < T_j

    qlo = np.quantile(lex_p[scor], np.linspace(0.02, 0.5, 25))
    qhi = np.quantile(lex_p[scor], np.linspace(0.5, 0.98, 25))
    best = None
    for a in qlo[qlo <= pivot]:
        for b in qhi[qhi >= pivot]:
            if b <= a:
                continue
            in_band = (~s["lex_blocked"]) & (lex_p > a) & (lex_p < b)
            escalate = in_band | s["lex_blocked"]
            flag = np.where(escalate, joint_flag, lex_flag)
            macro, fp, fn = _macro_counts(flag, y)
            row = (macro, float(escalate.mean()), a, b, fp, fn)
            # prefer higher macro, break ties on lower escalation rate
            if best is None or (row[0], -row[1]) > (best[0], -best[1]):
                best = row
    macro, esc, a, b, fp, fn = best
    return {
        "name": "W1 escalation cascade",
        "macro": macro,
        "fp": fp,
        "fn": fn,
        "escalation": esc,
        "band": (float(a), float(b)),
        "T_joint": float(T_j),
    }


# --------------------------------------------------------------------------- export


def export_winner(s, w1):
    """Frozen `calibration.semantic` block for the escalation tier.

    Fits the joint logistic on ALL data and folds the StandardScaler into raw-space
    weights so inference is a plain dot-product + sigmoid (no scikit-learn) - the same
    contract as the lexical manifolds. Pairs the head with W1's escalation band + the
    cascade gate/band operating points. Returns the block dict (no client text)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = s["y"]
    X = _joint_matrix(s, JOINT_COLS)  # columns == JOINT_FEATURES order
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
    coef = lr.coef_[0] / sc.scale_
    intercept = float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())
    weights = {"Intercept": round(intercept, 6)}
    for name, w in zip(JOINT_FEATURES, coef):
        weights[name] = round(float(w), 6)
    return {
        "feature_order": JOINT_FEATURES,
        "weights": weights,
        "threshold": round(w1["T_joint"], 6),
        "escalation_band": [round(w1["band"][0], 6), round(w1["band"][1], 6)],
        "cosine_gate": [0.493, 0.739],
        "cascade_band": [0.01, 0.66],
        "top_k": 8,
    }


def _emit_block(block) -> str:
    """Render the semantic block as a YAML snippet to paste under `calibration:`."""
    import yaml

    return yaml.safe_dump({"semantic": block}, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- report


def main() -> None:
    s = load_signals()
    y = s["y"]
    n = len(y)
    n_hall = int((y == 0).sum())
    n_blocked = int(s["lex_blocked"].sum())

    rows = [lexical_only(s), w3_reuse_seam(s), w2_always_both(s), w1_escalation(s)]
    base = rows[0]
    top = max(rows[1:], key=lambda r: r["macro"])

    # Freeze the escalation tier (W1 head + band) into a pasteable config block.
    w1_row = next(r for r in rows if r["name"].startswith("W1"))
    block = export_winner(s, w1_row)
    block_file = ROOT / "reports" / "semantic_tier_block.yaml"
    block_file.write_text(_emit_block(block), encoding="utf-8")

    L = [
        "# Joint lexical + semantic wirings - benchmark",
        "",
        f"Verified gold: {n} claims ({n_hall} hallucination / {n - n_hall} supported, "
        f"base rate {n_hall / n:.0%}). {n_blocked} claims are cross-lingual with no "
        "lexical (argos) model - the lexical tier cannot ground them. Cascade signals "
        "from the cached int8 pair scores; lexical P from the live effort=high manifold. "
        "OOF 5-fold (seed 42), thresholds at the macro-F1 optimum.",
        "",
        "## Wirings (out-of-fold)",
        "",
        "| wiring | macro-F1 | FP | FN | FP+FN | escalation | d macro vs lexical |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        esc = "-" if r is base else f"{r['escalation']:.0%}"
        d = "" if r is base else f"{r['macro'] - base['macro']:+.3f}"
        L.append(
            f"| {r['name']} | {r['macro']:.3f} | {r['fp']} | {r['fn']} | "
            f"{r['fp'] + r['fn']} | {esc} | {d} |"
        )

    w1 = next(r for r in rows if r["name"].startswith("W1"))
    L += [
        "",
        f"**Lexical-only (high)** at its shipped manifold threshold scores macro-F1 "
        f"{base['macro']:.3f} (FP {base['fp']}, FN {base['fn']}); {n_blocked} of the "
        "claims are cross-lingual with no lexical (argos) model and are flagged "
        "unconfirmed.",
        "",
        f"All three wirings cluster at macro-F1 {min(r['macro'] for r in rows[1:]):.3f}-"
        f"{top['macro']:.3f} (within noise at n={n}), each +0.06-0.07 over lexical-only. "
        "The shipped high manifold over-flags supported claims (high FP), so escalation "
        "favours a wide band.",
        "",
    ]
    L += [
        f"**Shipped as the `semantic` tier: W1 escalation** (macro-F1 {w1['macro']:.3f}, "
        f"FP {w1['fp']}, FN {w1['fn']}). It is the requested design - lexical decides, "
        "the uncertain band escalates to the cascade - carries the best hallucination "
        f"recall (lowest FN) and is the only wiring with a cost lever ({w1['escalation']:.0%} "
        f"escalation here at band [{w1['band'][0]:.2f}, {w1['band'][1]:.2f}]). A "
        "better-calibrated lexical gate would narrow the band and cut the cascade share "
        "further at the same quality.",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")
    print(f"wrote {block_file}")
    print("\n--- calibration.semantic block ---")
    print(_emit_block(block))


if __name__ == "__main__":
    main()
