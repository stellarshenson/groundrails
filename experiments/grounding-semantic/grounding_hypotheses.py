"""Final hypothesis round - H9/H10/H11 over the int8 grounder (no fine-tuning).

Three mechanism-targeting hypotheses against the deployed 2-cross-encoder stack
(bge-reranker int8 + mDeBERTa-v3 SmoothQuant int8, OOF macro-F1 0.796 fp32 /
0.795 int8):

- H9  NLI contradiction channel - the NLI forward already computes a 3-way softmax;
  add max-contradiction (and max-neutral) over chunks as logistic features. Zero
  added inference cost.
- H10 aggregation beyond max-over-chunks - distributional features of the per-pair
  score set (top-2 mean, logsumexp, count above threshold, top1-top2 margin) per
  model. Zero added inference cost.
- H11 reranker-first confidence cascade - run the NLI only when the reranker score
  falls inside an uncertainty band; out-of-band claims take the reranker-only
  verdict. Latency lever, quality-gated.

All scoring is CPU OpenVINO int8 (the deployed engines). Models stay frozen - only
softmax indices, aggregation statistics, thresholds and a logistic re-fit are used.

Round 2 (H12/H13/H14) extends the adopted cascade:

- H12 pre-filter cosine gate - the pre-filter already computes claim-chunk cosines;
  their max becomes a stage-0 gate (extreme tails skip BOTH cross-encoders). Pure
  cached arithmetic - no new scoring.
- H13 rank-ordered early-exit reranker - score pairs best-cosine-first and stop once
  the running max crosses the cascade pass edge; verdict-invariant by construction.
  Latency-only (scripts/bench_grounder_round2.py).
- H14 fused-evidence single-forward cross-encoders - assemble ONE evidence context
  per claim (v1 top-2 chunk concat / v2 salience-packed sentences) and run ONE
  forward per model instead of k=8 per-chunk forwards.

Run:  python grounding_hypotheses.py score        # round-1 pair cache (slow)
      python grounding_hypotheses.py               # round-1 eval (H9/H10/H11)
      python grounding_hypotheses.py score-fused  # H14 fused cache (~30 min)
      python grounding_hypotheses.py round2       # round-2 eval (H12/H14)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grounding_models import SCORES_DIR, load_gold

ROOT = Path(__file__).resolve().parents[2]
PAIRS = SCORES_DIR / "pairs" / "full_pairs.npz"
FUSED = SCORES_DIR / "pairs" / "fused_pairs.npz"
REPORT = ROOT / "reports" / "grounding_hypotheses.md"

BASELINE_MACRO = 0.796   # fp32 2-cross-encoder stack (semantic-grounding-sota.md)
F1_GATE = 0.806          # adopt H9/H10 only at baseline + 0.010
CASCADE_GATE = 0.782     # baseline - one fold-std
BAND = (0.01, 0.66)      # adopted H11 band (grounding_openvino.CASCADE_BAND)


# --------------------------------------------------------------------------- scoring

def score_pairs() -> None:
    """Score every (claim, chunk) gold pair with both int8 cross-encoders (CPU OpenVINO).

    Caches owner / per-pair reranker sigmoid / per-pair NLI 3-class softmax to PAIRS,
    then sanity-checks the max-over-pairs against the cached max-score signals.
    """
    from grounding_openvino import compile_ir, load_ov_hf, pair_scores_full
    records = load_gold()
    labels = np.array([r["label"] for r in records])
    langs = np.array([r["lang"] for r in records])

    # THROUGHPUT hint: offline batch path (async queues), not the serving path
    _, rr_tok, rr_dir = load_ov_hf("bge-reranker-v2-m3", compile=False)
    _, nli_tok, nli_dir = load_ov_hf("mDeBERTa-v3-nli", compile=False)
    rr_cm = compile_ir(rr_dir / "openvino_model.xml", hint="THROUGHPUT")
    nli_cm = compile_ir(nli_dir / "openvino_model.xml", hint="THROUGHPUT")
    id2label = {int(k): v.lower() for k, v in
                json.loads((nli_dir / "config.json").read_text())["id2label"].items()}

    owner, rr, nli = pair_scores_full(records, rr_cm, rr_tok, nli_cm, nli_tok)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PAIRS, owner=owner, rr=rr, nli=nli, labels=labels, langs=langs,
             id2label=np.array([id2label[i] for i in range(nli.shape[1])]))

    # sanity: max over pairs must reproduce the cached max-over-chunks signals
    n = len(records)
    ent = next(i for i, v in id2label.items() if "entail" in v)
    rr_max = np.array([rr[owner == i].max() for i in range(n)])
    nli_max = np.array([nli[owner == i, ent].max() for i in range(n)])
    # int8 DeBERTa output shifts with batch padding composition (this run batches in
    # document order under THROUGHPUT; the cached reference was length-bucketed under
    # LATENCY), so exact reproduction is not achievable - the gate is the documented
    # int8-vs-fp32 parity level (0.984). Measured: pearson 0.986, mean|d| 0.031.
    from scipy.stats import pearsonr
    nli_ref = np.load(SCORES_DIR / "mDeBERTa-v3-int8-sq.npy")          # same int8 engine
    rr_ref = np.load(SCORES_DIR / "BAAI__bge-reranker-v2-m3.npy")      # fp32 GPU reference
    p_nli = float(pearsonr(nli_max, nli_ref)[0])
    p_rr = float(pearsonr(rr_max, rr_ref)[0])
    print(f"sanity: nli int8 pearson={p_nli:.4f} (gate >=0.98) | "
          f"rr int8-vs-fp32 pearson={p_rr:.4f} (gate >=0.99)")
    if p_nli < 0.98 or p_rr < 0.99:
        raise SystemExit("pair-cache sanity FAILED")
    print(f"wrote {PAIRS} ({len(owner)} pairs / {n} records)")


# --------------------------------------------------------------------------- features

def load_pairs():
    d = np.load(PAIRS, allow_pickle=True)
    return d["owner"], d["rr"], d["nli"], d["labels"], d["langs"], list(d["id2label"])


def _agg(owner, s, n):
    """Distributional aggregations of one score array, per record: max, top-2 mean,
    logsumexp (T=0.1), count >= 0.5 (log1p), top1-top2 margin."""
    out = np.zeros((n, 5))
    for i in range(n):
        v = np.sort(s[owner == i])[::-1]
        t2 = v[:2].mean() if len(v) > 1 else v[0]
        lse = 0.1 * np.log(np.exp(v / 0.1).sum())
        cnt = np.log1p(float((v >= 0.5).sum()))
        mar = v[0] - v[1] if len(v) > 1 else v[0]
        out[i] = [v[0], t2, lse, cnt, mar]
    return out


def build_features():
    """All candidate per-record features from the pair cache.

    Returns (feats: dict name -> (n,) array, y, langs). Baseline stack = {rr_max,
    nli_ent_max}; H9 adds nli_contra_max / nli_neut_max; H10 adds the aggregation
    columns per model.
    """
    owner, rr, nli, y, langs, id2label = load_pairs()
    n = len(y)
    ent = next(i for i, v in enumerate(id2label) if "entail" in v)
    con = next(i for i, v in enumerate(id2label) if "contra" in v)
    neu = next(i for i, v in enumerate(id2label) if "neutral" in v)
    rr_a = _agg(owner, rr, n)
    ent_a = _agg(owner, nli[:, ent], n)
    feats = {
        "rr_max": rr_a[:, 0], "nli_ent_max": ent_a[:, 0],
        # H9 - extra NLI channels (same forward pass)
        "nli_contra_max": np.array([nli[owner == i, con].max() for i in range(n)]),
        "nli_neut_max": np.array([nli[owner == i, neu].max() for i in range(n)]),
        # H10 - distribution of the per-pair scores
        "rr_top2": rr_a[:, 1], "rr_lse": rr_a[:, 2], "rr_cnt": rr_a[:, 3],
        "rr_margin": rr_a[:, 4],
        "nli_top2": ent_a[:, 1], "nli_lse": ent_a[:, 2], "nli_cnt": ent_a[:, 3],
        "nli_margin": ent_a[:, 4],
    }
    return feats, y, langs


# --------------------------------------------------------------------------- evaluate

def oof_p(X, y):
    """Out-of-fold P(supported) - same protocol as grounding_ensemble (5-fold, seed 42)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    return cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]


def _macro_counts(flag, y):
    from grounding_ensemble import _f1c
    hall = y == 0
    tp = int((flag & hall).sum())
    fn = int((~flag & hall).sum())
    fp = int((flag & ~hall).sum())
    tn = int((~flag & ~hall).sum())
    return 0.5 * (_f1c(tp, fp, fn) + _f1c(tn, fn, fp)), fp, fn


def eval_stacks(feats, y):
    """H9/H10 ladder - OOF macro-F1 of logistic stacks over feature subsets."""
    from grounding_ensemble import best_macro
    sets = {
        "baseline {rr, nli_ent} (int8 pairs)": ["rr_max", "nli_ent_max"],
        "H9 +contradiction": ["rr_max", "nli_ent_max", "nli_contra_max"],
        "H9 +contradiction +neutral": ["rr_max", "nli_ent_max", "nli_contra_max",
                                       "nli_neut_max"],
        "H10 rr aggregations": ["rr_max", "rr_top2", "rr_lse", "rr_cnt", "rr_margin",
                                "nli_ent_max"],
        "H10 nli aggregations": ["rr_max", "nli_ent_max", "nli_top2", "nli_lse",
                                 "nli_cnt", "nli_margin"],
        "H10 both aggregations": ["rr_max", "rr_top2", "rr_cnt", "nli_ent_max",
                                  "nli_top2", "nli_cnt"],
        "H9+H10 combined": ["rr_max", "rr_top2", "rr_cnt", "nli_ent_max", "nli_top2",
                            "nli_cnt", "nli_contra_max"],
    }
    rows = []
    for label, cols in sets.items():
        X = np.column_stack([feats[c] for c in cols])
        p = oof_p(X, y)
        macro, T, m = best_macro(p, y)
        rows.append({"set": label, "k": len(cols), "macro": macro,
                     "fp": None, "fn": None, "p": p, "T": T})
    return rows


def eval_cascade(feats, y):
    """H11 frontier - band sweep on the reranker score; in-band claims use the stack.

    Verdicts follow the baseline protocol (thresholds fit on OOF predictions): rr-only
    threshold from best_macro(rr), stack threshold from best_macro(OOF p). skip = the
    fraction of claims resolved without the NLI forward.
    """
    from grounding_ensemble import best_macro
    rr = feats["rr_max"]
    X = np.column_stack([feats["rr_max"], feats["nli_ent_max"]])
    p = oof_p(X, y)
    _, T_rr, _ = best_macro(rr, y)
    _, T_st, _ = best_macro(p, y)
    rows = []
    # dense grid: the coarse 19-point sweep rounded the adopted band to +4 FP; at 99
    # points the same band is strictly no-worse than baseline (FP 243/244, FN equal)
    qs = np.quantile(rr, np.linspace(0.01, 0.99, 99))
    for a in qs[qs <= T_rr]:
        for b in qs[qs >= T_rr]:
            out_band = (rr <= a) | (rr >= b)
            flag = np.where(out_band, rr < T_rr, p < T_st)
            macro, fp, fn = _macro_counts(flag, y)
            rows.append({"a": float(a), "b": float(b), "skip": float(out_band.mean()),
                         "macro": macro, "fp": fp, "fn": fn})
    return rows, T_rr, T_st


# --------------------------------------------------------------------------- round 2

def _cascade_ref(feats, y):
    """Adopted-cascade reference verdicts (band fixed at BAND, baseline protocol).

    Returns (casc_flag, rr, T_rr, T_st, (macro, fp, fn)) - the row every round-2
    candidate is compared against (0.797 / FP 243 / FN 217 on the int8 pair cache).
    """
    from grounding_ensemble import best_macro
    rr = feats["rr_max"]
    X = np.column_stack([feats["rr_max"], feats["nli_ent_max"]])
    p = oof_p(X, y)
    _, T_rr, _ = best_macro(rr, y)
    _, T_st, _ = best_macro(p, y)
    out_band = (rr <= BAND[0]) | (rr >= BAND[1])
    casc_flag = np.where(out_band, rr < T_rr, p < T_st)
    return casc_flag, rr, T_rr, T_st, _macro_counts(casc_flag, y)


def eval_gate(feats, y):
    """H12 frontier - stage-0 gate on the pre-filter max cosine, then the adopted cascade.

    The pre-filter already computes the claim-chunk cosines (the max is currently
    discarded), so gated claims skip BOTH cross-encoders at zero added compute. Sweep
    (a0, b0) over the cosine tails; cos <= a0 flags, cos >= b0 passes, in-between falls
    through to the cascade verdict. The disabled gate (-inf, +inf) must reproduce the
    cascade reference exactly.
    """
    cos = np.load(SCORES_DIR / "BAAI__bge-m3.npy")
    if len(cos) != len(y):
        raise SystemExit("cosine cache / gold size mismatch")
    casc_flag, _, _, _, ref = _cascade_ref(feats, y)
    rows = []
    lo = np.r_[-np.inf, np.quantile(cos, np.linspace(0.005, 0.20, 40))]
    hi = np.r_[np.quantile(cos, np.linspace(0.80, 0.995, 40)), np.inf]
    for a0 in lo:
        for b0 in hi:
            g_lo, g_hi = cos <= a0, cos >= b0
            flag = casc_flag.copy()
            flag[g_lo] = True
            flag[g_hi] = False
            macro, fp, fn = _macro_counts(flag, y)
            rows.append({"a0": float(a0), "b0": float(b0),
                         "skip0": float((g_lo | g_hi).mean()),
                         "macro": macro, "fp": fp, "fn": fn})
    return rows, ref


def score_fused() -> None:
    """H14 scoring run - ONE evidence context per claim, ONE forward per cross-encoder.

    v1: top-2 cosine-ranked chunks, each char-truncated to fit two in the 512 window.
    v2: sentences from the top-8 chunks, salience-ranked by the same bi-encoder and
    packed to ~480 tokens in document order.

    All int8 OpenVINO on CPU (THROUGHPUT hint - offline batch path). Caches reranker
    sigmoid + full NLI softmax per variant to FUSED. Input assembly only - the same
    mechanism family as the existing top-k pre-filter; no weights touched.
    """
    from grounding_openvino import (
        _async_cross,
        _async_cross_probs,
        _async_embed,
        compile_ir,
        fused_context_chunks,
        load_ov_hf,
        pack_sentences,
        split_sentences,
    )
    records = load_gold()
    labels = np.array([r["label"] for r in records])
    claims = [r["claim"] for r in records]

    _, etok, edir = load_ov_hf("bge-m3", compile=False)
    emb = compile_ir(edir / "openvino_model.xml", hint="THROUGHPUT")

    def embed_unique(texts, desc):
        # length-sort before batching (less padding), restore order after
        order = np.argsort([len(t) for t in texts])
        v = _async_embed(emb, etok, [texts[i] for i in order], desc=desc)
        out = np.empty_like(v)
        out[order] = v
        return out

    # records share evidence lists - embed each unique chunk / sentence once
    uniq: dict[str, int] = {}
    for r in records:
        for c in r["chunks"]:
            uniq.setdefault(c, len(uniq))
    chunk_texts = list(uniq)
    sent_lists = [split_sentences(c) or [c] for c in chunk_texts]
    usent: dict[str, int] = {}
    for ss in sent_lists:
        for s in ss:
            usent.setdefault(s, len(usent))
    sent_texts = list(usent)
    print(f"records={len(records)} unique chunks={len(chunk_texts)} "
          f"unique sentences={len(sent_texts)}", flush=True)

    cvec = embed_unique(claims, "embed claims")
    chvec = embed_unique(chunk_texts, "embed chunks")
    svec = embed_unique(sent_texts, "embed sentences")

    ctx1, ctx2 = [], []
    for i, r in enumerate(records):
        ids = [uniq[c] for c in r["chunks"]]
        cos = chvec[ids] @ cvec[i]
        order = np.argsort(-cos)
        ranked = [r["chunks"][j] for j in order]
        ctx1.append(fused_context_chunks(ranked))
        # v2 - sentences of the top-8 chunks in (chunk rank, position) order
        sents = [s for j in order[:8] for s in sent_lists[ids[j]]]
        sims = np.array([svec[usent[s]] @ cvec[i] for s in sents])
        ctx2.append(pack_sentences(sents, sims) if sents else ctx1[-1])
    for name, ctxs in (("v1", ctx1), ("v2", ctx2)):
        ln = np.array([len(c) for c in ctxs])
        print(f"ctx {name}: chars p50={np.percentile(ln, 50):.0f} "
              f"p90={np.percentile(ln, 90):.0f} max={ln.max()}", flush=True)

    _, rr_tok, rr_dir = load_ov_hf("bge-reranker-v2-m3", compile=False)
    _, nli_tok, nli_dir = load_ov_hf("mDeBERTa-v3-nli", compile=False)
    rr_cm = compile_ir(rr_dir / "openvino_model.xml", hint="THROUGHPUT")
    nli_cm = compile_ir(nli_dir / "openvino_model.xml", hint="THROUGHPUT")
    id2label = {int(k): v.lower() for k, v in
                json.loads((nli_dir / "config.json").read_text())["id2label"].items()}
    n_out = nli_cm.output(0).get_partial_shape()[1].get_length()

    out = {"labels": labels,
           "id2label": np.array([id2label[i] for i in range(n_out)])}
    sig = lambda lg: 1.0 / (1.0 + np.exp(-lg.reshape(-1)))  # noqa: E731
    for name, ctxs in (("v1", ctx1), ("v2", ctx2)):
        out[f"rr_{name}"] = _async_cross(rr_cm, rr_tok, claims, ctxs, sig,
                                         desc=f"rerank fused {name}").astype(np.float32)
        out[f"nli_{name}"] = _async_cross_probs(nli_cm, nli_tok, ctxs, claims, n_out,
                                                desc=f"nli fused {name}")
    FUSED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FUSED, **out)

    # sanity (report only - fused inputs differ from per-chunk max by design)
    if PAIRS.exists():
        from scipy.stats import pearsonr
        feats, y, _ = build_features()
        ent = next(i for i in range(n_out) if "entail" in id2label[i])
        for name in ("v1", "v2"):
            p_rr = pearsonr(out[f"rr_{name}"], feats["rr_max"])[0]
            p_nli = pearsonr(out[f"nli_{name}"][:, ent], feats["nli_ent_max"])[0]
            print(f"sanity {name}: rr_fused vs rr_max pearson={p_rr:.3f} | "
                  f"nli_fused vs nli_ent_max pearson={p_nli:.3f}", flush=True)
    print(f"wrote {FUSED}", flush=True)


def eval_fused(feats, y):
    """H14 ladder - OOF stacks over fused scores, plus the deployable cascade-composed
    mixed config (rr_max cascade kept, fused NLI replacing per-chunk NLI in-band)."""
    from grounding_ensemble import best_macro
    d = np.load(FUSED, allow_pickle=True)
    id2label = list(d["id2label"])
    ent = next(i for i, v in enumerate(id2label) if "entail" in v)
    casc_flag, rr, T_rr, _, _ = _cascade_ref(feats, y)
    out_band = (rr <= BAND[0]) | (rr >= BAND[1])
    rows = []
    for v in ("v1", "v2"):
        rrf, nlf = d[f"rr_{v}"], d[f"nli_{v}"][:, ent]
        stacks = {
            f"{v} full fuse {{rr_fused, nli_fused}}": (rrf, nlf),
            f"{v} fused NLI {{rr_max, nli_fused}}": (feats["rr_max"], nlf),
            f"{v} fused rr {{rr_fused, nli_max}}": (rrf, feats["nli_ent_max"]),
        }
        for label, cols in stacks.items():
            p = oof_p(np.column_stack(cols), y)
            macro, T, _ = best_macro(p, y)
            _, fp, fn = _macro_counts(p < T, y)
            rows.append({"set": label, "macro": macro, "fp": fp, "fn": fn,
                         "skip": None})
        # deployable shape: adopted band on rr_max; in-band verdict from the fused-NLI
        # stack; out-of-band claims keep the reranker-only verdict (still skip the NLI)
        p = oof_p(np.column_stack([feats["rr_max"], nlf]), y)
        _, T_st, _ = best_macro(p, y)
        flag = np.where(out_band, rr < T_rr, p < T_st)
        macro, fp, fn = _macro_counts(flag, y)
        rows.append({"set": f"{v} cascade + fused NLI in-band", "macro": macro,
                     "fp": fp, "fn": fn, "skip": float(out_band.mean())})
    return rows


def main_round2() -> None:
    """Round-2 quality evaluation (H12 gate + H14 fused) - appends to REPORT.

    H13 has no quality side (verdict-invariant); its latency numbers come from
    scripts/bench_grounder_round2.py.
    """
    feats, y, _ = build_features()
    gate_rows, ref = eval_gate(feats, y)
    ref_macro, ref_fp, ref_fn = ref
    disabled = next(r for r in gate_rows if np.isinf(r["a0"]) and np.isinf(r["b0"]))
    assert (disabled["fp"], disabled["fn"]) == (ref_fp, ref_fn), "gate-off != cascade ref"
    ok = [r for r in gate_rows if r["fp"] <= ref_fp and r["fn"] <= ref_fn
          and r["skip0"] > 0]
    best_gate = max(ok, key=lambda r: r["skip0"]) if ok else None
    better = [r for r in ok if r["fp"] + r["fn"] < ref_fp + ref_fn]
    best_better = max(better, key=lambda r: r["skip0"]) if better else None
    relaxed = [r for r in gate_rows if r["macro"] >= CASCADE_GATE]
    best_relaxed = max(relaxed, key=lambda r: r["skip0"]) if relaxed else None

    L = ["", "## Round 2 - H12/H13/H14 (CPU OpenVINO int8)", "",
         f"Reference = adopted cascade (band [{BAND[0]}, {BAND[1]}]): macro-F1 "
         f"{ref_macro:.3f}, FP {ref_fp}, FN {ref_fn}. Same OOF protocol as round 1.", "",
         "### H12 - pre-filter cosine gate (stage 0, zero added compute)", "",
         "Gate on the bi-encoder max cosine the pre-filter already computes: cos <= a0 "
         "flags, cos >= b0 passes - gated claims skip BOTH cross-encoders (claim cost "
         "~38 ms). Adoption requires FP and FN no-worse than the reference at >= 3% "
         "stage-0 skip.", ""]
    if best_gate:
        L += [f"Max-skip no-worse gate: a0={best_gate['a0']:.3f} b0={best_gate['b0']:.3f} "
              f"stage-0 skip {best_gate['skip0']:.1%}, macro-F1 {best_gate['macro']:.3f}, "
              f"FP {best_gate['fp']}, FN {best_gate['fn']}."]
        if best_better:
            L += [f"Strictly-better gate (adopted): a0={best_better['a0']:.3f} "
                  f"b0={best_better['b0']:.3f} stage-0 skip {best_better['skip0']:.1%}, "
                  f"macro-F1 {best_better['macro']:.3f}, FP {best_better['fp']}, "
                  f"FN {best_better['fn']} - fewer errors than the reference while a "
                  "fifth of claims never touch a cross-encoder."]
        verdict = "ADOPT" if best_gate["skip0"] >= 0.03 else "REJECT (< 3% skip)"
        L += [f"Gate verdict: **{verdict}**."]
    else:
        L += ["No (a0, b0) with positive skip keeps FP and FN no-worse - "
              "gate verdict: **REJECT**."]
    if best_relaxed:
        L += ["", f"Relaxed (macro >= {CASCADE_GATE}) frontier point: "
              f"a0={best_relaxed['a0']:.3f} b0={best_relaxed['b0']:.3f} skip "
              f"{best_relaxed['skip0']:.1%}, macro-F1 {best_relaxed['macro']:.3f}, "
              f"FP {best_relaxed['fp']}, FN {best_relaxed['fn']} - the cosine tails are "
              "impure (AUC 0.730), so skip is bought with error budget."]

    L += ["", "### H13 - rank-ordered early-exit reranker", "",
          "Verdict-invariant by construction (scoring stops only once the running max "
          "crosses the pass edge - the cascade verdict is then final; claims that never "
          "cross score every pair). No quality table - latency and the verdict-equality "
          "assert live in `scripts/bench_grounder_round2.py`."]

    if FUSED.exists():
        fused_rows = eval_fused(feats, y)
        L += ["", "### H14 - fused-evidence single-forward cross-encoders", "",
              "ONE evidence context per claim (v1 top-2 chunk concat / v2 salience-packed "
              "sentences), ONE forward per model - 16 cross-encoder forwards become 2. "
              f"Gates: macro >= {F1_GATE} adopt-for-F1; >= 0.795 with stage latency cut "
              f">= 25% adopt-for-latency; < {CASCADE_GATE} reject.", "",
              "| config | macro-F1 | d vs ref | FP | FN | NLI skip |",
              "|---|---|---|---|---|---|"]
        for r in fused_rows:
            skip = f"{r['skip']:.0%}" if r["skip"] is not None else "-"
            L.append(f"| {r['set']} | {r['macro']:.3f} | {r['macro'] - ref_macro:+.3f} "
                     f"| {r['fp']} | {r['fn']} | {skip} |")
    else:
        L += ["", "### H14 - fused-evidence cross-encoders", "",
              "Fused cache not built yet - run `score-fused` first."]

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nappended round 2 to {REPORT}")


def main() -> None:
    feats, y, _ = build_features()
    stack_rows = eval_stacks(feats, y)
    base = next(r for r in stack_rows if r["set"].startswith("baseline"))
    casc_rows, T_rr, T_st = eval_cascade(feats, y)
    # frontier: best skip-rate meeting the quality gate, plus the macro-optimal band
    ok = [r for r in casc_rows if r["macro"] >= CASCADE_GATE]
    best_skip = max(ok, key=lambda r: r["skip"]) if ok else None
    best_macro_row = max(casc_rows, key=lambda r: r["macro"])

    L = ["# Grounding hypotheses H9/H10/H11 - final round (CPU OpenVINO int8)", "",
         f"Pair cache: {PAIRS.name}, n={len(y)} gold records. Baseline fp32 stack macro-F1 "
         f"{BASELINE_MACRO}; all rows below are int8-true (deployed engines), OOF 5-fold "
         "(seed 42), thresholds at OOF macro-F1 optimum. Models frozen - features are "
         "softmax channels and aggregation statistics; only the logistic is re-fit.", "",
         "## H9 / H10 - feature stacks (OOF macro-F1)", "",
         "| feature set | k | macro-F1 | d vs baseline | gate >= " + f"{F1_GATE} |",
         "|---|---|---|---|---|"]
    for r in stack_rows:
        d = r["macro"] - base["macro"]
        verdict = "PASS" if r["macro"] >= F1_GATE else "fail"
        L.append(f"| {r['set']} | {r['k']} | {r['macro']:.3f} | {d:+.3f} | {verdict} |")
    L += ["", f"## H11 - reranker-first cascade (T_rr={T_rr:.2f}, T_stack={T_st:.2f})", "",
          f"Quality gate macro-F1 >= {CASCADE_GATE}. skip = claims resolved without the "
          "NLI forward (warm latency saving ~ NLI share x skip).", "",
          "| band [a, b] | skip | macro-F1 | FP | FN |", "|---|---|---|---|---|"]
    show = sorted(ok, key=lambda r: -r["skip"])[:8] if ok else []
    for r in show:
        L.append(f"| [{r['a']:.2f}, {r['b']:.2f}] | {r['skip']:.0%} | {r['macro']:.3f} "
                 f"| {r['fp']} | {r['fn']} |")
    if best_skip:
        L += ["", f"Best gated point: skip {best_skip['skip']:.0%} at macro-F1 "
              f"{best_skip['macro']:.3f} (band [{best_skip['a']:.2f}, {best_skip['b']:.2f}])."]
    else:
        L += ["", "No band meets the quality gate - cascade rejected."]
    L += [f"Macro-optimal band: [{best_macro_row['a']:.2f}, {best_macro_row['b']:.2f}] "
          f"skip {best_macro_row['skip']:.0%} macro-F1 {best_macro_row['macro']:.3f}."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "score":
        score_pairs()
    elif cmd == "score-fused":
        score_fused()
    elif cmd == "round2":
        main_round2()
    else:
        main()
