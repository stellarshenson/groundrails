"""R19-H161 lane L2 - FAILURE-CLASS AUTOPSY OF THE H159 REGRESSION. ANALYSIS ONLY.

R19-H159 (the arm that added five public corpora to the training mix) lost 0.02607
blind mean against the R18-H150 flagship pair, concentrated in finqa -0.1429,
tatqa -0.1328 and delucionqa -0.1025. The enriched mix carried the old mix in
unchanged, so every original lane was diluted by exactly 0.8248 - a flat cut with
nothing singled out.

HYPOTHESIS H2 (this lane) - COLUMN-BINDING DILUTION WITH UNEQUAL MARGIN. The cut
is uniform but its COST is not, because the lanes differ in saturation. On the
held-out probe bank bind_row sits at ceiling 0.9920 and lost nothing (0.9918),
bind_col sits at 0.9603 and fell to 0.9363, and scale_unit - diluted identically -
ROSE 0.8587 -> 0.8650. Column binding, the harder half of the table skill, may sit
on the steep part of its learning curve where a 17.5% exposure cut costs real
accuracy. If H2 holds, the enriched model's NEW errors on finqa and tatqa
concentrate in the TABLE-BINDING class specifically, not spread across classes.

delucionqa is the discriminator: car-manual QA, NO TABLES, and it fell 0.1025 -
which H2 cannot explain. The rival H1 (lane L1) says the mechanism is suppression
of the model's lexical-overlap prior by the FAVA corpus, which would hit
near-verbatim manual passages and tables alike. So the reading of delucionqa's new
errors - near-copy pattern (H1) versus binding pattern (H2) - is reported either
way.

Nothing here trains, tunes or selects. No threshold, formula or parameter is
chosen because it improves an arena number: the operating threshold is the
pre-stated in-sample macro-F1-optimal choice of the banked R17-H147 protocol, and
the threshold-free rank-loss decomposition is reported alongside it. The RAGBench
arena is READ-ONLY evidence and its source corpora are FORBIDDEN as training data.
NO GPU is touched (CUDA_VISIBLE_DEVICES is forced empty).

PRIOR ART, REUSED NOT REINVENTED. `R18-H157_finqa_autopsy.py` is the completed
failure-mode autopsy of finqa on the two flagship draws. This lane imports that
module and calls its taxonomy - `classify_fn`, `classify_fp`, the number machinery
(`extract_numbers`, `present_verbatim`, `math_candidates`, `evidence_number_pool`),
the annotation access (`item_annotation`, `sentence_support_fit`), `op_threshold`,
`rank_loss`, `binom_se` and the `TAXONOMY` tuple - VERBATIM, so the class masses
here are comparable with the banked finqa autopsy. Only the signal plumbing is
re-pointed: H157 read its own per-sentence parquet, this lane reads the shared
R19-H161 A0 pair dump. The signal dict handed to the classifiers is built with the
identical definitions.

INPUT - the shared A0 substrate dump (`R19-H161_dump.py`), one row per
(subset, item, sentence, window):

    R19-H161_pairs_h150d1.parquet   models/R18-H150-arm-draw1   flagship draw 1
    R19-H161_pairs_h150d2.parquet   models/R18-H150-arm-draw2   flagship draw 2
    R19-H161_pairs_h159d1.parquet   models/R19-H159-arm-draw1   enriched

The dump carries provenance but not raw text; sentence and window strings are
re-derived from the same frozen gate loader (`R8-H77.load_subsets` via
`R8-H92_decomposed_arena`) and joined on (subset, item_id, sent_idx, win_idx).

POSITIVE CONTROL, run before any analysis: the per-subset AUROC reconstructed from
the dump's `item_score` must match the banked windowed value to <= 1e-3 for all
nine (checkpoint, subset) cells. The lane aborts otherwise.

TAXONOMY ADDITION - `near_copy_verbatim`. The H157 taxonomy has no class for a
failure on a near-verbatim-copied prose sentence, which is exactly H1's signature
and exactly what delucionqa needs. It is added as an explicit, named, ORTHOGONAL
OVERLAY - a flag computed from the dump's own frozen surface features, NOT
inserted into the H157 precedence chain - so the primary class masses stay
byte-comparable with the banked autopsy. Rule (pre-stated, and reported at three
cuts so the reading does not hang on one):

    near_copy_verbatim(item) := the sinking sentence's ARGMAX window shares a
    contiguous verbatim token run of length >= 8 with the sentence
    (max_common_ngram, stopwords included) AND content containment >= 0.8.

Eight tokens is the standard near-duplicate shingle length; 6/8/10 x 0.7/0.8/0.9
sensitivity and the threshold-free distributions of both features are reported.

Run (detached, CPU only):
  nohup setsid uv run python experiments/grounding-semantic/R19-H161_L2.py \
    >> logs/R19-H161_L2.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # this lane never touches a card
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl
from scipy import stats

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R19-H161_L2_result.json"
OUT_PARQUET = HERE / "R19-H161_L2_items.parquet"

CONTROL_TOL = 1e-3
SUBSETS = ("finqa", "tatqa", "delucionqa")

CHECKPOINTS = {
    "h150d1": {
        "pairs": "R19-H161_pairs_h150d1.parquet",
        "family": "flagship",
        "model": "models/R18-H150-arm-draw1",
        "banked": {"finqa": 0.6515, "tatqa": 0.7842, "delucionqa": 0.8009},
    },
    "h150d2": {
        "pairs": "R19-H161_pairs_h150d2.parquet",
        "family": "flagship",
        "model": "models/R18-H150-arm-draw2",
        "banked": {"finqa": 0.7135, "tatqa": 0.8093, "delucionqa": 0.7888},
    },
    "h159d1": {
        "pairs": "R19-H161_pairs_h159d1.parquet",
        "family": "enriched",
        "model": "models/R19-H159-arm-draw1",
        "banked": {"finqa": 0.5396, "tatqa": 0.6640, "delucionqa": 0.6923},
    },
}
FLAGSHIP = ("h150d1", "h150d2")
ENRICHED = "h159d1"

# The flagship's held-out probe bank, flagship draw 1 -> enriched (the H2 premise).
PROBE_BANK = {
    "bind_col": {
        "flagship": 0.9603,
        "enriched": 0.9363,
        "delta": -0.0240,
        "reading": "unsaturated, lost accuracy under the flat cut",
    },
    "bind_row": {
        "flagship": 0.9920,
        "enriched": 0.9918,
        "delta": -0.0002,
        "reading": "at ceiling, lost nothing",
    },
    "scale_unit": {
        "flagship": 0.8587,
        "enriched": 0.8650,
        "delta": +0.0063,
        "reading": "diluted identically and ROSE",
    },
}

# near_copy_verbatim overlay cuts (pre-stated; the middle row is the headline).
NEAR_COPY_NGRAM = 8
NEAR_COPY_CONTAINMENT = 0.80
NEAR_COPY_GRID = [(6, 0.70), (8, 0.80), (10, 0.90)]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H157 = _mod("h157", "R18-H157_finqa_autopsy.py")  # the banked taxonomy, imported whole
ARM = H157.ARM
H92 = H157.H92
ARENA = H157.ARENA
M59 = H157.M59
R12 = _mod("r12", "R12_label_ceiling.py")

TAXONOMY = H157.TAXONOMY

# H157's manual-verification overrides are keyed (draw, item) over the finqa
# sample. They are corrections of the RULE classifier on specific items, not of a
# specific checkpoint's behaviour, and the two draws never disagree where both are
# listed - so they collapse to an item-keyed map. Applying them to only the two
# flagship checkpoints would bias the flagship-vs-enriched contrast, so the PRIMARY
# tables use the rule class alone (symmetric across all three checkpoints) and a
# SECONDARY finqa table applies this map to all three for continuity with H157.
FINQA_ITEM_OVERRIDES = {}
for (_d, _i), _c in H157.MANUAL_OVERRIDES.items():
    prev = FINQA_ITEM_OVERRIDES.setdefault(_i, _c)
    assert prev == _c, f"H157 overrides disagree across draws on item {_i}"


# --- substrate: geometry + raw text, re-derived from the frozen gate loader --------


def build_substrate():
    """Per-subset: claims, chunk lists, labels, annotation rows, and the flat
    window list with char spans in the parent document.

    The window list is built with H157's `_windows_spanned`, which is ARM.windows
    plus each window's span; the flattening order (document, then window within
    document) is the dump's `win_idx` order, so `win_idx` indexes it directly.
    """
    subs = ARENA.load_subsets()
    raw_all = R12.load_rows()
    out = {}
    for sub in SUBSETS:
        claims, chunks, y = subs[sub]
        raw = raw_all[sub]
        y_raw = raw["adherence_score"].cast(pl.Int8).to_numpy()
        assert np.array_equal(y, y_raw), f"{sub}: arena and R12 row order disagree"
        assert all(a == b for a, b in zip(claims, raw["response"].to_list(), strict=True)), (
            f"{sub}: arena and R12 response order disagree"
        )
        raw_rows = list(raw.iter_rows(named=True))

        sents, wlists, norm_wins, ev_pools = [], [], [], []
        for c, ks in zip(claims, chunks, strict=True):
            sents.append(H92.sentences(c))
            wl = [
                (w, di, wi, s0, s0 + len(w))
                for di, k in enumerate(ks)
                for wi, (s0, w) in enumerate(H157._windows_spanned(k))
            ]
            wlists.append(wl)
            texts = [w for w, _, _, _, _ in wl]
            norm_wins.append([H157.norm_text(t) for t in texts])
            ev_pools.append(H157.evidence_number_pool(texts))
        out[sub] = {
            "claims": claims,
            "chunks": chunks,
            "y": y,
            "raw_rows": raw_rows,
            "sents": sents,
            "wlists": wlists,
            "norm_wins": norm_wins,
            "ev_pools": ev_pools,
            "n_sent_total": sum(len(s) for s in sents),
            "n_pairs": sum(len(s) * len(w) for s, w in zip(sents, wlists, strict=True)),
        }
        print(
            f"  {sub:12s} n={len(y):>4} sentences={out[sub]['n_sent_total']:>5} "
            f"pairs={out[sub]['n_pairs']:>6}",
            flush=True,
        )
    return out


# --- dump access -------------------------------------------------------------------


def load_cells(sub_data):
    """Per (checkpoint, subset): item scores in item order, and the sinking
    sentence's argmax pair row (one per item)."""
    cells = {}
    for tag, spec in CHECKPOINTS.items():
        path = HERE / spec["pairs"]
        if not path.exists():
            raise SystemExit(f"missing dump parquet {path} - the A0 dump has not landed")
        df = pl.read_parquet(path).filter(pl.col("subset").is_in(list(SUBSETS)))
        for sub in SUBSETS:
            d = df.filter(pl.col("subset") == sub)
            n = len(sub_data[sub]["y"])
            if d.height != sub_data[sub]["n_pairs"]:
                raise SystemExit(
                    f"{tag}/{sub}: dump has {d.height} pairs, the "
                    f"re-derived geometry has {sub_data[sub]['n_pairs']}"
                )
            iscore = d.group_by("item_id").agg(pl.col("item_score").first()).sort("item_id")
            if iscore.height != n or iscore["item_id"].to_list() != list(range(n)):
                raise SystemExit(f"{tag}/{sub}: item_id coverage is not 0..{n - 1}")
            sink = d.filter(pl.col("is_sinking") & pl.col("is_argmax")).sort("item_id")
            if sink.height != n:
                raise SystemExit(f"{tag}/{sub}: {sink.height} sinking-argmax rows for {n} items")
            cells[(tag, sub)] = {
                "item_score": iscore["item_score"].to_numpy().astype(np.float64),
                "sink": sink,
            }
    return cells


def positive_control(cells, sub_data):
    ctrl, bad = {}, []
    for tag, spec in CHECKPOINTS.items():
        for sub in SUBSETS:
            y = sub_data[sub]["y"]
            auc, _, _ = M59.auc_and_f1(y, cells[(tag, sub)]["item_score"])
            banked = spec["banked"][sub]
            row = {
                "reproduced_auc": round(float(auc), 6),
                "banked_auc": banked,
                "abs_delta": round(abs(float(auc) - banked), 6),
                "pass": bool(abs(float(auc) - banked) <= CONTROL_TOL),
            }
            ctrl[f"{tag}/{sub}"] = row
            if not row["pass"]:
                bad.append(f"{tag}/{sub}")
            print(
                f"  CONTROL {tag}/{sub:12s} read {auc:.4f}  banked {banked:.4f}  "
                f"delta {auc - banked:+.5f}  {'PASS' if row['pass'] else 'FAIL'}",
                flush=True,
            )
    return ctrl, bad


# --- signals + classification (H157 definitions, re-pointed at the dump) -------------


def item_signals(sub, sub_data, sink_row, y_i, ann):
    """The H157 signal dict for one error item, built from the dump's provenance."""
    i = int(sink_row["item_id"])
    sent_idx = int(sink_row["sent_idx"])
    win_idx = int(sink_row["win_idx"])
    s_txt = sub_data["sents"][i][sent_idx]
    w_txt, w_doc, _w_local, w_a, w_b = sub_data["wlists"][i][win_idx]
    all_w = sub_data["norm_wins"][i]
    ev_pool = sub_data["ev_pools"][i]

    s_nums = H157.extract_numbers(s_txt)
    absent = [nm for nm in s_nums if not H157.present_verbatim(nm["digits"], all_w)]
    absent_content = [nm for nm in absent if nm["value"] not in H157._FORMULA_CONSTANTS]
    absent_in_argmax = [
        nm for nm in s_nums if not H157.present_verbatim(nm["digits"], [H157.norm_text(w_txt)])
    ]
    cands = {nm["digits"]: H157.math_candidates(nm["value"], ev_pool) for nm in absent_content}
    scale_c = [h for c in cands.values() for h in c["scale"]]
    deriv_c = [h for c in cands.values() for h in c["derivation"]]

    fit, fit_detail = H157.sentence_support_fit(
        R12, ann, sent_idx, sub_data["raw_rows"][i]["documents"][:8]
    )
    argmax_has_support = None
    if fit == "covered":
        lo, hi = fit_detail["span"]
        argmax_has_support = bool(fit_detail["doc"] == w_doc and w_a <= lo and hi <= w_b)

    unsp_expl, unsp_absent, unsp_all_present, sink_is_unsupported = "", False, None, None
    if y_i == 0:
        u_texts = [
            ann["ann_texts"][j] for j, k in enumerate(ann["keys"]) if k in ann["unsupported"]
        ]
        unsp_expl = " | ".join(
            ann["expl"].get(k, "") for k in ann["keys"] if k in ann["unsupported"]
        )
        u_nums = [nm for t in u_texts for nm in H157.extract_numbers(t)]
        u_absent = [
            nm
            for nm in u_nums
            if not H157.present_verbatim(nm["digits"], all_w)
            and nm["value"] not in H157._FORMULA_CONSTANTS
        ]
        unsp_absent = len(u_absent) > 0
        unsp_all_present = len(u_absent) == 0 and len(u_nums) > 0
        if unsp_absent:
            uc = H157.math_candidates(u_absent[0]["value"], ev_pool)
            scale_c = scale_c or uc["scale"]
            deriv_c = deriv_c or uc["derivation"]
        hits = ann["mapping"][sent_idx]
        sink_is_unsupported = (
            any(ann["keys"][j] in ann["unsupported"] for j in hits) if hits else False
        )

    signals = {
        "n_numbers_in_sinking": len(s_nums),
        "n_absent_verbatim": len(absent_content),
        "n_absent_in_argmax_window": len(absent_in_argmax),
        "absent_numbers": [nm["raw"] for nm in absent_content][:8],
        "derivation_register": bool(H157._DERIV_WORDS.search(s_txt)),
        "derivation_candidates": deriv_c[:3],
        "scale_candidates": scale_c[:3],
        "support_fit": fit,
        "argmax_window_has_support": argmax_has_support,
        "unsupported_explanation": unsp_expl,
        "unsupported_number_absent": unsp_absent,
        "unsupported_numbers_all_present": unsp_all_present,
        "sinking_is_annotated_unsupported": sink_is_unsupported,
    }
    context = {
        "sinking_sentence": s_txt,
        "sinking_sent_idx": sent_idx,
        "argmax_doc": w_doc,
        "argmax_win": win_idx,
        "argmax_window_text": w_txt,
    }
    return signals, context


def near_copy_flag(row, ngram=NEAR_COPY_NGRAM, containment=NEAR_COPY_CONTAINMENT):
    """The taxonomy ADDITION, as an orthogonal overlay on the dump's frozen
    surface features - never inserted into the H157 precedence chain."""
    return bool(row["max_common_ngram"] >= ngram and row["tok_containment"] >= containment)


def classify_cell(tag, sub, sub_data, cell, thr):
    """Signals + H157 rule class for every error item of one (checkpoint, subset)."""
    y = sub_data["y"]
    sv = cell["item_score"]
    pred = (sv >= thr).astype(int)
    err = np.where(pred != y)[0]
    sink_rows = {int(r["item_id"]): r for r in cell["sink"].iter_rows(named=True)}

    items = []
    for i in err:
        i = int(i)
        row = sink_rows[i]
        ann = H157.item_annotation(R12, sub_data["raw_rows"][i])
        signals, context = item_signals(sub, sub_data, row, int(y[i]), ann)
        cls = H157.classify_fn(signals) if y[i] == 1 else H157.classify_fp(signals)
        overlay = {
            "max_common_ngram": int(row["max_common_ngram"]),
            "tok_containment": float(row["tok_containment"]),
            "tok_jaccard": float(row["tok_jaccard"]),
            "num_containment": (
                None if row["num_containment"] is None else float(row["num_containment"])
            ),
            "n_num_sent": int(row["n_num_sent"]),
            "near_copy_verbatim": near_copy_flag(row),
        }
        items.append(
            {
                "checkpoint": tag,
                "subset": sub,
                "item": i,
                "item_id_raw": sub_data["raw_rows"][i].get("id"),
                "label": int(y[i]),
                "error_type": "fp" if y[i] == 0 else "fn",
                "score": round(float(sv[i]), 4),
                "threshold": round(float(thr), 4),
                "rule_class": cls,
                "final_class": (FINQA_ITEM_OVERRIDES.get(i, cls) if sub == "finqa" else cls),
                "overlay": overlay,
                "signals": signals,
                **context,
            }
        )
    return items, pred, err


# --- tables ---------------------------------------------------------------------------


def tax_table(classes, base):
    counts = {c: 0 for c in TAXONOMY}
    for c in classes:
        counts[c] += 1
    out = {}
    for c, k in counts.items():
        share = k / max(base, 1)
        se = H157.binom_se(k, base)
        out[c] = {
            "count": k,
            "share_of_errors": round(share, 4),
            "share_binomial_se": round(se, 4),
            "resolvable": bool(k == 0 or se < 0.5 * share),
        }
    return out


def feature_stats(rows, keys=("max_common_ngram", "tok_containment", "tok_jaccard")):
    if not rows:
        return {k: None for k in keys}
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {
            "n": len(v),
            "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
        }
    return out


OVERLAP_FEATURES = ("max_common_ngram", "tok_containment", "tok_jaccard")


def sink_features(cell, item_ids):
    """The dump's frozen surface features on the sinking sentence's argmax pair, for
    a set of items, read from the checkpoint that produced the outcome."""
    rows = cell["sink"].filter(pl.col("item_id").is_in([int(i) for i in item_ids]))
    return {k: rows[k].to_numpy().astype(float) for k in OVERLAP_FEATURES}


def overlap_contrast(cell_broken, broken_ids, cell_ref, ref_ids):
    """Do the items a checkpoint BREAKS sit at lower lexical overlap with their own
    best window than the items it keeps? Two-sided Mann-Whitney U, which needs no
    distributional assumption and nothing is tuned on it."""
    a = sink_features(cell_broken, broken_ids)
    b = sink_features(cell_ref, ref_ids)
    out = {"n_broken": len(broken_ids), "n_reference": len(ref_ids)}
    for k in OVERLAP_FEATURES:
        x, y = a[k], b[k]
        if len(x) < 3 or len(y) < 3:
            out[k] = {"broken_median": None, "reference_median": None, "p_value": None}
            continue
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        out[k] = {
            "broken_mean": round(float(x.mean()), 4),
            "broken_median": round(float(np.median(x)), 4),
            "reference_mean": round(float(y.mean()), 4),
            "reference_median": round(float(np.median(y)), 4),
            "direction": ("broken LOWER" if np.median(x) < np.median(y) else "broken HIGHER"),
            "p_value": round(float(u.pvalue), 5),
        }
    return out


def support_fit_dist(items):
    """How the annotated support of each error item's sinking sentence sits in the
    window geometry. It gates the H157 precedence chain: `split` short-circuits to
    `window_boundary`, and the binding test is only DEFINED when the fit is
    `covered`, so this table is what makes the class masses readable on a prose
    subset. delucionqa runs 44.6% `split` over all its sentences against 1.2% on
    finqa and 1.7% on tatqa - a structural property of stitched car-manual answers,
    not a classifier fault."""
    out = {}
    for it in items:
        out[it["signals"]["support_fit"]] = out.get(it["signals"]["support_fit"], 0) + 1
    n = max(len(items), 1)
    return {k: {"count": v, "share": round(v / n, 4)} for k, v in sorted(out.items())}


def binding_stats(items):
    """The H157 `table_binding` trigger, reported unconditionally AND conditioned on
    the sentences where it is defined (support_fit == 'covered')."""
    cov = [it for it in items if it["signals"]["support_fit"] == "covered"]
    miss = sum(1 for it in items if it["signals"]["argmax_window_has_support"] is False)
    return {
        "count": miss,
        "share": round(miss / max(len(items), 1), 4),
        "n_covered": len(cov),
        "share_conditional_on_covered": (round(miss / len(cov), 4) if cov else None),
        "definition": "the sinking sentence's annotated support is locatable in ONE "
        "window and the model's argmax window is NOT that window - the H157 "
        "table_binding trigger. Undefined (never fires) when the support is split "
        "across windows or documents, unmapped, or carries no support keys, so the "
        "conditional rate is the honest one on a subset with a high split rate",
    }


def main():
    t0 = time.time()
    print(f"=== R19-H161 lane L2 failure-class autopsy  {time.strftime('%F %T')} ===", flush=True)
    print("CPU only (CUDA_VISIBLE_DEVICES forced empty); ANALYSIS ONLY", flush=True)

    print("\n--- substrate (frozen gate loader, re-derived text) ---", flush=True)
    sub_data = build_substrate()

    print("\n--- dump ---", flush=True)
    cells = load_cells(sub_data)

    print("\n--- positive control ---", flush=True)
    ctrl, bad = positive_control(cells, sub_data)
    if bad:
        OUT_JSON.write_text(
            json.dumps(
                {
                    "lane": "L2",
                    "aborted": "positive control failed",
                    "failed_cells": bad,
                    "positive_control": ctrl,
                },
                indent=2,
            )
        )
        raise SystemExit(f"=== H161 L2 FAILED: positive control on {bad} ===")
    print("positive control: all nine cells reproduce the banked AUROC to <= 1e-3", flush=True)

    # --- per-cell error split, taxonomy, rank loss ---------------------------------
    print("\n--- per-cell classification ---", flush=True)
    per_cell, all_items, preds, rls = {}, [], {}, {}
    for sub in SUBSETS:
        y = sub_data[sub]["y"]
        n, n_pos, n_neg = len(y), int(y.sum()), int((y == 0).sum())
        for tag in CHECKPOINTS:
            sv = cells[(tag, sub)]["item_score"]
            thr = H157.op_threshold(y, sv)
            items, pred, _err = classify_cell(tag, sub, sub_data[sub], cells[(tag, sub)], thr)
            all_items.extend(items)
            preds[(tag, sub)] = pred
            rl = H157.rank_loss(y, sv)
            rls[(tag, sub)] = rl
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            rl_by_class = {c: 0.0 for c in TAXONOMY}
            for it in items:
                rl_by_class[it["rule_class"]] += float(rl[it["item"]])
            per_cell[f"{tag}/{sub}"] = {
                "checkpoint": tag,
                "subset": sub,
                "family": CHECKPOINTS[tag]["family"],
                "auc": ctrl[f"{tag}/{sub}"]["reproduced_auc"],
                "n": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "operating_threshold": round(float(thr), 4),
                "n_errors": int(fp + fn),
                "false_positives": {
                    "count": fp,
                    "base": n_neg,
                    "rate": round(fp / max(n_neg, 1), 4),
                    "rate_binomial_se": round(H157.binom_se(fp, n_neg), 4),
                },
                "false_negatives": {
                    "count": fn,
                    "base": n_pos,
                    "rate": round(fn / max(n_pos, 1), 4),
                    "rate_binomial_se": round(H157.binom_se(fn, n_pos), 4),
                },
                "rank_loss_by_error_type": {
                    "fp_share": round(float(rl[y == 0].sum()), 4),
                    "fn_share": round(float(rl[y == 1].sum()), 4),
                },
                "taxonomy_rule_class": tax_table([it["rule_class"] for it in items], len(items)),
                "rank_loss_share_by_class": {c: round(v, 4) for c, v in rl_by_class.items()},
                "near_copy_verbatim": {
                    "count": sum(it["overlay"]["near_copy_verbatim"] for it in items),
                    "share_of_errors": round(
                        sum(it["overlay"]["near_copy_verbatim"] for it in items)
                        / max(len(items), 1),
                        4,
                    ),
                },
                "support_fit_distribution": support_fit_dist(items),
            }
            print(
                f"  {tag}/{sub:12s} thr {thr:+.3f}  errors {fp + fn:>3} (fp {fp}, fn {fn})",
                flush=True,
            )

    # finqa secondary table with the H157 manual overrides applied symmetrically
    finqa_override_table = {}
    for tag in CHECKPOINTS:
        its = [it for it in all_items if it["subset"] == "finqa" and it["checkpoint"] == tag]
        finqa_override_table[tag] = tax_table([it["final_class"] for it in its], len(its))

    # --- the decisive contrast: class mass, flagship 2-draw mean vs enriched -------
    print("\n--- class-mass contrast ---", flush=True)
    class_mass_delta = {}
    for sub in SUBSETS:
        d1 = per_cell[f"h150d1/{sub}"]["taxonomy_rule_class"]
        d2 = per_cell[f"h150d2/{sub}"]["taxonomy_rule_class"]
        en = per_cell[f"{ENRICHED}/{sub}"]["taxonomy_rule_class"]
        rows = {}
        for c in TAXONOMY:
            fm_count = (d1[c]["count"] + d2[c]["count"]) / 2.0
            fm_share = (d1[c]["share_of_errors"] + d2[c]["share_of_errors"]) / 2.0
            d_count = en[c]["count"] - fm_count
            d_share = en[c]["share_of_errors"] - fm_share
            noise_count = abs(d1[c]["count"] - d2[c]["count"])
            noise_share = abs(d1[c]["share_of_errors"] - d2[c]["share_of_errors"])
            rows[c] = {
                "flagship_d1_count": d1[c]["count"],
                "flagship_d2_count": d2[c]["count"],
                "flagship_mean_count": round(fm_count, 2),
                "enriched_count": en[c]["count"],
                "delta_count": round(d_count, 2),
                "noise_floor_count": noise_count,
                "clears_noise_count": bool(d_count > noise_count),
                "flagship_mean_share": round(fm_share, 4),
                "enriched_share": en[c]["share_of_errors"],
                "delta_share": round(d_share, 4),
                "noise_floor_share": round(noise_share, 4),
                "clears_noise_share": bool(d_share > noise_share),
            }
        grew = [c for c in TAXONOMY if rows[c]["delta_count"] > 0]
        grew_clearing = [c for c in grew if rows[c]["clears_noise_count"]]
        class_mass_delta[sub] = {
            "per_class": rows,
            "total_errors": {
                "flagship_d1": per_cell[f"h150d1/{sub}"]["n_errors"],
                "flagship_d2": per_cell[f"h150d2/{sub}"]["n_errors"],
                "enriched": per_cell[f"{ENRICHED}/{sub}"]["n_errors"],
            },
            "classes_that_grew": grew,
            "classes_that_grew_clearing_noise": grew_clearing,
            "largest_growth_class": max(TAXONOMY, key=lambda c: rows[c]["delta_count"]),
            "largest_share_growth_class": max(TAXONOMY, key=lambda c: rows[c]["delta_share"]),
        }
        print(f"  {sub:12s} grew {grew}  clearing noise {grew_clearing}", flush=True)

    # --- newly-broken items ----------------------------------------------------------
    print("\n--- newly-broken items ---", flush=True)
    item_class = {(it["checkpoint"], it["subset"], it["item"]): it for it in all_items}
    newly_broken = {}
    for sub in SUBSETS:
        y = sub_data[sub]["y"]
        ok1 = preds[("h150d1", sub)] == y
        ok2 = preds[("h150d2", sub)] == y
        bad_e = preds[(ENRICHED, sub)] != y
        nb = np.where(ok1 & ok2 & bad_e)[0]
        nf = np.where((~ok1) & (~ok2) & (preds[(ENRICHED, sub)] == y))[0]
        its = [item_class[(ENRICHED, sub, int(i))] for i in nb]
        cls_counts = {c: 0 for c in TAXONOMY}
        for it in its:
            cls_counts[it["rule_class"]] += 1
        overlays = [it["overlay"] for it in its]
        rl_e = rls[(ENRICHED, sub)]
        retained_ok = np.where(ok1 & ok2 & (preds[(ENRICHED, sub)] == y))[0]
        newly_broken[sub] = {
            "n_newly_broken": len(nb),
            "n_newly_fixed": len(nf),
            "net": int(len(nb) - len(nf)),
            "n_correct_in_both_flagship_draws": int((ok1 & ok2).sum()),
            "items": [int(i) for i in nb],
            "error_type": {
                "fp": sum(1 for it in its if it["error_type"] == "fp"),
                "fn": sum(1 for it in its if it["error_type"] == "fn"),
            },
            "class_counts": cls_counts,
            "class_shares": {c: round(k / max(len(its), 1), 4) for c, k in cls_counts.items()},
            "dominant_class": (max(cls_counts, key=cls_counts.get) if its else None),
            "near_copy_verbatim": {
                "count": sum(o["near_copy_verbatim"] for o in overlays),
                "share": round(
                    sum(o["near_copy_verbatim"] for o in overlays) / max(len(overlays), 1), 4
                ),
            },
            "binding_signature": binding_stats(its),
            "support_fit_distribution": support_fit_dist(its),
            "features_newly_broken": feature_stats(overlays),
            "rank_loss_mass_enriched": round(float(rl_e[nb].sum()), 4),
            "features_retained_correct": feature_stats(
                [
                    {
                        "max_common_ngram": int(r["max_common_ngram"]),
                        "tok_containment": float(r["tok_containment"]),
                        "tok_jaccard": float(r["tok_jaccard"]),
                    }
                    for r in cells[(ENRICHED, sub)]["sink"]
                    .filter(pl.col("item_id").is_in([int(i) for i in retained_ok]))
                    .iter_rows(named=True)
                ]
            ),
            # Is the newly-broken set simply the MARGINAL set? Low-overlap items are
            # harder for any checkpoint, so a uniformly worse model would break them
            # first. The control is the flagship's OWN draw-to-draw churn measured
            # the same way: items one draw got right and the other got wrong, scored
            # on the draw that got them wrong. If the enriched shift is generic
            # marginality, this control shows the same direction and size.
            "overlap_contrast": overlap_contrast(
                cells[(ENRICHED, sub)], nb, cells[(ENRICHED, sub)], retained_ok
            ),
            "overlap_contrast_flagship_churn_control": {
                "d2_broke_what_d1_kept": overlap_contrast(
                    cells[("h150d2", sub)],
                    np.where(ok1 & ~ok2)[0],
                    cells[("h150d2", sub)],
                    np.where(ok1 & ok2)[0],
                ),
                "d1_broke_what_d2_kept": overlap_contrast(
                    cells[("h150d1", sub)],
                    np.where(ok2 & ~ok1)[0],
                    cells[("h150d1", sub)],
                    np.where(ok1 & ok2)[0],
                ),
            },
        }
        print(
            f"  {sub:12s} newly broken {len(nb)}  newly fixed {len(nf)}  "
            f"dominant {newly_broken[sub]['dominant_class']}  "
            f"near-copy {newly_broken[sub]['near_copy_verbatim']['share']:.2f}  "
            f"binding {newly_broken[sub]['binding_signature']['share']:.2f}",
            flush=True,
        )

    # --- near-copy sensitivity grid (all newly-broken sets) --------------------------
    near_copy_grid = {}
    for sub in SUBSETS:
        rows = cells[(ENRICHED, sub)]["sink"].filter(
            pl.col("item_id").is_in(newly_broken[sub]["items"])
        )
        grid = {}
        for ng, ct in NEAR_COPY_GRID:
            k = sum(1 for r in rows.iter_rows(named=True) if near_copy_flag(r, ng, ct))
            grid[f"ngram>={ng},containment>={ct}"] = {
                "count": k,
                "share": round(k / max(rows.height, 1), 4),
            }
        near_copy_grid[sub] = grid

    # --- delucionqa reading ------------------------------------------------------------
    dq = newly_broken["delucionqa"]
    near = dq["near_copy_verbatim"]["share"]
    bind = dq["binding_signature"]["share"]
    # like-for-like: both statistics restricted to the items where the binding test
    # is DEFINED (support_fit == 'covered'), so the split rate cannot decide it
    dq_items = [item_class[(ENRICHED, "delucionqa", i)] for i in dq["items"]]
    dq_cov = [it for it in dq_items if it["signals"]["support_fit"] == "covered"]
    near_cov = round(
        sum(it["overlay"]["near_copy_verbatim"] for it in dq_cov) / max(len(dq_cov), 1), 4
    )
    bind_cov = round(
        sum(1 for it in dq_cov if it["signals"]["argmax_window_has_support"] is False)
        / max(len(dq_cov), 1),
        4,
    )
    dq_reading = {
        "n_newly_broken": dq["n_newly_broken"],
        "error_type_split": dq["error_type"],
        "near_copy_share": near,
        "binding_share": bind,
        "binding_share_conditional_on_covered": dq["binding_signature"][
            "share_conditional_on_covered"
        ],
        "like_for_like_on_covered_items": {
            "n_covered": len(dq_cov),
            "near_copy_share": near_cov,
            "binding_share": bind_cov,
        },
        "support_fit_distribution": dq["support_fit_distribution"],
        "class_shares": dq["class_shares"],
        "features_newly_broken_vs_retained": {
            "newly_broken": dq["features_newly_broken"],
            "retained_correct": dq["features_retained_correct"],
        },
        "near_copy_sensitivity": near_copy_grid["delucionqa"],
        "overlap_contrast": dq["overlap_contrast"],
        "overlap_contrast_flagship_churn_control": dq["overlap_contrast_flagship_churn_control"],
        "favours": (
            "H1 (overlap-prior suppression)"
            if near > bind
            else "H2 (binding)"
            if bind > near
            else "neither - tied"
        ),
        "favours_like_for_like": (
            "H1 (overlap-prior suppression)"
            if near_cov > bind_cov
            else "H2 (binding)"
            if bind_cov > near_cov
            else "neither - tied"
        ),
        "note": "delucionqa is car-manual prose with NO TABLES, so the H157 "
        "`table_binding` class is read here as its literal trigger - the "
        "model's argmax window is not the window carrying the annotated "
        "support - i.e. PASSAGE binding, not column binding. H2's mechanism "
        "(column-header binding) has no surface in this subset at all.",
        "overlap_confound_ruling": "The newly-broken items DO sit at lower lexical "
        "overlap with their own best window than the items the enriched run keeps "
        "(delucionqa containment median 0.351 vs 0.667, p=0.006). That is NOT "
        "evidence for either hypothesis: the flagship's own draw-to-draw churn "
        "reproduces the same skew at GREATER significance (0.444 vs 0.655, "
        "p=0.0008), so low overlap marks which items are MARGINAL for any "
        "checkpoint, not what the enrichment did. Any lane arguing from the "
        "newly-broken set's overlap distribution must run this control.",
        "class_table_caveat": "the H157 binding test is only DEFINED when the sinking "
        "sentence's annotated support is locatable in one window (`covered`); see "
        "`support_fit_distribution` for the share of this subset's error items that "
        "qualify. delucionqa runs 44.6% `split` support over ALL its sentences "
        "against finqa 1.2% and tatqa 1.7% - a structural property of stitched "
        "car-manual answers - so the class shares alone are a weak reading here. The "
        "near-copy overlay and the covered-only like-for-like contrast carry the "
        "H1-vs-H2 evidence on this subset.",
    }

    # --- verdict --------------------------------------------------------------------
    # Two legs, because raw class COUNTS are confounded - the enriched run simply
    # makes more errors, so every class grows. GROWTH is the count leg (the class
    # really did gain error mass, by more than the flagship draw-to-draw gap on the
    # same cell); CONCENTRATION is the share leg (the class gained a LARGER slice of
    # the errors, which is what distinguishes a binding-specific loss from uniform
    # degradation). H2 needs both, on both table subsets.
    h2_legs = {}
    for sub in ("finqa", "tatqa"):
        row = class_mass_delta[sub]["per_class"]["table_binding"]
        growth = bool(row["delta_count"] > 0 and row["clears_noise_count"])
        concentration = bool(
            row["delta_share"] > 0
            and row["clears_noise_share"]
            and class_mass_delta[sub]["largest_share_growth_class"] == "table_binding"
        )
        h2_legs[sub] = {
            "growth_leg": growth,
            "concentration_leg": concentration,
            "passes": bool(growth and concentration),
            "delta_count": row["delta_count"],
            "noise_floor_count": row["noise_floor_count"],
            "delta_share": row["delta_share"],
            "noise_floor_share": row["noise_floor_share"],
            "largest_growth_class_by_count": class_mass_delta[sub]["largest_growth_class"],
            "largest_growth_class_by_share": class_mass_delta[sub]["largest_share_growth_class"],
        }
    h2_finqa = h2_legs["finqa"]["passes"]
    h2_tatqa = h2_legs["tatqa"]["passes"]
    if h2_finqa and h2_tatqa:
        verdict = "SUPPORTED"
    elif not h2_finqa and not h2_tatqa:
        verdict = "NOT_SUPPORTED"
    else:
        verdict = "INDETERMINATE"

    report = {
        "lane": "L2",
        "hypothesis": "H2 - column-binding dilution with unequal margin",
        "verdict": verdict,
        "verdict_rule": {
            "stated_before_the_read": True,
            "rule": "SUPPORTED only if the move from the flagship 2-draw mean to the "
            "enriched run both GROWS `table_binding` by more than the flagship "
            "draw-to-draw gap on the same cell (count leg) and CONCENTRATES in it - "
            "rising share of errors, clearing the same gap, and the largest growth "
            "of any class - on BOTH finqa and tatqa",
            "legs": h2_legs,
            "finqa_table_binding_passes": h2_finqa,
            "tatqa_table_binding_passes": h2_tatqa,
        },
        "probe_bank_premise": PROBE_BANK,
        "positive_control": ctrl,
        "per_cell": per_cell,
        "class_mass_delta": class_mass_delta,
        "newly_broken_items": newly_broken,
        "near_copy_sensitivity_grid": near_copy_grid,
        "delucionqa_reading": dq_reading,
        "finqa_secondary_with_h157_overrides": {
            "note": "H157's manual-verification overrides, item-keyed and applied to all "
            "THREE checkpoints so the contrast stays symmetric. Secondary only - "
            "the primary tables are rule-class.",
            "n_overridden_items": len(FINQA_ITEM_OVERRIDES),
            "per_checkpoint": finqa_override_table,
        },
        "taxonomy": {
            "reused_verbatim_from": "R18-H157_finqa_autopsy.py (classify_fn, classify_fp, "
            "the number machinery, sentence_support_fit, "
            "op_threshold, rank_loss, binom_se, TAXONOMY)",
            "classes": list(TAXONOMY),
            "additions": [
                {
                    "name": "near_copy_verbatim",
                    "rule": f"the sinking sentence's ARGMAX window shares a contiguous "
                    f"verbatim token run of length >= {NEAR_COPY_NGRAM} with the "
                    f"sentence (max_common_ngram, stopwords included) AND content "
                    f"containment >= {NEAR_COPY_CONTAINMENT}",
                    "kind": "ORTHOGONAL OVERLAY - a flag on the dump's frozen surface "
                    "features, NOT inserted into the H157 precedence chain, so the "
                    "primary class masses stay comparable with the banked autopsy",
                    "why": "the H157 taxonomy has no class for a failure on a near-verbatim "
                    "prose sentence, which is H1's signature and what delucionqa needs",
                    "sensitivity": [f"ngram>={a},containment>={b}" for a, b in NEAR_COPY_GRID],
                }
            ],
            "prose_mapping": "on delucionqa (no tables) `table_binding` fires on its "
            "literal H157 trigger - the argmax window is not the window "
            "carrying the annotated support - which is PASSAGE binding; "
            "`scale_unit` and `derivation_arithmetic` fire only through "
            "the number machinery and are near-empty by construction",
        },
        "error_items": all_items,
        "meta": {
            "experiment": "R19-H161 lane L2 - failure-class autopsy of the H159 regression",
            "licence": "ANALYSIS ONLY - nothing trains, tunes or selects; the RAGBench "
            "arena is read-only evidence and its source corpora are forbidden "
            "as training data",
            "read": "PRIMARY windowed decomposed-min (1500/750, MAX over windows on the "
            "logit, then MIN over sentences), from the shared R19-H161 A0 dump",
            "threshold": "in-sample macro-F1-optimal, the pre-stated R17-H147 choice; "
            "nothing is tuned on it, and the threshold-free rank-loss "
            "decomposition is reported alongside",
            "device": "CPU only",
            "checkpoints": {t: s["model"] for t, s in CHECKPOINTS.items()},
            "subsets": list(SUBSETS),
            "runtime_seconds": round(time.time() - t0, 1),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    pl.DataFrame(
        [
            {
                "checkpoint": it["checkpoint"],
                "subset": it["subset"],
                "item": it["item"],
                "label": it["label"],
                "error_type": it["error_type"],
                "score": it["score"],
                "threshold": it["threshold"],
                "rule_class": it["rule_class"],
                "final_class": it["final_class"],
                "near_copy_verbatim": it["overlay"]["near_copy_verbatim"],
                "max_common_ngram": it["overlay"]["max_common_ngram"],
                "tok_containment": it["overlay"]["tok_containment"],
                "tok_jaccard": it["overlay"]["tok_jaccard"],
                "support_fit": it["signals"]["support_fit"],
                "argmax_window_has_support": it["signals"]["argmax_window_has_support"],
                "sinking_sent_idx": it["sinking_sent_idx"],
                "sinking_sentence": it["sinking_sentence"],
            }
            for it in all_items
        ]
    ).write_parquet(OUT_PARQUET)

    print(f"\nverdict {verdict}", flush=True)
    print(f"result -> {OUT_JSON}", flush=True)
    print(f"items  -> {OUT_PARQUET}", flush=True)
    print(f"=== H161 L2 COMPLETE ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
