"""R19-H162 - HOTPOTQA MECHANISM DISSECTION. ANALYSIS ONLY. CPU ONLY.

Executor M3 of the R19-H162 mechanism-dissection wave. hotpotqa is multi-hop
question answering (answer requires combining facts across two or more retrieved
documents) and the flagship's third-lowest arena subset (2-draw AUROC 0.6706).

The brief's structural question: the serving read scores ONE sentence against
ONE window and takes the MAX over windows, so a claim supported only by the
CONJUNCTION of window A and window B can never score high. The R16-H140 arm
replaced the max with a learned attention readout over window embeddings
specifically to enable composition, and hotpotqa was the WORST mover (-0.052,
seed-replicated -0.0427 +/- over 4 seeds). This script decides whether the
bottleneck is composition at all.

Nothing here trains. No GPU. Model scores are read from the banked R19-H161
per-pair dump (`R19-H161_pairs_h150d1.parquet`, control-verified bit-exact
against the banked windowed read). Text comes from the frozen gate sample via
the banked loader (`R8-H77.load_subsets` -> `R8-H92.sentences`), CPU-only.

Stages:
  1. structural fingerprint + positive control against the H161 dump
  2. hop census: per claim-sentence, how many DOCUMENTS are needed to cover the
     sentence's lexical anchors (greedy set cover), plus a containment-based
     restatement classifier that separates "one doc carries the whole sentence"
     from "the sentence spans a bridge"
  3. partial-support saturation: positive vs negative separation of the max
     window logit, split by hop class
  4. pooling counterfactual: recompute the hotpotqa AUROC under mean / logsumexp
     / softmax-attention pooling over the SAME per-window logits, to reproduce
     the H140 readout kill deterministically and read its mechanism
  5. argmax provenance: does the winning window come from the doc that carries
     the most of the sentence
  6. exemplar dump for manual reading

Run:  uv run python experiments/grounding-semantic/R19-H162_hotpotqa_probe.py
"""

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R19-H162_hotpotqa_probe.json"
OUT_SENT = HERE / "R19-H162_hotpotqa_sentences.parquet"
OUT_EYEBALL = HERE / "R19-H162_hotpotqa_eyeball.md"
DUMP = HERE / "R19-H161_pairs_h150d1.parquet"

SUBSET = "hotpotqa"
# Structural fingerprint of the banked H150 draw-1 windowed read on hotpotqa
# (R19-H161 dump: 250 items, 293 sentences, 1,177 pairs, AUROC 0.6766).
FINGERPRINT = {"n": 250, "n_sent": 293, "n_pairs": 1177}
BANKED_AUC_D1 = 0.6766
FLAGSHIP_2DRAW = 0.6706


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- anchors: verbatim from R16-H140_G0_census so the hop census is comparable

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "along",
    "already",
    "also",
    "although",
    "always",
    "among",
    "another",
    "back",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "cannot",
    "come",
    "could",
    "does",
    "doing",
    "done",
    "down",
    "during",
    "each",
    "either",
    "else",
    "enough",
    "even",
    "ever",
    "every",
    "from",
    "further",
    "give",
    "goes",
    "gone",
    "have",
    "having",
    "here",
    "hers",
    "herself",
    "himself",
    "however",
    "into",
    "itself",
    "just",
    "keep",
    "know",
    "less",
    "like",
    "made",
    "make",
    "many",
    "may",
    "maybe",
    "might",
    "more",
    "most",
    "much",
    "must",
    "near",
    "need",
    "never",
    "next",
    "none",
    "only",
    "onto",
    "other",
    "others",
    "ours",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "rather",
    "same",
    "seem",
    "shall",
    "should",
    "since",
    "some",
    "still",
    "such",
    "sure",
    "take",
    "than",
    "that",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "thing",
    "things",
    "this",
    "those",
    "though",
    "through",
    "thus",
    "together",
    "toward",
    "towards",
    "under",
    "unless",
    "until",
    "upon",
    "used",
    "using",
    "very",
    "want",
    "well",
    "were",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "whom",
    "whose",
    "will",
    "with",
    "within",
    "without",
    "would",
    "your",
    "yours",
    "yourself",
}

_WORD = re.compile(r"[a-z][a-z\-']{3,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def anchors_of(sentence):
    low = sentence.lower()
    words = {w for w in _WORD.findall(low) if w not in STOPWORDS}
    nums = set(_NUM.findall(low))
    return sorted(words | nums)


def anchor_forms(a):
    """Surface forms an anchor may take in a document."""
    forms = {a}
    if "," in a:
        forms.add(a.replace(",", ""))
    return forms


def doc_has(low_doc, anchor):
    return any(f in low_doc for f in anchor_forms(anchor))


def greedy_cover(anchor_hits, n_docs):
    """Minimum number of docs (greedy) to cover every MATCHED anchor.

    anchor_hits: list over anchors of the set of doc indices containing it.
    Anchors matched by NO doc are dropped (unmatched anchors carry no evidence).
    Returns (n_docs_needed, covering_doc_indices, n_matched_anchors).
    """
    matched = [h for h in anchor_hits if h]
    if not matched:
        return 0, [], 0
    remaining = set(range(len(matched)))
    chosen = []
    while remaining:
        best, best_gain = None, -1
        for d in range(n_docs):
            gain = sum(1 for i in remaining if d in matched[i])
            if gain > best_gain:
                best, best_gain = d, gain
        if best_gain <= 0:
            break
        chosen.append(best)
        remaining = {i for i in remaining if best not in matched[i]}
    return len(chosen), chosen, len(matched)


def containment(anchors, low_doc):
    """Fraction of the sentence's anchors present in this document."""
    if not anchors:
        return 0.0
    return sum(1 for a in anchors if doc_has(low_doc, a)) / len(anchors)


# --- pooling counterfactuals ---------------------------------------------------


def pool_variants(logits):
    """Aggregations over one sentence's per-window logits."""
    lg = np.asarray(logits, dtype=np.float64)
    out = {"max": float(lg.max()), "mean": float(lg.mean())}
    for t in (0.5, 1.0, 2.0, 4.0):
        m = lg.max()
        out[f"lse_t{t}"] = float(m + t * np.log(np.exp((lg - m) / t).sum()))
        w = np.exp((lg - m) / t)
        w = w / w.sum()
        out[f"softmax_t{t}"] = float((w * lg).sum())
    out["top2mean"] = float(np.sort(lg)[-2:].mean()) if lg.size >= 2 else out["max"]
    return out


def boot_auc_ci(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, pos.size, replace=True)
        n = rng.choice(neg, neg.size, replace=True)
        idx = np.concatenate([p, n])
        vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    arena = _mod("arena", "R8-H77_unseen_arena.py")
    h92 = _mod("h92", "R8-H92_decomposed_arena.py")

    subs = arena.load_subsets()
    claims, chunk_lists, y = subs[SUBSET]
    print(f"loaded {SUBSET}: {len(y)} items, {int(y.sum())} positive", flush=True)

    dump = pl.read_parquet(DUMP).filter(pl.col("subset") == SUBSET)
    fp = {
        "n": len(y),
        "n_sent": dump.select(["item_id", "sent_idx"]).unique().height,
        "n_pairs": dump.height,
    }
    print(f"fingerprint {fp} vs banked {FINGERPRINT}", flush=True)
    if fp != FINGERPRINT:
        raise SystemExit(f"FINGERPRINT ABORT: {fp} != {FINGERPRINT}")

    # positive control: reconstruct the banked read from the dump
    sent = (
        dump.group_by(["item_id", "sent_idx"])
        .agg(pl.col("logit").max().alias("smax"), pl.col("label").first())
        .sort(["item_id", "sent_idx"])
    )
    item = (
        sent.group_by("item_id")
        .agg(pl.col("smax").min().alias("iscore"), pl.col("label").first())
        .sort("item_id")
    )
    auc_ctl = roc_auc_score(item["label"].to_numpy(), item["iscore"].to_numpy())
    print(f"CONTROL max-pool AUROC {auc_ctl:.4f} vs banked {BANKED_AUC_D1:.4f}", flush=True)
    if abs(auc_ctl - BANKED_AUC_D1) > 1e-3:
        raise SystemExit("CONTROL ABORT")
    lo, hi = boot_auc_ci(item["label"].to_numpy(), item["iscore"].to_numpy())
    print(
        f"  bootstrap 95% CI [{lo:.4f}, {hi:.4f}]  (n_neg={int((1 - item['label'].to_numpy()).sum())})",
        flush=True,
    )

    # --- per-sentence census ---------------------------------------------------
    rows = []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        lows = [k.lower() for k in ks]
        for si, s in enumerate(h92.sentences(c)):
            anc = anchors_of(s)
            hits = [{d for d in range(len(ks)) if doc_has(lows[d], a)} for a in anc]
            n_need, cover, n_matched = greedy_cover(hits, len(ks))
            conts = [containment(anc, ld) for ld in lows]
            order = np.argsort(conts)[::-1]
            best_doc = int(order[0]) if conts else -1
            top1 = float(conts[order[0]]) if conts else 0.0
            top2 = float(conts[order[1]]) if len(conts) > 1 else 0.0
            # union containment of the best two docs
            if len(lows) > 1:
                u = sum(1 for a in anc if doc_has(lows[order[0]], a) or doc_has(lows[order[1]], a))
                union2 = u / len(anc) if anc else 0.0
            else:
                union2 = top1
            rows.append(
                {
                    "item_id": i,
                    "sent_idx": si,
                    "label": int(y[i]),
                    "sent": s,
                    "n_docs": len(ks),
                    "n_anchors": len(anc),
                    "n_matched_anchors": n_matched,
                    "docs_needed": n_need,
                    "cover_docs": json.dumps(sorted(cover)),
                    "best_doc": best_doc,
                    "cont_top1": top1,
                    "cont_top2": top2,
                    "cont_union2": union2,
                    "cont_gain2": union2 - top1,
                }
            )
    cen = pl.DataFrame(rows)

    # join model provenance
    argm = (
        dump.filter(pl.col("is_argmax"))
        .group_by(["item_id", "sent_idx"])
        .agg(
            pl.col("doc_idx").first().alias("argmax_doc"),
            pl.col("logit").first().alias("argmax_logit"),
            pl.col("tok_containment").first().alias("argmax_tok_cont"),
        )
    )
    smax = dump.group_by(["item_id", "sent_idx"]).agg(
        pl.col("logit").max().alias("smax"),
        pl.col("logit").mean().alias("smean"),
        pl.col("logit").std().alias("sstd"),
        pl.col("logit").len().alias("n_win"),
        pl.col("is_sinking").any().alias("is_sinking"),
    )
    cen = cen.join(argm, on=["item_id", "sent_idx"], how="left").join(
        smax, on=["item_id", "sent_idx"], how="left"
    )
    # margin between best and second-best window logit
    win = dump.select(["item_id", "sent_idx", "logit"]).sort(
        ["item_id", "sent_idx", "logit"], descending=[False, False, True]
    )
    margin = (
        win.group_by(["item_id", "sent_idx"], maintain_order=True)
        .agg(pl.col("logit").head(2).alias("top2"))
        .with_columns(
            (pl.col("top2").list.get(0) - pl.col("top2").list.get(1)).alias("win_margin")
        )
        .drop("top2")
    )
    cen = cen.join(margin, on=["item_id", "sent_idx"], how="left")
    cen = cen.with_columns(
        (pl.col("argmax_doc") == pl.col("best_doc")).alias("argmax_on_best_doc"),
        pl.when(pl.col("docs_needed") >= 2)
        .then(pl.lit("multi_doc"))
        .when(pl.col("docs_needed") == 1)
        .then(pl.lit("single_doc"))
        .otherwise(pl.lit("unanchored"))
        .alias("hop_class"),
    )
    cen.write_parquet(OUT_SENT)

    res = {
        "subset": SUBSET,
        "flagship_2draw_auroc": FLAGSHIP_2DRAW,
        "draw1_auroc": round(auc_ctl, 5),
        "draw1_auroc_ci95": [round(lo, 4), round(hi, 4)],
        "n_items": len(y),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_sentences": cen.height,
        "fingerprint": fp,
    }

    # --- hop census -----------------------------------------------------------
    hc = cen.group_by("hop_class").agg(pl.len().alias("n")).sort("n", descending=True)
    res["hop_census"] = {r["hop_class"]: int(r["n"]) for r in hc.to_dicts()}
    n_s = cen.height
    res["hop_census_pct"] = {k: round(100.0 * v / n_s, 2) for k, v in res["hop_census"].items()}
    res["docs_needed_hist"] = {
        str(r["docs_needed"]): int(r["n"])
        for r in cen.group_by("docs_needed")
        .agg(pl.len().alias("n"))
        .sort("docs_needed")
        .to_dicts()
    }

    # containment view: how much of the sentence does the BEST single doc carry
    for cut in (0.7, 0.8, 0.9, 1.0):
        res[f"share_cont_top1_ge_{cut}"] = round(float((cen["cont_top1"] >= cut).mean()), 4)
    res["cont_top1_mean"] = round(float(cen["cont_top1"].mean()), 4)
    res["cont_union2_mean"] = round(float(cen["cont_union2"].mean()), 4)
    res["cont_gain2_mean"] = round(float(cen["cont_gain2"].mean()), 4)
    # the decisive split: multi-doc by cover, but one doc still carries most of it
    res["multi_doc_but_top1_ge_0.8"] = int(
        cen.filter((pl.col("hop_class") == "multi_doc") & (pl.col("cont_top1") >= 0.8)).height
    )
    res["multi_doc_and_top1_lt_0.6"] = int(
        cen.filter((pl.col("hop_class") == "multi_doc") & (pl.col("cont_top1") < 0.6)).height
    )

    # --- partial-support saturation -------------------------------------------
    sat = {}
    for cls in ("single_doc", "multi_doc"):
        sub = cen.filter(pl.col("hop_class") == cls)
        p = sub.filter(pl.col("label") == 1)["smax"].to_numpy()
        n = sub.filter(pl.col("label") == 0)["smax"].to_numpy()
        sat[cls] = {
            "n_sent": int(sub.height),
            "n_pos_sent": int(p.size),
            "n_neg_sent": int(n.size),
            "smax_pos_mean": round(float(p.mean()), 4) if p.size else None,
            "smax_neg_mean": round(float(n.mean()), 4) if n.size else None,
            "smax_gap": round(float(p.mean() - n.mean()), 4) if p.size and n.size else None,
            "smax_pos_sd": round(float(p.std()), 4) if p.size else None,
        }
        if p.size and n.size and n.size >= 3:
            yy = np.concatenate([np.ones(p.size), np.zeros(n.size)])
            ss = np.concatenate([p, n])
            sat[cls]["sent_auroc"] = round(float(roc_auc_score(yy, ss)), 4)
    res["saturation_by_hop"] = sat

    # item-level AUROC restricted to items whose sentences are all single-doc / any multi-doc
    itcls = cen.group_by("item_id").agg(
        (pl.col("hop_class") == "multi_doc").any().alias("any_multi"),
        pl.col("label").first(),
    )
    it = item.join(itcls, on="item_id")
    for tag, flt in (
        ("items_any_multi_doc", pl.col("any_multi")),
        ("items_all_single_doc", ~pl.col("any_multi")),
    ):
        s = it.filter(flt)
        yv, sv = s["label"].to_numpy(), s["iscore"].to_numpy()
        res[tag] = {
            "n": int(s.height),
            "n_neg": int((1 - yv).sum()),
            "auroc": round(float(roc_auc_score(yv, sv)), 4)
            if len(set(yv.tolist())) == 2 and (1 - yv).sum() >= 3
            else None,
        }

    # --- pooling counterfactual ------------------------------------------------
    per_sent = (
        dump.group_by(["item_id", "sent_idx"])
        .agg(pl.col("logit").alias("lgs"), pl.col("label").first())
        .sort(["item_id", "sent_idx"])
    )
    variants = {}
    names = list(pool_variants([0.0, 1.0]).keys())
    for nm in names:
        vals = [pool_variants(r)[nm] for r in per_sent["lgs"].to_list()]
        tmp = per_sent.with_columns(pl.Series("v", vals))
        iv = (
            tmp.group_by("item_id")
            .agg(pl.col("v").min().alias("iv"), pl.col("label").first())
            .sort("item_id")
        )
        variants[nm] = round(float(roc_auc_score(iv["label"].to_numpy(), iv["iv"].to_numpy())), 4)
    res["pooling_counterfactual"] = variants
    res["pooling_counterfactual_note"] = (
        "AUROC of the hotpotqa gate sample under alternative window aggregations "
        "of the SAME banked per-window logits (flagship draw 1); min-over-sentences "
        "unchanged. Deterministic, no retraining - isolates the aggregation axis."
    )

    # same counterfactual on every subset, for contrast
    allsub = pl.read_parquet(DUMP)
    contrast = {}
    for sname in sorted(allsub["subset"].unique().to_list()):
        d = allsub.filter(pl.col("subset") == sname)
        ps = (
            d.group_by(["item_id", "sent_idx"])
            .agg(pl.col("logit").alias("lgs"), pl.col("label").first())
            .sort(["item_id", "sent_idx"])
        )
        row = {}
        for nm in ("max", "mean", "softmax_t1.0"):
            vals = [pool_variants(r)[nm] for r in ps["lgs"].to_list()]
            t = ps.with_columns(pl.Series("v", vals))
            iv = t.group_by("item_id").agg(pl.col("v").min().alias("iv"), pl.col("label").first())
            row[nm] = round(float(roc_auc_score(iv["label"].to_numpy(), iv["iv"].to_numpy())), 4)
        row["mean_minus_max"] = round(row["mean"] - row["max"], 4)
        row["n_neg"] = int(
            d.group_by("item_id").agg(pl.col("label").first()).filter(pl.col("label") == 0).height
        )
        contrast[sname] = row
    res["pooling_contrast_all_subsets"] = contrast

    # --- argmax provenance -----------------------------------------------------
    mm = cen.filter(pl.col("hop_class") == "multi_doc")
    res["argmax_provenance"] = {
        "share_argmax_on_highest_containment_doc_all": round(
            float(cen["argmax_on_best_doc"].mean()), 4
        ),
        "share_argmax_on_highest_containment_doc_multi_doc": round(
            float(mm["argmax_on_best_doc"].mean()), 4
        )
        if mm.height
        else None,
        "n_docs_mean": round(float(cen["n_docs"].mean()), 3),
        "n_win_mean": round(float(cen["n_win"].mean()), 3),
        "win_margin_mean": round(float(cen["win_margin"].mean()), 4),
        "win_margin_pos_mean": round(
            float(cen.filter(pl.col("label") == 1)["win_margin"].mean()), 4
        ),
        "win_margin_neg_mean": round(
            float(cen.filter(pl.col("label") == 0)["win_margin"].mean()), 4
        ),
    }

    # correlation of the model's max logit with single-doc containment vs union
    for f in ("cont_top1", "cont_union2", "cont_gain2", "argmax_tok_cont"):
        v = cen[f].to_numpy().astype(float)
        s = cen["smax"].to_numpy().astype(float)
        ok = np.isfinite(v) & np.isfinite(s)
        res.setdefault("smax_correlations", {})[f] = round(
            float(np.corrcoef(v[ok], s[ok])[0, 1]), 4
        )

    # --- eyeball dump ----------------------------------------------------------
    lines = [
        "# R19-H162 hotpotqa eyeball sample",
        "",
        "Frozen gate sample, flagship draw 1 per-window logits from the R19-H161 dump.",
        "",
    ]
    for tag, flt in (
        (
            "multi_doc, LOW single-doc containment (composition genuinely needed)",
            (pl.col("hop_class") == "multi_doc") & (pl.col("cont_top1") < 0.6),
        ),
        (
            "multi_doc, HIGH single-doc containment (restatement, one doc nearly suffices)",
            (pl.col("hop_class") == "multi_doc") & (pl.col("cont_top1") >= 0.8),
        ),
        ("single_doc", pl.col("hop_class") == "single_doc"),
        ("NEGATIVES (label 0)", pl.col("label") == 0),
    ):
        sub = cen.filter(flt).head(12)
        lines.append(f"## {tag}  (n shown {sub.height})")
        lines.append("")
        for r in sub.to_dicts():
            lines.append(
                f"- item {r['item_id']} sent {r['sent_idx']} label {r['label']} "
                f"docs_needed {r['docs_needed']} cont_top1 {r['cont_top1']:.2f} "
                f"union2 {r['cont_union2']:.2f} smax {r['smax']:.3f} "
                f"argmax_doc {r['argmax_doc']} best_doc {r['best_doc']}"
            )
            lines.append(f"    {r['sent'][:400]}")
        lines.append("")
    OUT_EYEBALL.write_text("\n".join(lines))

    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("pooling_contrast_all_subsets",)}, indent=2
        )
    )
    print(f"\nwrote {OUT_JSON}\n      {OUT_SENT}\n      {OUT_EYEBALL}")


if __name__ == "__main__":
    main()
