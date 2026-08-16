"""R19-H162 stage 2 - HOTPOTQA CLAIM-FAMILY TAXONOMY. ANALYSIS ONLY. CPU ONLY.

Stage 1 (`R19-H162_hotpotqa_probe.py`) established that 71.3% of hotpotqa claim
sentences need two or more documents to cover their lexical anchors, and that on
exactly those sentences the flagship's max-window logit carries NO label signal
(positive mean -3.665 vs negative mean -3.664). This stage asks WHICH cross-
document structure the claims have, because the lever differs by structure:

  conjoin_attrs   two named entities, one attribute each, in different documents,
                  joined by a conjunction or a comparative ("X is Cuban, while Y
                  is French"; "both are magazines"; "X has more species than Y")
  bridge_entity   two endpoints named in the claim, linked through an
                  intermediate entity that is ELIDED from the claim and appears
                  in both cover documents ("the memoir written by the honoree of
                  the Black and White Ball is 'Personal History'")
  single_hop      one document covers the claim

Measurements: family counts; per-family positive/negative score separation;
clause-level document split for the conjunction family; elided-bridge detection;
and confound controls (sentence length, anchor count) so the score collapse on
multi-document sentences is not attributed to length.

Run:  uv run python experiments/grounding-semantic/R19-H162_hotpotqa_families.py
"""

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
IN_SENT = HERE / "R19-H162_hotpotqa_sentences.parquet"
OUT_JSON = HERE / "R19-H162_hotpotqa_families.json"
OUT_EYEBALL = HERE / "R19-H162_hotpotqa_families_eyeball.md"
CACHE = HERE / "R16-H140_cache" / "arena_hotpotqa.npz"

SUBSET = "hotpotqa"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PROBE = _mod("probe", "R19-H162_hotpotqa_probe.py")

# Conjunction / comparison surface markers. A hotpotqa comparison answer almost
# always carries one of these; the clause split below is what actually decides
# the family, the marker only proposes the split point.
SPLIT_MARKERS = [
    " while ",
    " whereas ",
    " compared to ",
    " compared with ",
    " but ",
    " and ",
    "; ",
    ", while ",
    " than ",
]
COMPARE_MARKERS = re.compile(
    r"\b(while|whereas|compared to|compared with|both|neither|either|more than|"
    r"less than|higher|lower|larger|smaller|longer|shorter|older|younger|earlier|"
    r"later|closer|farther|further|greater|fewer|first|same|different|"
    r"respectively|only)\b|\bthan\b|^(yes|no)\b",
    re.IGNORECASE,
)


def clauses_of(sentence):
    """Split a sentence at the first conjunction/comparison marker found."""
    low = sentence.lower()
    for m in SPLIT_MARKERS:
        i = low.find(m)
        if 12 < i < len(sentence) - 12:
            return [sentence[:i].strip(), sentence[i + len(m) :].strip()]
    return [sentence]


def best_doc_for(text, lows):
    anc = PROBE.anchors_of(text)
    if not anc:
        return -1, 0.0
    conts = [PROBE.containment(anc, ld) for ld in lows]
    b = int(np.argmax(conts))
    return b, float(conts[b])


def elided_bridge(anchors_claim, lows, cover):
    """Content tokens shared by the cover documents but absent from the claim.

    The signature of a bridge hop: the intermediate entity that links the claim's
    two endpoints is named in both documents and dropped from the answer.
    """
    if len(cover) < 2:
        return []
    sets = []
    for d in cover[:3]:
        toks = {w for w in PROBE._WORD.findall(lows[d]) if w not in PROBE.STOPWORDS}
        sets.append(toks)
    shared = set.intersection(*sets)
    return sorted(shared - set(anchors_claim))


def boot_gap_ci(pos, neg, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [
        rng.choice(pos, pos.size, replace=True).mean()
        - rng.choice(neg, neg.size, replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    arena = _mod("arena", "R8-H77_unseen_arena.py")
    subs = arena.load_subsets()
    _claims, chunk_lists, _y = subs[SUBSET]

    cen = pl.read_parquet(IN_SENT)
    res = {"subset": SUBSET, "n_sentences": cen.height}

    rows = []
    for r in cen.to_dicts():
        lows = [k.lower() for k in chunk_lists[r["item_id"]]]
        s = r["sent"]
        anc = PROBE.anchors_of(s)
        cls = clauses_of(s)
        has_marker = bool(COMPARE_MARKERS.search(s))
        cover = json.loads(r["cover_docs"])
        # clause-level document split
        if len(cls) == 2:
            d0, c0 = best_doc_for(cls[0], lows)
            d1, c1 = best_doc_for(cls[1], lows)
            split = (d0 != d1) and d0 >= 0 and d1 >= 0
            clause_min_cont = min(c0, c1)
        else:
            d0 = d1 = -1
            split = False
            clause_min_cont = 0.0
        bridge = elided_bridge(anc, lows, cover)
        if r["hop_class"] == "single_doc":
            fam = "single_hop"
        elif has_marker and split:
            fam = "conjoin_attrs"
        elif has_marker and len(cls) == 2:
            fam = "conjoin_attrs_same_doc"
        elif bridge:
            fam = "bridge_entity"
        else:
            fam = "multi_doc_other"
        rows.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "item_id",
                        "sent_idx",
                        "label",
                        "sent",
                        "hop_class",
                        "cont_top1",
                        "cont_union2",
                        "smax",
                        "n_anchors",
                        "docs_needed",
                        "argmax_on_best_doc",
                        "win_margin",
                    )
                },
                "family": fam,
                "has_compare_marker": has_marker,
                "clause_docs_differ": split,
                "clause_min_cont": clause_min_cont,
                "n_elided_bridge_tokens": len(bridge),
                "elided_bridge_sample": ", ".join(bridge[:6]),
                "char_len": len(s),
            }
        )
    fam = pl.DataFrame(rows)
    fam.write_parquet(HERE / "R19-H162_hotpotqa_families.parquet")

    # --- family census ---------------------------------------------------------
    cnt = fam.group_by("family").agg(pl.len().alias("n")).sort("n", descending=True)
    res["family_census"] = {r["family"]: int(r["n"]) for r in cnt.to_dicts()}
    res["family_census_pct"] = {
        k: round(100.0 * v / fam.height, 2) for k, v in res["family_census"].items()
    }

    per_fam = {}
    for f in res["family_census"]:
        sub = fam.filter(pl.col("family") == f)
        p = sub.filter(pl.col("label") == 1)["smax"].to_numpy()
        n = sub.filter(pl.col("label") == 0)["smax"].to_numpy()
        d = {
            "n_sent": int(sub.height),
            "n_pos": int(p.size),
            "n_neg": int(n.size),
            "smax_pos_mean": round(float(p.mean()), 4) if p.size else None,
            "smax_neg_mean": round(float(n.mean()), 4) if n.size else None,
            "smax_gap": round(float(p.mean() - n.mean()), 4) if p.size and n.size else None,
            "mean_char_len": round(float(sub["char_len"].mean()), 1),
            "mean_n_anchors": round(float(sub["n_anchors"].mean()), 2),
            "share_argmax_on_best_doc": round(float(sub["argmax_on_best_doc"].mean()), 4),
        }
        if p.size and n.size >= 3:
            lo, hi = boot_gap_ci(p, n)
            d["smax_gap_ci95"] = [round(lo, 3), round(hi, 3)]
            yy = np.concatenate([np.ones(p.size), np.zeros(n.size)])
            d["sent_auroc"] = round(float(roc_auc_score(yy, np.concatenate([p, n]))), 4)
        per_fam[f] = d
    res["per_family"] = per_fam

    # clause-split detail for the conjunction family
    cj = fam.filter(pl.col("family").str.starts_with("conjoin_attrs"))
    res["conjunction_detail"] = {
        "n_with_marker": int(fam["has_compare_marker"].sum()),
        "share_with_marker": round(float(fam["has_compare_marker"].mean()), 4),
        "n_marker_and_clauses_in_different_docs": int(fam["clause_docs_differ"].sum()),
        "share_of_multi_doc_that_is_conjunction": round(
            float(
                (
                    fam.filter(pl.col("hop_class") == "multi_doc")["family"] == "conjoin_attrs"
                ).mean()
            ),
            4,
        ),
        "clause_min_containment_mean": round(float(cj["clause_min_cont"].mean()), 4)
        if cj.height
        else None,
    }

    # bridge detail
    br = fam.filter(pl.col("family") == "bridge_entity")
    res["bridge_detail"] = {
        "n": int(br.height),
        "mean_elided_tokens": round(float(br["n_elided_bridge_tokens"].mean()), 2)
        if br.height
        else None,
        "share_of_multi_doc": round(
            float(
                (
                    fam.filter(pl.col("hop_class") == "multi_doc")["family"] == "bridge_entity"
                ).mean()
            ),
            4,
        ),
    }

    # --- confound controls -----------------------------------------------------
    # (a) length / anchor count are matched between labels WITHIN the multi_doc
    #     class, so the collapsed gap there is not a length artefact
    ctl = {}
    for cls in ("single_doc", "multi_doc"):
        sub = fam.filter(pl.col("hop_class") == cls)
        p = sub.filter(pl.col("label") == 1)
        n = sub.filter(pl.col("label") == 0)
        ctl[cls] = {
            "char_len_pos": round(float(p["char_len"].mean()), 1),
            "char_len_neg": round(float(n["char_len"].mean()), 1),
            "n_anchors_pos": round(float(p["n_anchors"].mean()), 2),
            "n_anchors_neg": round(float(n["n_anchors"].mean()), 2),
        }
    res["length_control"] = ctl

    # (b) anchor-count-stratified smax: does score fall with anchors REGARDLESS
    #     of hop class, or specifically with the need for a second document
    strat = []
    for lo, hi in ((0, 6), (6, 9), (9, 12), (12, 100)):
        b = fam.filter((pl.col("n_anchors") >= lo) & (pl.col("n_anchors") < hi))
        row = {"anchors": f"{lo}-{hi}", "n": int(b.height)}
        for cls in ("single_doc", "multi_doc"):
            s = b.filter(pl.col("hop_class") == cls)["smax"]
            row[cls] = round(float(s.mean()), 3) if s.len() else None
            row[f"n_{cls}"] = int(s.len())
        strat.append(row)
    res["smax_by_anchor_bin"] = strat

    # (c) partial correlation of smax with docs_needed controlling for n_anchors
    x = fam["docs_needed"].to_numpy().astype(float)
    z = fam["n_anchors"].to_numpy().astype(float)
    yv = fam["smax"].to_numpy().astype(float)

    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    res["partial_corr_smax_vs_docs_needed_given_n_anchors"] = round(
        float(np.corrcoef(resid(yv, z), resid(x, z))[0, 1]), 4
    )
    res["raw_corr_smax_vs_docs_needed"] = round(float(np.corrcoef(yv, x)[0, 1]), 4)
    res["raw_corr_smax_vs_n_anchors"] = round(float(np.corrcoef(yv, z)[0, 1]), 4)

    # --- H140 cache cross-check ------------------------------------------------
    if CACHE.exists():
        z140 = np.load(CACHE)
        sc = z140["sent_cross"]
        mine = (cen.sort(["item_id", "sent_idx"])["hop_class"] == "multi_doc").to_numpy()
        res["h140_cache_crosscheck"] = {
            "n_sent_cache": int(sc.size),
            "h140_cross_share": round(float(sc.mean()), 4),
            "h162_multi_doc_share": round(float(mine.mean()), 4),
            "agreement": round(float((sc == mine).mean()), 4) if sc.size == mine.size else None,
            "note": (
                "H140 flagged cross-WINDOW at anchor level; H162 flags "
                "cross-DOCUMENT by greedy set cover. Independent methods, "
                "same population."
            ),
        }

    # --- eyeball ---------------------------------------------------------------
    lines = ["# R19-H162 hotpotqa claim families", ""]
    for f in res["family_census"]:
        sub = fam.filter(pl.col("family") == f)
        lines += [f"## {f}  (n {sub.height})", ""]
        for r in sub.head(10).to_dicts():
            lines.append(
                f"- item {r['item_id']} label {r['label']} smax {r['smax']:.2f} "
                f"cont_top1 {r['cont_top1']:.2f} bridge_toks {r['n_elided_bridge_tokens']}"
            )
            lines.append(f"    {r['sent'][:300]}")
        # every negative in the family, they are the scarce class
        neg = sub.filter(pl.col("label") == 0)
        if neg.height:
            lines += ["", f"  NEGATIVES in {f} (n {neg.height}):"]
            for r in neg.to_dicts():
                lines.append(f"  - item {r['item_id']} smax {r['smax']:.2f}: {r['sent'][:250]}")
        lines.append("")
    OUT_EYEBALL.write_text("\n".join(lines))

    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT_JSON}\n      {OUT_EYEBALL}")


if __name__ == "__main__":
    main()
