"""R20-H177 Lane B - is the eval_B contamination LOAD-BEARING?  CPU only, zero GPU.

The blast-radius sweep (`R20-H175b_eval_contamination_sweep.json`) found 33 of
736 `R20-H177_eval_B.parquet` passages present in the assembled training mix.
eval_B carries Lane B's PRIMARY gate: held-out AUROC >= 0.80 against the banked
baseline-leg floor 0.5064 (2-draw mean on the flagship checkpoints
`R18-H150-arm-draw{1,2}`).  Nothing has trained on Lane B yet, so the only
question is whether the INSTRUMENT is sound.

What this measures, and nothing else:

  1. WHICH passages leak, against the FLAGSHIP mix the baseline-leg checkpoints
     actually trained on (`public_train()` untruncated + `R17-H146_lane` +
     `R18-H150_scaleunit_lane` = 721,210 rows), in all six string forms - the
     same three-form method (raw / truncated to `CFG.chunk_max_chars` /
     whitespace-collapsed case-folded) crossed both ways.  The sweep's mix
     additionally carried `R20-H175b_qlane`; that lane is PsiloQA and cannot
     supply a TabFact/EDGAR passage, but the flagship mix is the correct
     denominator for a read taken on flagship checkpoints, so it is rebuilt here
     rather than inherited.
  2. A SECOND check the sweep did not run: eval_B against `R20-H177_lane_B`
     itself in the normalised form.  The lane/eval split was proven doc-disjoint
     by blake2b; the normalised-form mode is exactly what an exact-string or
     id-level check cannot see, and Lane B enters the mix when the arm trains.
  3. The baseline leg RECOMPUTED with the contaminated rows excluded, from the
     banked per-row score arrays `R20_baseline_legs_scores_eval_B_h150d{1,2}.npy`
     (claim-level, max-over-windows, eval-parquet row order).  Recomputing the
     banked AUROC from those arrays first is the integrity check; a mismatch
     aborts rather than producing a number.
  4. The memorisation-feature analogue.  On the H175b eval the feature read
     0.6230: overlap between the eval claim and the `llm_answer` the mix pairs
     with that leg's QUESTION over that same passage.  eval_B has no question
     channel, so the analogue keys on the passage alone - what CLAIM(s) does the
     mix pair with this passage, and does overlap with the eval claim separate
     the labels?

Run:  uv run python experiments/grounding-semantic/R20-H177_evalB_contamination_assessment.py
"""

import os

# CPU ONLY - GPU0/GPU1 are training H174 draws 2 and 3 and are not to be touched;
# `R10-H108_lane` imports torch and would otherwise `setdefault` a device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

HERE = Path(__file__).parent
OUT = HERE / "R20-H177_evalB_contamination_assessment.json"

EVAL_B = HERE / "R20-H177_eval_B.parquet"
LANE_B = HERE / "R20-H177_lane_B.parquet"
SCORES = {"h150d1": HERE / "R20_baseline_legs_scores_eval_B_h150d1.npy",
          "h150d2": HERE / "R20_baseline_legs_scores_eval_B_h150d2.npy"}
BANKED = {"h150d1": 0.5090, "h150d2": 0.5038}
BANKED_MEAN = 0.5064
BANKED_FAMILY = {                      # 2-draw means, from R20-H177_baseline_leg.json
    "cmp_order": 0.5126, "cmp_amount": 0.5013,
    "cmp_extreme": 0.4778, "cmp_trend": 0.7930,
}


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """The sweep's normalised form - whitespace-collapsed, case-folded."""
    return " ".join(s.split()).casefold()


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(s)))


# --------------------------------------------------------------------------- #
# the FLAGSHIP mix - what R18-H150-arm-draw{1,2} actually trained on
# --------------------------------------------------------------------------- #
def flagship_mix():
    """`R20_baseline_legs.flagship_mix_text()` recipe, keeping the claims too.

    Returns the chunk sets in the three forms plus a normalised-chunk -> claims
    map (built lazily by the caller for only the passages that matter).
    """
    arm = _mod("g1arm", HERE / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", HERE / "R10-H108_lane.py")
    chunk_max = H108.M59.CFG.chunk_max_chars
    print(f"mix: chunk_max_chars = {chunk_max}", flush=True)

    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    labels = list(np.asarray(y, dtype="float64"))
    print(f"mix: clean public {len(claims)} rows over {len(set(tags))} groups",
          flush=True)
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows, expected 685,670")

    for fname in ("R17-H146_lane.parquet", "R18-H150_scaleunit_lane.parquet"):
        p = HERE / fname
        if not p.exists():
            raise SystemExit(f"MIX ABORT: lane {fname} absent")
        d = pl.read_parquet(p)
        ch_col = "chunk" if "chunk" in d.columns else "evidence"
        lcol = next((c for c in ("label", "y") if c in d.columns), None)
        claims += d["claim"].to_list()
        chunks += d[ch_col].to_list()
        labels += ([float(v) for v in d[lcol].to_list()] if lcol
                   else [float("nan")] * d.height)
        tags += [fname] * d.height
        print(f"mix: lane {fname} {d.height} rows (label column {lcol})", flush=True)

    if len(claims) != 721_210:
        raise SystemExit(f"MIX ABORT: flagship mix {len(claims)} rows, expected 721,210")
    print(f"mix: flagship total {len(claims)} rows", flush=True)
    return claims, chunks, labels, tags, chunk_max


def forms(chunks, chunk_max):
    raw = set(chunks)
    trunc = {c[:chunk_max] for c in raw}
    return {"raw": raw, "trunc": trunc,
            "nraw": {norm(c) for c in raw}, "ntrunc": {norm(c) for c in trunc},
            "chunk_max": chunk_max}


def six_forms(passages, mix):
    """The sweep's six checks, per passage.  Returns (per-form counts, hit set)."""
    cut = mix["chunk_max"]
    tests = (
        ("raw_in_mix_raw", lambda p: p in mix["raw"]),
        ("raw_in_mix_truncated", lambda p: p in mix["trunc"]),
        ("truncated_in_mix_raw", lambda p: p[:cut] in mix["raw"]),
        ("truncated_in_mix_truncated", lambda p: p[:cut] in mix["trunc"]),
        ("normalised_in_mix_normalised_raw", lambda p: norm(p) in mix["nraw"]),
        ("normalised_in_mix_normalised_truncated", lambda p: norm(p) in mix["ntrunc"]),
    )
    counts, hit = {}, set()
    for name, test in tests:
        h = {p for p in passages if test(p)}
        counts[name] = len(h)
        hit |= h
    return counts, hit


# --------------------------------------------------------------------------- #
# memorisation-feature analogue
# --------------------------------------------------------------------------- #
def memorisation_feature(df, mix_claims, mix_labels, hit_norm):
    """Pure recall over the PASSAGE, the only key eval_B has.

    For each eval row the mix is asked: what claim(s) did you carry over THIS
    passage?  Overlap of the best-matching mix claim with the eval claim is
    scored against the eval label.  Reported alongside is the structural reason
    the concept is weaker here than on the H175b eval: both legs of an eval_B
    pair carry the SAME passage and a claim differing in ONE relation word, so a
    passage-keyed lookup returns the identical claim bag to both legs and can
    only separate them through that one word.
    """
    Q = _mod("h175bqlane", HERE / "R20-H175b_qlane.py")

    y = np.asarray(df["label"].to_list())
    lookup = [mix_claims.get(norm(c), []) for c in df["chunk"].to_list()]
    lookup_y = [mix_labels.get(norm(c), []) for c in df["chunk"].to_list()]
    covered = sum(1 for v in lookup if v)
    out = {"rows": df.height, "rows_with_a_mix_claim": covered,
           "coverage": round(covered / df.height, 4) if df.height else 0.0}
    if covered == 0:
        out["auroc"] = None
        out["note"] = "the mix carries no claim over any eval_B passage"
        return out

    variants = {
        "jaccard": lambda c, a: Q.jaccard(c, a),
        "claim_into_mixclaim_containment": lambda c, a: Q.containment(c, a),
        "mixclaim_into_claim_containment": lambda c, a: Q.containment(a, c),
        "shared_token_count": lambda c, a: float(len(set(Q.tok(c)) & set(Q.tok(a)))),
    }
    for vname, fn in variants.items():
        s = np.array([max((fn(c, a) for a in v), default=0.0)
                      for c, v in zip(df["claim"].to_list(), lookup)])
        out[vname] = round(float(auroc(y, s)), 4)
    out["auroc"] = max(out[v] for v in variants)
    out["strongest_variant"] = max(variants, key=lambda v: out[v])

    # label-aware variant: the label the mix attached to its best-overlapping
    # claim over this passage - the strongest recall signal a memoriser could use
    best_lab = []
    for c, v, ly in zip(df["claim"].to_list(), lookup, lookup_y):
        if not v:
            best_lab.append(0.0)
            continue
        j = int(np.argmax([Q.jaccard(c, a) for a in v]))
        best_lab.append(float(ly[j]))
    out["nearest_mix_claim_label"] = round(float(auroc(y, np.array(best_lab))), 4)

    # how far apart are the two legs of a pair under this feature at all?
    s = np.array([max((Q.jaccard(c, a) for a in v), default=0.0)
                  for c, v in zip(df["claim"].to_list(), lookup)])
    d = (df.with_columns(pl.Series("f", s))
           .group_by("pair_id")
           .agg((pl.col("f").max() - pl.col("f").min()).alias("spread")))
    out["within_pair_feature_spread"] = {
        "pairs": d.height,
        "pairs_with_zero_spread": int((d["spread"] == 0.0).sum()),
        "mean_spread": round(float(d["spread"].mean()), 6),
        "max_spread": round(float(d["spread"].max()), 6),
    }
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=== R20-H177 eval_B contamination assessment (CPU only) ===", flush=True)
    ev = pl.read_parquet(EVAL_B)
    print(f"eval_B: {ev.height} rows / {ev['pair_id'].n_unique()} pairs / "
          f"{ev['doc_id'].n_unique()} docs / {ev['chunk'].n_unique()} passages",
          flush=True)

    claims, chunks, labels, tags, chunk_max = flagship_mix()
    mix = forms(chunks, chunk_max)
    print(f"mix: {len(mix['raw'])} distinct raw chunks", flush=True)

    passages = sorted({c for c in ev["chunk"].to_list() if c and c.strip()})
    counts, hit = six_forms(passages, mix)
    print(f"flagship-mix forms: {json.dumps(counts)}", flush=True)
    print(f"flagship-mix hits: {len(hit)}/{len(passages)}", flush=True)

    # which mix group supplies each contaminated passage
    hit_norm = {norm(p) for p in hit}
    by_group = collections.Counter()
    mix_claims = collections.defaultdict(list)
    lab_map = collections.defaultdict(list)
    for cl, ch, lv, tg in zip(claims, chunks, labels, tags, strict=True):
        n = norm(ch)
        if n in hit_norm:
            by_group[tg] += 1
            mix_claims[n].append(cl)
            lab_map[n].append(lv)
    del claims, chunks, labels

    # Lane B itself - the sweep's mix pooled Lane B away; the lane/eval split was
    # proven doc-disjoint by blake2b over `doc_id`, which is an ID-level proof.
    lane_counts, lane_hit, lane_detail = None, set(), None
    if LANE_B.exists():
        lb = pl.read_parquet(LANE_B)
        lb_forms = forms(lb["chunk"].to_list(), chunk_max)
        lane_counts, lane_hit = six_forms(passages, lb_forms)
        print(f"lane_B forms: {json.dumps(lane_counts)}  hits {len(lane_hit)}",
              flush=True)
        shared = ev.filter(pl.col("chunk").is_in(lane_hit)) if lane_hit else None
        # TabFact serialises a table under both a `1-` and a `2-` csv id; a split
        # keyed on the id string sees those as two documents.
        def stem(d):
            return d[10:] if d.startswith("tabfact:") and d[8] in "12" else d
        e_st = {stem(d) for d in
                ev.filter(pl.col("source") == "tabfact")["doc_id"].unique().to_list()}
        l_st = {stem(d) for d in
                lb.filter(pl.col("source") == "tabfact")["doc_id"].unique().to_list()}
        lane_detail = {
            "doc_id_overlap": len(set(ev["doc_id"]) & set(lb["doc_id"])),
            "byte_identical_passages": len(lane_hit),
            "eval_rows_on_them": 0 if shared is None else shared.height,
            "eval_pairs_on_them": 0 if shared is None else shared["pair_id"].n_unique(),
            "eval_doc_ids": [] if shared is None else shared["doc_id"].unique().to_list(),
            "lane_doc_ids": [] if not lane_hit else
                lb.filter(pl.col("chunk").is_in(lane_hit))["doc_id"].unique().to_list(),
            "tabfact_doc_id_stem_collisions": len(e_st & l_st),
            "note": "the blake2b doc-disjoint split keys on the doc_id STRING; "
                    "TabFact's `1-`/`2-` csv-id prefixes make one serialised table "
                    "two document ids, so an id-level disjointness proof does not "
                    "imply passage disjointness",
        }
        print(f"lane_B detail: {json.dumps(lane_detail)}", flush=True)
        del lb, lb_forms

    # ---- rows / pairs affected ------------------------------------------ #
    ev = ev.with_columns(
        pl.col("chunk").map_elements(lambda c: c in hit, return_dtype=pl.Boolean)
          .alias("contaminated"))
    n_rows_c = int(ev["contaminated"].sum())
    pairs_c = ev.filter(pl.col("contaminated"))["pair_id"].n_unique()
    docs_c = ev.filter(pl.col("contaminated"))["doc_id"].n_unique()
    src_c = dict(ev.filter(pl.col("contaminated")).group_by("source").len().iter_rows())
    fam_c = dict(ev.filter(pl.col("contaminated"))
                   .group_by("neg_family").len().iter_rows())
    print(f"contaminated: {n_rows_c} rows / {pairs_c} pairs / {docs_c} docs "
          f"sources={src_c} families={fam_c}", flush=True)

    # ---- baseline leg, with and without --------------------------------- #
    y = ev["label"].to_numpy()
    fam = np.array(ev["neg_family"].to_list())
    keep = ~ev["contaminated"].to_numpy()

    leg = {}
    for tag, p in SCORES.items():
        s = np.load(p)
        if len(s) != ev.height:
            raise SystemExit(f"ABORT: {p.name} has {len(s)} scores, eval has {ev.height}")
        full = auroc(y, s)
        if abs(round(full, 4) - BANKED[tag]) > 1e-9:
            raise SystemExit(f"ABORT: {tag} recomputed {full:.6f} != banked {BANKED[tag]} "
                             "- the score array does not reproduce the banked leg")
        block = {"checkpoint": tag, "auroc_all_rows": round(full, 6),
                 "auroc_clean_rows": round(auroc(y[keep], s[keep]), 6),
                 "n_rows_clean": int(keep.sum())}
        if len(set(y[~keep].tolist())) == 2:
            block["auroc_contaminated_rows_only"] = round(auroc(y[~keep], s[~keep]), 6)
            block["n_rows_contaminated"] = int((~keep).sum())
        f_all, f_clean = {}, {}
        for f in sorted(set(fam.tolist())):
            m = fam == f
            if len(set(y[m].tolist())) == 2:
                f_all[f] = {"n_rows": int(m.sum()), "auroc": round(auroc(y[m], s[m]), 6)}
            mc = m & keep
            if len(set(y[mc].tolist())) == 2:
                f_clean[f] = {"n_rows": int(mc.sum()),
                              "auroc": round(auroc(y[mc], s[mc]), 6)}
        block["by_neg_family_all_rows"] = f_all
        block["by_neg_family_clean_rows"] = f_clean
        leg[tag] = block
        print(f"{tag}: all {block['auroc_all_rows']:.6f} -> clean "
              f"{block['auroc_clean_rows']:.6f}", flush=True)

    mean_all = float(np.mean([leg[t]["auroc_all_rows"] for t in SCORES]))
    mean_clean = float(np.mean([leg[t]["auroc_clean_rows"] for t in SCORES]))
    fam_mean_all, fam_mean_clean = {}, {}
    for f in sorted(set(fam.tolist())):
        vs = [leg[t]["by_neg_family_all_rows"][f]["auroc"] for t in SCORES
              if f in leg[t]["by_neg_family_all_rows"]]
        if len(vs) == 2:
            fam_mean_all[f] = round(float(np.mean(vs)), 6)
        vs = [leg[t]["by_neg_family_clean_rows"][f]["auroc"] for t in SCORES
              if f in leg[t]["by_neg_family_clean_rows"]]
        if len(vs) == 2:
            fam_mean_clean[f] = round(float(np.mean(vs)), 6)
    mean_dirty = float(np.mean([leg[t]["auroc_contaminated_rows_only"] for t in SCORES
                                if "auroc_contaminated_rows_only" in leg[t]]))
    print(f"2-draw mean: all {mean_all:.6f} (banked {BANKED_MEAN}) -> "
          f"clean {mean_clean:.6f}  delta {mean_clean - mean_all:+.6f}  "
          f"contaminated-rows-only {mean_dirty:.6f}", flush=True)

    # ---- memorisation-feature analogue ---------------------------------- #
    mem_contaminated = memorisation_feature(
        ev.filter(pl.col("contaminated")), mix_claims, lab_map, hit_norm)
    mem_all = memorisation_feature(ev, mix_claims, lab_map, hit_norm)

    # ---- bank ------------------------------------------------------------ #
    res = {
        "experiment": "R20-H177 Lane B - is the eval_B contamination load-bearing? "
                      "CPU-only assessment, zero GPU, zero training",
        "scope": "measurement and reading only - no adjudication, no bar changed",
        "eval": {"parquet": EVAL_B.name, "rows": ev.height,
                 "pairs": ev["pair_id"].n_unique(), "docs": ev["doc_id"].n_unique(),
                 "distinct_passages": len(passages)},
        "mix": {"recipe": "R10-H108_lane.public_train() under "
                          "R16-H142_G1_arm.untruncated_evidence() + R17-H146_lane + "
                          "R18-H150_scaleunit_lane = 721,210 rows - the FLAGSHIP mix "
                          "the baseline-leg checkpoints trained on",
                "distinct_raw_chunks": len(mix["raw"]),
                "chunk_max_chars": chunk_max},
        "contamination": {
            "by_form_against_flagship_mix": counts,
            "passages_in_the_mix": len(hit),
            "share_of_passages": round(len(hit) / len(passages), 4),
            "rows_affected": n_rows_c, "pairs_affected": int(pairs_c),
            "docs_affected": int(docs_c),
            "rows_by_source": src_c, "rows_by_neg_family": fam_c,
            "mix_rows_carrying_a_contaminated_passage_by_group": dict(by_group),
            "against_lane_B_by_form": lane_counts,
            "passages_shared_with_lane_B": len(lane_hit),
            "lane_B_detail": lane_detail,
            "correction_to_the_canonical_log": (
                "the EVAL CONTAMINATION block records eval_B as leaking 'ONLY "
                "through the whitespace-normalised form - 0 raw, 0 truncated'. The "
                "banked sweep JSON it cites reads 19 raw / 19 truncated / 33 "
                "normalised for eval_B, reproduced here against the flagship mix. "
                "The 0-raw/0-truncated description is true of R17-H143_evalset "
                "(0/0/10) but not of eval_B: 19 of the 33 are BYTE-IDENTICAL to a "
                "mix passage and 14 are additionally reachable only after "
                "normalisation"),
        },
        "baseline_leg": {
            "protocol": "banked per-row score arrays "
                        "R20_baseline_legs_scores_eval_B_h150d{1,2}.npy, eval-parquet "
                        "row order; the banked AUROC is reproduced exactly before any "
                        "exclusion is applied",
            "banked": {"per_draw": BANKED, "two_draw_mean": BANKED_MEAN,
                       "per_family_two_draw_mean": BANKED_FAMILY},
            "per_draw": leg,
            "two_draw_mean_all_rows": round(mean_all, 6),
            "two_draw_mean_clean_rows": round(mean_clean, 6),
            "two_draw_mean_contaminated_rows_only": round(mean_dirty, 6),
            "delta_mean": round(mean_clean - mean_all, 6),
            "per_family_two_draw_mean_all_rows": fam_mean_all,
            "per_family_two_draw_mean_clean_rows": fam_mean_clean,
        },
        "memorisation_feature": {
            "definition": "the H175b analogue with the only key eval_B has - the "
                          "PASSAGE. For each eval row, the overlap between the eval "
                          "claim and the best-matching claim the mix carries over the "
                          "same (normalised) passage, scored against the eval label",
            "contaminated_rows_only": mem_contaminated,
            "whole_eval": mem_all,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"=== banked -> {OUT.name} ({res['elapsed_s']}s) ===", flush=True)


if __name__ == "__main__":
    main()
