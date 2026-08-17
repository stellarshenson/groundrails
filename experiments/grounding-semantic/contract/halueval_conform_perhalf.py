"""PER-HALF SEARCH - the last conformance search.  CPU ONLY.

The pooled search found exactly one subset clearing C5's pooled conjunction:
`peel_shape_margin` at 240 pairs.  It collapses to the QA half (440 of 480 rows)
and its 40-row summarization half reads claim_char_length AUROC 0.2913, outside
the [0.45, 0.55] parity band the phase-1 report applied PER HALF.

This search asks whether any subset clears every leg of C5 pooled AND on both
halves, with each half large enough for the reading to mean anything.  Peels are
run INSIDE each half so neither half can be starved, and the barred surface
channels are checked on the pooled member and on each half.

Run: uv run python experiments/grounding-semantic/contract/halueval_conform_perhalf.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HCF = _mod("halueval_conform", HERE / "halueval_conform.py")
C = HCF.C

MIN_HALF_ROWS = 400          # below this a per-half parity read is not a reading
RETENTIONS = (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.100,
              0.150, 0.200)


def per_half_orders(df, key_by_pair, pids):
    """Ordering of pair ids inside each half, ascending on `key_by_pair`."""
    pt = HCF.pair_table(df).sort("pair_id")
    half = np.asarray(pt["half"].to_list())
    order = {}
    for h in ("qa", "summarization"):
        m = half == h
        k = key_by_pair[m]
        order[h] = pids[m][np.argsort(k, kind="stable")]
    return order


def main():
    d, df = HCF.load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, aux = HCF.rankings(df, z)
    pids = aux["pair_ids"]

    pt = HCF.pair_table(df).sort("pair_id")
    cp, cn = pt["claim_pos"].to_list(), pt["claim_neg"].to_list()
    lp = np.array([len(c) for c in cp], dtype=float)
    ln = np.array([len(c) for c in cn], dtype=float)
    rel_len = np.abs(lp - ln) / np.maximum(np.maximum(lp, ln), 1)

    pid_row = np.asarray(df["pair_id"].to_list())
    lab = df["label"].to_numpy()
    idx = {p: i for i, p in enumerate(pids)}

    def margin(score):
        sp, sn = np.zeros(len(pids)), np.zeros(len(pids))
        for i in range(len(pid_row)):
            j = idx[pid_row[i]]
            (sp if lab[i] == 1 else sn)[j] = score[i]
        return np.abs(sp - sn)

    KEYS = {
        "per_half_peel_shape_margin": margin(z["shape_score"]),
        "per_half_peel_probe_margin": margin(z["score"]),
        "per_half_length_matched": rel_len,
        "per_half_length_matched_then_shape_peel": None,   # composite, built below
    }
    m_shape = margin(z["shape_score"])
    comp = rel_len.copy()
    # inside each half: keep the length-matched half of the pairs, order those by
    # |shape margin|, push the rest behind them
    pt_half = np.asarray(pt["half"].to_list())
    comp_key = np.empty(len(pids))
    for h in ("qa", "summarization"):
        m = pt_half == h
        sub_rel = rel_len[m]
        cut = np.quantile(sub_rel, 0.5)
        inside = sub_rel <= cut
        k = np.where(inside, m_shape[m], m_shape[m].max() + 1.0 + sub_rel)
        comp_key[m] = k
    KEYS["per_half_length_matched_then_shape_peel"] = comp_key

    out = {"note": HCF.NOTE,
           "what": "largest subset clearing every leg of C5 pooled AND on both "
                   "halves, with each half at or above "
                   f"{MIN_HALF_ROWS} rows so a per-half reading is a reading",
           "bars": {"claim_only": 0.55, "within_pair": 0.60,
                    "surface_parity_barred_channels": [0.45, 0.55],
                    "minimum_rows_per_half": MIN_HALF_ROWS},
           "levels": {}}
    path = HERE / "halueval_conform_perhalf.json"
    if path.exists():
        out = json.loads(path.read_text())

    for name, key in KEYS.items():
        orders = per_half_orders(df, key, pids)
        for r in RETENTIONS:
            tag = f"{name}@{r:.3f}"
            if tag in out["levels"]:
                continue
            keep = np.concatenate([orders[h][:int(round(r * len(orders[h])))]
                                   for h in ("qa", "summarization")])
            sub = df.filter(pl.col("pair_id").is_in(keep.tolist()))
            halves = {k: int(v) for k, v in sub.group_by("half").len().iter_rows()}
            if min(halves.get("qa", 0), halves.get("summarization", 0)) < MIN_HALF_ROWS:
                out["levels"][tag] = {"rows": sub.height, "half_composition": halves,
                                      "skipped": "a half is below the minimum"}
                path.write_text(json.dumps(out, indent=2))
                continue
            subi = sub.with_columns(pl.col("label").cast(pl.Int64))
            parity = {"all": C.surface_parity(
                subi, report_only=("claim_chunk_containment",))}
            for h in ("qa", "summarization"):
                parity[h] = C.surface_parity(
                    subi.filter(pl.col("half") == h),
                    report_only=("claim_chunk_containment",))
            parity_ok = all(v["pass"] for v in parity.values())
            rec = {"strategy": name, "retention_pairs_per_half": r,
                   "rows": sub.height, "pairs": int(sub["pair_id"].n_unique()),
                   "retention_of_member_rows": round(sub.height / HCF.ORIG_ROWS, 4),
                   "half_composition": halves,
                   "surface_parity": {k: {"auroc": v["auroc"],
                                          "worst_barred_deviation": v["worst_deviation"],
                                          "pass": v["pass"]} for k, v in parity.items()},
                   "clears_surface_parity_pooled_and_both_halves": parity_ok}
            if parity_ok:
                claims = sub["claim"].to_list()
                labels = sub["label"].to_numpy()
                g = HCF.groups_of(sub)
                aur, wps = [], []
                for seed in range(5):
                    a, s = HCF.probe(claims, labels, g, seed=seed)
                    aur.append(round(a, 4))
                    wps.append(C.within_pair_accuracy(subi, s)["all"]["acc"])
                rec.update({"claim_only_auroc_seeds": aur,
                            "claim_only_auroc_max": max(aur),
                            "within_pair_max": max(wps),
                            "clears_claim_only_all_seeds": bool(max(aur) < 0.55),
                            "clears_within_pair_all_seeds": bool(max(wps) < 0.60)})
                rec["clears_every_leg_of_C5"] = bool(
                    parity_ok and max(aur) < 0.55 and max(wps) < 0.60)
            else:
                rec["clears_every_leg_of_C5"] = False
            out["levels"][tag] = rec
            print(f"{tag}: rows {rec['rows']} halves {halves} parity_ok {parity_ok} "
                  f"auroc {rec.get('claim_only_auroc_seeds')} "
                  f"C5 {rec['clears_every_leg_of_C5']}", flush=True)
            path.write_text(json.dumps(out, indent=2))

    passing = [v for v in out["levels"].values() if v.get("clears_every_leg_of_C5")]
    out["largest_subset_clearing_every_leg"] = (
        max(passing, key=lambda v: v["rows"]) if passing else None)
    out["subsets_clearing_every_leg"] = len(passing)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["largest_subset_clearing_every_leg"], indent=2), flush=True)
    print("perhalf written", flush=True)


if __name__ == "__main__":
    main()
