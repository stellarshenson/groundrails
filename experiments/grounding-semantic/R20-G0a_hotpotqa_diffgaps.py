"""R20 gate 4 (hotpotqa G0a) - bank the difference-of-label-gaps statistic.

Recipe, verbatim from `docs/experiments/briefs/R20-fanout-hotpotqa-composition-
hypotheses.md` (design-pass block, line 21): join
`R19-H162_hotpotqa_families.parquet` on (item_id, sent_idx) to the per-window
dumps `R19-H161_pairs_h150d{1,2}.parquet` and `R18-H151_scores_{1142,2142}.parquet`,
per-checkpoint z-scored sentence-max, 4-checkpoint pooled, bootstrap over
sentences.

Statistic: label gap = mean(z | label 1) - mean(z | label 0) within a hop_class;
the reported number is gap(single_doc) - gap(multi_doc), averaged over the four
checkpoints. PASS = bootstrap CI95 excludes zero.

Design-pass expectation: +0.886, CI95 [+0.043, +1.657], per-checkpoint
+0.908 / +0.835 / +1.081 / +0.718.

Run:  uv run python experiments/grounding-semantic/R20-G0a_hotpotqa_diffgaps.py
"""

import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20-G0a_hotpotqa_diffgaps.json"
N_BOOT = 10000
SEED = 20260816


def sentence_max(tag):
    """Per-(item_id, sent_idx) max window logit for hotpotqa on one checkpoint."""
    if tag.startswith("h150"):
        d = (pl.read_parquet(HERE / f"R19-H161_pairs_{tag}.parquet")
             .filter(pl.col("subset") == "hotpotqa")
             .group_by(["item_id", "sent_idx"])
             .agg(pl.col("logit").max().alias("smax")))
    else:
        d = (pl.read_parquet(HERE / f"R18-H151_scores_{tag}.parquet")
             .filter(pl.col("subset") == "hotpotqa")
             .group_by(["item_id", "sentence_id"])
             .agg(pl.col("score").max().alias("smax"))
             .rename({"sentence_id": "sent_idx"}))
    return d.with_columns(pl.col("item_id").cast(pl.Int64),
                          pl.col("sent_idx").cast(pl.Int64))


def gaps(z, lab, hop):
    out = {}
    for h in ("single_doc", "multi_doc"):
        m = hop == h
        out[h] = float(z[m & (lab == 1)].mean() - z[m & (lab == 0)].mean())
    return out


def main():
    fam = (pl.read_parquet(HERE / "R19-H162_hotpotqa_families.parquet")
           .select(["item_id", "sent_idx", "label", "hop_class"])
           .with_columns(pl.col("item_id").cast(pl.Int64),
                         pl.col("sent_idx").cast(pl.Int64)))
    ckpts = ("h150d1", "h150d2", "1142", "2142")
    Z, per_ckpt = {}, {}
    base = None
    for t in ckpts:
        j = fam.join(sentence_max(t), on=["item_id", "sent_idx"], how="inner")
        j = j.sort(["item_id", "sent_idx"])
        if base is None:
            base = j.select(["item_id", "sent_idx", "label", "hop_class"])
        else:
            assert j.select(["item_id", "sent_idx"]).equals(
                base.select(["item_id", "sent_idx"])), f"row alignment broke on {t}"
        s = j["smax"].to_numpy().astype(float)
        Z[t] = (s - s.mean()) / s.std(ddof=0)
        g = gaps(Z[t], j["label"].to_numpy(), np.array(j["hop_class"].to_list()))
        per_ckpt[t] = {
            "n_sent": j.height,
            "gap_single_doc": round(g["single_doc"], 4),
            "gap_multi_doc": round(g["multi_doc"], 4),
            "diff_of_gaps": round(g["single_doc"] - g["multi_doc"], 4),
        }
    lab = base["label"].to_numpy()
    hop = np.array(base["hop_class"].to_list())
    n = len(lab)
    point = float(np.mean([per_ckpt[t]["diff_of_gaps"] for t in ckpts]))

    def stat(idx, l_, h_):
        return float(np.mean([
            (lambda g: g["single_doc"] - g["multi_doc"])(gaps(Z[t][idx], l_, h_))
            for t in ckpts]))

    def boot_plain(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            l_, h_ = lab[idx], hop[idx]
            if any(((h_ == h) & (l_ == v)).sum() < 1
                   for h in ("single_doc", "multi_doc") for v in (0, 1)):
                continue
            out.append(stat(idx, l_, h_))
        return np.array(out)

    strata = [np.where((hop == h) & (lab == v))[0]
              for h in ("single_doc", "multi_doc") for v in (0, 1)]

    def boot_strat(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            out.append(stat(idx, lab[idx], hop[idx]))
        return np.array(out)

    boots = boot_plain(SEED)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = [per_ckpt[t]["diff_of_gaps"] > 0 for t in ckpts]

    # seed and variant sensitivity - the registered recipe is the plain
    # resample-over-sentences bootstrap; its lower bound sits ON zero, so both
    # readings are banked rather than a single coin-flip verdict.
    sens = {"plain_over_sentences": {}, "stratified_by_hopclass_label": {}}
    for s_ in (0, 1, 2, SEED, 42):
        b = boot_plain(s_)
        sens["plain_over_sentences"][str(s_)] = {
            "ci95": [round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])],
            "p_le_zero": round(float((b <= 0).mean()), 4)}
    for s_ in (0, SEED):
        b = boot_strat(s_)
        sens["stratified_by_hopclass_label"][str(s_)] = {
            "ci95": [round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])],
            "p_le_zero": round(float((b <= 0).mean()), 4)}

    plain_lows = [v["ci95"][0] for v in sens["plain_over_sentences"].values()]
    if lo > 0 or hi < 0:
        verdict = "PASS"
    elif min(plain_lows) < 0 < max(plain_lows):
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    payload = {
        "gate": "R20 gate 4 - hotpotqa G0a difference-of-label-gaps (single_doc - multi_doc)",
        "recipe_summary": ("join R19-H162_hotpotqa_families on (item_id, sent_idx) to the "
                           "per-window dumps R19-H161_pairs_h150d{1,2} and "
                           "R18-H151_scores_{1142,2142}; per-checkpoint z-scored sentence-max "
                           "(max over windows); label gap = mean(z|1) - mean(z|0) within "
                           "hop_class; statistic = gap(single_doc) - gap(multi_doc) averaged "
                           f"over the 4 checkpoints; {N_BOOT} bootstrap resamples over the "
                           f"{n} sentences, percentile CI95"),
        "n_sentences": n,
        "n_negatives": int((lab == 0).sum()),
        "per_checkpoint": per_ckpt,
        "pooled_diff_of_gaps": round(point, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "sign_stability": f"{sum(signs)}/4 positive",
        "bootstrap_draws_used": int(len(boots)),
        "sensitivity": sens,
        "threshold": "PASS if the bootstrap CI95 excludes zero",
        "verdict": verdict,
        "verdict_note": ("the point estimate reproduces the design pass (+0.8872 vs +0.886) "
                         "and the sign is 4/4, but the CI95 does NOT exclude zero under the "
                         "registered plain resample-over-sentences bootstrap: at 10,000 "
                         "draws all five seeds give a NEGATIVE lower bound (-0.006 to "
                         "-0.060, p(stat<=0) 0.026-0.031). The failure is narrow and "
                         "estimator-dependent - at 5,000 draws two of five seeds flipped "
                         "the bound positive, and a bootstrap stratified by hop_class x "
                         "label puts it at +0.04..+0.08. The design pass's [+0.043, +1.657] "
                         "sits inside that spread, i.e. it was not reproducible at the "
                         "registered recipe's own resolution. Binding limit: the single_doc "
                         "negative cell holds 6 sentences"),
        "design_pass_reference": {"point": 0.886, "ci95": [0.043, 1.657],
                                  "per_checkpoint": [0.908, 0.835, 1.081, 0.718]},
        "timestamp": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("pooled_diff_of_gaps", "ci95", "sign_stability", "verdict")}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
