"""R19-H162 - PROCEDURAL-REGISTER MECHANISM DISSECTION (emanual + techqa). ANALYSIS ONLY.

Executor M4 of the R19-H162 mechanism-dissection wave. Targets the two
procedural / technical-documentation subsets of the blind arena:

    emanual   consumer-electronics manual QA      flagship 2-draw 0.6780
    techqa    enterprise technical-support QA     flagship 2-draw 0.7335

Nothing here trains, tunes, or selects on arena statistics (the H141
discipline). No GPU: every model score is read from the banked R19-H161
per-pair logit dump (`R19-H161_pairs_{h150d1,h150d2,h159d1}.parquet`), whose own
positive control reproduced the banked windowed AUROCs to <= 1e-3.

Read convention (inherited, not re-derived): the frozen gate sample from
`R8-H77.load_subsets` (adherence non-null, response > 20 chars, documents
non-empty, sample(min(250, n), seed=0), documents[:8]); each H92 sentence of the
response against every 1,500-char window (stride 750) of every retained
document; MAX over windows per sentence, then MIN over sentences per item.

Stages (all CPU):

  geometry  - rebuild the (sentence, window) text for both subsets, byte-identical
              to the dump's pair order, and join it onto the dump
  inflate   - the max-over-many-windows inflation test on `n_win_sent`
  errors    - error split at the in-sample macro-F1-optimal threshold, sinking
              sentence extraction, surface-feature contrast
  export    - error records with full text for manual reading

Run:  uv run python experiments/grounding-semantic/R19-H162_procedural_autopsy.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

SUBSETS = ("emanual", "techqa")
PEER = "delucionqa"  # the sibling procedural subset, read as a cross-check only

DUMPS = {
    "h150d1": HERE / "R19-H161_pairs_h150d1.parquet",
    "h150d2": HERE / "R19-H161_pairs_h150d2.parquet",
    "h159d1": HERE / "R19-H161_pairs_h159d1.parquet",
}

OUT_GEOM = HERE / "R19-H162_procedural_geometry.parquet"
OUT_JSON = HERE / "R19-H162_procedural_inflation.json"
OUT_ERRORS = HERE / "R19-H162_procedural_errors.parquet"

RNG_SEED = 20260814


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA


def available_dumps():
    return {k: v for k, v in DUMPS.items() if v.exists()}


# --- geometry -----------------------------------------------------------------------


def build_geometry(subs, subset):
    """Sentence and window text keyed by (item_id, sent_idx) / (item_id, win_idx).

    Rebuilds exactly the pair enumeration of R19-H161_dump.build_subset: windows
    are flattened document-major, `win_idx` indexes that flat list.
    """
    claims, chunk_lists, _y = subs[subset]
    sent_rows, win_rows = [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wi = 0
        for di, k in enumerate(ks):
            for w in ARM.windows(k):
                win_rows.append({"item_id": i, "win_idx": wi, "doc_idx": di, "win_text": w})
                wi += 1
        for si, s in enumerate(H92.sentences(c)):
            sent_rows.append({"item_id": i, "sent_idx": si, "sent_text": s})
    return (
        pl.DataFrame(sent_rows, schema_overrides={"item_id": pl.Int32, "sent_idx": pl.Int16}),
        pl.DataFrame(
            win_rows,
            schema_overrides={"item_id": pl.Int32, "win_idx": pl.Int16, "doc_idx": pl.Int16},
        ),
    )


# --- shared helpers -----------------------------------------------------------------


def op_threshold(y, s):
    """Macro-F1-optimal threshold, in-sample - the R17-H147/R18-H157 stated choice.

    Nothing is tuned on it; the threshold-free AUROC is reported alongside.
    """
    cand = np.unique(s)
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 400))
    best, best_t = -1.0, float(np.median(s))
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro", zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def item_scores(d):
    """MAX over windows per sentence, then MIN over sentences per item, on the logit."""
    sent = (
        d.group_by(["item_id", "sent_idx"])
        .agg(pl.col("logit").max().alias("s"), pl.col("label").first())
        .sort(["item_id", "sent_idx"])
    )
    item = (
        sent.group_by("item_id")
        .agg(pl.col("s").min().alias("score"), pl.col("label").first())
        .sort("item_id")
    )
    return item


def auroc(y, s):
    return float(roc_auc_score(y, s))


def auroc_se(y, s):
    """Hanley-McNeil standard error - the instrument's resolution at this base rate."""
    a = auroc(y, s)
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


# --- inflation test -----------------------------------------------------------------


def inflation_test(d, rng, n_boot=200):
    """Does MAX over many windows inflate scores and cost AUROC?

    Three readings:
      1. mechanical  - E[sent_score | n_win_sent], positives vs negatives
      2. item-level  - partial correlation of item_score with the item's window
                       count, and AUROC of window count alone (a label leak check)
      3. counterfactual - subsample K windows per sentence uniformly at random and
                       recompute the item score; if AUROC rises as K falls, the
                       extra windows are net noise, not net evidence
    """
    out = {}

    sent = d.group_by(["item_id", "sent_idx"]).agg(
        pl.col("logit").max().alias("s"),
        pl.col("label").first(),
        pl.col("n_win_sent").first(),
        pl.col("tok_containment").max().alias("tok_cont_max"),
    )

    # 1. mechanical - sentence max vs window count
    binned = (
        sent.with_columns(
            pl.when(pl.col("n_win_sent") <= 5)
            .then(pl.lit("<=5"))
            .when(pl.col("n_win_sent") <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col("n_win_sent") <= 20)
            .then(pl.lit("11-20"))
            .when(pl.col("n_win_sent") <= 40)
            .then(pl.lit("21-40"))
            .otherwise(pl.lit(">40"))
            .alias("bin")
        )
        .group_by(["bin", "label"])
        .agg(
            pl.len().alias("n"),
            pl.col("s").mean().alias("mean_sent_logit"),
            pl.col("tok_cont_max").mean().alias("mean_tok_cont"),
        )
        .sort(["bin", "label"])
    )
    out["sent_max_by_window_count"] = [
        {
            "bin": r["bin"],
            "label": int(r["label"]),
            "n": int(r["n"]),
            "mean_sent_logit": round(float(r["mean_sent_logit"]), 4),
            "mean_argmax_tok_containment": round(float(r["mean_tok_cont"]), 4),
        }
        for r in binned.iter_rows(named=True)
    ]
    sp = sent.select(
        pl.corr("n_win_sent", "s", method="spearman").alias("all"),
    )
    out["spearman_windowcount_vs_sentmax"] = round(float(sp["all"][0]), 4)

    # 2. item level
    item = item_scores(d)
    wc = d.group_by("item_id").agg(pl.col("n_win_sent").mean().alias("mean_win"))
    item = item.join(wc, on="item_id").sort("item_id")
    y = item["label"].to_numpy()
    s = item["score"].to_numpy()
    w = item["mean_win"].to_numpy()
    out["auroc"] = round(auroc(y, s), 5)
    out["auroc_se"] = round(auroc_se(y, s), 5)
    out["auroc_of_window_count_alone"] = round(auroc(y, w), 5)
    out["spearman_item_windowcount_vs_score"] = round(
        float(pl.DataFrame({"w": w, "s": s}).select(pl.corr("w", "s", method="spearman"))[0, 0]), 4
    )
    # partial: within-label correlation, so a label-window confound cannot drive it
    for lab in (0, 1):
        m = y == lab
        if m.sum() > 5:
            out[f"spearman_item_windowcount_vs_score_label{lab}"] = round(
                float(
                    pl.DataFrame({"w": w[m], "s": s[m]}).select(
                        pl.corr("w", "s", method="spearman")
                    )[0, 0]
                ),
                4,
            )

    # 3. counterfactual - cap the max's window set
    caps = {}
    pairs = d.select(["item_id", "sent_idx", "label", "logit", "n_win_sent"])
    for K in (3, 5, 10, 20, 40):
        aucs = []
        for _ in range(n_boot):
            r = pairs.with_columns(pl.Series("r", rng.random(pairs.height), dtype=pl.Float64))
            kept = r.sort("r").group_by(["item_id", "sent_idx"], maintain_order=True).head(K)
            it = (
                kept.group_by(["item_id", "sent_idx"])
                .agg(pl.col("logit").max().alias("s"), pl.col("label").first())
                .group_by("item_id")
                .agg(pl.col("s").min().alias("score"), pl.col("label").first())
                .sort("item_id")
            )
            aucs.append(auroc(it["label"].to_numpy(), it["score"].to_numpy()))
        caps[f"K={K}"] = {
            "mean_auroc": round(float(np.mean(aucs)), 5),
            "sd": round(float(np.std(aucs)), 5),
            "delta_vs_full": round(float(np.mean(aucs)) - out["auroc"], 5),
        }
    out["random_window_cap"] = caps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()

    have = available_dumps()
    print(f"dumps available: {sorted(have)}", flush=True)
    if "h150d1" not in have:
        raise SystemExit("h150d1 dump missing - nothing to read")

    subs = ARENA.load_subsets()
    rng = np.random.default_rng(RNG_SEED)

    report = {
        "read": "PRIMARY windowed decomposed-min (1500/750, MAX over windows, MIN over sentences)",
        "source": "banked R19-H161 per-pair logit dump; no GPU, no re-scoring",
        "dumps_used": sorted(have),
        "subsets": {},
    }

    geom_frames = []
    for sub in SUBSETS + (PEER,):
        _claims, chunk_lists, y = subs[sub]
        y = np.asarray(y)
        sent_txt, win_txt = build_geometry(subs, sub)
        geom_frames.append(
            sent_txt.with_columns(pl.lit(sub).alias("subset")).select(
                ["subset", "item_id", "sent_idx", "sent_text"]
            )
        )
        blk = {
            "n_items": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "n_sentences": int(sent_txt.height),
            "n_windows_total": int(win_txt.height),
            "mean_docs_per_item": float(np.mean([len(k) for k in chunk_lists])),
            "mean_doc_chars": float(np.mean([len(dd) for k in chunk_lists for dd in k])),
        }
        report["subsets"][sub] = blk
        print(f"{sub}: {blk}", flush=True)

    pl.concat(geom_frames).write_parquet(OUT_GEOM)
    print(f"wrote {OUT_GEOM}", flush=True)

    for tag, path in sorted(have.items()):
        df = pl.read_parquet(path)
        for sub in SUBSETS + (PEER,):
            d = df.filter(pl.col("subset") == sub)
            if d.height == 0:
                continue
            res = inflation_test(d, rng, n_boot=args.boot)
            report["subsets"][sub].setdefault("inflation", {})[tag] = res
            print(
                f"  {tag}/{sub} auroc {res['auroc']:.4f} (SE {res['auroc_se']:.4f}) "
                f"| wc-alone {res['auroc_of_window_count_alone']:.4f} "
                f"| caps {[(k, v['delta_vs_full']) for k, v in res['random_window_cap'].items()]}",
                flush=True,
            )

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
