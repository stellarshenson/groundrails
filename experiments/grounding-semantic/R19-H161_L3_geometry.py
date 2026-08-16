"""R19-H161 lane L3 - ADVERSARIAL GROUP GEOMETRY. ANALYSIS ONLY.

Hypothesis H3: the R19-H159 enriched mix added four prose corpora and one
table corpus, changing WHAT the domain adversary can separate. If gradient
reversal at 19 groups erases more of the prose-versus-table distinction than
it did at 14 groups, the trunk's representation should make table-evidence
versus prose-evidence LESS linearly decodable in the enriched checkpoint.

Nothing trains. Each trunk is frozen, the `[CLS]` vector the scoring head
actually reads is extracted through the banked encode path, and a logistic
regression is fitted on those frozen vectors.

Checkpoints (all banked, none re-trained):

    h150d1  models/R18-H150-arm-draw1   flagship pair draw 1, 14 DANN groups
    h150d2  models/R18-H150-arm-draw2   flagship pair draw 2, 14 DANN groups
    h159d1  models/R19-H159-arm-draw1   enriched arm draw 1, 19 DANN groups

Representation: `R16-H142_G1_arm.load_run` rebuilds the run and
`model.encode(enc)` is the trunk's `last_hidden_state[:, 0]` in fp32 - the
exact vector `pair_logits` and `domain_head` both consume. The tokenizer call
matches `ARM.score_sets` byte for byte (`tok(sentence, window)`, padding,
truncation, `max_length=ARM.MAX_LEN`).

Mix: assembled by the banked `R19-H159_arm_run.make_build_mix` unchanged, so
the 19-group row set, every lane abort and the window-census cross-check
against `R19-H159_window_census.json` all fire here. Rows carrying one of the
14 flagship groups are byte-identical to the H150 mix (the H159 build appends
the five R19 lanes AFTER the flagship's three sources), so the 14-group subset
is the same row set all three checkpoints trained on.

Register map (spec):

    TABLE  tabfact, quant_misbind, quant_scale_unit, findver
    PROSE  everything else

`findver` is NEW and table-like, so the enriched mix added table supply as
well as prose supply. The PRIMARY read therefore restricts to the 14 SHARED
groups, where the row set and the label space are identical for all three
checkpoints; the 19-group read is reported as a secondary, since h150 never
saw the five new lanes.

No holdout exists: the trainer runs one epoch over the full mix and keeps no
eval split, so these rows were trained on by every checkpoint that owns them.
Decodability on trained rows is an upper bound. The comparison stays valid
because all three checkpoints see the SAME rows on the primary read.

Stages (idempotent - each skips when its artifact is on disk):
    sample  CPU only. Assemble the mix, draw the per-group sample, pick one
            window per row -> R19-H161_L3_sample.parquet
    embed   GPU0 only. Frozen [CLS] per checkpoint -> R19-H161_L3_cls_*.npy
    probe   CPU only. 5-fold CV logistic probes -> R19-H161_L3_result.json

Run (detached, GPU0, capped small - a peer agent shares the card):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python \
    experiments/grounding-semantic/R19-H161_L3_geometry.py \
    >> logs/R19-H161_L3.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

SAMPLE_PARQUET = HERE / "R19-H161_L3_sample.parquet"
RESULT = HERE / "R19-H161_L3_result.json"
CLS_NPY = "R19-H161_L3_cls_{name}.npy"

SEED = 191613
N_PER_GROUP = 500  # every group in the 19-group mix has >= 2,400 rows
BATCH = 32  # small on purpose: GPU0 is shared, peak must stay well under 6 GB
N_FOLDS = 5
PROBE_C = 1.0
PROBE_MAX_ITER = 500

CKPTS = {
    "h150d1": "R18-H150-arm-draw1",
    "h150d2": "R18-H150-arm-draw2",
    "h159d1": "R19-H159-arm-draw1",
}

GROUPS_19 = (
    "attributionbench",
    "fava",
    "findver",
    "halueval",
    "minicheck",
    "psiloqa",
    "pubhealth",
    "quant_misbind",
    "quant_scale_unit",
    "ragtruth_cn",
    "ragtruth_de",
    "ragtruth_en",
    "ragtruth_es",
    "ragtruth_fr",
    "ragtruth_hu",
    "ragtruth_it",
    "ragtruth_pl",
    "tabfact",
    "vitaminc",
)
GROUPS_NEW = ("attributionbench", "fava", "findver", "minicheck", "pubhealth")
GROUPS_14 = tuple(g for g in GROUPS_19 if g not in GROUPS_NEW)
TABLE_GROUPS = frozenset({"tabfact", "quant_misbind", "quant_scale_unit", "findver"})

# Per-register sample sizes for the two table-vs-prose reads. Even spread over
# the register's groups, so no single corpus carries the register's signal.
N_PER_REGISTER = 1500


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def register_of(group):
    return "table" if group in TABLE_GROUPS else "prose"


# --- stage: sample --------------------------------------------------------------


def build_sample():
    """Draw N_PER_GROUP rows per DANN group from the banked 19-group mix and
    pick ONE window per row - the (sentence, window) pair the trunk sees."""
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    h159 = _mod("h159", "R19-H159_arm_run.py")
    # The closure carries h159's own group map, row count and census cross-check,
    # so the mix is assembled and verified by the arm's own code, unmodified.
    build_mix = h159.make_build_mix(arm)

    t0 = time.time()
    claims, wsets, y, tags = build_mix()
    print(f"mix assembled: {len(y)} rows in {time.time() - t0:.0f}s", flush=True)

    tags_arr = np.asarray(tags)
    rng = np.random.default_rng(SEED)
    rows = []
    for g in GROUPS_19:
        idx = np.flatnonzero(tags_arr == g)
        if idx.size < N_PER_GROUP:
            raise SystemExit(f"SAMPLE ABORT: group {g} has {idx.size} rows < {N_PER_GROUP}")
        take = rng.permutation(idx)[:N_PER_GROUP]
        for rank, i in enumerate(take):
            i = int(i)
            nw = len(wsets[i])
            wj = int(rng.integers(nw))
            rows.append(
                {
                    "row_id": i,
                    "group": g,
                    "register": register_of(g),
                    "rank_in_group": rank,
                    "label": float(y[i]),
                    "n_windows": nw,
                    "win_idx": wj,
                    "sentence": claims[i],
                    "window": wsets[i][wj],
                }
            )
        print(f"  {g:<18} pool {idx.size:>7}  sampled {N_PER_GROUP}", flush=True)

    df = pl.DataFrame(rows)
    df.write_parquet(SAMPLE_PARQUET)
    print(
        f"sample -> {SAMPLE_PARQUET.name}  {df.height} pairs "
        f"({df.filter(pl.col('register') == 'table').height} table)",
        flush=True,
    )
    return df


# --- stage: embed ---------------------------------------------------------------


def embed_all(df):
    """Frozen [CLS] per checkpoint, through the banked load_run + encode path."""
    import torch

    dev = torch.cuda.get_device_name(0)
    print(
        f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True
    )
    if "RTX PRO 4000" not in dev:
        raise SystemExit(
            f"WRONG GPU: {dev} - R19-H161 L3 is pinned to card 0 "
            "(cards 1 and 2 carry the R19-H160 draws)"
        )

    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    sents = df["sentence"].to_list()
    wins = df["window"].to_list()
    n = len(sents)

    for name, ckpt in CKPTS.items():
        out = HERE / CLS_NPY.format(name=name)
        if out.exists():
            print(f"  {name}: cached {out.name}", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        model, tok = arm.load_run(ROOT / "models" / ckpt)
        d = model.trunk.config.hidden_size
        cls = np.zeros((n, d), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, n, BATCH):
                enc = tok(
                    sents[i : i + BATCH],
                    wins[i : i + BATCH],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=arm.MAX_LEN,
                )
                enc = {k: v.cuda() for k, v in enc.items()}
                cls[i : i + BATCH] = model.encode(enc).float().cpu().numpy()
                if (i // BATCH) % 50 == 0 and i:
                    print(
                        f"    {name} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)",
                        flush=True,
                    )
        peak = torch.cuda.max_memory_allocated() / 2**30
        np.save(out, cls)
        del model
        torch.cuda.empty_cache()
        print(
            f"  {name}: {n} pairs in {time.time() - t0:.0f}s, peak {peak:.2f} GB -> {out.name}",
            flush=True,
        )


# --- stage: probe ---------------------------------------------------------------


def _fit_cv(X, lab, task):
    """5-fold stratified CV logistic probe. Binary -> AUROC, multiclass ->
    accuracy. Scaler is fitted inside each fold, never on the test rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold, oof = [], np.zeros(len(lab), dtype=np.float64)
    oof_pred = np.empty(len(lab), dtype=object)
    for tr, te in skf.split(X, lab):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=PROBE_MAX_ITER, C=PROBE_C, n_jobs=-1)
        m.fit(sc.transform(X[tr]), lab[tr])
        if task == "binary":
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            oof[te] = p
            per_fold.append(float(roc_auc_score(lab[te], p)))
        else:
            pred = m.predict(sc.transform(X[te]))
            oof_pred[te] = pred
            per_fold.append(float((pred == lab[te]).mean()))
    pooled = (
        float(roc_auc_score(lab, oof)) if task == "binary" else float((oof_pred == lab).mean())
    )
    return {
        "pooled": round(pooled, 5),
        "fold_mean": round(float(np.mean(per_fold)), 5),
        "fold_sd": round(float(np.std(per_fold, ddof=1)), 5),
        "folds": [round(v, 5) for v in per_fold],
        "n": len(lab),
    }, oof_pred


def even_slice(df, groups, total):
    """Take `total` rows spread as evenly as possible over `groups`, using the
    sample's own draw order (`rank_in_group`) so the pick is deterministic and
    identical for every checkpoint."""
    base, rem = divmod(total, len(groups))
    keep = []
    for k, g in enumerate(sorted(groups)):
        n = base + (1 if k < rem else 0)
        keep.append(df.filter((pl.col("group") == g) & (pl.col("rank_in_group") < n)))
    return pl.concat(keep).sort("sample_idx")


def probe_all(df):
    df = df.with_row_index("sample_idx")
    embeds = {name: np.load(HERE / CLS_NPY.format(name=name)) for name in CKPTS}
    for name, X in embeds.items():
        if X.shape[0] != df.height:
            raise SystemExit(f"SHAPE ABORT: {name} has {X.shape[0]} rows, sample has {df.height}")

    table14 = sorted(g for g in GROUPS_14 if g in TABLE_GROUPS)
    prose14 = sorted(g for g in GROUPS_14 if g not in TABLE_GROUPS)
    table19 = sorted(g for g in GROUPS_19 if g in TABLE_GROUPS)
    prose19 = sorted(g for g in GROUPS_19 if g not in TABLE_GROUPS)

    reads = {
        "table_vs_prose_shared14": {
            "kind": "binary",
            "rows": pl.concat(
                [even_slice(df, table14, N_PER_REGISTER), even_slice(df, prose14, N_PER_REGISTER)]
            ).sort("sample_idx"),
            "label": lambda d: (d["register"] == "table").cast(pl.Int8).to_numpy(),
            "note": "PRIMARY - only the 14 flagship groups, so the row set is "
            "identical for all three checkpoints and every row was trained on "
            "by all three",
        },
        "table_vs_prose_all19": {
            "kind": "binary",
            "rows": pl.concat(
                [even_slice(df, table19, N_PER_REGISTER), even_slice(df, prose19, N_PER_REGISTER)]
            ).sort("sample_idx"),
            "label": lambda d: (d["register"] == "table").cast(pl.Int8).to_numpy(),
            "note": "SECONDARY - includes the five new lanes (findver on the table "
            "side), which h150d1/h150d2 never saw; not a like-for-like read",
        },
        "group14way": {
            "kind": "multi",
            "rows": df.filter(pl.col("group").is_in(list(GROUPS_14))),
            "label": lambda d: d["group"].to_numpy(),
            "note": "identical 14-way label space and identical rows across all three "
            "checkpoints - the comparable multi-way read",
        },
        "group19way": {
            "kind": "multi",
            "rows": df,
            "label": lambda d: d["group"].to_numpy(),
            "note": "INFORMATIONAL ONLY - a 19-way read is not comparable with the "
            "14-way one and h150 never saw five of the classes",
        },
    }

    out = {}
    for read, spec in reads.items():
        rows = spec["rows"]
        idx = rows["sample_idx"].to_numpy()
        lab = spec["label"](rows)
        blk = {
            "note": spec["note"],
            "n": len(idx),
            "groups": sorted(rows["group"].unique().to_list()),
        }
        if spec["kind"] == "binary":
            blk["n_table"] = int((lab == 1).sum())
            blk["n_prose"] = int((lab == 0).sum())
        for name in CKPTS:
            t0 = time.time()
            stats, oof_pred = _fit_cv(embeds[name][idx], lab, spec["kind"])
            blk[name] = stats
            if spec["kind"] == "multi":
                rec = {}
                for g in sorted(set(lab)):
                    m = lab == g
                    rec[str(g)] = round(float((oof_pred[m] == g).mean()), 4)
                blk[name]["per_group_recall"] = rec
            metric = "auroc" if spec["kind"] == "binary" else "acc"
            print(
                f"  {read:<24} {name}  {metric} {stats['pooled']:.5f} "
                f"(fold mean {stats['fold_mean']:.5f} sd {stats['fold_sd']:.5f}, "
                f"{time.time() - t0:.0f}s)",
                flush=True,
            )
        out[read] = blk
    return out


def adjudicate(reads):
    """The flagship's own two draws are the noise floor. H3 is SUPPORTED only
    if the enriched checkpoint falls below BOTH flagship draws by more than the
    gap between them."""
    p = reads["table_vs_prose_shared14"]
    d1, d2, e = (p["h150d1"]["pooled"], p["h150d2"]["pooled"], p["h159d1"]["pooled"])
    floor = abs(d1 - d2)
    lo = min(d1, d2)
    delta = e - lo  # negative = enriched is LESS decodable, H3's direction
    supported = e < lo - floor
    if supported:
        verdict, reading = (
            "SUPPORTED",
            (
                "the enriched trunk makes table-vs-prose decodably weaker than either "
                "flagship draw, by more than the flagship's own 2-draw gap"
            ),
        )
    elif e > max(d1, d2) + floor:
        verdict, reading = (
            "NOT_SUPPORTED",
            (
                "the enriched trunk makes table-vs-prose MORE decodable than either "
                "flagship draw - the opposite of H3's prediction"
            ),
        )
    else:
        verdict, reading = (
            "NOT_SUPPORTED",
            (
                "the enriched trunk's table-vs-prose decodability sits inside the "
                "flagship's own 2-draw noise band - no extra register erasure to find"
            ),
        )
    return {
        "verdict": verdict,
        "reading": reading,
        "primary_read": "table_vs_prose_shared14",
        "h150d1": d1,
        "h150d2": d2,
        "h159d1": e,
        "flagship_2draw_gap": round(floor, 5),
        "enriched_minus_nearest_flagship": round(delta, 5),
        "bar": "h159d1 < min(h150d1, h150d2) - |h150d1 - h150d2|",
        "bar_value": round(lo - floor, 5),
        "clears_noise_floor": bool(supported),
    }


# --- driver ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("sample", "embed", "probe", "all"), default="all")
    args = ap.parse_args()

    print(f"=== R19-H161 L3 adversarial group geometry  {time.strftime('%F %T')} ===", flush=True)

    if args.stage in ("sample", "all"):
        if SAMPLE_PARQUET.exists():
            print(f"sample cached -> {SAMPLE_PARQUET.name}", flush=True)
        else:
            build_sample()
    if args.stage == "sample":
        return

    df = pl.read_parquet(SAMPLE_PARQUET)
    if args.stage in ("embed", "all"):
        embed_all(df)
    if args.stage == "embed":
        return

    reads = probe_all(df)
    ruling = adjudicate(reads)
    res = {
        "lane": "L3",
        "hypothesis": "H3 adversarial group geometry",
        "question": "does the 19-group adversary erase more of the prose-versus-table "
        "distinction from the trunk than the 14-group adversary did",
        "method": "frozen trunks, [CLS] from the banked R16-H142_G1_arm.load_run + "
        "model.encode path (the vector both pair_logits and domain_head read), "
        f"logistic probes at C={PROBE_C} under {N_FOLDS}-fold stratified CV "
        "with the scaler fitted inside each fold",
        "checkpoints": {k: f"models/{v}" for k, v in CKPTS.items()},
        "data": f"banked R19-H159 19-group training mix (assembled by "
        f"R19-H159_arm_run.make_build_mix unchanged, window census cross-checked), "
        f"{N_PER_GROUP} rows per group, one uniformly drawn window per row, "
        f"seed {SEED}",
        "register_map": {
            "table": sorted(TABLE_GROUPS),
            "prose": sorted(set(GROUPS_19) - TABLE_GROUPS),
        },
        "new_groups_in_h159": list(GROUPS_NEW),
        "shared_groups": list(GROUPS_14),
        "reads": reads,
        "ruling": ruling,
        "caveats": [
            "No holdout exists - the trainer runs one epoch over the full mix and keeps "
            "no eval split, so every sampled row was trained on by the checkpoints that "
            "own it. Decodability on trained rows is an upper bound; the primary read "
            "keeps the comparison fair by using only rows all three checkpoints saw.",
            "findver is a NEW group and also table-like, so the enriched mix added table "
            "supply as well as prose supply. The primary read excludes it, so it measures "
            "erasure on the flagship's own register geometry rather than the enriched "
            "mix's larger table pool.",
            "One window per row is sampled, not the whole window set - the probe reads "
            "the per-pair representation, which is what the domain head is trained on.",
            "ANALYSIS ONLY - nothing trains, nothing is tuned on arena statistics, and "
            "the arena is not touched at all.",
        ],
        "note": "Numbers recorded, not adjudicated - the coordinator holds the verdict.",
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 92)
    print("R19-H161 L3  ADVERSARIAL GROUP GEOMETRY")
    print("=" * 92)
    p = reads["table_vs_prose_shared14"]
    print(
        f"  table-vs-prose AUROC (14 shared groups, {p['n']} pairs): "
        f"h150d1 {p['h150d1']['pooled']:.5f}  h150d2 {p['h150d2']['pooled']:.5f}  "
        f"h159d1 {p['h159d1']['pooled']:.5f}"
    )
    q = reads["table_vs_prose_all19"]
    print(
        f"  table-vs-prose AUROC (all 19 groups, {q['n']} pairs):    "
        f"h150d1 {q['h150d1']['pooled']:.5f}  h150d2 {q['h150d2']['pooled']:.5f}  "
        f"h159d1 {q['h159d1']['pooled']:.5f}"
    )
    r = reads["group14way"]
    print(
        f"  14-way group accuracy ({r['n']} rows, chance {1 / 14:.4f}):   "
        f"h150d1 {r['h150d1']['pooled']:.5f}  h150d2 {r['h150d2']['pooled']:.5f}  "
        f"h159d1 {r['h159d1']['pooled']:.5f}"
    )
    print(f"\n  noise floor (flagship 2-draw gap): {ruling['flagship_2draw_gap']:.5f}")
    print(f"  bar: h159d1 < {ruling['bar_value']:.5f}   actual {ruling['h159d1']:.5f}")
    print(f"  VERDICT: {ruling['verdict']} - {ruling['reading']}")
    print(f"  -> {RESULT}")
    print("=== R19-H161 L3 DONE ===", flush=True)


if __name__ == "__main__":
    main()
