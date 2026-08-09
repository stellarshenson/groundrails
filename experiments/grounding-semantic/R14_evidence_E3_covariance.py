"""R14 Evidence E3 - config x subset covariance / trade-structure analysis.

ANALYSIS ONLY. No training, no GPU. Reads banked arena read JSONs, builds the
config x subset AUROC matrix with Polars, and computes subset-subset covariance
of per-config deltas vs the clean baseline of the same read type.
"""

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

D = Path("/home/lab/workspace/private/ai-assistants/groundrails/experiments/grounding-semantic")
SUBSETS = [
    "covidqa", "delucionqa", "emanual", "expertqa", "finqa",
    "hagrid", "hotpotqa", "pubmedqa", "tatqa", "techqa",
]


def load(fn):
    return json.loads((D / fn).read_text())


def ps_auc(fn, key="per_subset", field="auc"):
    d = load(fn)
    for part in key.split("/"):
        d = d[part]
    out = {}
    for s in SUBSETS:
        v = d[s]
        out[s] = float(v[field]) if isinstance(v, dict) else float(v)
    return out


def ps_flat(fn, key):
    d = load(fn)
    for part in key.split("/"):
        d = d[part]
    return {s: float(d[s]) for s in SUBSETS}


# ---------------------------------------------------------------- windowed rows
# family: TRAIN = distinct trained checkpoint read through the standard windowed
#         decomposed-min read; READVAR = frozen weights, non-standard read/ensemble.
W = []


def add(cfg, family, era, lane, vec):
    W.append((cfg, family, era, lane, vec))


add("h90_full", "TRAIN", "full", "none", ps_auc("R8-H101_result.json"))
add("h100_replicate", "TRAIN", "full", "none", ps_auc("R8-H101_replicate_result.json"))
add("h100_draw3", "TRAIN", "full", "none", ps_auc("R8-H101_draw3_result.json"))
add("h102_twohead_score", "TRAIN", "full", "twohead",
    ps_auc("R8-H102_reads.json", "reads/score_windowed/per_subset"))
add("h105d1_clean", "TRAIN", "clean", "none", ps_auc("R9-H105_windowed_result.json"))
add("h105d2_clean", "TRAIN", "clean", "none", ps_auc("R9-H105_draw2_windowed_result.json"))
add("h107d1_procedural", "TRAIN", "clean", "procedural",
    ps_auc("R10-H107_lane_draw1_windowed_result.json"))
add("h107d2_procedural", "TRAIN", "clean", "procedural",
    ps_auc("R10-H107_lane_draw2_windowed_result.json"))
add("h108d1_quant", "TRAIN", "clean", "quant",
    ps_auc("R10-H108_lane_draw1_windowed_result.json"))
add("h108d2_quant", "TRAIN", "clean", "quant",
    ps_auc("R10-H108_lane_draw2_windowed_result.json"))
add("drd1_control", "TRAIN", "clean", "DR",
    ps_auc("DR_lane_draw1_control_windowed_result.json"))
add("drd2_control", "TRAIN", "clean", "DR",
    ps_auc("DR_lane_draw2_control_windowed_result.json"))
add("drd1_margin", "TRAIN", "clean", "DR+margin",
    ps_auc("DR_lane_draw1_margin_windowed_result.json"))
add("h118_soup", "TRAIN", "clean", "soup(h105 d1+d2)",
    ps_auc("R11-H118_soup_h105_windowed_result.json"))

# read / ensemble variants on frozen weights
add("h102_token_head", "READVAR", "full", "twohead",
    ps_auc("R8-H102_reads.json", "reads/token_windowed/per_subset"))
add("h104_fused_head", "READVAR", "full", "twohead",
    ps_auc("R8-H104_result.json", "reads/fused_windowed/per_subset"))
add("anchor_teacher_h105pair", "READVAR", "clean", "prob-mean(h105 d1,d2)",
    ps_flat("R13_anchor_teacher_result.json", "anchor_per_subset"))
for tag in ["h108d1", "h108d2", "h105d1", "h105d2"]:
    add(f"h124_consensus_{tag}", "READVAR", "clean", "top2-window consensus read",
        ps_flat("R13-H124_result.json", f"per_checkpoint/{tag}/consensus_per_subset"))
add("h125_union_h108d1", "READVAR", "clean", "union read",
    ps_flat("R13-H125_result_h108d1.json", "union_per_subset"))
for tag in ["h105d1", "h105d2", "h108d1", "h108d2"]:
    for arm in ["strip", "add"]:
        add(f"h119_{arm}_{tag}", "READVAR", "clean", f"numeric-canon {arm}",
            ps_auc(f"R12-H119_{tag}_{arm}_windowed_result.json"))

# --------------------------------------------------------------- truncated rows
T = []
dec = load("R8_decomposed_reads.json")
DEC_META = {
    "R8-H73": ("full", "none"), "R8-H90": ("full", "none"), "R8-H91": ("full", "none"),
    "R8-H95": ("full", "groupdro"), "R8-H96": ("full", "groupdro"),
    "R8-H99": ("full", "lambda"), "R8-H100": ("full", "none"),
    "R8-H100-draw3": ("full", "none"), "R9-H105": ("clean", "none"),
    "R9-H105-draw2": ("clean", "none"), "R9-H106": ("clean", "fusion"),
    "R10-H107-lane-draw1": ("clean", "procedural"),
    "R10-H107-lane-draw2": ("clean", "procedural"),
    "R10-H108-lane-draw1": ("clean", "quant"),
    "R10-H108-lane-draw2": ("clean", "quant"),
    "DR-lane-draw1-control": ("clean", "DR"),
    "DR-lane-draw2-control": ("clean", "DR"),
    "DR-lane-draw1-margin": ("clean", "DR+margin"),
}
for k, (era, lane) in DEC_META.items():
    T.append((k, "TRAIN", era, lane,
              {s: float(dec[k]["per_subset"][s]["auc"]) for s in SUBSETS}))
T.append(("R8-H102-token-trunc", "READVAR", "full", "twohead",
          ps_auc("R8-H102_reads.json", "reads/token_truncated/per_subset")))
T.append(("R8-H102-score-trunc", "READVAR", "full", "twohead",
          ps_auc("R8-H102_reads.json", "reads/score_truncated/per_subset")))
T.append(("R8-H104-fused-trunc", "READVAR", "full", "twohead",
          ps_auc("R8-H104_result.json", "reads/fused_truncated/per_subset")))


def frame(rows):
    recs = []
    for cfg, fam, era, lane, vec in rows:
        r = {"config": cfg, "family": fam, "era": era, "lane": lane}
        r.update({s: vec[s] for s in SUBSETS})
        r["mean"] = round(sum(vec[s] for s in SUBSETS) / 10, 5)
        recs.append(r)
    return pl.DataFrame(recs)


wdf = frame(W)
tdf = frame(T)

# ------------------------------------------------------------------- baselines
w_base = {s: (wdf.filter(pl.col("config") == "h105d1_clean")[s][0]
              + wdf.filter(pl.col("config") == "h105d2_clean")[s][0]) / 2 for s in SUBSETS}
t_base = {s: (tdf.filter(pl.col("config") == "R9-H105")[s][0]
              + tdf.filter(pl.col("config") == "R9-H105-draw2")[s][0]) / 2 for s in SUBSETS}


def deltas(df, base):
    return df.with_columns([(pl.col(s) - base[s]).round(4).alias(s) for s in SUBSETS]).with_columns(
        pl.mean_horizontal([pl.col(s) for s in SUBSETS]).round(4).alias("mean_delta")
    )


wd = deltas(wdf, w_base)
td = deltas(tdf, t_base)


def corr_block(df, subsets=SUBSETS, demean_rows=False):
    M = df.select(subsets).to_numpy().astype(float)
    if demean_rows:
        M = M - M.mean(axis=1, keepdims=True)
    n = M.shape[0]
    R = np.corrcoef(M, rowvar=False)
    P = np.ones_like(R)
    for i in range(len(subsets)):
        for j in range(len(subsets)):
            if i != j:
                r = R[i, j]
                r = max(min(r, 0.999999), -0.999999)
                t = r * math.sqrt((n - 2) / (1 - r * r))
                P[i, j] = 2 * (1 - stats.t.cdf(abs(t), n - 2))
    return R, P, n


def show(R, subsets=SUBSETS, title=""):
    print(f"\n--- {title}")
    print("           " + " ".join(f"{s[:6]:>7}" for s in subsets))
    for i, s in enumerate(subsets):
        print(f"{s:>11}" + " ".join(f"{R[i, j]:7.2f}" for j in range(len(subsets))))


def pair_table(R, P, n, subsets=SUBSETS):
    rows = []
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            rows.append({"a": subsets[i], "b": subsets[j],
                         "r": round(float(R[i, j]), 3), "p": round(float(P[i, j]), 4)})
    return pl.DataFrame(rows).sort("r")


pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_float("mixed")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

out = {}

print("=== WINDOWED matrix (TRAIN family) ===")
wt = wd.filter(pl.col("family") == "TRAIN")
print(wt.select(["config", "era", "lane"] + SUBSETS + ["mean_delta"]))
Rw, Pw, nw = corr_block(wt)
show(Rw, title=f"windowed TRAIN raw delta corr (n={nw} configs)")
Rwc, Pwc, _ = corr_block(wt, demean_rows=True)
show(Rwc, title="windowed TRAIN LEVEL-REMOVED (row-demeaned) corr")

print("\nranked pairs raw:")
print(pair_table(Rw, Pw, nw))
print("\nranked pairs level-removed:")
print(pair_table(Rwc, Pwc, nw))

print("\n=== WINDOWED ALL (TRAIN + READVAR) ===")
Rwa, Pwa, nwa = corr_block(wd)
show(Rwa, title=f"windowed ALL raw delta corr (n={nwa})")
Rwac, Pwac, _ = corr_block(wd, demean_rows=True)
show(Rwac, title="windowed ALL level-removed corr")

print("\n=== TRUNCATED matrix (TRAIN family) ===")
tt = td.filter(pl.col("family") == "TRAIN")
print(tt.select(["config", "era", "lane"] + SUBSETS + ["mean_delta"]))
Rt, Pt, nt = corr_block(tt)
show(Rt, title=f"truncated TRAIN raw delta corr (n={nt})")
Rtc, Ptc, _ = corr_block(tt, demean_rows=True)
show(Rtc, title="truncated TRAIN level-removed corr")
print("\nranked pairs truncated raw:")
print(pair_table(Rt, Pt, nt))
print("\nranked pairs truncated level-removed:")
print(pair_table(Rtc, Ptc, nt))

# ------------------------------------------------------- (b) joint finqa+deluc
print("\n=== (b) configs lifting finqa AND delucionqa vs clean baseline ===")
for name, df in [("windowed", wd), ("truncated", td)]:
    j = df.filter((pl.col("finqa") > 0) & (pl.col("delucionqa") > 0)).select(
        ["config", "family", "lane", "finqa", "delucionqa", "hotpotqa", "mean_delta"])
    print(f"\n{name}: both>0")
    print(j)
    j2 = df.filter((pl.col("finqa") > 0.03) & (pl.col("delucionqa") > 0.03))
    print(f"{name}: both > +0.03 (noise band): {j2['config'].to_list()}")

# ------------------------------------------------------------------ (c) anchor
print("\n=== (c) anchor teacher per-subset delta vs H105 pair mean ===")
print(wd.filter(pl.col("config") == "anchor_teacher_h105pair").select(SUBSETS + ["mean_delta"]))
a = load("R13_anchor_teacher_result.json")
d1 = a["reproduction_guard"]["per_draw"]["h105d1"]["per_subset"]
d2 = a["reproduction_guard"]["per_draw"]["h105d2"]["per_subset"]
rows = []
for s in SUBSETS:
    x, y, an = d1[s]["banked"], d2[s]["banked"], a["anchor_per_subset"][s]
    rows.append({"subset": s, "h105d1": x, "h105d2": y, "pair_mean": round((x + y) / 2, 5),
                 "anchor": an, "d_vs_pairmean": round(an - (x + y) / 2, 4),
                 "d_vs_best_draw": round(an - max(x, y), 4),
                 "d_vs_worst_draw": round(an - min(x, y), 4),
                 "draw_spread": round(abs(x - y), 4)})
adf = pl.DataFrame(rows)
print(adf)
print("anchor beats BOTH draws on:", adf.filter(pl.col("d_vs_best_draw") > 0)["subset"].to_list())
print("anchor below worst draw on:", adf.filter(pl.col("d_vs_worst_draw") < 0)["subset"].to_list())

# --------------------------------------------------------------- (d) noise vs structure
print("\n=== (d) seed-replicate noise vs config structure (windowed) ===")
REPL = [("h105d1_clean", "h105d2_clean", "H105 clean"),
        ("h107d1_procedural", "h107d2_procedural", "H107 procedural"),
        ("h108d1_quant", "h108d2_quant", "H108 quant"),
        ("drd1_control", "drd2_control", "DR control"),
        ("h100_replicate", "h100_draw3", "H100 full-era")]
rows = []
for a_, b_, lbl in REPL:
    va = wdf.filter(pl.col("config") == a_)
    vb = wdf.filter(pl.col("config") == b_)
    for s in SUBSETS:
        rows.append({"pair": lbl, "subset": s, "diff": round(va[s][0] - vb[s][0], 4)})
rep = pl.DataFrame(rows)
noise = rep.group_by("subset").agg(
    (pl.col("diff").pow(2).sum() / (2 * pl.len())).sqrt().alias("sigma_seed"),
    pl.col("diff").abs().max().alias("max_abs_diff"),
).sort("subset")
print(rep.pivot(on="pair", index="subset", values="diff"))
print(noise)

struct = wt.select(SUBSETS).to_numpy().astype(float)
sd_cfg = struct.std(axis=0, ddof=1)
sig = noise.sort("subset")["sigma_seed"].to_numpy()
order = {s: i for i, s in enumerate(sorted(SUBSETS))}
sig = np.array([sig[order[s]] for s in SUBSETS])
var_tbl = pl.DataFrame({
    "subset": SUBSETS,
    "sd_across_configs": np.round(sd_cfg, 4),
    "sigma_seed": np.round(sig, 4),
    "var_ratio_noise_over_total": np.round(sig ** 2 / sd_cfg ** 2, 3),
    "structure_share": np.round(1 - sig ** 2 / sd_cfg ** 2, 3),
})
print(var_tbl)

# paired-seed margin arm (H117 margin vs control, seed 1117)
print("\npaired-seed DR draw1: margin - control (same seed 1117)")
mc = []
for s in SUBSETS:
    m = wdf.filter(pl.col("config") == "drd1_margin")[s][0]
    c = wdf.filter(pl.col("config") == "drd1_control")[s][0]
    mc.append({"subset": s, "margin": m, "control": c, "delta": round(m - c, 4),
               "sigma_seed": round(float(sig[SUBSETS.index(s)]), 4),
               "z_vs_seed_noise": round((m - c) / float(sig[SUBSETS.index(s)]), 2)})
print(pl.DataFrame(mc))

# lane-effect paired contrasts (same lane recipe, draw-matched vs clean draw-matched)
print("\ndraw-matched lane contrasts (windowed):")
lanes = [("h107d1_procedural", "h105d1_clean", "H107 d1 - H105 d1"),
         ("h107d2_procedural", "h105d2_clean", "H107 d2 - H105 d2"),
         ("h108d1_quant", "h105d1_clean", "H108 d1 - H105 d1"),
         ("h108d2_quant", "h105d2_clean", "H108 d2 - H105 d2"),
         ("drd1_control", "h105d1_clean", "DRctl d1 - H105 d1"),
         ("drd2_control", "h105d2_clean", "DRctl d2 - H105 d2")]
lrows = []
for a_, b_, lbl in lanes:
    r = {"contrast": lbl}
    for s in SUBSETS:
        r[s] = round(wdf.filter(pl.col("config") == a_)[s][0]
                     - wdf.filter(pl.col("config") == b_)[s][0], 4)
    lrows.append(r)
ldf = pl.DataFrame(lrows)
print(ldf)
Rl, Pl, nl = corr_block(ldf, SUBSETS)
show(Rl, title=f"lane-contrast corr (n={nl} contrasts)")

# finqa / delucionqa / hotpotqa focus
print("\n=== focus triple, windowed TRAIN ===")
for lab, R_, P_ in [("raw", Rw, Pw), ("level-removed", Rwc, Pwc)]:
    fi, de, ho, ta = (SUBSETS.index(x) for x in ["finqa", "delucionqa", "hotpotqa", "tatqa"])
    print(f"{lab}: finqa~delucionqa r={R_[fi, de]:.3f} p={P_[fi, de]:.4f} | "
          f"finqa~hotpotqa r={R_[fi, ho]:.3f} p={P_[fi, ho]:.4f} | "
          f"finqa~tatqa r={R_[fi, ta]:.3f} p={P_[fi, ta]:.4f} | "
          f"deluc~hotpot r={R_[de, ho]:.3f} p={P_[de, ho]:.4f}")

# Spearman cross-check on windowed TRAIN
Mw = wt.select(SUBSETS).to_numpy().astype(float)
sp = stats.spearmanr(Mw).statistic
print("\nSpearman (windowed TRAIN raw): finqa~delucionqa "
      f"{sp[SUBSETS.index('finqa'), SUBSETS.index('delucionqa')]:.3f}, "
      f"finqa~hotpotqa {sp[SUBSETS.index('finqa'), SUBSETS.index('hotpotqa')]:.3f}")

# PCA of level-removed delta matrix -> trade axes
Mc = Mw - Mw.mean(axis=1, keepdims=True)
Mc = Mc - Mc.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
print("\nPCA of level-removed windowed TRAIN deltas - explained variance:",
      np.round(ev[:4], 3))
for k in range(3):
    load_ = pl.DataFrame({"subset": SUBSETS, f"PC{k+1}": np.round(Vt[k], 3)})
    print(load_.sort(f"PC{k+1}"))
# raw (level-in) PCA for reference
Mr = Mw - Mw.mean(axis=0, keepdims=True)
U2, S2, Vt2 = np.linalg.svd(Mr, full_matrices=False)
ev2 = S2 ** 2 / (S2 ** 2).sum()
print("\nPCA with level retained - explained variance:", np.round(ev2[:4], 3))
print(pl.DataFrame({"subset": SUBSETS, "PC1": np.round(Vt2[0], 3), "PC2": np.round(Vt2[1], 3)}))

wdf.write_parquet(D / "R14_E3_windowed_matrix.parquet")
tdf.write_parquet(D / "R14_E3_truncated_matrix.parquet")
print("\nwrote matrices to parquet")

# ------------------------------------------------- (d2) is the seed noise itself structured?
print("\n=== (d2) covariance OF THE SEED NOISE (5 replicate difference vectors) ===")
dif = rep.pivot(on="subset", index="pair", values="diff").select(["pair"] + SUBSETS)
print(dif)
Rn, Pn, nn_ = corr_block(dif, SUBSETS)
show(Rn, title=f"seed-difference-vector corr (n={nn_} replicate pairs, sign-arbitrary)")
print(pair_table(Rn, Pn, nn_).head(6))
print(pair_table(Rn, Pn, nn_).tail(6))

print("\n=== (d3) how many config deltas clear 2*sigma_seed? ===")
sig_map = {s: float(sig[SUBSETS.index(s)]) for s in SUBSETS}
rows = []
for s in SUBSETS:
    col = wt[s].to_numpy()
    rows.append({"subset": s, "two_sigma": round(2 * sig_map[s], 4),
                 "n_configs_beyond_2sigma": int((np.abs(col) > 2 * sig_map[s]).sum()),
                 "n_configs": len(col),
                 "max_abs_delta": round(float(np.abs(col).max()), 4)})
print(pl.DataFrame(rows))

print("\n=== (d4) joint finqa+delucionqa lift vs 1-sigma and 2-sigma ===")
fs, ds = sig_map["finqa"], sig_map["delucionqa"]
print(f"sigma finqa={fs:.4f} delucionqa={ds:.4f}")
for k in [1, 2]:
    hit = wd.filter((pl.col("finqa") > k * fs) & (pl.col("delucionqa") > k * ds))
    print(f"both > {k} sigma: {hit['config'].to_list()}")

print("\n=== (d5) baseline anchor cross-check for H108 (doc uses H90 anchor) ===")
h108 = load("R10-H108_lane_draw1_windowed_result.json")["per_subset"]
print({s: {"auc": h108[s]["auc"], "delta_vs_h90": h108[s]["delta"],
           "baseline_h90": h108[s]["baseline_h90"]} for s in ["finqa", "delucionqa"]})
print("clean H105 pair-mean anchor used here:",
      {s: round(w_base[s], 4) for s in ["finqa", "delucionqa"]})

# --------------------------------------- (d6) analytic AUROC SE (Hanley-McNeil) per subset
print("\n=== (d6) analytic AUROC standard error vs measured sigma_seed ===")
arena = load("R8-H77_arena.json")["per_subset"]
base_auc = {s: w_base[s] for s in SUBSETS}
rows = []
for s in SUBSETS:
    n = int(arena[s]["n"])
    gr = float(arena[s]["grounded_rate"])
    n1 = round(n * gr)          # grounded (positive-score class under this read)
    n0 = n - n1
    A = base_auc[s]
    A_ = max(A, 1 - A)
    q1 = A_ / (2 - A_)
    q2 = 2 * A_ ** 2 / (1 + A_)
    se = math.sqrt((A_ * (1 - A_) + (n1 - 1) * (q1 - A_ ** 2) + (n0 - 1) * (q2 - A_ ** 2)) / (n0 * n1))
    rows.append({"subset": s, "n": n, "n_grounded": n1, "n_halluc": n0,
                 "auc_clean": round(A, 4), "analytic_SE": round(se, 4),
                 "SE_of_diff_indep": round(se * math.sqrt(2), 4),
                 "sigma_seed": round(float(sig[SUBSETS.index(s)]), 4)})
se_df = pl.DataFrame(rows)
print(se_df.with_columns(
    (pl.col("sigma_seed") / pl.col("analytic_SE")).round(2).alias("sigma_seed_over_SE")))

print("\n=== (a-supp) focus pairs restricted to CLEAN-era TRAIN configs ===")
wc = wt.filter(pl.col("era") == "clean")
print("configs:", wc["config"].to_list())
Rc, Pc, nc = corr_block(wc)
for a_, b_ in [("finqa", "delucionqa"), ("finqa", "hotpotqa"), ("finqa", "techqa"),
               ("finqa", "pubmedqa"), ("delucionqa", "techqa"), ("delucionqa", "tatqa")]:
    i, j = SUBSETS.index(a_), SUBSETS.index(b_)
    print(f"  clean-era n={nc}: {a_}~{b_} r={Rc[i, j]:.3f} p={Pc[i, j]:.4f}")
Rcc, Pcc, _ = corr_block(wc, demean_rows=True)
print("  level-removed:")
for a_, b_ in [("finqa", "delucionqa"), ("finqa", "hotpotqa"), ("finqa", "pubmedqa"),
               ("delucionqa", "techqa")]:
    i, j = SUBSETS.index(a_), SUBSETS.index(b_)
    print(f"    {a_}~{b_} r={Rcc[i, j]:.3f} p={Pcc[i, j]:.4f}")

print("\n=== (a-supp2) lane-contrast focus pairs (n=6 draw-matched contrasts) ===")
for a_, b_ in [("finqa", "delucionqa"), ("finqa", "hotpotqa"), ("finqa", "techqa"),
               ("finqa", "pubmedqa"), ("delucionqa", "techqa")]:
    i, j = SUBSETS.index(a_), SUBSETS.index(b_)
    print(f"  {a_}~{b_} r={Rl[i, j]:.3f} p={Pl[i, j]:.4f}")

print("\n=== (a-supp3) partial correlations and jackknife robustness (clean-era TRAIN, n=10) ===")
Mc10 = wc.select(SUBSETS).to_numpy().astype(float)
cfgs = wc["config"].to_list()


def pcorr(M, a_, b_, ctrl):
    ia, ib = SUBSETS.index(a_), SUBSETS.index(b_)
    ic = [SUBSETS.index(c) for c in ctrl]
    X = M[:, ic]
    X = np.column_stack([np.ones(len(M)), X])
    ra = M[:, ia] - X @ np.linalg.lstsq(X, M[:, ia], rcond=None)[0]
    rb = M[:, ib] - X @ np.linalg.lstsq(X, M[:, ib], rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


print("partial finqa~delucionqa | techqa      :", round(pcorr(Mc10, "finqa", "delucionqa", ["techqa"]), 3))
print("partial finqa~delucionqa | mean-proxy  :",
      round(pcorr(np.column_stack([Mc10]), "finqa", "delucionqa", ["covidqa"]), 3))
print("partial finqa~techqa | pubmedqa        :", round(pcorr(Mc10, "finqa", "techqa", ["pubmedqa"]), 3))
print("partial delucionqa~techqa | tatqa      :", round(pcorr(Mc10, "delucionqa", "techqa", ["tatqa"]), 3))

print("\njackknife (drop one config) on clean-era TRAIN:")
for a_, b_ in [("finqa", "pubmedqa"), ("finqa", "techqa"), ("delucionqa", "techqa"),
               ("finqa", "delucionqa")]:
    ia, ib = SUBSETS.index(a_), SUBSETS.index(b_)
    rs = []
    for k in range(len(Mc10)):
        idx = [x for x in range(len(Mc10)) if x != k]
        rs.append(float(np.corrcoef(Mc10[idx, ia], Mc10[idx, ib])[0, 1]))
    worst = int(np.argmin(np.abs(rs)))
    print(f"  {a_}~{b_}: full r={np.corrcoef(Mc10[:, ia], Mc10[:, ib])[0, 1]:.3f} "
          f"jackknife range [{min(rs):.3f}, {max(rs):.3f}] "
          f"most-influential drop={cfgs[worst]} -> r={rs[worst]:.3f}")
