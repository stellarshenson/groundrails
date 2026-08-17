"""R20 BASELINE LEGS - three pre-registered flagship reads, measurement only, zero training.

Each leg is a "read this BEFORE the arm trains" step written into its arm's
registration in docs/experiments/semantic-grounding-experiments.md. Each arm's
PRIMARY is a mechanism gate stated as a rise FROM a measured baseline, so without
the baseline the later gate is unattributable. Nothing here trains or tunes.

    LEG 1  R20-H177 numeric verification (block "R20-H177 NUMERIC-VERIFICATION
           PORTFOLIO ARM" + "STAGE 0 COMPLETE"): flagship AUROC on the two banked
           held-out mechanism evals `R20-H177_eval_B.parquet` (compare/direction)
           and `R20-H177_eval_C.parquet` (role/sign/period misbind).
           Registered prediction: near-chance. Arm PRIMARY after training: >= 0.80

    LEG 2  R20-H175b question relevance (block "R20-H175b QUESTION CONDITIONING"
           + "STAGE 0 COMPLETE"): flagship AUROC on `R20-H175b_qlane_eval.parquet`.
           Both legs of a pair carry the SAME claim and SAME chunk - the label
           lives entirely in the question, and the flagship has NO question
           channel. Exactly 0.5000 is the CORRECT result, not a defect; the read
           exists to confirm the eval loads through the standard path and to
           establish the floor as exactly chance. The banked surface floor 0.5816
           (stage-0 disposition 3) is carried through for context, not recomputed

    LEG 3  R19-H166 AMENDMENT A1: flagship BINARY serving scalar on a held-out
           VitaminC REFUTES-vs-NEI split. Registered prediction: near-chance - the
           binary objective cannot express contradiction-versus-absence. Arm
           PRIMARY after training: >= 0.85 on the new `con_head` channel

Read protocol - the shipped one, byte-identical to R20-H176_findver_read.py:
    evidence UNTRUNCATED, presented as 1,500-char windows at stride 750
    (R8-H101 / R16-H142 G0 `windows`), claim scored against every window,
    MAX over windows. All four sets are CLAIM-level, so the arena read's
    MIN-over-response-sentences stage does not apply: one claim, one bag, one
    score. Frozen trunk + task head via R15_gate_common.load_ckpt / .score.

Checkpoints: models/R18-H150-arm-draw1 and models/R18-H150-arm-draw2 - the banked
flagship weights.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20_baseline_legs.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

DRAWS = {"h150d1": "R18-H150-arm-draw1", "h150d2": "R18-H150-arm-draw2"}
WIN, STRIDE = 1500, 750

# Banked, carried through - NOT recomputed here (R20-H175b stage-0 disposition 3).
H175B_SURFACE_FLOOR = 0.5816

OUT1 = HERE / "R20-H177_baseline_leg.json"
OUT2 = HERE / "R20-H175b_baseline_leg.json"
OUT3 = HERE / "R19-H166-A1_baseline_leg.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(chunk):
    """R8-H101 / R16-H142 G0 `windows`, byte-identical."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s: s + WIN] for s in starts]


def flatten(claims, chunks):
    """Flatten to (claim, window) pairs; `starts` are the per-row bag boundaries."""
    flat_c, flat_w, starts = [], [], []
    for cl, ch in zip(claims, chunks, strict=True):
        starts.append(len(flat_c))
        for w in windows(ch):
            flat_c.append(cl)
            flat_w.append(w)
    return flat_c, flat_w, np.array(starts, dtype=np.int64)


def max_over_windows(s_pair, starts):
    """MAX over each row's window bag - reduceat on the contiguous bag boundaries."""
    return np.maximum.reduceat(np.asarray(s_pair, dtype=np.float64), starts)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(s)))


# --------------------------------------------------------------------------- #
# LEG 3 split construction - VitaminC held-out, disjoint from the training mix
# --------------------------------------------------------------------------- #
def flagship_mix_text():
    """Every (claim, evidence) the banked flagship actually trained on.

    The flagship recipe is fixed by `R18-H150_arm_run.py`: the clean public mix
    (`R10-H108_lane.public_train`, 685,670 rows) plus two lanes - `R17-H146_lane`
    (quant_misbind, 30,000) and `R18-H150_scaleunit_lane` (quant_scale_unit,
    5,540) - for 721,210 rows over 14 DANN groups. This rebuilds the clean mix
    through the banked loader under `untruncated_evidence()` (so the evidence
    strings are the raw corpus text, directly comparable to the VitaminC zip) and
    reads the two lane parquets, then returns the text sets used for the
    disjointness proof.
    """
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", "R10-H108_lane.py")
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    print(f"  clean mix rebuilt: {len(claims)} rows, groups {tuple(sorted(set(tags)))}",
          flush=True)
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows, expected 685,670")

    lane_rows = 0
    for fname in ("R17-H146_lane.parquet", "R18-H150_scaleunit_lane.parquet"):
        d = pl.read_parquet(HERE / fname)
        cl_col = "claim"
        ch_col = "chunk" if "chunk" in d.columns else "evidence"
        claims += d[cl_col].to_list()
        chunks += d[ch_col].to_list()
        tags += [fname] * d.height
        lane_rows += d.height
        print(f"  lane {fname}: {d.height} rows", flush=True)
    print(f"  flagship training mix total: {len(claims)} rows "
          f"(685,670 clean + {lane_rows} lane)", flush=True)

    n_vit = sum(1 for t in tags if t == "vitaminc")
    return {
        "n_rows": len(claims),
        "n_vitaminc_rows": n_vit,
        "claims": set(claims),
        "evidence": set(chunks),
        "pairs": set(zip(claims, chunks, strict=True)),
    }


def vitaminc_holdout(mix):
    """A REFUTES-vs-NEI split with zero overlap against the training mix.

    Construction, stated so it is auditable:
      1. The mix takes VitaminC from `tals__vitaminc__train.parquet` ONLY
         (`R10-H108_lane.py:150-165`, single `endswith("__train.parquet")`
         selection; the count is pinned by the `R19-H166_labels3` alignment
         assertion at 370,653). `__test` and `__validation` are untouched by
         every flagship-mix path.
      2. Candidate held-out pool = `__test` + `__validation` (118,251 rows).
      3. Hard key filter: drop any candidate row whose `page`, `claim`,
         `evidence` or `wiki_revision_id` value occurs ANYWHERE in the VitaminC
         train split. `unique_id` and `case_id` are already 0-shared by the
         official split; page/claim/evidence/revision are not, so they are
         filtered rather than assumed.
      4. Text filter against the ASSEMBLED flagship mix (all 14 DANN groups plus
         both lanes, not just VitaminC): drop any candidate whose claim string,
         evidence string, or (claim, evidence) pair appears in the mix. This
         closes any route by which another corpus could carry the same text.
      5. Keep label in {REFUTES, NOT ENOUGH INFO}; REFUTES is the positive class
         (the contradiction the amended arm's `con_head` must separate), NOT
         ENOUGH INFO the negative (absence).
    """
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    tr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    if tr.height != 370_653:
        raise SystemExit(f"VITAMINC ABORT: train split {tr.height} rows, expected 370,653")

    parts = []
    for split in ("test", "validation"):
        d = pl.read_parquet(io.BytesIO(z.read(f"tals__vitaminc__{split}.parquet")))
        parts.append(d.with_columns(pl.lit(split).alias("split")))
    cand = pl.concat(parts)
    n0 = cand.height

    key_shared = {}
    for col in ("unique_id", "case_id", "page", "claim", "evidence", "wiki_revision_id"):
        s_tr = set(tr[col].to_list())
        key_shared[col] = int(cand[col].is_in(list(s_tr)).sum())
    print(f"  candidate pool {n0} rows; raw key collisions vs train: {key_shared}", flush=True)

    keep = np.ones(cand.height, dtype=bool)
    for col in ("page", "claim", "evidence", "wiki_revision_id", "unique_id", "case_id"):
        s_tr = set(tr[col].to_list())
        keep &= ~cand[col].is_in(list(s_tr)).to_numpy()
    cand = cand.filter(keep)
    n1 = cand.height
    print(f"  after VitaminC-train key filter: {n1} rows (dropped {n0 - n1})", flush=True)

    m_cl = ~cand["claim"].is_in(list(mix["claims"])).to_numpy()
    m_ev = ~cand["evidence"].is_in(list(mix["evidence"])).to_numpy()
    m_pr = np.array([(c, e) not in mix["pairs"]
                     for c, e in zip(cand["claim"].to_list(), cand["evidence"].to_list(),
                                     strict=True)])
    cand = cand.filter(m_cl & m_ev & m_pr)
    n2 = cand.height
    print(f"  after assembled-mix text filter: {n2} rows (dropped {n1 - n2})", flush=True)

    held = cand.filter(pl.col("label").is_in(["REFUTES", "NOT ENOUGH INFO"]))
    held = held.with_columns((pl.col("label") == "REFUTES").cast(pl.Int64).alias("y"))

    # Post-hoc verification: zero overlap on every key, recomputed on the final set.
    verify = {}
    for col in ("unique_id", "case_id", "page", "claim", "evidence", "wiki_revision_id"):
        s_tr = set(tr[col].to_list())
        verify[f"shared_{col}_with_vitaminc_train"] = int(held[col].is_in(list(s_tr)).sum())
    verify["shared_claim_with_flagship_mix"] = int(
        held["claim"].is_in(list(mix["claims"])).sum())
    verify["shared_evidence_with_flagship_mix"] = int(
        held["evidence"].is_in(list(mix["evidence"])).sum())
    verify["shared_pair_with_flagship_mix"] = int(sum(
        (c, e) in mix["pairs"] for c, e in
        zip(held["claim"].to_list(), held["evidence"].to_list(), strict=True)))
    if any(v != 0 for v in verify.values()):
        raise SystemExit(f"H166-A1 DISJOINTNESS ABORT: residual overlap {verify}")
    print(f"  held-out split: {held.height} rows "
          f"(REFUTES {int((held['y'] == 1).sum())} / NEI {int((held['y'] == 0).sum())}), "
          f"{held['page'].n_unique()} pages; disjointness verified {verify}", flush=True)
    return held, {"candidate_pool_rows": n0, "after_key_filter": n1,
                  "after_mix_text_filter": n2, "raw_key_collisions_vs_train": key_shared,
                  "verify_zero_overlap": verify}


# --------------------------------------------------------------------------- #
def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    t_all = time.time()
    print(f"=== R20 BASELINE LEGS  {time.strftime('%F %T')} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", flush=True)

    # ---- load the three registered evals -------------------------------- #
    ev_b = pl.read_parquet(HERE / "R20-H177_eval_B.parquet")
    ev_c = pl.read_parquet(HERE / "R20-H177_eval_C.parquet")
    ev_q = pl.read_parquet(HERE / "R20-H175b_qlane_eval.parquet")
    print(f"eval_B {ev_b.height} rows / {ev_b['pair_id'].n_unique()} pairs / "
          f"{ev_b['doc_id'].n_unique()} docs", flush=True)
    print(f"eval_C {ev_c.height} rows / {ev_c['pair_id'].n_unique()} pairs / "
          f"{ev_c['doc_id'].n_unique()} docs", flush=True)
    print(f"qlane_eval {ev_q.height} rows / {ev_q['pair_id'].n_unique()} pairs / "
          f"{ev_q['doc_id'].n_unique()} docs", flush=True)

    # ---- LEG 3 split construction (CPU, before any GPU work) ------------- #
    print("--- leg 3: building the disjoint VitaminC held-out split ---", flush=True)
    t0 = time.time()
    mix = flagship_mix_text()
    held, split_report = vitaminc_holdout(mix)
    del mix
    print(f"  split built in {time.time() - t0:.0f}s", flush=True)

    sets = {
        "eval_B": (ev_b["claim"].to_list(), ev_b["chunk"].to_list(),
                   ev_b["label"].to_numpy()),
        "eval_C": (ev_c["claim"].to_list(), ev_c["chunk"].to_list(),
                   ev_c["label"].to_numpy()),
        "qlane_eval": (ev_q["claim"].to_list(), ev_q["chunk"].to_list(),
                       ev_q["label"].to_numpy()),
        "vitaminc_holdout": (held["claim"].to_list(), held["evidence"].to_list(),
                             held["y"].to_numpy()),
    }
    flat = {}
    for k, (cl, ch, _) in sets.items():
        fc, fw, st = flatten(cl, ch)
        flat[k] = (fc, fw, st, len(cl))
        print(f"windowed {k}: {len(fc)} (claim, window) pairs over {len(cl)} rows "
              f"(mean {len(fc) / len(cl):.3f} windows/row)", flush=True)

    # ---- score every set on both banked flagship draws ------------------- #
    scores = {tag: {} for tag in DRAWS}
    for tag, ckpt in DRAWS.items():
        t0 = time.time()
        tok, trunk, head = C.load_ckpt(ckpt)
        for k, (fc, fw, st, n) in flat.items():
            s_pair = C.score(tok, trunk, head, fc, fw)
            scores[tag][k] = max_over_windows(s_pair, st)
            assert len(scores[tag][k]) == n
            np.save(HERE / f"R20_baseline_legs_scores_{k}_{tag}.npy", scores[tag][k])
        del trunk, head
        torch.cuda.empty_cache()
        print(f"  {tag} ({ckpt}) scored all four sets in {time.time() - t0:.0f}s",
              flush=True)

    # ---- LEG 1 - R20-H177 ------------------------------------------------ #
    leg1 = {}
    for k in ("eval_B", "eval_C"):
        y = sets[k][2]
        per_draw = {}
        for tag, ckpt in DRAWS.items():
            s = scores[tag][k]
            fam = {}
            fcol = ev_b["neg_family"] if k == "eval_B" else ev_c["neg_family"]
            famv = np.array(fcol.to_list())
            for f in sorted(set(famv.tolist())):
                m = famv == f
                if len(set(y[m].tolist())) == 2:
                    fam[f] = {"n_rows": int(m.sum()), "auroc": round(auroc(y[m], s[m]), 4)}
            per_draw[tag] = {"checkpoint": ckpt, "auroc": round(auroc(y, s), 4),
                             "by_neg_family": fam}
        leg1[k] = {
            "n_rows": len(y), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
            "per_draw": per_draw,
            "two_draw_mean_auroc": round(
                float(np.mean([per_draw[t]["auroc"] for t in DRAWS])), 4),
        }
    OUT1.write_text(json.dumps({
        "experiment": "R20-H177 BASELINE LEG - flagship read on both held-out "
                      "mechanism evals (measurement only, zero training)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, blocks "
                         "'R20-H177 NUMERIC-VERIFICATION PORTFOLIO ARM' and "
                         "'R20-H177 STAGE 0 COMPLETE' (2026-08-16)"),
        "registered_prediction": "near-chance",
        "arm_primary_after_training": ">= 0.80 per lane eval",
        "protocol": (f"untruncated evidence windowed {WIN}/{STRIDE}, claim scored vs every "
                     "window, MAX over windows; claim-level rows so no min-over-sentences "
                     "stage; frozen trunk + task head (R15_gate_common.load_ckpt/.score)"),
        "results": leg1,
        "timestamp": time.strftime("%F %T"),
    }, indent=2))
    for k in ("eval_B", "eval_C"):
        print(f"LEG1 {k}: d1 {leg1[k]['per_draw']['h150d1']['auroc']:.4f}  "
              f"d2 {leg1[k]['per_draw']['h150d2']['auroc']:.4f}  "
              f"mean {leg1[k]['two_draw_mean_auroc']:.4f}", flush=True)
    print(f"wrote {OUT1.name}", flush=True)

    # ---- LEG 2 - R20-H175b ----------------------------------------------- #
    y_q = sets["qlane_eval"][2]
    pid = ev_q["pair_id"].to_numpy()
    order = np.argsort(pid, kind="stable")
    leg2_draws = {}
    for tag, ckpt in DRAWS.items():
        s = scores[tag]["qlane_eval"]
        so = s[order]
        po = pid[order]
        a = so[0::2]
        b = so[1::2]
        pa = po[0::2]
        pb = po[1::2]
        if not np.array_equal(pa, pb):
            raise SystemExit("LEG2 ABORT: rows do not pair up two-per-pair_id")
        ident = int(np.sum(a == b))
        diff = np.abs(a - b)
        leg2_draws[tag] = {
            "checkpoint": ckpt,
            "auroc": round(auroc(y_q, s), 6),
            "n_pairs": int(len(a)),
            "pairs_bit_identical": ident,
            "pairs_not_bit_identical": int(len(a) - ident),
            "max_abs_within_pair_delta": float(diff.max()),
            "mean_abs_within_pair_delta": float(diff.mean()),
        }
    OUT2.write_text(json.dumps({
        "experiment": "R20-H175b BASELINE LEG - flagship read on the question-relevance "
                      "contrast eval (measurement only, zero training)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, blocks "
                         "'R20-H175b QUESTION CONDITIONING' and 'R20-H175b STAGE 0 "
                         "COMPLETE' (2026-08-16/17)"),
        "structural_note": ("both rows of a pair carry the SAME claim and SAME evidence "
                            "chunk; the label lives entirely in the question and the "
                            "flagship has no question channel, so exactly 0.5000 is the "
                            "CORRECT result. The read confirms the eval loads and scores "
                            "through the standard path and fixes the floor at exactly "
                            "chance rather than approximately chance"),
        "registered_prediction": "near-chance",
        "arm_primary_after_training": ">= 0.80, read against the banked surface floor",
        "banked_surface_floor": H175B_SURFACE_FLOOR,
        "banked_surface_floor_provenance": ("R20-H175b STAGE 0 COMPLETE, coordinator "
                                            "disposition 3 - carried through, NOT "
                                            "recomputed here"),
        "n_rows": int(len(y_q)),
        "per_draw": leg2_draws,
        "timestamp": time.strftime("%F %T"),
    }, indent=2))
    for tag in DRAWS:
        d = leg2_draws[tag]
        print(f"LEG2 {tag}: auroc {d['auroc']:.6f}  bit-identical pairs "
              f"{d['pairs_bit_identical']}/{d['n_pairs']}  max|delta| "
              f"{d['max_abs_within_pair_delta']:.3e}", flush=True)
    print(f"wrote {OUT2.name}", flush=True)

    # ---- LEG 3 - R19-H166-A1 --------------------------------------------- #
    y_v = sets["vitaminc_holdout"][2]
    split_col = np.array(held["split"].to_list())
    leg3_draws = {}
    for tag, ckpt in DRAWS.items():
        s = scores[tag]["vitaminc_holdout"]
        by_split = {}
        for sp in ("test", "validation"):
            m = split_col == sp
            by_split[sp] = {"n_rows": int(m.sum()), "auroc": round(auroc(y_v[m], s[m]), 4)}
        leg3_draws[tag] = {
            "checkpoint": ckpt,
            "auroc": round(auroc(y_v, s), 4),
            "by_source_split": by_split,
            "mean_score_refutes": round(float(s[y_v == 1].mean()), 4),
            "mean_score_nei": round(float(s[y_v == 0].mean()), 4),
        }
    OUT3.write_text(json.dumps({
        "experiment": "R19-H166 AMENDMENT A1 BASELINE LEG - flagship BINARY serving "
                      "scalar on a held-out VitaminC REFUTES-vs-NEI split "
                      "(measurement only, zero training)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, block "
                         "'R19-H166 AMENDMENT A1' (2026-08-16)"),
        "registered_prediction": ("near-chance - the binary objective cannot express "
                                  "contradiction-versus-absence"),
        "arm_primary_after_training": ">= 0.85 on the new con_head contradiction channel",
        "channel_read_here": ("the BINARY task head (the shipped serving scalar); the "
                              "arm's con_head does not exist on the banked flagship"),
        "positive_class": "REFUTES (contradiction)",
        "negative_class": "NOT ENOUGH INFO (absence)",
        "protocol": (f"untruncated evidence windowed {WIN}/{STRIDE}, MAX over windows; "
                     "frozen trunk + task head (R15_gate_common.load_ckpt/.score)"),
        "split_construction": vitaminc_holdout.__doc__,
        "split_report": split_report,
        "n_rows": int(len(y_v)),
        "n_refutes": int((y_v == 1).sum()),
        "n_nei": int((y_v == 0).sum()),
        "n_pages": int(held["page"].n_unique()),
        "per_draw": leg3_draws,
        "two_draw_mean_auroc": round(
            float(np.mean([leg3_draws[t]["auroc"] for t in DRAWS])), 4),
        "timestamp": time.strftime("%F %T"),
    }, indent=2))
    for tag in DRAWS:
        print(f"LEG3 {tag}: auroc {leg3_draws[tag]['auroc']:.4f}  "
              f"(REFUTES mean {leg3_draws[tag]['mean_score_refutes']:.4f} vs "
              f"NEI mean {leg3_draws[tag]['mean_score_nei']:.4f})", flush=True)
    print(f"wrote {OUT3.name}", flush=True)

    print(f"total {time.time() - t_all:.0f}s", flush=True)
    print("=== R20 BASELINE LEGS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
