"""R19-H160 SEED-DIVERSE WEIGHT AVERAGING - the soups and their reads.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R19-H160 SEED-DIVERSE WEIGHT AVERAGING AS THE SERVED ARTIFACT". The PRIMARY is
soup B: the uniform 0.5/0.5 elementwise average of trunk + task head over the two
FRESH draws of the H150 flagship recipe (seeds 3150 and 4150). The H158 cross-init
soup that motivated the arm was read BEFORE registration, so it is exploratory and
cannot confirm itself; soup B is the unseen confirmation.

THE AVERAGING MACHINERY IS THE BANKED H158 ONE. `R18-H158_soup.py` is imported and
its helpers are called directly - `load_ingredient`, `verify_pair` (fingerprint
agreement, key alignment, shape and dtype checks, non-float buffer equality,
adapter-zero verification), `basin_diagnostic` (L2, raw cosine, update-space
cosine against the pretrained mmBERT anchor), `write_soup` (the 0.5/0.5 average)
and `null_criterion`. Nothing in that file is edited. This module adds exactly
three things the H158 cell table cannot express:

  * the H160 cell configuration (which checkpoints, which banked reads)
  * `write_soup_k` - the uniform 1/k generalisation for the registered k-sweep
    secondaries (k = 3 and 4); the k = 2 PRIMARY goes through H158's own
    `write_soup`, byte-identical to the banked implementation
  * H160-namespaced output paths, so no H158 artifact is touched

Cells:

    soupB   R19-H160-arm-draw3 + R19-H160-arm-draw4          PRIMARY, k = 2
    k3      R18-H150-arm-draw1 + draw2 + R19-H160-arm-draw3   secondary, k = 3
    k4      the same four draws                               secondary, k = 4

UNIFORM AVERAGE ONLY. Choosing which draws enter a soup by their arena scores is
tuning on arena statistics and is BARRED by the registration - there is no greedy
path and no selection of any kind in this file.

READS: the soup is read by the BANKED blind windowed arena reader
(`R16-H142_G1_reads.py`, `--run twin --mode windowed`) reused unchanged - only its
checkpoint binding and output path are rebound, the R18-H150 wrapper pattern - so
the frozen R8-H77 gate is byte-identical to every banked read. gold_full is read
through the banked `R16-H142_G1_arm.score_claims` on `H108.gold_full()`, the same
call the banked `evaluate()` makes. Each draw is ALSO re-read here only if its own
banked windowed result is missing; normally the draws' reads come from the
campaign's own stage and are consumed from disk.

Idempotent across container restarts: a cell whose `R19-H160_soup_cell_<name>.json`
is on disk is skipped, an existing soup checkpoint is reused, and a read whose
result JSON is on disk is not repeated. Relaunch = the same command.

Run detached (GPU0 or GPU2 - GPU1 is reserved for R19-H159):
    GPU=2 nohup setsid uv run python \
        experiments/grounding-semantic/R19-H160_soup.py --stage run \
        >> logs/R19-H160_soup.log 2>&1 &
"""

import os

# GPU1 carries R19-H159 exclusively. An empty CUDA_VISIBLE_DEVICES is legitimate
# (the verify stage is CPU-only); an unset one is not, because the banked modules
# imported below default it to "1" and would land on H159's card.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if "GPU" not in os.environ:
        raise SystemExit("GPU PLACEMENT ABORT: set CUDA_VISIBLE_DEVICES (0 or 2, or "
                         "empty for the CPU verify stage) - GPU1 is reserved for "
                         "R19-H159")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
# GPU1 reservation LIFTED 2026-08-14 16:38 - R19-H159 was killed at draw 1 and
# released the card; index 1 is a legal placement for this arm. The unset check
# above stays: the banked trainer silently defaults to "1" at import.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import time
import traceback

from safetensors.torch import save_file
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
MODELS = ROOT / "models"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H158 = _mod("h158soup", "R18-H158_soup.py")  # the banked averaging machinery
BAND = H158.BAND  # the campaign's standing 0.005 reporting band

D3 = "R19-H160-arm-draw3"
D4 = "R19-H160-arm-draw4"
H150_D1 = "R18-H150-arm-draw1"
H150_D2 = "R18-H150-arm-draw2"

# (checkpoint, label, banked windowed result, banked train result)
INGREDIENTS = {
    H150_D1: ("H150 draw 1 (seed 1150, monolithic executor)",
              "R18-H150_arm_draw1_windowed_result.json",
              "R18-H150_arm_draw1_result.json"),
    H150_D2: ("H150 draw 2 (seed 2150, monolithic executor)",
              "R18-H150_arm_draw2_windowed_result.json",
              "R18-H150_arm_draw2_result.json"),
    D3: ("H160 draw 3 (seed 3150, split-cotangent executor)",
         "R19-H160_arm_draw3_windowed_result.json",
         "R19-H160_arm_draw3_result.json"),
    D4: ("H160 draw 4 (seed 4150, split-cotangent executor)",
         "R19-H160_arm_draw4_windowed_result.json",
         "R19-H160_arm_draw4_result.json"),
}

CELLS = {
    "soupB": {
        "members": [D3, D4],
        "role": "PRIMARY - the unseen confirmatory soup",
        "soup_ckpt": "R19-H160-soup-B",
        "full_indomain": True,  # the registered holds need non-EN as well as gold_full
        "expect": "blind windowed mean >= 0.72049 GRADUATE, < 0.71549 KILL",
    },
    "k3": {
        "members": [H150_D1, H150_D2, D3],
        "role": "registered secondary - k-sweep, reads only, no promotion route",
        "soup_ckpt": "R19-H160-soup-k3",
        "expect": "monotone non-decreasing in k (pre-stated direction)",
    },
    "k4": {
        "members": [H150_D1, H150_D2, D3, D4],
        "role": "registered secondary - k-sweep, reads only, no promotion route",
        "soup_ckpt": "R19-H160-soup-k4",
        "expect": "monotone non-decreasing in k (pre-stated direction)",
    },
}

RESULT = HERE / "R19-H160_soup_result.json"

_READS = None


class CellAbort(Exception):
    """A structural abort - recorded, and the cell is not retried on relaunch."""


def get_reads():
    global _READS
    if _READS is None:
        _READS = _mod("g1reads", "R16-H142_G1_reads.py")
    return _READS


# --- verification (the banked H158 checks, extended over k members) -----------------


def verify_members(cell, members):
    """Every structural precondition of the uniform average. For k = 2 this is
    exactly H158's `verify_pair` with no shared-init requirement; for k > 2 the
    same checks run pairwise against member 0, which is what the average needs."""
    A = members[0]
    checks = {"n_members": len(members), "members": [], "k": len(members),
              "weights": [round(1.0 / len(members), 8)] * len(members)}
    for m in members[1:]:
        H158.check_keys("trunk", A["trunk"], m["trunk"])
        H158.check_shapes("trunk", A["trunk"], m["trunk"])
        H158.check_keys("task_head", A["student"]["task_head"],
                        m["student"]["task_head"])
        H158.check_shapes("task_head", A["student"]["task_head"],
                          m["student"]["task_head"])
        H158.check_keys("adapter", A["adapter"]["adapter"], m["adapter"]["adapter"])
        da = {str(t.dtype) for t in A["trunk"].values()}
        db = {str(t.dtype) for t in m["trunk"].values()}
        if da != db:
            raise H158.CellAbort(
                f"DTYPE ABORT (trunk): a={sorted(da)} b={sorted(db)}")
        for k, t in A["trunk"].items():
            if not t.is_floating_point() and not torch.equal(t, m["trunk"][k]):
                raise H158.CellAbort(
                    f"NON-FLOAT ABORT: integer trunk buffer {k} differs between "
                    "the ingredients and cannot be averaged")
    fps, perms, seeds = [], [], []
    for m in members:
        if not H158.adapter_is_zero(m["adapter"]):
            raise H158.CellAbort(
                f"ADAPTER ABORT: {m['dir'].name} output layer is not zero")
        fp = m["fp"]
        fps.append(fp.get("blake2b_128"))
        perms.append(fp.get("perm_fingerprint"))
        seeds.append(fp.get("seed"))
        checks["members"].append({
            "checkpoint": m["dir"].name,
            "seed": fp.get("seed"),
            "n_groups": fp.get("n_groups"),
            "init_fingerprint": fp.get("blake2b_128"),
            "perm_fingerprint": fp.get("perm_fingerprint"),
            "executor": fp.get("executor", "monolithic"),
            "adapter_zero": True,
        })
    checks.update({
        "init_fingerprints_all_distinct": len(set(fps)) == len(fps),
        "perm_fingerprints_all_distinct": len(set(perms)) == len(perms),
        "seeds": seeds,
        "n_trunk_tensors": len(A["trunk"]),
        "trunk_dtypes": sorted({str(t.dtype) for t in A["trunk"].values()}),
        "task_head_keys": sorted(A["student"]["task_head"]),
        "domain_head_note": ("training-only, NOT averaged - copied from the first "
                             "member verbatim"),
        "selection_note": ("UNIFORM average over ALL k members - no selection, no "
                           "greedy soup; barred by the registration"),
    })
    if not checks["init_fingerprints_all_distinct"]:
        raise H158.CellAbort(
            f"INIT-FINGERPRINT ABORT: the members share an init draw {fps} - the "
            "arm's mechanism is seed-diverse averaging, a shared init is the "
            "banked negative cell")
    if not checks["perm_fingerprints_all_distinct"]:
        raise H158.CellAbort(
            f"PERM-FINGERPRINT ABORT: the members share a data order {perms}")
    return checks


# --- the k-way soup ------------------------------------------------------------------


def write_soup_k(members, out_dir):
    """Uniform 1/k elementwise average of trunk + task_head into a NEW directory.
    The k = 2 case goes through H158's own `write_soup` instead; this is the
    identical arithmetic generalised for the registered k-sweep. Non-float trunk
    buffers are copied (already asserted identical); the domain head, the adapter
    and the tokenizer come from the first member unchanged."""
    A = members[0]
    w = 1.0 / len(members)
    (out_dir / "trunk").mkdir(parents=True, exist_ok=True)

    soup_trunk = {}
    for k, ta in A["trunk"].items():
        if ta.is_floating_point():
            acc = torch.zeros_like(ta, dtype=torch.float64)
            for m in members:
                acc += w * m["trunk"][k].double()
            soup_trunk[k] = acc.to(ta.dtype)
        else:
            soup_trunk[k] = ta.clone()
    save_file(soup_trunk, str(out_dir / "trunk" / "model.safetensors"))

    st = dict(A["student"])
    st["trunk"] = {k: v.clone() for k, v in soup_trunk.items()}
    head = {}
    for k, v in A["student"]["task_head"].items():
        acc = torch.zeros_like(v, dtype=torch.float64)
        for m in members:
            acc += w * m["student"]["task_head"][k].double()
        head[k] = acc.to(v.dtype)
    st["task_head"] = head
    st["domain_head"] = A["student"]["domain_head"]  # training-only, not averaged
    torch.save(st, out_dir / "dann_student.pt")

    ad = dict(A["adapter"])
    ad["adapter_active"] = False  # frozen at zero in every member, carried through
    torch.save(ad, out_dir / "adapter.pt")

    shutil.copy(A["dir"] / "trunk" / "config.json", out_dir / "trunk" / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(A["dir"] / f, out_dir / f)
    return len(soup_trunk)


# --- reads -----------------------------------------------------------------------------


def windowed_read(soup_name, cell):
    """The banked blind windowed arena read, reused UNCHANGED - only the
    checkpoint binding and the output path are rebound (the R18-H150 wrapper
    pattern), so the frozen R8-H77 gate is byte-identical to every banked read."""
    out = HERE / f"R19-H160_soup_{cell}_windowed_result.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"  SKIP windowed read (on disk: {out.name})", flush=True)
        return json.loads(out.read_text())
    reads = get_reads()
    reads.ARM.RUNS["twin"]["ckpt"] = soup_name
    reads.out_path = lambda run, mode: out
    argv = sys.argv
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    try:
        reads.main()
    finally:
        sys.argv = argv
    torch.cuda.empty_cache()
    return json.loads(out.read_text())


def indomain_read(soup_dir, cell, full):
    """The in-domain reads through the banked path. `full` runs the banked
    `R16-H142_G1_arm.evaluate` unchanged - gold, gold_full, RAGTruth EN and the
    seven translations - which is what the PRIMARY's registered holds need
    (gold_full >= 0.84 AND non-EN >= 0.82). The k-sweep secondaries need only
    gold_full, which is the same `score_claims` call the banked `evaluate` makes
    for that row."""
    out = HERE / f"R19-H160_soup_{cell}_goldfull_result.json"
    if out.exists() and out.stat().st_size > 0:
        res = json.loads(out.read_text())
        if not full or "ragtruth_nonen" in res:
            print(f"  SKIP in-domain read (on disk: {out.name})", flush=True)
            return res
    reads = get_reads()
    arm = reads.ARM
    model, tok = arm.load_run(soup_dir)
    if full:
        res = arm.evaluate(model, tok)
        res["checkpoint"] = str(soup_dir)
    else:
        cl, ck, y = arm.H108.gold_full()
        s = arm.score_claims(model, tok, cl, ck, tag=f"{cell}/gold_full")
        auc, f1, _ = arm.M59.auc_and_f1(y, s)
        res = {"checkpoint": str(soup_dir), "gold_full": {
            "auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}}
    out.write_text(json.dumps(res, indent=2))
    del model
    torch.cuda.empty_cache()
    gf = res["gold_full"]
    extra = (f"  ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}"
             if "ragtruth_nonen" in res else "")
    print(f"  gold_full {gf['auc']:.4f} (n={gf['n']}){extra} -> {out.name}",
          flush=True)
    return res


def banked(name):
    label, win_f, train_f = INGREDIENTS[name]
    wp, tp = HERE / win_f, HERE / train_f
    if not wp.exists() or not tp.exists():
        raise CellAbort(f"ingredient {name}: banked reads missing "
                        f"({win_f} / {train_f}) - run the draw's campaign first")
    w, t = json.loads(wp.read_text()), json.loads(tp.read_text())
    return {
        "label": label, "checkpoint": name,
        "windowed_mean": w["mean"],
        "per_subset": {k: v["auc"] for k, v in w["per_subset"].items()},
        "gold_full": t["gold_full"]["auc"],
        "ragtruth_nonen": t["ragtruth_nonen"]["auc"],
        "source_windowed": win_f, "source_train": train_f,
    }


# --- driver ------------------------------------------------------------------------


def cell_path(cell):
    return HERE / f"R19-H160_soup_cell_{cell}.json"


def run_cell(cell, cfg, stage, base):
    t0 = time.time()
    print(f"\n=== CELL {cell}  {' + '.join(cfg['members'])}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"  role: {cfg['role']}", flush=True)

    ings = [banked(n) for n in cfg["members"]]
    members = [H158.load_ingredient(n) for n in cfg["members"]]
    checks = verify_members(cell, members)
    print(f"  verified: {checks['n_trunk_tensors']} trunk tensors aligned over "
          f"k={checks['k']} members, task_head aligned, adapter zero in all, "
          f"init fingerprints distinct "
          f"{[m['init_fingerprint'][:8] for m in checks['members']]}", flush=True)

    # the basin diagnostic is a PAIRWISE quantity - reported for every pair
    diag = {}
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            key = f"{cfg['members'][i]}__vs__{cfg['members'][j]}"
            diag[key] = H158.basin_diagnostic(members[i], members[j], base)
            us = diag[key].get("update_space") or {}
            print(f"  basin {cfg['members'][i]} vs {cfg['members'][j]}: "
                  f"L2(trunk+task_head) "
                  f"{diag[key]['l2_diff_trunk_plus_task_head']:.4f}  "
                  f"cos_raw {diag[key]['cosine_trunk_raw']:.9f}  "
                  f"cos_update {us.get('cosine_trunk_update')}  "
                  f"L2diff/mean|update| {us.get('l2_diff_over_mean_update')}",
                  flush=True)

    ing_means = [i["windowed_mean"] for i in ings]
    record = {
        "cell": cell, "status": "verified", "role": cfg["role"],
        "k": len(members),
        "ingredients": ings,
        "checks": checks,
        "basin_pairwise": diag,
        "expectation": cfg["expect"],
    }

    if stage == "verify":
        print(f"  verify only - no soup written  ({time.time() - t0:.0f}s)", flush=True)
        return record

    soup_name = cfg["soup_ckpt"]
    soup_dir = MODELS / soup_name
    if (soup_dir / "trunk" / "model.safetensors").exists():
        print(f"  SKIP soup build (on disk: {soup_dir})", flush=True)
    else:
        if len(members) == 2:
            n = H158.write_soup(members[0], members[1], soup_dir)  # banked, verbatim
            impl = "R18-H158_soup.write_soup (banked, 0.5/0.5)"
        else:
            n = write_soup_k(members, soup_dir)
            impl = f"R19-H160_soup.write_soup_k (uniform 1/{len(members)})"
        (soup_dir / "soup_manifest.json").write_text(json.dumps({
            "arm": "R19-H160 seed-diverse weight averaging",
            "cell": cell, "role": cfg["role"],
            "k": len(members), "weights": checks["weights"],
            "implementation": impl,
            "ingredients": [str(m["dir"]) for m in members],
            "averaged_scope": ["trunk", "task_head"],
            "copied_from_first_member": ["domain_head", "adapter", "h_norm",
                                         "ctx_norm", "tokenizer", "trunk/config.json"],
            "checks": checks,
        }, indent=2))
        print(f"  soup written -> {soup_dir}  ({n} trunk tensors averaged, "
              f"{impl})", flush=True)

    del members
    win = windowed_read(soup_name, cell)
    gf = indomain_read(soup_dir, cell, full=cfg.get("full_indomain", False))

    soup_mean = win["mean"]
    per_sub_mean = {k: sum(i["per_subset"][k] for i in ings) / len(ings)
                    for k in win["per_subset"]}
    record.update({
        "status": "read",
        "soup_checkpoint": str(soup_dir),
        "soup": {
            "windowed_mean": soup_mean,
            "per_subset": {k: v["auc"] for k, v in win["per_subset"].items()},
            "gold_full": gf["gold_full"]["auc"],
            "gold": gf.get("gold", {}).get("auc"),
            "ragtruth_en": gf.get("ragtruth_en", {}).get("auc"),
            "ragtruth_nonen": gf.get("ragtruth_nonen", {}).get("auc"),
            "ragtruth_nonen_per_lang": gf.get("ragtruth_nonen", {}).get("per_lang"),
        },
        "ingredient_windowed_means": ing_means,
        "ingredient_mean_of_means": round(sum(ing_means) / len(ing_means), 5),
        "per_subset_delta_vs_ingredient_mean": {
            k: round(v["auc"] - per_sub_mean[k], 5)
            for k, v in win["per_subset"].items()
        },
        "arena": H158.null_criterion(soup_mean, min(ing_means), max(ing_means))
        if len(ings) == 2 else k_null(soup_mean, ing_means),
        "gold_full_vs_ingredients": k_null(
            gf["gold_full"]["auc"], [i["gold_full"] for i in ings]),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    a = record["arena"]
    print(f"  ARENA soup {soup_mean:.5f}  ingredients "
          f"{' / '.join(f'{m:.5f}' for m in ing_means)}  "
          f"mean {a['ingredient_mean']:.5f}  "
          f"delta {a['soup_minus_ingredient_mean']:+.5f}  "
          f"vs best {a['soup_minus_better_ingredient']:+.5f}  -> {a['verdict']}",
          flush=True)
    return record


def k_null(soup, ings):
    """The H158 null criterion generalised over k ingredients - same bands, same
    wording, the mean and the best taken over all k."""
    ing_mean = sum(ings) / len(ings)
    best = max(ings)
    d_mean, d_best = soup - ing_mean, soup - best
    if d_best >= BAND:
        verdict, reason = "POSITIVE", (
            f"soup is {d_best:+.5f} above the best ingredient ({best:.5f}), at or "
            f"over the {BAND} reporting band")
    elif d_mean <= -BAND:
        verdict, reason = "DEGRADED", (
            f"soup is {d_mean:+.5f} below the ingredient mean ({ing_mean:.5f}), at "
            f"or over the {BAND} reporting band")
    else:
        verdict, reason = "NULL", (
            f"soup sits {d_mean:+.5f} from the ingredient mean ({ing_mean:.5f}) and "
            f"{d_best:+.5f} from the best ingredient ({best:.5f}) - inside the "
            f"{BAND} band")
    return {"ingredient_mean": round(ing_mean, 5),
            "better_ingredient": round(best, 5),
            "soup_minus_ingredient_mean": round(d_mean, 5),
            "soup_minus_better_ingredient": round(d_best, 5),
            "reporting_band": BAND, "verdict": verdict, "reason": reason}


def merge(selected):
    cells = {}
    for c in CELLS:
        p = cell_path(c)
        if p.exists():
            cells[c] = json.loads(p.read_text())
    k_sweep = {}
    for c in ("soupB", "k3", "k4"):
        r = cells.get(c, {})
        if r.get("status") == "read":
            k_sweep[c] = {"k": r["k"], "windowed_mean": r["soup"]["windowed_mean"],
                          "gold_full": r["soup"]["gold_full"]}
    monotone = None
    if {"k3", "k4"} <= set(k_sweep) and "soupB" in k_sweep:
        seq = [k_sweep["k3"]["windowed_mean"], k_sweep["k4"]["windowed_mean"]]
        monotone = bool(seq[1] >= seq[0])
    payload = {
        "arm": "R19-H160 seed-diverse weight averaging as the served artifact",
        "primary": ("soup B - the uniform 0.5/0.5 average of the two FRESH draws "
                    "(seeds 3150, 4150) of the H150 flagship recipe; UNSEEN, the "
                    "confirmatory read the arm is registered on"),
        "exploratory_note": ("the H158 cross-init soup (0.72306) motivated this arm "
                             "and was read before registration - exploratory, it "
                             "cannot confirm itself and is not re-read here"),
        "method": ("uniform elementwise average of trunk + task_head over the k "
                   "members; the k = 2 primary uses the BANKED R18-H158 write_soup "
                   "verbatim, k > 2 uses its uniform 1/k generalisation; adapter "
                   "frozen at zero in every member and carried through unchanged; "
                   "domain head training-only, copied from the first member; "
                   "LayerNorm architecture, no running statistics"),
        "reads": ("blind windowed decomposed-min arena read through the BANKED "
                  "R16-H142_G1_reads.py reused unchanged (checkpoint binding and "
                  "output path rebound only), plus gold_full through the banked "
                  "R16-H142_G1_arm.score_claims on H108.gold_full()"),
        "selection_barred": ("uniform average over all k draws - greedy or "
                             "score-selected soups are barred by the registration"),
        "cells_requested": selected,
        "cells": cells,
        "k_sweep": {"cells": k_sweep,
                    "predicted": "monotone non-decreasing in k",
                    "k3_to_k4_monotone": monotone},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "written": time.strftime("%F %T"),
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(f"\nresults -> {RESULT}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=("verify", "run"))
    ap.add_argument("--cells", default=",".join(CELLS),
                    help="comma-separated subset of " + ",".join(CELLS))
    args = ap.parse_args()
    selected = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in selected if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known: {list(CELLS)}")

    print(f"=== R19-H160 SOUPS ({args.stage})  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"cells: {selected}", flush=True)
    if args.stage == "run":
        print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"({torch.cuda.get_device_name(0)})", flush=True)

    print("\nloading the pretrained trunk anchor for the update-space diagnostics...",
          flush=True)
    base = H158._pretrained_trunk()

    for cell in selected:
        p = cell_path(cell)
        if args.stage == "run" and p.exists() and p.stat().st_size > 0:
            print(f"\n--- SKIP cell {cell} (already on disk: {p.name}) ---", flush=True)
            continue
        try:
            rec = run_cell(cell, CELLS[cell], args.stage, base)
        except (CellAbort, H158.CellAbort) as exc:
            rec = {"cell": cell, "status": "ABORTED", "reason": str(exc),
                   "members": CELLS[cell]["members"]}
            print(f"  === CELL {cell} ABORTED: {exc} ===", flush=True)
        except Exception:  # noqa: BLE001 - one cell must not lose the others
            print(f"  === CELL {cell} FAILED (not recorded - a relaunch retries it) ===",
                  flush=True)
            traceback.print_exc()
            continue
        if args.stage == "run" or rec["status"] == "ABORTED":
            p.write_text(json.dumps(rec, indent=2))
            print(f"  cell record -> {p.name}", flush=True)
        if args.stage == "run":
            merge(selected)

    if args.stage == "run":
        payload = merge(selected)
        print("\n=== SUMMARY ===", flush=True)
        for c, r in payload["cells"].items():
            if r.get("status") == "read":
                a = r["arena"]
                print(f"  {c:6s} k={r['k']}  soup {r['soup']['windowed_mean']:.5f}  "
                      f"ing-mean {a['ingredient_mean']:.5f}  "
                      f"delta {a['soup_minus_ingredient_mean']:+.5f}  "
                      f"gold_full {r['soup']['gold_full']:.4f}  {a['verdict']}",
                      flush=True)
            else:
                print(f"  {c:6s} {r.get('status')}: {r.get('reason', '')}", flush=True)
    print("\n=== R19-H160 SOUPS DONE ===", flush=True)


if __name__ == "__main__":
    main()
