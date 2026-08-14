"""R18-H158 WEIGHT-SOUP DIAGNOSTIC - elementwise 0.5/0.5 averages of banked endpoints.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R18-H158
WEIGHT-SOUP DIAGNOSTIC". DIAGNOSTIC ONLY, binding: zero training, banked
checkpoints only, and NO number this script produces can ever be a record, a
flagship or a publication claim. The arm's output is a mechanism answer for the
reproducible in-run averaging question (EMA banked; SWA the candidate upgrade).

Three cells, each a 0.5/0.5 average of two endpoints already on disk:

    same_init   R18-H155-d1-arm-draw1 + R18-H155-d2-arm-draw2
                shared init fingerprint cd8417f37e347a05301cec5c8e9deebd,
                distinct permutations - the cell ABORTS if the two
                init_fingerprint.json files disagree
    cross_init  R18-H150-arm-draw1 + R18-H150-arm-draw2   (different init draws)
    cross_arm   R18-H150-arm-draw1 + R18-H152-ema-draw1   (different init draws
                AND different recipes: 3-source mix / 14 DANN groups vs clean mix
                / 12 groups, EMA-served)

WHAT IS AVERAGED: the trunk (`trunk/model.safetensors`, 134 tensors) and the task
head (`dann_student.pt["task_head"]`) - the registered scope, and the campaign's
own init-fingerprint scope. LayerNorm architecture, so there are no running
statistics to re-estimate. The domain head is training-only and is copied from
ingredient A verbatim (it is not on the read path beyond supplying `n_groups` to
the banked loader; in cell `cross_arm` the two ingredients disagree on its output
width, 14 vs 12, which is recorded and is NOT an abort because the tensor is not
averaged). The adapter is frozen at its zero init in every ingredient: the script
VERIFIES the output layer is exactly zero in both before averaging anything and
carries A's `adapter.pt` through unchanged.

KEY ALIGNMENT IS A HARD ABORT: if the averaged scope's key sets differ between
the two ingredients the cell aborts with the key diff recorded - keys are never
silently intersected.

READS: the soup is read by the BANKED blind windowed arena reader
(`R16-H142_G1_reads.py`, `--run twin --mode windowed`) reused unchanged - only
its checkpoint binding and output path are rebound, the R18-H150 wrapper pattern,
so the frozen R8-H77 gate is byte-identical to every banked read. gold_full is
read through the banked `R16-H142_G1_arm.score_claims` on `H108.gold_full()`,
the same call the banked `evaluate()` makes.

BASIN DIAGNOSTIC (free, per cell): the L2 norm of the ingredient parameter
difference over the averaged scope, and the cosine similarity of the flattened
trunk parameter vectors. Both are also reported in UPDATE space - the ingredient
minus the pretrained mmBERT trunk every run starts from - because all four
endpoints descend from the SAME pretrained trunk, so the raw-parameter cosine is
saturated near 1.0 by shared pretrained mass and cannot discriminate the cells.

NULL CRITERION, applied per cell: a soup that merely matches its ingredients'
mean is a NULL, not a win. Only a soup meaningfully ABOVE the better ingredient
is a positive signal. The reporting band is the campaign's standing promotion
margin, 0.005 - an executor reporting convention, not a bar; the arm has no bars
and no promotion path, and the coordinator adjudicates.

Stages:
    verify  CPU only, no GPU touched: fingerprint agreement, key alignment,
            adapter-zero verification and the basin diagnostics for every cell,
            then exit before any soup is written
    run     the same checks, then write each soup and read it

Idempotent across container restarts: a cell whose `R18-H158_soup_cell_<name>.json`
is on disk is skipped, an existing soup checkpoint is reused, and a read whose
result JSON is on disk is not repeated. Relaunch = the same command.

Run detached (GPU1, the 96 GB card):
    GPU=1 nohup setsid uv run python \
        experiments/grounding-semantic/R18-H158_soup.py --stage run \
        >> logs/R18-H158_soup.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "1"))

import argparse
import json
import pathlib
import shutil
import sys
import time
import traceback

from safetensors.torch import load_file, save_file
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
MODELS = ROOT / "models"

# The campaign's standing promotion margin, reused here purely as the reporting
# band for the null criterion. NOT a bar - this arm has none.
BAND = 0.005

CELLS = {
    "same_init": {
        "a": "R18-H155-d1-arm-draw1",
        "b": "R18-H155-d2-arm-draw2",
        "a_label": "H155 draw 1 (init 5155, perm 5155a)",
        "b_label": "H155 draw 2 (init 5155, perm 5155b)",
        "a_windowed": "R18-H155_twin_draw1_windowed_result.json",
        "b_windowed": "R18-H155_twin_draw2_windowed_result.json",
        "a_train": "R18-H155_twin_draw1_result.json",
        "b_train": "R18-H155_twin_draw2_result.json",
        "require_shared_init": "cd8417f37e347a05301cec5c8e9deebd",
        "expect": "smallest basin distance, best soup behaviour",
    },
    "cross_init": {
        "a": "R18-H150-arm-draw1",
        "b": "R18-H150-arm-draw2",
        "a_label": "H150 draw 1 (seed 1150)",
        "b_label": "H150 draw 2 (seed 2150)",
        "a_windowed": "R18-H150_arm_draw1_windowed_result.json",
        "b_windowed": "R18-H150_arm_draw2_windowed_result.json",
        "a_train": "R18-H150_arm_draw1_result.json",
        "b_train": "R18-H150_arm_draw2_result.json",
        "require_shared_init": None,
        "expect": "middle basin distance",
    },
    "cross_arm": {
        "a": "R18-H150-arm-draw1",
        "b": "R18-H152-ema-draw1",
        "a_label": "H150 draw 1 (seed 1150, 3-source mix, 14 groups)",
        "b_label": "H152 EMA draw 1 (seed 3151, clean mix, 12 groups, EMA-served)",
        "a_windowed": "R18-H150_arm_draw1_windowed_result.json",
        "b_windowed": "R18-H152_arm_draw1_windowed_result.json",
        "a_train": "R18-H150_arm_draw1_result.json",
        "b_train": "R18-H152_arm_draw1_result.json",
        "require_shared_init": None,
        "expect": "largest basin distance, worst soup behaviour",
    },
}

RESULT = HERE / "R18-H158_soup_result.json"

_READS = None


class CellAbort(Exception):
    """A structural abort - recorded, and the cell is not retried on relaunch."""


def get_reads():
    """The banked reader module, loaded once. Importing it also executes the
    banked trainer and the arena module, so it is deferred out of the CPU-only
    verify stage."""
    global _READS
    if _READS is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "g1reads", HERE / "R16-H142_G1_reads.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _READS = m
    return _READS


# --- verification ------------------------------------------------------------------


def load_ingredient(name):
    d = MODELS / name
    if not d.exists():
        raise CellAbort(f"ingredient {name} is not on disk at {d}")
    sf = load_file(d / "trunk" / "model.safetensors")
    st = torch.load(d / "dann_student.pt", map_location="cpu", weights_only=False)
    ad = torch.load(d / "adapter.pt", map_location="cpu", weights_only=False)
    fp = json.loads((d / "init_fingerprint.json").read_text())
    return {"dir": d, "trunk": sf, "student": st, "adapter": ad, "fp": fp}


def check_keys(scope, ka, kb):
    only_a, only_b = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    if only_a or only_b:
        raise CellAbort(
            f"KEY-ALIGNMENT ABORT ({scope}): only_in_a={only_a[:20]} "
            f"only_in_b={only_b[:20]} (|only_a|={len(only_a)}, |only_b|={len(only_b)})")


def check_shapes(scope, a, b):
    bad = {k: (tuple(a[k].shape), tuple(b[k].shape)) for k in a
           if tuple(a[k].shape) != tuple(b[k].shape)}
    if bad:
        raise CellAbort(f"SHAPE ABORT ({scope}): {bad}")


def adapter_is_zero(ad):
    w, b = ad["adapter"]["2.weight"], ad["adapter"]["2.bias"]
    return bool(torch.all(w == 0).item() and torch.all(b == 0).item())


def verify_pair(cell, cfg, A, B):
    """Every structural precondition of the average, in one place. Raises
    CellAbort on any failure; returns the record of what was checked."""
    fa, fb = A["fp"].get("blake2b_128"), B["fp"].get("blake2b_128")
    want = cfg["require_shared_init"]
    if want is not None and not (fa == fb == want):
        raise CellAbort(
            f"INIT-FINGERPRINT ABORT: cell {cell} requires the shared init "
            f"{want}, got a={fa} b={fb}")

    check_keys("trunk", A["trunk"], B["trunk"])
    check_shapes("trunk", A["trunk"], B["trunk"])
    check_keys("task_head", A["student"]["task_head"], B["student"]["task_head"])
    check_shapes("task_head", A["student"]["task_head"], B["student"]["task_head"])
    check_keys("adapter", A["adapter"]["adapter"], B["adapter"]["adapter"])

    dtypes = {str(t.dtype) for t in A["trunk"].values()}
    dtypes_b = {str(t.dtype) for t in B["trunk"].values()}
    if dtypes != dtypes_b:
        raise CellAbort(f"DTYPE ABORT (trunk): a={sorted(dtypes)} b={sorted(dtypes_b)}")

    non_float = sorted(k for k, t in A["trunk"].items() if not t.is_floating_point())
    for k in non_float:
        if not torch.equal(A["trunk"][k], B["trunk"][k]):
            raise CellAbort(
                f"NON-FLOAT ABORT: integer trunk buffer {k} differs between the "
                "ingredients and cannot be averaged")

    if not adapter_is_zero(A["adapter"]):
        raise CellAbort(f"ADAPTER ABORT: {A['dir'].name} output layer is not zero")
    if not adapter_is_zero(B["adapter"]):
        raise CellAbort(f"ADAPTER ABORT: {B['dir'].name} output layer is not zero")

    dh_a = {k: tuple(v.shape) for k, v in A["student"]["domain_head"].items()}
    dh_b = {k: tuple(v.shape) for k, v in B["student"]["domain_head"].items()}
    return {
        "init_fingerprint_a": fa,
        "init_fingerprint_b": fb,
        "init_fingerprint_shared": fa == fb,
        "n_trunk_tensors": len(A["trunk"]),
        "trunk_dtypes": sorted(dtypes),
        "non_float_trunk_buffers": non_float,
        "task_head_keys": sorted(A["student"]["task_head"]),
        "adapter_zero_a": True,
        "adapter_zero_b": True,
        "domain_head_shapes_a": {k: list(v) for k, v in dh_a.items()},
        "domain_head_shapes_b": {k: list(v) for k, v in dh_b.items()},
        "domain_head_shapes_agree": dh_a == dh_b,
        "domain_head_note": (
            "training-only, NOT averaged - copied from ingredient A verbatim"
            + ("" if dh_a == dh_b else
               "; the ingredients disagree on its output width, which is why the "
               "average is scoped to trunk + task_head")),
        "n_groups_a": A["fp"].get("n_groups"),
        "n_groups_b": B["fp"].get("n_groups"),
        "seed_a": A["fp"].get("seed"),
        "seed_b": B["fp"].get("seed"),
        "perm_fingerprint_a": A["fp"].get("perm_fingerprint"),
        "perm_fingerprint_b": B["fp"].get("perm_fingerprint"),
    }


# --- basin diagnostics --------------------------------------------------------------


def _pretrained_trunk():
    """The pretrained mmBERT trunk every run in this campaign starts from -
    the anchor for the update-space diagnostics. Returns None (recorded, never
    fatal) if it cannot be loaded offline."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("h108", HERE / "R10-H108_lane.py")
        h108 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h108)
        from transformers import AutoModel

        base = AutoModel.from_pretrained(h108.STUDENT)
        base.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs
        return {k: v.detach().cpu() for k, v in base.state_dict().items()}
    except Exception as exc:  # noqa: BLE001 - a missing anchor degrades, never aborts
        print(f"  update-space anchor unavailable: {exc}", flush=True)
        return None


def basin_diagnostic(A, B, base):
    """L2 distance over the averaged scope and cosine over the flattened trunk,
    in raw parameter space and (the discriminating form) in update space."""
    sq = 0.0
    dot = na = nb = 0.0
    for k, ta in A["trunk"].items():
        if not ta.is_floating_point():
            continue
        a, b = ta.double(), B["trunk"][k].double()
        sq += float((a - b).pow(2).sum())
        dot += float((a * b).sum())
        na += float(a.pow(2).sum())
        nb += float(b.pow(2).sum())
    trunk_sq, trunk_dot, trunk_na, trunk_nb = sq, dot, na, nb
    for k, ta in A["student"]["task_head"].items():
        sq += float((ta.double() - B["student"]["task_head"][k].double()).pow(2).sum())

    out = {
        "l2_diff_trunk_plus_task_head": round(sq**0.5, 6),
        "l2_diff_trunk_only": round(trunk_sq**0.5, 6),
        "cosine_trunk_raw": round(trunk_dot / (trunk_na**0.5 * trunk_nb**0.5), 9),
        "raw_cosine_note": (
            "saturated by shared pretrained mass - every endpoint in this campaign "
            "fine-tunes the SAME pretrained mmBERT trunk, so this number cannot "
            "discriminate the cells; the update-space cosine below is the one that does"),
    }
    if base is None:
        out["update_space"] = None
        return out
    missing = sorted(set(A["trunk"]) - set(base))
    extra = sorted(set(base) - set(A["trunk"]))
    if missing or extra:
        out["update_space"] = {
            "available": False,
            "reason": f"anchor key mismatch: missing={missing[:10]} extra={extra[:10]}",
        }
        return out
    dot = na = nb = 0.0
    for k, ta in A["trunk"].items():
        if not ta.is_floating_point():
            continue
        p = base[k].double()
        da, db = ta.double() - p, B["trunk"][k].double() - p
        dot += float((da * db).sum())
        na += float(da.pow(2).sum())
        nb += float(db.pow(2).sum())
    out["update_space"] = {
        "available": True,
        "anchor": "pretrained mmBERT-base trunk",
        "cosine_trunk_update": round(dot / (na**0.5 * nb**0.5), 6),
        "l2_update_a": round(na**0.5, 6),
        "l2_update_b": round(nb**0.5, 6),
        "l2_diff_over_mean_update": round(trunk_sq**0.5 / ((na**0.5 + nb**0.5) / 2), 6),
    }
    return out


# --- the soup ------------------------------------------------------------------------


def write_soup(A, B, out_dir):
    """0.5/0.5 elementwise average of trunk + task_head into a NEW directory.
    Non-float trunk buffers are copied (already asserted identical); the domain
    head, the adapter and the tokenizer come from ingredient A unchanged."""
    (out_dir / "trunk").mkdir(parents=True, exist_ok=True)

    soup_trunk = {}
    for k, ta in A["trunk"].items():
        tb = B["trunk"][k]
        if ta.is_floating_point():
            soup_trunk[k] = (0.5 * ta.double() + 0.5 * tb.double()).to(ta.dtype)
        else:
            soup_trunk[k] = ta.clone()
    save_file(soup_trunk, str(out_dir / "trunk" / "model.safetensors"))

    st = dict(A["student"])
    st["trunk"] = {k: v.clone() for k, v in soup_trunk.items()}
    st["task_head"] = {
        k: (0.5 * v.double() + 0.5 * B["student"]["task_head"][k].double()).to(v.dtype)
        for k, v in A["student"]["task_head"].items()
    }
    st["domain_head"] = A["student"]["domain_head"]  # training-only, not averaged
    torch.save(st, out_dir / "dann_student.pt")

    ad = dict(A["adapter"])
    ad["adapter_active"] = False  # frozen at zero in both ingredients, carried through
    torch.save(ad, out_dir / "adapter.pt")

    shutil.copy(A["dir"] / "trunk" / "config.json", out_dir / "trunk" / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(A["dir"] / f, out_dir / f)
    return len(soup_trunk)


# --- reads ---------------------------------------------------------------------------


def windowed_read(soup_name, cell):
    """The banked blind windowed arena read, reused UNCHANGED - only the
    checkpoint binding and the output path are rebound (the R18-H150 wrapper
    pattern), so the frozen gate is byte-identical to every banked read."""
    out = HERE / f"R18-H158_soup_{cell}_windowed_result.json"
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


def goldfull_read(soup_dir, cell):
    """gold_full through the banked in-domain path - the same three calls the
    banked `evaluate()` makes for that row."""
    out = HERE / f"R18-H158_soup_{cell}_goldfull_result.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"  SKIP gold_full read (on disk: {out.name})", flush=True)
        return json.loads(out.read_text())
    reads = get_reads()
    arm = reads.ARM
    model, tok = arm.load_run(soup_dir)
    cl, ck, y = arm.H108.gold_full()
    s = arm.score_claims(model, tok, cl, ck, tag=f"{cell}/gold_full")
    auc, f1, _ = arm.M59.auc_and_f1(y, s)
    res = {"checkpoint": str(soup_dir), "gold_full": {
        "auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}}
    out.write_text(json.dumps(res, indent=2))
    del model
    torch.cuda.empty_cache()
    print(f"  gold_full {auc:.4f} (n={len(y)}) -> {out.name}", flush=True)
    return res


# --- adjudication-free reporting -------------------------------------------------------


def null_criterion(soup_mean, a_mean, b_mean):
    ing_mean = (a_mean + b_mean) / 2
    best = max(a_mean, b_mean)
    d_mean = soup_mean - ing_mean
    d_best = soup_mean - best
    if d_best >= BAND:
        verdict = "POSITIVE"
        reason = (f"soup is {d_best:+.5f} above the better ingredient ({best:.5f}), "
                  f"at or over the {BAND} reporting band")
    elif d_mean <= -BAND:
        verdict = "DEGRADED"
        reason = (f"soup is {d_mean:+.5f} below the ingredient mean ({ing_mean:.5f}), "
                  f"at or over the {BAND} reporting band")
    else:
        verdict = "NULL"
        reason = (f"soup sits {d_mean:+.5f} from the ingredient mean ({ing_mean:.5f}) "
                  f"and {d_best:+.5f} from the better ingredient ({best:.5f}) - inside "
                  f"the {BAND} band, so it matches rather than beats its ingredients")
    return {
        "ingredient_mean": round(ing_mean, 5),
        "better_ingredient": round(best, 5),
        "soup_minus_ingredient_mean": round(d_mean, 5),
        "soup_minus_better_ingredient": round(d_best, 5),
        "reporting_band": BAND,
        "verdict": verdict,
        "reason": reason,
    }


def banked(cfg, which):
    w = json.loads((HERE / cfg[f"{which}_windowed"]).read_text())
    t = json.loads((HERE / cfg[f"{which}_train"]).read_text())
    return {
        "label": cfg[f"{which}_label"],
        "checkpoint": cfg[which],
        "windowed_mean": w["mean"],
        "per_subset": {k: v["auc"] for k, v in w["per_subset"].items()},
        "gold_full": t["gold_full"]["auc"],
        "source_windowed": cfg[f"{which}_windowed"],
        "source_train": cfg[f"{which}_train"],
    }


# --- driver ------------------------------------------------------------------------


def cell_path(cell):
    return HERE / f"R18-H158_soup_cell_{cell}.json"


def run_cell(cell, cfg, stage, base):
    t0 = time.time()
    print(f"\n=== CELL {cell}  {cfg['a']} + {cfg['b']}  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"  registered expectation: {cfg['expect']}", flush=True)

    A, B = load_ingredient(cfg["a"]), load_ingredient(cfg["b"])
    checks = verify_pair(cell, cfg, A, B)
    print(f"  verified: {checks['n_trunk_tensors']} trunk tensors aligned, "
          f"task_head aligned, adapter zero in both, init fp "
          f"{'SHARED' if checks['init_fingerprint_shared'] else 'distinct'} "
          f"(a={checks['init_fingerprint_a']} b={checks['init_fingerprint_b']})",
          flush=True)
    if not checks["domain_head_shapes_agree"]:
        print(f"  note: domain heads differ in width "
              f"({checks['n_groups_a']} vs {checks['n_groups_b']} groups) - "
              "training-only, not averaged, A's copied through", flush=True)

    diag = basin_diagnostic(A, B, base)
    us = diag.get("update_space") or {}
    print(f"  basin: L2(trunk+task_head) {diag['l2_diff_trunk_plus_task_head']:.4f}  "
          f"cos_raw {diag['cosine_trunk_raw']:.9f}  "
          f"cos_update {us.get('cosine_trunk_update')}  "
          f"L2diff/mean|update| {us.get('l2_diff_over_mean_update')}", flush=True)

    ing_a, ing_b = banked(cfg, "a"), banked(cfg, "b")
    record = {
        "cell": cell,
        "status": "verified",
        "diagnostic_only": ("no number in this record is a record, a flagship or a "
                            "publication claim - the arm cannot promote by construction"),
        "ingredients": {"a": ing_a, "b": ing_b},
        "checks": checks,
        "basin": diag,
        "expectation": cfg["expect"],
    }

    if stage == "verify":
        print(f"  verify only - no soup written  ({time.time() - t0:.0f}s)", flush=True)
        return record

    soup_name = f"R18-H158-soup-{cell}"
    soup_dir = MODELS / soup_name
    if (soup_dir / "trunk" / "model.safetensors").exists():
        print(f"  SKIP soup build (on disk: {soup_dir})", flush=True)
    else:
        n = write_soup(A, B, soup_dir)
        (soup_dir / "soup_manifest.json").write_text(json.dumps({
            "arm": "R18-H158 weight-soup diagnostic",
            "cell": cell, "alpha": 0.5,
            "ingredient_a": str(A["dir"]), "ingredient_b": str(B["dir"]),
            "averaged_scope": ["trunk", "task_head"],
            "copied_from_a": ["domain_head", "adapter", "h_norm", "ctx_norm",
                              "tokenizer", "trunk/config.json"],
            "checks": checks,
            "diagnostic_only": True,
        }, indent=2))
        print(f"  soup written -> {soup_dir}  ({n} trunk tensors averaged)", flush=True)

    del A, B
    win = windowed_read(soup_name, cell)
    gf = goldfull_read(soup_dir, cell)

    soup_mean = win["mean"]
    record.update({
        "status": "read",
        "soup_checkpoint": str(soup_dir),
        "soup": {
            "windowed_mean": soup_mean,
            "per_subset": {k: v["auc"] for k, v in win["per_subset"].items()},
            "gold_full": gf["gold_full"]["auc"],
        },
        "per_subset_delta_vs_ingredient_mean": {
            k: round(v["auc"] - (ing_a["per_subset"][k] + ing_b["per_subset"][k]) / 2, 5)
            for k, v in win["per_subset"].items()
        },
        "arena": null_criterion(soup_mean, ing_a["windowed_mean"], ing_b["windowed_mean"]),
        "gold_full_vs_ingredients": null_criterion(
            gf["gold_full"]["auc"], ing_a["gold_full"], ing_b["gold_full"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    a_ver = record["arena"]
    print(f"  ARENA soup {soup_mean:.5f}  ingredients {ing_a['windowed_mean']:.5f} / "
          f"{ing_b['windowed_mean']:.5f}  mean {a_ver['ingredient_mean']:.5f}  "
          f"delta {a_ver['soup_minus_ingredient_mean']:+.5f}  "
          f"vs best {a_ver['soup_minus_better_ingredient']:+.5f}  -> "
          f"{a_ver['verdict']}", flush=True)
    return record


def ordering_check(cells):
    """The registered prediction: same_init smallest distance and best soup
    behaviour, cross_arm largest and worst."""
    have = {c: r for c, r in cells.items() if r.get("basin")}
    if len(have) < 3:
        return {"available": False, "reason": "not all three cells completed"}

    def key(c, path):
        v = have[c]["basin"]
        for p in path:
            if v is None:
                return None
            v = v.get(p)
        return v

    dist_raw = {c: key(c, ["l2_diff_trunk_plus_task_head"]) for c in have}
    dist_rel = {c: key(c, ["update_space", "l2_diff_over_mean_update"]) for c in have}
    order_raw = sorted(dist_raw, key=lambda c: dist_raw[c])
    predicted = ["same_init", "cross_init", "cross_arm"]
    out = {
        "available": True,
        "predicted_distance_order": predicted,
        "l2_diff_trunk_plus_task_head": dist_raw,
        "observed_distance_order": order_raw,
        "distance_ordering_holds": order_raw == predicted,
    }
    if all(v is not None for v in dist_rel.values()):
        order_rel = sorted(dist_rel, key=lambda c: dist_rel[c])
        out["l2_diff_over_mean_update"] = dist_rel
        out["observed_relative_distance_order"] = order_rel
        out["relative_distance_ordering_holds"] = order_rel == predicted
    deltas = {c: have[c].get("arena", {}).get("soup_minus_ingredient_mean")
              for c in have}
    if all(v is not None for v in deltas.values()):
        order_b = sorted(deltas, key=lambda c: -deltas[c])
        out["soup_minus_ingredient_mean"] = deltas
        out["observed_behaviour_order_best_first"] = order_b
        out["behaviour_ordering_holds"] = order_b == predicted
        out["verdicts"] = {c: have[c]["arena"]["verdict"] for c in have}
    return out


def merge(selected):
    cells = {}
    for c in CELLS:
        p = cell_path(c)
        if p.exists():
            cells[c] = json.loads(p.read_text())
    payload = {
        "arm": "R18-H158 weight-soup diagnostic",
        "status_label": ("DIAGNOSTIC ONLY, binding - no number in this file is a "
                         "record, a flagship or a publication claim; the arm has no "
                         "bars and cannot promote by construction"),
        "method": ("elementwise 0.5/0.5 average of trunk + task_head over two banked "
                   "checkpoints; adapter frozen at zero in every ingredient and "
                   "carried through unchanged; domain head training-only, copied from "
                   "ingredient A; LayerNorm architecture, no running statistics"),
        "reads": ("blind windowed decomposed-min arena read through the BANKED "
                  "R16-H142_G1_reads.py reused unchanged (checkpoint binding and "
                  "output path rebound only), plus gold_full through the banked "
                  "R16-H142_G1_arm.score_claims on H108.gold_full()"),
        "null_criterion": ("a soup that merely matches its ingredients' mean is a "
                           f"NULL, not a win; POSITIVE requires >= {BAND} above the "
                           "better ingredient. The band is the campaign's standing "
                           "promotion margin used as a reporting convention - it is "
                           "NOT a bar, and the coordinator adjudicates"),
        "cells_requested": selected,
        "cells": cells,
        "ordering": ordering_check(cells),
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

    print(f"=== R18-H158 WEIGHT-SOUP DIAGNOSTIC ({args.stage})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print("DIAGNOSTIC ONLY - zero training, banked checkpoints only, no number here "
          "can ever promote", flush=True)
    print(f"cells: {selected}", flush=True)
    if args.stage == "run":
        print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"({torch.cuda.get_device_name(0)})", flush=True)

    print("\nloading the pretrained trunk anchor for the update-space diagnostics...",
          flush=True)
    base = _pretrained_trunk()

    for cell in selected:
        p = cell_path(cell)
        if args.stage == "run" and p.exists() and p.stat().st_size > 0:
            print(f"\n--- SKIP cell {cell} (already on disk: {p.name}) ---", flush=True)
            continue
        try:
            rec = run_cell(cell, CELLS[cell], args.stage, base)
        except CellAbort as exc:
            rec = {"cell": cell, "status": "ABORTED", "reason": str(exc),
                   "ingredients": {"a": CELLS[cell]["a"], "b": CELLS[cell]["b"]}}
            print(f"  === CELL {cell} ABORTED: {exc} ===", flush=True)
        except Exception:  # noqa: BLE001 - one cell must not lose the others
            print(f"  === CELL {cell} FAILED (not recorded - a relaunch retries it) ===",
                  flush=True)
            traceback.print_exc()
            continue
        if args.stage == "run" or rec["status"] == "ABORTED":
            p.write_text(json.dumps(rec, indent=2))
            print(f"  cell record -> {p.name}", flush=True)
        else:
            print(f"  (verify stage - cell record not written for {cell})", flush=True)
        if args.stage == "run":
            merge(selected)

    if args.stage == "run":
        payload = merge(selected)
        print("\n=== SUMMARY ===", flush=True)
        for c, r in payload["cells"].items():
            if r.get("status") == "read":
                a = r["arena"]
                print(f"  {c:11s} soup {r['soup']['windowed_mean']:.5f}  "
                      f"ing-mean {a['ingredient_mean']:.5f}  "
                      f"delta {a['soup_minus_ingredient_mean']:+.5f}  "
                      f"gold_full {r['soup']['gold_full']:.4f}  {a['verdict']}",
                      flush=True)
            else:
                print(f"  {c:11s} {r.get('status')}: {r.get('reason', '')}", flush=True)
        print(f"  ordering: {json.dumps(payload['ordering'].get('distance_ordering_holds'))}"
              f" (distance), "
              f"{json.dumps(payload['ordering'].get('behaviour_ordering_holds'))}"
              " (behaviour)", flush=True)
    print("\n=== R18-H158 SOUP DIAGNOSTIC DONE ===", flush=True)


if __name__ == "__main__":
    main()
