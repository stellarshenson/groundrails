"""R20-H173 SOUP CLOSURE READS - the k=6 uniform soup and four pair soups.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R20-H173 SOUP CLOSURE READS", adopted from `docs/experiments/briefs/
R20-sweep-D-weight-averaging.md` section (d). ZERO TRAINING: uniform elementwise
weight averaging over six already-banked flagship endpoints, then deterministic
blind reads.

PRIMARY (one read): S6, the uniform 1/6 average of ALL six flagship draws -
trunk + task head, equal weights, NO selection, NO tuning, NO weighting - read
through the standard windowed decomposed-min arena protocol and compared to the
BANKED k=6 single-draw mean M6 = 0.71218. Pre-registered branches:

    delta >= +0.005   averaging re-enters as a registered candidate under a
                      fresh confirmation pair
    delta <= -0.005   the R19-H160 kill is recipe-general and the class closes
    strictly between  NULL; the class closes as a mean lever

SECONDARY (four reads, mechanism only, NO promotion route): two-ingredient
soups, same uniform averaging, same read, each compared to the mean of its OWN
two ingredients - split-only pairs (d5,d6) and (d3,d5) against mixed pairs
(d1,d5) and (d2,d6). A consistent sign split across the two groups is the
licensed conclusion; magnitudes are indicative only.

FREE RIDER (runs FIRST, before any soup is written): the read path is verified
against the banked per-draw reads by re-reading draws 3 and 4 through the same
reader and diffing per subset. Bar <= 1e-4 per subset. A failure ABORTS the run
before a single soup exists - the soup numbers would be uninterpretable. The
same two reads are the identity proof for the d3/d4 checkpoints: reproducing
`R19-H160_arm_draw{3,4}_windowed_result.json` per subset is direct evidence that
the weights on disk are the weights behind those banked reads.

THE MACHINERY IS BANKED, NOT REWRITTEN. `R19-H160_soup.py` (which itself imports
`R18-H158_soup.py`) is loaded and its functions are called directly -
`verify_members` (fingerprint distinctness, key/shape/dtype alignment, non-float
buffer equality, adapter-zero), `write_soup_k` (uniform 1/k average over trunk +
task head), `H158.write_soup` (the byte-identical k=2 average), `H158.basin_
diagnostic`, `H158.null_criterion` and `k_null`, and `run_cell` itself. This
module adds exactly three things:

  * the six-ingredient table (draws 5 and 6 are new since H160)
  * the H173 cell configuration
  * H173-namespaced read paths, so no H158/H160 artifact is touched

The averaged scope is trunk + task_head, confirmed against the banked cell
artifacts (`R18-H158_soup_cell_*.json`, `R19-H160_soup_cell_*.json`, key
`averaged_scope`): the domain head is training-only and copied from the first
member verbatim, the adapter is frozen at zero in every member and carried
through.

NO SELECTION OF ANY KIND. Greedy soups, best-subset search and fitted weights
are barred by the registration pending a non-arena surface with >= 5/6 sign
agreement on the banked soup reads, and none exists. Only the six cells named
above are read - an unregistered extra soup read is arena-shopping.

Idempotent across container restarts: a cell whose `R20-H173_soup_cell_<name>.json`
is on disk is skipped, an existing soup checkpoint is reused, and a read whose
result JSON is on disk is not repeated. Relaunch = the same command.

Run detached (GPU2 ONLY - GPUs 0 and 1 carry R20-H174 training draws):
    GPU=2 nohup setsid uv run python \
        experiments/grounding-semantic/R20-H173_soup.py --stage run \
        2>&1 | tee -a logs/R20-H173_soup_reads.log &
"""

import os

# GPUs 0 and 1 carry the R20-H174 training draws. This arm is GPU2 only; an
# empty CUDA_VISIBLE_DEVICES is legitimate (the verify stage is CPU-only), an
# unset one is not - the banked modules default it to "1" at import.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if "GPU" not in os.environ:
        raise SystemExit("GPU PLACEMENT ABORT: set CUDA_VISIBLE_DEVICES (2, or "
                         "empty for the CPU verify stage) - GPUs 0 and 1 carry "
                         "the R20-H174 training draws")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import pathlib
import sys
import time
import traceback

import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
MODELS = ROOT / "models"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H160 = _mod("h160soup", "R19-H160_soup.py")  # banked; imports H158 itself
H158 = H160.H158
BAND = H158.BAND  # the campaign's standing 0.005 reporting band

D1, D2 = "R18-H150-arm-draw1", "R18-H150-arm-draw2"
D3, D4 = "R19-H160-arm-draw3", "R19-H160-arm-draw4"
D5, D6 = "R20-H172-arm-draw5", "R20-H172-arm-draw6"

# The k=6 single-draw mean banked by R20-H172 COMPLETE - the PRIMARY comparator.
M6 = 0.71218

# (checkpoint, label, banked windowed result, banked train result)
INGREDIENTS = {
    D1: ("H150 draw 1 (seed 1150, monolithic executor)",
         "R18-H150_arm_draw1_windowed_result.json",
         "R18-H150_arm_draw1_result.json"),
    D2: ("H150 draw 2 (seed 2150, monolithic executor)",
         "R18-H150_arm_draw2_windowed_result.json",
         "R18-H150_arm_draw2_result.json"),
    D3: ("H160 draw 3 (seed 3150, split-cotangent executor)",
         "R19-H160_arm_draw3_windowed_result.json",
         "R19-H160_arm_draw3_result.json"),
    D4: ("H160 draw 4 (seed 4150, split-cotangent executor)",
         "R19-H160_arm_draw4_windowed_result.json",
         "R19-H160_arm_draw4_result.json"),
    D5: ("H172 draw 5 (seed 5150, split-cotangent executor)",
         "R20-H172_arm_draw5_windowed_result.json",
         "R20-H172_arm_draw5_result.json"),
    D6: ("H172 draw 6 (seed 6150, split-cotangent executor)",
         "R20-H172_arm_draw6_windowed_result.json",
         "R20-H172_arm_draw6_result.json"),
}

# The free-rider control: banked draws re-read through this process's own read
# path. d3 and d4 are chosen because the same two reads double as the identity
# proof the task requires for those two checkpoints.
CONTROLS = [D3, D4]
CONTROL_BAR = 1e-4

CELLS = {
    "S6": {
        "members": [D1, D2, D3, D4, D5, D6],
        "role": ("PRIMARY - uniform k=6 soup over all six flagship endpoints, "
                 "compared to the banked k=6 single-draw mean 0.71218"),
        "soup_ckpt": "R20-H173-soup-S6",
        "expect": ("delta vs M6 >= +0.005 -> candidate under a fresh confirmation "
                   "pair; <= -0.005 -> H160 kill is recipe-general, class closes; "
                   "inside -> NULL, class closed as a mean lever"),
        "group": "primary",
    },
    "p56": {
        "members": [D5, D6],
        "role": "SECONDARY - split-only pair, mechanism only, no promotion route",
        "soup_ckpt": "R20-H173-soup-p56",
        "expect": "negative if split-executor endpoints are average-destructive",
        "group": "split_only",
    },
    "p35": {
        "members": [D3, D5],
        "role": "SECONDARY - split-only pair, mechanism only, no promotion route",
        "soup_ckpt": "R20-H173-soup-p35",
        "expect": "negative if split-executor endpoints are average-destructive",
        "group": "split_only",
    },
    "p15": {
        "members": [D1, D5],
        "role": "SECONDARY - mixed-executor pair, mechanism only, no promotion route",
        "soup_ckpt": "R20-H173-soup-p15",
        "expect": "at or above the pooled soup mean -0.0035 under the hypothesis",
        "group": "mixed",
    },
    "p26": {
        "members": [D2, D6],
        "role": "SECONDARY - mixed-executor pair, mechanism only, no promotion route",
        "soup_ckpt": "R20-H173-soup-p26",
        "expect": "at or above the pooled soup mean -0.0035 under the hypothesis",
        "group": "mixed",
    },
}

RESULT = HERE / "R20-H173_soup_result.json"
CONTROL_RESULT = HERE / "R20-H173_readpath_control.json"

# The pooled empirical soup effect over the six banked reads (brief D section c).
POOLED_SOUP_DELTA = -0.00350


class ControlAbort(Exception):
    """The read path does not reproduce the banked reads - nothing else runs."""


# --- reads (banked reader, H173 paths) ------------------------------------------------


def windowed_read_ckpt(ckpt_name, out):
    """The banked blind windowed arena read, reused UNCHANGED - only the
    checkpoint binding and the output path are rebound (the R18-H150 wrapper
    pattern), so the frozen R8-H77 gate is byte-identical to every banked read."""
    if out.exists() and out.stat().st_size > 0:
        print(f"  SKIP windowed read (on disk: {out.name})", flush=True)
        return json.loads(out.read_text())
    reads = H160.get_reads()
    reads.ARM.RUNS["twin"]["ckpt"] = ckpt_name
    reads.out_path = lambda run, mode: out
    argv = sys.argv
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    try:
        reads.main()
    finally:
        sys.argv = argv
    torch.cuda.empty_cache()
    return json.loads(out.read_text())


def windowed_read(soup_name, cell):
    return windowed_read_ckpt(
        soup_name, HERE / f"R20-H173_soup_{cell}_windowed_result.json")


def indomain_read(soup_dir, cell, full):
    """gold_full through the banked `R16-H142_G1_arm.score_claims` on
    `H108.gold_full()` - the same call the banked `evaluate()` makes for that
    row. Recorded as a diagnostic; this arm has no in-domain hold."""
    out = HERE / f"R20-H173_soup_{cell}_goldfull_result.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"  SKIP in-domain read (on disk: {out.name})", flush=True)
        return json.loads(out.read_text())
    reads = H160.get_reads()
    arm = reads.ARM
    model, tok = arm.load_run(soup_dir)
    cl, ck, y = arm.H108.gold_full()
    s = arm.score_claims(model, tok, cl, ck, tag=f"{cell}/gold_full")
    auc, f1, _ = arm.M59.auc_and_f1(y, s)
    res = {"checkpoint": str(soup_dir),
           "gold_full": {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}}
    out.write_text(json.dumps(res, indent=2))
    del model
    torch.cuda.empty_cache()
    print(f"  gold_full {auc:.4f} (n={len(y)}) -> {out.name}", flush=True)
    return res


# --- the free-rider control -----------------------------------------------------------


def run_control():
    """Re-read the control draws through THIS process's read path and diff them
    against their banked reads, per subset. Runs before any soup is written."""
    if CONTROL_RESULT.exists() and CONTROL_RESULT.stat().st_size > 0:
        rec = json.loads(CONTROL_RESULT.read_text())
        print(f"--- SKIP read-path control (on disk: {CONTROL_RESULT.name}) "
              f"verdict {rec['verdict']} ---", flush=True)
    else:
        draws = {}
        for name in CONTROLS:
            label, win_f, _ = INGREDIENTS[name]
            banked = json.loads((HERE / win_f).read_text())
            out = HERE / f"R20-H173_control_{name}_windowed_result.json"
            print(f"\n--- CONTROL re-read {name}  {time.strftime('%F %T')} ---",
                  flush=True)
            got = windowed_read_ckpt(name, out)
            per_sub = {k: round(v["auc"] - banked["per_subset"][k]["auc"], 6)
                       for k, v in got["per_subset"].items()}
            worst = max(per_sub, key=lambda k: abs(per_sub[k]))
            draws[name] = {
                "label": label,
                "banked_source": win_f,
                "banked_checkpoint": banked["checkpoint"],
                "banked_mean": banked["mean"],
                "reread_mean": got["mean"],
                "mean_delta": round(got["mean"] - banked["mean"], 6),
                "per_subset_delta": per_sub,
                "worst_subset": worst,
                "worst_abs_delta": abs(per_sub[worst]),
                "reread_result": out.name,
                "passes": abs(per_sub[worst]) <= CONTROL_BAR,
            }
            print(f"  {name}: banked {banked['mean']:.5f} re-read {got['mean']:.5f} "
                  f"worst per-subset |delta| {abs(per_sub[worst]):.6f} ({worst}) "
                  f"-> {'PASS' if draws[name]['passes'] else 'FAIL'}", flush=True)
        worst_all = max(d["worst_abs_delta"] for d in draws.values())
        rec = {
            "control": "read-path reproduction of banked per-draw arena reads",
            "bar": CONTROL_BAR,
            "bar_note": ("<= 1e-4 per subset; the banked reads round AUROC to 4 "
                         "decimals, so 1e-4 is one unit in the last recorded digit"),
            "draws": draws,
            "worst_abs_per_subset_delta": worst_all,
            "verdict": "PASS" if worst_all <= CONTROL_BAR else "FAIL",
            "identity_note": ("reproducing the banked per-subset AUROC from the "
                              "checkpoint on disk is the identity proof that these "
                              "weights are the ones behind the banked reads"),
            "written": time.strftime("%F %T"),
        }
        CONTROL_RESULT.write_text(json.dumps(rec, indent=2))
        print(f"\ncontrol -> {CONTROL_RESULT.name}  verdict {rec['verdict']} "
              f"(worst |delta| {worst_all:.6f})", flush=True)
    if rec["verdict"] != "PASS":
        raise ControlAbort(
            f"read-path control FAILED: worst per-subset |delta| "
            f"{rec['worst_abs_per_subset_delta']:.6f} > {CONTROL_BAR} - the soup "
            "numbers would be uninterpretable and no soup is built")
    return rec


# --- merge ------------------------------------------------------------------------


def cell_path(cell):
    return HERE / f"R20-H173_soup_cell_{cell}.json"


def merge(selected, control):
    cells = {}
    for c in CELLS:
        p = cell_path(c)
        if p.exists():
            cells[c] = json.loads(p.read_text())

    primary = None
    r = cells.get("S6", {})
    if r.get("status") == "read":
        soup = r["soup"]["windowed_mean"]
        delta = round(soup - M6, 5)
        if delta >= 0.005:
            branch = ("delta >= +0.005 - weight averaging re-enters as a registered "
                      "candidate under a fresh confirmation pair")
        elif delta <= -0.005:
            branch = ("delta <= -0.005 - the R19-H160 kill is recipe-general and "
                      "the class closes")
        else:
            branch = ("strictly between -0.005 and +0.005 - NULL; the class closes "
                      "as a mean lever")
        primary = {
            "soup_arena_mean": soup,
            "k6_single_draw_mean_M6": M6,
            "delta_vs_M6": delta,
            "reporting_band": BAND,
            "branch_fired": branch,
            "delta_vs_ingredient_mean": r["arena"]["soup_minus_ingredient_mean"],
            "note": ("the ingredient mean over the six draws IS M6 up to rounding; "
                     "both comparisons are recorded"),
        }

    secondary = {}
    for c in ("p56", "p35", "p15", "p26"):
        r = cells.get(c, {})
        if r.get("status") == "read":
            secondary[c] = {
                "group": CELLS[c]["group"],
                "members": CELLS[c]["members"],
                "soup_arena_mean": r["soup"]["windowed_mean"],
                "ingredient_mean": r["arena"]["ingredient_mean"],
                "delta_vs_own_ingredient_mean":
                    r["arena"]["soup_minus_ingredient_mean"],
                "gold_full": r["soup"]["gold_full"],
                "verdict_band": r["arena"]["verdict"],
            }
    sign_pattern = None
    if len(secondary) == 4:
        split = [secondary[c]["delta_vs_own_ingredient_mean"]
                 for c in ("p56", "p35")]
        mixed = [secondary[c]["delta_vs_own_ingredient_mean"]
                 for c in ("p15", "p26")]
        consistent = all(d < 0 for d in split) and all(
            d >= POOLED_SOUP_DELTA for d in mixed)
        sign_pattern = {
            "hypothesis": ("split-executor endpoints are average-destructive: "
                           "split-only pairs read negative while mixed pairs read "
                           "at or above the pooled soup mean -0.0035"),
            "split_only_deltas": {"p56": split[0], "p35": split[1]},
            "mixed_deltas": {"p15": mixed[0], "p26": mixed[1]},
            "split_only_all_negative": all(d < 0 for d in split),
            "mixed_all_at_or_above_pooled": all(
                d >= POOLED_SOUP_DELTA for d in mixed),
            "consistent_split_vs_mixed": consistent,
            "licensed_conclusion": (
                "a consistent sign split across the two groups is the licensed "
                "conclusion; anything less is not, and magnitudes are indicative "
                "only"),
            "soupB_for_context": {
                "cell": "R19-H160 soupB (d3,d4), both split-cotangent",
                "delta": -0.01696,
                "source": "R19-H160_soup_cell_soupB.json"},
        }

    payload = {
        "arm": "R20-H173 soup closure reads",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, block "
                         "'R20-H173 SOUP CLOSURE READS'; design brief "
                         "docs/experiments/briefs/R20-sweep-D-weight-averaging.md"),
        "training": "NONE - weight averaging over banked checkpoints only",
        "method": ("uniform elementwise average of trunk + task_head over the k "
                   "members through the BANKED R19-H160/R18-H158 machinery "
                   "(k=2 via H158.write_soup verbatim, k=6 via H160.write_soup_k); "
                   "adapter frozen at zero in every member and carried through; "
                   "domain head training-only, copied from the first member"),
        "reads": ("blind windowed decomposed-min arena read through the BANKED "
                  "R16-H142_G1_reads.py reused unchanged (checkpoint binding and "
                  "output path rebound only), plus gold_full through the banked "
                  "R16-H142_G1_arm.score_claims on H108.gold_full()"),
        "selection_barred": ("uniform average over exactly the registered "
                             "ingredient sets - no greedy soup, no best-subset "
                             "search, no fitted weights"),
        "read_path_control": {
            "verdict": control["verdict"],
            "worst_abs_per_subset_delta": control["worst_abs_per_subset_delta"],
            "bar": control["bar"],
            "draws": list(control["draws"]),
        },
        "cells_requested": selected,
        "cells": cells,
        "primary": primary,
        "secondary": secondary,
        "secondary_sign_pattern": sign_pattern,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "written": time.strftime("%F %T"),
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(f"\nresults -> {RESULT}", flush=True)
    return payload


# --- driver ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=("verify", "control", "run"))
    ap.add_argument("--cells", default=",".join(CELLS),
                    help="comma-separated subset of " + ",".join(CELLS))
    args = ap.parse_args()
    selected = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in selected if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known: {list(CELLS)}")

    print(f"=== R20-H173 SOUP CLOSURE READS ({args.stage})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"cells: {selected}", flush=True)
    if args.stage != "verify":
        print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"({torch.cuda.get_device_name(0)})", flush=True)

    # the banked machinery, rebound to the H173 ingredient table and paths
    H160.INGREDIENTS = INGREDIENTS
    H160.windowed_read = windowed_read
    H160.indomain_read = indomain_read

    control = None
    if args.stage != "verify":
        try:
            control = run_control()
        except ControlAbort as exc:
            print(f"\n=== READ-PATH CONTROL ABORT ===\n{exc}", flush=True)
            print("=== R20-H173 ABORTED ===", flush=True)
            raise SystemExit(2) from exc
    if args.stage == "control":
        print("\n=== R20-H173 CONTROL ONLY - DONE ===", flush=True)
        return

    print("\nloading the pretrained trunk anchor for the update-space diagnostics...",
          flush=True)
    base = H158._pretrained_trunk()

    for cell in selected:
        p = cell_path(cell)
        if args.stage == "run" and p.exists() and p.stat().st_size > 0:
            print(f"\n--- SKIP cell {cell} (already on disk: {p.name}) ---",
                  flush=True)
            continue
        try:
            rec = H160.run_cell(cell, CELLS[cell], args.stage, base)
        except (H160.CellAbort, H158.CellAbort) as exc:
            rec = {"cell": cell, "status": "ABORTED", "reason": str(exc),
                   "members": CELLS[cell]["members"]}
            print(f"  === CELL {cell} ABORTED: {exc} ===", flush=True)
        except Exception:  # noqa: BLE001 - one cell must not lose the others
            print(f"  === CELL {cell} FAILED (not recorded - a relaunch retries it) "
                  "===", flush=True)
            traceback.print_exc()
            continue
        if args.stage == "run" or rec["status"] == "ABORTED":
            p.write_text(json.dumps(rec, indent=2))
            print(f"  cell record -> {p.name}", flush=True)
        if args.stage == "run":
            merge(selected, control)

    if args.stage == "run":
        payload = merge(selected, control)
        print("\n=== SUMMARY ===", flush=True)
        for c, r in payload["cells"].items():
            if r.get("status") == "read":
                a = r["arena"]
                print(f"  {c:4s} k={r['k']}  soup {r['soup']['windowed_mean']:.5f}  "
                      f"ing-mean {a['ingredient_mean']:.5f}  "
                      f"delta {a['soup_minus_ingredient_mean']:+.5f}  "
                      f"gold_full {r['soup']['gold_full']:.4f}  {a['verdict']}",
                      flush=True)
            else:
                print(f"  {c:4s} {r.get('status')}: {r.get('reason', '')}", flush=True)
        if payload["primary"]:
            pr = payload["primary"]
            print(f"\n  PRIMARY S6 {pr['soup_arena_mean']:.5f} vs M6 "
                  f"{pr['k6_single_draw_mean_M6']:.5f}  delta "
                  f"{pr['delta_vs_M6']:+.5f}\n  BRANCH: {pr['branch_fired']}",
                  flush=True)
        sp = payload["secondary_sign_pattern"]
        if sp:
            print(f"  SECONDARY split-only {sp['split_only_deltas']} mixed "
                  f"{sp['mixed_deltas']}  consistent="
                  f"{sp['consistent_split_vs_mixed']}", flush=True)
    print("\n=== R20-H173 SOUP CLOSURE READS DONE ===", flush=True)


if __name__ == "__main__":
    main()
