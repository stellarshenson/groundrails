"""R18-H155 INIT-VS-ORDER ATTRIBUTION PAIR - shared init, different data order.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H155 INIT-VS-ORDER ATTRIBUTION PAIR" (2026-08-12 ~18:40): the twin draws
of the banked pair differ in BOTH init and permutation, so the measured 0.0243
two-seed arena-mean spread conflates Dodge 2020's two comparable-magnitude
components. This pair holds the init FIXED and varies only the flat-shuffle
data order, so its two-draw spread is the ORDER component; the init component
is the remainder against the twin pair's 0.0243. Attribution measurement, not
a candidate - no accuracy bars; the holds are read for context only.

Protocol = the twin recipe VERBATIM (banked trainer `R16-H142_G1_arm.py`, run
`twin`): the clean 685,670-row mix via the H108 public_train lineage, evidence
UNTRUNCATED, 1,500/750 windowed presentation, MIL max-over-windows BCE,
12-group DANN, full trunk at lr 1e-5 OneCycleLR 1 epoch, adapter FROZEN at its
zero init (TWIN INTEGRITY ABORT guard). NO H152 regularizers - the anchor is
the plain twin pair.

THE DECOUPLING (the one difference from every prior campaign run): the banked
trainer derives BOTH the init and the permutation from its single module
global SEED -

    torch.manual_seed(SEED)  before model construction, re-issued after
                             (H126 ruling 8) -> trunk+task_head init draw AND
                             the training-time dropout stream
    np.random.seed(SEED)     legacy numpy global (the mix build consumes no
                             numpy RNG - verified)
    perm = np.random.default_rng(SEED).permutation(n_rows)

This wrapper rebinds SEED to the SHARED init seed 5155 for both draws and
remaps the trainer's ONE default_rng call site (the permutation - the only
`default_rng` in the whole dependency path) to the draw's own perm seed:

    init seed  5155  BOTH draws -> identical init fingerprint AND identical
                     per-step dropout stream; the ONLY difference between the
                     draws is the data order
    perm seed  51551 draw 1 (the registered 5155a)
    perm seed  51552 draw 2 (the registered 5155b)

The training code itself runs byte-identical - the banked `main()` is
dispatched untouched (the R18-H150 draw-2 wrapper-rebind pattern); only the
module globals and the one RNG factory call are rebound. The resume path is
unaffected: a resumed run recomputes the permutation through the same rebind
and the banked fingerprint check still fires. After the banked `main()`
returns, the wrapper rewrites the seed/perm fields of the result JSON and the
checkpoint's init_fingerprint.json so the record names the decoupled seeds
(the banked writer records the single SEED it can see).

Draws:

    draw 1  (5155a)  perm seed 51551  models/R18-H155-initpair-draw1
    draw 2  (5155b)  perm seed 51552  models/R18-H155-initpair-draw2

Stages:
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read (dispatched
               into the banked reader, byte-identical to the twin's reads)
    census     CPU-only dry run: mix + window census, then the decoupling
               proof - BOTH draws construct under init seed 5155 and must show
               the SAME init fingerprint, DIFFERENT perm fingerprints, the
               perms distinct from every banked fingerprint, and the rebound
               default_rng call site reproducing each draw's census perm

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R18-H155_arm_run.py \
          --stage train --draw 1
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent

INIT_SEED = 5155  # shared: both draws construct AND re-issue under this seed
DRAWS = {
    1: {"label": "5155a", "perm_seed": 51551,
        "ckpt": "R18-H155-initpair-draw1",
        "train_out": "R18-H155_twin_draw1_result.json",
        "read_out": "R18-H155_twin_draw1_{mode}_result.json"},
    2: {"label": "5155b", "perm_seed": 51552,
        "ckpt": "R18-H155-initpair-draw2",
        "train_out": "R18-H155_twin_draw2_result.json",
        "read_out": "R18-H155_twin_draw2_{mode}_result.json"},
}

# Every banked permutation fingerprint this pair must stay distinct from:
# 1142 (G1 twin), 2142 (H142-T d2), 1150 (H150 d1), 2150 (H150 d2),
# 3151 (H152 d1), 3152 (H152 d2).
BANKED_PERM_FPS = {"a8b2cf491a236bba", "eebe673dabeef46f", "7d13f9ac86a79574",
                   "8fb06248240a78e1", "5e3de18e48c57632", "70d71966b2f7ebcb"}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(arm, draw):
    """Shared init seed on the module global, draw-specific permutation via the
    one default_rng call site - nothing else. The twin guard stands: this
    wrapper only ever dispatches the adapter-frozen run."""
    if arm.RUNS["twin"]["use_adapter"]:
        raise SystemExit("TWIN INTEGRITY ABORT: the dispatched run trains the adapter")
    cfg = DRAWS[draw]
    arm.SEED = INIT_SEED  # init draw + H126 re-issue + legacy numpy, all shared
    arm.RUNS["twin"]["ckpt"] = cfg["ckpt"]
    arm.RUNS["twin"]["out"] = cfg["train_out"]

    real_default_rng = np.random.default_rng

    def decoupled_rng(seed=None):
        """The banked trainer's sole default_rng call is the permutation; a
        call with the init seed gets the draw's perm seed, everything else
        passes through untouched."""
        if seed == INIT_SEED:
            return real_default_rng(cfg["perm_seed"])
        return real_default_rng(seed)

    np.random.default_rng = decoupled_rng
    return arm


def _fix_record(draw):
    """The banked writers record the single SEED they can see; rewrite the
    seed/perm fields of both records so they name the decoupled seeds."""
    cfg = DRAWS[draw]
    conv = (f"np.random.default_rng({cfg['perm_seed']}).permutation(n_rows), flat "
            f"(DECOUPLED: init seed {INIT_SEED} shared by both draws, H126 facility)")
    out = HERE / cfg["train_out"]
    res = json.loads(out.read_text())
    res["seed"] = cfg["label"]
    res["seeds"] = {"init": INIT_SEED, "perm": cfg["perm_seed"]}
    res["perm_convention"] = conv
    res["arm"] = f"h155_initpair_twin_draw{draw}"
    res["experiment"] = ("R18-H155 init-vs-order attribution pair - twin protocol "
                         "verbatim, init seed 5155 SHARED by both draws (init weights "
                         "and the dropout stream identical), only the flat-shuffle "
                         "data order differs; no regularizers")
    out.write_text(json.dumps(res, indent=2))
    fp_path = HERE.parent.parent / "models" / cfg["ckpt"] / "init_fingerprint.json"
    rec = json.loads(fp_path.read_text())
    rec["arm"] = f"h155_initpair_twin_draw{draw}"
    rec["seed"] = cfg["label"]
    rec["seeds"] = {"init": INIT_SEED, "perm": cfg["perm_seed"]}
    rec["perm_convention"] = conv
    fp_path.write_text(json.dumps(rec, indent=2))
    print(f"records relabelled for the decoupled seeds -> {out} and {fp_path}",
          flush=True)


# --- train + in-domain suite (banked main, byte-identical) ---------------------


def train(draw):
    cfg = DRAWS[draw]
    print(f"=== R18-H155 INIT-PAIRED TWIN draw {draw} ({cfg['label']}: init seed "
          f"{INIT_SEED} shared, perm seed {cfg['perm_seed']})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), draw)
    sys.argv = ["g1arm", "--run", "twin"]
    arm.main()
    _fix_record(draw)


# --- windowed arena read (banked reader dispatch) --------------------------------


def windowed(draw):
    cfg = DRAWS[draw]
    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    rebind(reads.ARM, draw)
    reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    reads.main()


# --- census + decoupling proof (CPU only) -----------------------------------------


def census():
    print(f"=== R18-H155 CPU census (dry run, no GPU)  {time.strftime('%F %T')} ===",
          flush=True)
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    rng_before = torch.get_rng_state()
    _claims, wsets, y, tags = arm.build_mix()
    rng_after = torch.get_rng_state()
    n_rows = len(y)
    cens, sizes = arm.window_census(wsets, tags)
    print(f"build_mix consumed no torch RNG: {torch.equal(rng_before, rng_after)}",
          flush=True)

    print(f"train: {n_rows} rows over {len(arm.EXPECTED_GROUPS)} domains  "
          f"mean target {y.mean():.3f}", flush=True)
    for t in arm.EXPECTED_GROUPS:
        g = cens["per_group"][t]
        print(f"  {t:<18} {g['rows']:>7}  mean win {g['mean_windows']:.2f}  "
              f"max {g['max_windows']:.2f}", flush=True)
    print(f"window census (UNTRUNCATED evidence, {arm.WIN}/{arm.STRIDE}): "
          f"{cens['total_pairs']} pairs, mean set {cens['mean_windows_per_row']:.3f}, "
          f"median {cens['median_windows_per_row']}, p90 {cens['p90_windows_per_row']}, "
          f"max {cens['max_windows_per_row']}, "
          f"{cens['rows_with_multi_window_set']} rows ({cens['multi_window_share']:.1%}) "
          f"multi-window", flush=True)
    print(f"  set-size histogram (clipped at 10): {cens['histogram_clipped_at_10']}",
          flush=True)
    if cens["mean_windows_per_row"] < arm.MIN_MEAN_WINDOWS:
        raise SystemExit("WINDOW-CENSUS ABORT: untruncated evidence did not "
                         "produce multi-window sets")
    if cens["max_windows_per_row"] > arm.PAIRS_PER_BATCH:
        raise SystemExit("BATCH-CAP ABORT: a training set exceeds the pair cap")
    print(flush=True)

    from transformers import AutoModel, AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    n_groups = len(arm.EXPECTED_GROUPS)

    init_fps, perm_fps = {}, {}
    for draw in (1, 2):
        cfg = DRAWS[draw]
        # the run's construction path, verbatim: init seed before AND after
        torch.manual_seed(INIT_SEED)
        np.random.seed(INIT_SEED)
        model = arm.DANNAdapterStudent(base, n_groups)
        torch.manual_seed(INIT_SEED)  # H126 ruling 8 re-issue, as in the run
        fp, fp_numel = arm.init_fingerprint(model)
        if not arm.zero_init_ok(model):
            raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
        perm = np.random.default_rng(cfg["perm_seed"]).permutation(n_rows)
        pfp = arm.perm_fingerprint(perm)
        n_steps = len(arm.pack_batches(perm, sizes))
        init_fps[draw], perm_fps[draw] = fp, pfp
        print(f"draw {draw} ({cfg['label']}): init seed {INIT_SEED}  "
              f"perm seed {cfg['perm_seed']}\n"
              f"  init fingerprint {fp} ({fp_numel} params)\n"
              f"  perm fingerprint {pfp}  {n_steps} steps", flush=True)
        del model

    # the decoupling bars: same init, different perms, perms off every banked fp
    assert init_fps[1] == init_fps[2], "DECOUPLING FAIL: init fingerprints differ"
    assert perm_fps[1] != perm_fps[2], "DECOUPLING FAIL: permutations identical"
    bad = BANKED_PERM_FPS & {perm_fps[1], perm_fps[2]}
    assert not bad, f"DECOUPLING FAIL: perm collides with banked {bad}"
    print(f"\nDECOUPLING OK: init fingerprint IDENTICAL across draws "
          f"({init_fps[1]}); perm fingerprints differ and are distinct from "
          f"every banked fingerprint {sorted(BANKED_PERM_FPS)}", flush=True)

    # the engagement proof: the rebound call site reproduces each census perm
    print("\n=== rebind engagement proof (the run-time default_rng path) ===",
          flush=True)
    for draw in (1, 2):
        arm_r = rebind(_mod(f"g1arm_proof{draw}", "R16-H142_G1_arm.py"), draw)
        assert arm_r.SEED == INIT_SEED
        perm = np.random.default_rng(arm_r.SEED).permutation(n_rows)
        pfp = arm.perm_fingerprint(perm)
        assert pfp == perm_fps[draw], (f"ENGAGEMENT FAIL: rebound rng gave {pfp}, "
                                       f"want {perm_fps[draw]}")
        print(f"  draw {draw}: np.random.default_rng({INIT_SEED}) under the rebind "
              f"-> perm fingerprint {pfp} == census value", flush=True)
    np.random.default_rng = real_default_rng  # restore
    del base, tok
    print("\n=== CENSUS ONLY - no training ===", flush=True)


real_default_rng = np.random.default_rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    ap.add_argument("--draw", type=int, choices=(1, 2), default=None)
    args = ap.parse_args()

    if args.stage == "census":
        census()
        return
    if args.draw is None:
        ap.error("--draw is required for the train and windowed stages")
    if args.stage == "windowed":
        windowed(args.draw)
        return
    train(args.draw)


if __name__ == "__main__":
    main()
