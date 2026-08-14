"""R19-H160 SPLIT EQUIVALENCE PROOF - split vs monolithic, step by step.

Proves that `R19-H160_split_exec.py` (cotangent window-chunked, memory-bounded)
produces the SAME training trajectory as the monolithic reference
(`R19-H160_arm_run.train_step` - the banked G1 twin step transcribed verbatim
from `R16-H142_G1_arm.main()`'s loop body) on the H160 draw-3 config: seed 3150
init, the H150 3-source mix (721,210 rows, 14 DANN groups), the seed-3150
permutation replayed and packed by the banked `pack_batches`.

REDUCED GEOMETRY. The monolithic reference cannot fit the registered batch
geometry (<= 48 sets / <= 96 pairs) on either free card - the banked H152 vram
probe priced the full window stack at 36.96 GB allocated. The proof therefore
runs at SETS_PER_BATCH 8 / PAIRS_PER_BATCH 16 (process-level rebind, the H155 /
H156 pattern) with the split at pass-A 8 / pass-B 4 windows per chunk, so the
deepest batches exercise 4 pass-B chunks (> 3) and every multi-window set still
routes through the same code. The chunking identity is SCALE-FREE in P: the
chain-rule factorization dL/dtheta = sum_i cs_i ds_i/dtheta holds per window
regardless of how the P windows are partitioned into chunks, and the per-step RNG
draw sequence (one dropout([P, hidden]) capture) is chunk-independent - so the
reduction weakens nothing; it only reprices the batch. This note is recorded in
the proof JSON.

Three runs over the IDENTICAL step sequence - the first 50 reduced-geometry steps
plus 5 hand-picked DEEP batches carrying the mix's deepest window sets that still
pack under the reduced pair cap (the multi-window path where the split execution
actually differs):

    reference   the monolithic path, verbatim (arm.encode_batch + train_step)
    refnoise    the monolithic path again under a benign kernel variation (the
                cuDNN sdpa attention backend where the default pick is the
                mem-efficient one) - the reference-vs-reference diff IS the
                noise floor
    split       the cotangent window-chunked split executor

Recorded per step: task loss, domain loss, domain accuracy, pre-clip gradient
norm. RNG lock-step is proven by the END-OF-RUN CUDA RNG state being identical
across all three runs (the per-step draw sequence - exactly one
dropout([P, hidden]) capture - is then identical by construction; H160 carries no
window dropout, so the H152 proof's drop-mask comparison has no analogue). End of
run: raw trunk+task_head states (the banked fingerprint scope) are banked per
mode and the report diffs them. Init fingerprints must be identical across runs.

PASS bar (all conditions, evaluated in --mode report):
  * final CUDA RNG state identical across all three runs
  * split-vs-reference per-step task/domain loss diffs are the SAME ORDER as the
    reference-vs-reference noise floor (max and median within 10x)
  * no systematic offset: sign consistency of the split's signed task-loss diffs
    < 0.8, or |mean signed diff| inside the noise's own offset scale
  * end-of-run raw trunk+task_head weight max-abs-diff <= 1e-3 (bf16-noise scale)
  * init fingerprints identical across the three runs

Run (one process per mode, sequential - a clean CUDA heap per run):
    for m in reference refnoise split report; do \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R19-H160_exec_equivalence.py \
        --mode $m; done
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit("GPU PLACEMENT ABORT: CUDA_VISIBLE_DEVICES is unset - set it "
                     "explicitly (GPU0 or GPU2; GPU1 is reserved for R19-H159)")
if os.environ["CUDA_VISIBLE_DEVICES"].strip() == "1":
    raise SystemExit("GPU PLACEMENT ABORT: GPU1 is reserved exclusively for R19-H159")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

DRAW = 3      # the H160 draw-3 config
N_REGISTERED_STEPS = 50
N_DEEP_BATCHES = 5

# the reduced proof geometry (the registered 48/96 does not fit either free
# card monolithically); pass-B 4 keeps > 3 grad chunks on the deepest batches
PROOF_SETS_PER_BATCH = 8
PROOF_PAIRS_PER_BATCH = 16
PROOF_PASS_A_CHUNK = 8
PROOF_PASS_B_CHUNK = 4

# verdict thresholds (documented in the proof JSON) - the banked H152/H156 values
SAME_ORDER_FACTOR = 10.0   # split diff vs noise-floor diff, max and median
BIAS_SIGN_CONSISTENCY = 0.8
WEIGHT_MAX_ABS_BAR = 1e-3  # bf16-noise scale over 55 accumulated steps
FLOOR_FALLBACK = 1e-6      # if the noise floor comes out exactly zero

METRICS = ("task_loss", "domain_loss", "grad_norm")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pick_deep_batches(arm, perm, sizes, n=N_DEEP_BATCHES):
    """The n deepest-window rows THAT STILL PACK under the reduced pair cap
    (ties: lowest row index), each packed with its successors in permutation
    order under the reduced caps - the multi-window batches where the split
    execution actually differs."""
    cap = arm.PAIRS_PER_BATCH
    order = sorted((i for i in range(len(sizes)) if int(sizes[i]) <= cap),
                   key=lambda i: (-int(sizes[i]), i))
    deep_rows = [int(r) for r in order[:n]]
    if len(deep_rows) < n:
        raise SystemExit(f"DEEP-BATCH ABORT: only {len(deep_rows)} rows with "
                         f"<= {cap} windows")
    pos = {int(r): p for p, r in enumerate(perm)}
    deep = []
    for r in deep_rows:
        cur, tot, k = [], 0, 0
        p = pos[r]
        while k < len(perm) and len(cur) < arm.SETS_PER_BATCH:
            row = int(perm[(p + k) % len(perm)])
            s = int(sizes[row])
            if cur and tot + s > cap:
                break
            cur.append(row)
            tot += s
            k += 1
        deep.append(cur)
    return deep, deep_rows


def run_mode(mode):
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    w160 = _mod("h160", "R19-H160_arm_run.py")
    split_exec = _mod("h160split", "R19-H160_split_exec.py")
    arm = w160.rebind(arm, DRAW)  # the H150 mix + 14-group map, draw-3 seed/ckpt
    seed = w160.DRAWS[DRAW]["seed"]
    arm.SETS_PER_BATCH = PROOF_SETS_PER_BATCH    # reduced proof geometry -
    arm.PAIRS_PER_BATCH = PROOF_PAIRS_PER_BATCH  # see the module docstring

    print(f"=== R19-H160 EXEC EQUIVALENCE - mode {mode}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    print(f"REDUCED GEOMETRY: <= {PROOF_SETS_PER_BATCH} sets / "
          f"<= {PROOF_PAIRS_PER_BATCH} pairs per batch; split chunks "
          f"{PROOF_PASS_A_CHUNK}/{PROOF_PASS_B_CHUNK} - the chunking identity "
          f"is scale-free in P, the reduction reprices but does not weaken",
          flush=True)
    split_exec.kernel_self_check()

    if mode == "refnoise":
        print("noise-floor variation: sdpa CUDNN_ATTENTION backend "
              "(default is EFFICIENT)", flush=True)

    # the trainer prelude, verbatim ordering (seed -> mix -> model -> re-issue),
    # so the init and the RNG stream are the registered draw-3 ones
    torch.manual_seed(seed)
    np.random.seed(seed)
    claims, wsets, y, tags = arm.build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(arm.EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(arm.EXPECTED_GROUPS)
    cens, sizes = arm.window_census(wsets, tags)
    print(f"window census: {cens['total_pairs']} pairs, mean set "
          f"{cens['mean_windows_per_row']:.3f}, max {cens['max_windows_per_row']}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    base = base.cuda()
    model = arm.DANNAdapterStudent(base, n_groups)
    model = model.cuda()
    torch.manual_seed(seed)  # H126 ruling 8: re-issue after construction
    fp, fp_numel = arm.init_fingerprint(model)
    if not arm.zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    print(f"init fingerprint {fp} ({fp_numel} params)", flush=True)

    perm = np.random.default_rng(seed).permutation(n_rows)
    perm_fp = arm.perm_fingerprint(perm)
    batches = arm.pack_batches(perm, sizes)
    n_steps = len(batches)
    print(f"perm fingerprint {perm_fp}  {n_steps} steps at reduced geometry",
          flush=True)

    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=n_steps, pct_start=arm.WARMUP_FRAC,
        anneal_strategy="linear")

    deep, deep_rows = pick_deep_batches(arm, perm, sizes, n=N_DEEP_BATCHES)
    step_batches = batches[:N_REGISTERED_STEPS] + deep
    kinds = ["registered"] * N_REGISTERED_STEPS + ["deep"] * N_DEEP_BATCHES
    print(f"deep batches: rows {deep_rows} with window counts "
          f"{[int(sizes[r]) for r in deep_rows]}, batch pair totals "
          f"{[sum(int(sizes[i]) for i in b) for b in deep]}", flush=True)

    # capture the pre-clip grad norm inside the reference step without touching
    # banked files: spy on the module-level clip function
    clip_calls = []
    orig_clip = torch.nn.utils.clip_grad_norm_
    if mode != "split":
        def spy(params, max_norm, *a, **kw):
            nrm = orig_clip(params, max_norm, *a, **kw)
            clip_calls.append(float(nrm))
            return nrm
        torch.nn.utils.clip_grad_norm_ = spy

    records = []
    model.train()
    t_run = time.time()
    try:
        for step, batch in enumerate(step_batches):
            p = step / max(n_steps - 1, 1)
            lam = arm.LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if mode == "split":
                m = split_exec.split_train_step(
                    arm, model, opt, sched, tok, claims, wsets, batch,
                    y[batch], groups[batch], lam, audit_rng=True,
                    pass_a_chunk=PROOF_PASS_A_CHUNK,
                    pass_b_chunk=PROOF_PASS_B_CHUNK)
                rec = {"task_loss": m["task_loss"], "domain_loss": m["domain_loss"],
                       "dom_acc": m["dom_correct"] / m["dom_total"],
                       "grad_norm": m["grad_norm"], "n_pairs": m["n_pairs"]}
            else:
                enc, si = arm.encode_batch(tok, claims, wsets, batch)
                yy = torch.as_tensor(y[batch], device="cuda")
                gg = torch.as_tensor(groups[batch], device="cuda")
                if mode == "refnoise":
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        t_loss, d_loss, domain_logit = w160.train_step(
                            arm, model, opt, sched, enc, si, len(batch), yy, gg, lam)
                else:
                    t_loss, d_loss, domain_logit = w160.train_step(
                        arm, model, opt, sched, enc, si, len(batch), yy, gg, lam)
                rec = {"task_loss": float(t_loss.detach()),
                       "domain_loss": float(d_loss.detach()),
                       "dom_acc": float((domain_logit.argmax(-1) == gg[si])
                                        .float().mean()),
                       "grad_norm": clip_calls[-1], "n_pairs": int(si.numel())}
            rec.update({
                "step": step, "kind": kinds[step], "n_sets": len(batch),
                "lam": lam,
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "secs": round(time.time() - t0, 2),
            })
            records.append(rec)
            print(f"  [{mode}] step {step:2d} ({kinds[step]:10s}) pairs "
                  f"{rec['n_pairs']:3d} task {rec['task_loss']:.6f} domain "
                  f"{rec['domain_loss']:.6f} gnorm {rec['grad_norm']:.4f} "
                  f"peak {rec['peak_alloc_gb']:.1f}/{rec['peak_reserved_gb']:.1f} GB "
                  f"({rec['secs']:.1f}s)", flush=True)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip

    rng_final = hashlib.blake2b(torch.cuda.get_rng_state().numpy().tobytes(),
                                digest_size=8).hexdigest()
    state = {"raw": {n: p.detach().cpu().clone()
                     for n, p in model.named_parameters()
                     if n.startswith(("trunk.", "task_head."))}}
    torch.save(state, HERE / f"_equiv_state_h160_{mode}.pt")
    payload = {
        "mode": mode, "seed": seed, "init_fp": fp, "perm_fp": perm_fp,
        "n_steps_total": n_steps,
        "geometry": {"sets_per_batch": PROOF_SETS_PER_BATCH,
                     "pairs_per_batch": PROOF_PAIRS_PER_BATCH,
                     "pass_a_chunk": PROOF_PASS_A_CHUNK,
                     "pass_b_chunk": PROOF_PASS_B_CHUNK},
        "noise_variation": ("sdpa CUDNN_ATTENTION backend (default EFFICIENT)"
                            if mode == "refnoise" else None),
        "gpu": torch.cuda.get_device_name(0),
        "deep_rows": deep_rows,
        "deep_row_windows": [int(sizes[r]) for r in deep_rows],
        "deep_batch_pairs": [sum(int(sizes[i]) for i in b) for b in deep],
        "deep_batches": deep,
        "final_cuda_rng_fp": rng_final,
        "records": records,
        "run_seconds": round(time.time() - t_run, 1),
    }
    (HERE / f"_equiv_records_h160_{mode}.json").write_text(
        json.dumps(payload, indent=2))
    print(f"mode {mode} done in {payload['run_seconds']}s; final CUDA RNG "
          f"fingerprint {rng_final}", flush=True)


# --- report ----------------------------------------------------------------------


def diff_stats(signed):
    d = np.asarray(signed, dtype=np.float64)
    nz = d[d != 0]
    sign_cons = 0.0
    if len(nz):
        sign_cons = max((nz > 0).mean(), (nz < 0).mean())
    return {"max_abs": float(np.abs(d).max()),
            "median_abs": float(np.median(np.abs(d))),
            "mean_signed": float(d.mean()),
            "sign_consistency": float(sign_cons)}


def report():
    runs = {}
    for mode in ("reference", "refnoise", "split"):
        p = HERE / f"_equiv_records_h160_{mode}.json"
        if not p.exists():
            raise SystemExit(f"REPORT ABORT: missing {p} - run --mode {mode} first")
        runs[mode] = json.loads(p.read_text())
    ref, refnoise, split = (runs[m]["records"] for m in
                            ("reference", "refnoise", "split"))
    n = len(ref)
    if not (len(refnoise) == len(split) == n):
        raise SystemExit("REPORT ABORT: record count mismatch across modes")
    if [r["step"] for r in ref] != [r["step"] for r in split]:
        raise SystemExit("REPORT ABORT: step sequences differ")

    rng_ok = len({runs[m]["final_cuda_rng_fp"] for m in runs}) == 1
    init_ok = len({runs[m]["init_fp"] for m in runs}) == 1

    noise_signed, split_signed = {}, {}
    for k in METRICS:
        noise_signed[k] = [refnoise[i][k] - ref[i][k] for i in range(n)]
        split_signed[k] = [split[i][k] - ref[i][k] for i in range(n)]
    noise_stats = {k: diff_stats(v) for k, v in noise_signed.items()}
    split_stats = {k: diff_stats(v) for k, v in split_signed.items()}

    states = {m: torch.load(HERE / f"_equiv_state_h160_{m}.pt",
                            map_location="cpu", weights_only=False) for m in runs}
    weight_diffs = {}
    for cmp_name, cmp_mode in (("split_vs_ref", "split"),
                               ("refnoise_vs_ref", "refnoise")):
        per_tensor = {}
        for name, t in states["reference"]["raw"].items():
            per_tensor[name] = float((states[cmp_mode]["raw"][name].float()
                                      - t.float()).abs().max())
        top = sorted(per_tensor.items(), key=lambda kv: -kv[1])[:5]
        weight_diffs[f"raw_{cmp_name}"] = {
            "max_abs": max(per_tensor.values()),
            "top_tensors": [{"tensor": k, "max_abs": v} for k, v in top],
        }

    verdicts = {}
    for k in ("task_loss", "domain_loss"):
        floor = noise_stats[k]["max_abs"] or FLOOR_FALLBACK
        floor_med = noise_stats[k]["median_abs"] or FLOOR_FALLBACK
        verdicts[k] = {
            "magnitude_ok": (split_stats[k]["max_abs"] <= SAME_ORDER_FACTOR * floor
                             and split_stats[k]["median_abs"]
                             <= SAME_ORDER_FACTOR * floor_med),
            "floor_max_abs_used": floor,
        }
    s, nz = split_stats["task_loss"], noise_stats["task_loss"]
    bias_detected = (s["sign_consistency"] >= BIAS_SIGN_CONSISTENCY
                     and abs(s["mean_signed"])
                     > max(3 * abs(nz["mean_signed"]), nz["median_abs"]))
    weights_ok = weight_diffs["raw_split_vs_ref"]["max_abs"] <= WEIGHT_MAX_ABS_BAR
    verdict = {
        "final_rng_identical": rng_ok,
        "init_fingerprints_identical": bool(init_ok),
        "task_loss": verdicts["task_loss"],
        "domain_loss": verdicts["domain_loss"],
        "split_bias_detected": bool(bias_detected),
        "weight_max_abs_ok": bool(weights_ok),
        "thresholds": {"same_order_factor": SAME_ORDER_FACTOR,
                       "bias_sign_consistency": BIAS_SIGN_CONSISTENCY,
                       "weight_max_abs_bar": WEIGHT_MAX_ABS_BAR,
                       "floor_fallback": FLOOR_FALLBACK},
    }
    verdict["PASS"] = bool(
        rng_ok and init_ok
        and verdicts["task_loss"]["magnitude_ok"]
        and verdicts["domain_loss"]["magnitude_ok"] and not bias_detected
        and weights_ok)

    split_recs = runs["split"]["records"]
    deep_recs = [r for r in split_recs if r["kind"] == "deep"]
    memory = {
        "split_overall_peak_alloc_gb": max(r["peak_alloc_gb"] for r in split_recs),
        "split_overall_peak_reserved_gb": max(r["peak_reserved_gb"] for r in split_recs),
        "split_deep_peak_alloc_gb": max(r["peak_alloc_gb"] for r in deep_recs),
        "split_deep_peak_reserved_gb": max(r["peak_reserved_gb"] for r in deep_recs),
        "deepest_batch_pairs": max(runs["split"]["deep_batch_pairs"]),
        "deepest_single_row_windows": max(runs["split"]["deep_row_windows"]),
        "reference_overall_peak_alloc_gb": max(
            r["peak_alloc_gb"] for r in runs["reference"]["records"]),
        "geometry": "REDUCED - see the scale-free note; the registered-geometry "
                    "split peak is measured by the full draw and the banked H152 "
                    "proof (8.53 GB alloc / 10.35 GB reserved)",
    }

    hashes = {}
    for f in ("R16-H142_G1_arm.py", "R18-H150_arm_run.py", "R19-H160_arm_run.py",
              "R19-H160_split_exec.py", "R19-H160_exec_equivalence.py"):
        hashes[f] = hashlib.sha256((HERE / f).read_bytes()).hexdigest()

    proof = {
        "experiment": "R19-H160 SPLIT equivalence proof - cotangent window-chunked "
                      "memory-bounded executor vs the monolithic MIL max-BCE "
                      "reference, draw-3 config, REDUCED geometry",
        "seed": runs["reference"]["seed"], "init_fp": runs["reference"]["init_fp"],
        "perm_fp": runs["reference"]["perm_fp"],
        "n_steps_total": runs["reference"]["n_steps_total"],
        "steps_compared": n,
        "n_registered_steps": N_REGISTERED_STEPS, "n_deep_batches": N_DEEP_BATCHES,
        "geometry": {
            "sets_per_batch": PROOF_SETS_PER_BATCH,
            "pairs_per_batch": PROOF_PAIRS_PER_BATCH,
            "pass_a_chunk": PROOF_PASS_A_CHUNK,
            "pass_b_chunk": PROOF_PASS_B_CHUNK,
            "registered_geometry": {"sets_per_batch": 48, "pairs_per_batch": 96},
            "scale_free_note": "the cotangent identity dL/dtheta = sum_i cs_i "
                               "ds_i/dtheta factorizes PER WINDOW and is invariant "
                               "to how the P windows are partitioned into pass-B "
                               "chunks, and the per-step RNG sequence (one "
                               "dropout([P, hidden]) capture) is chunk-independent "
                               "- the reduction to 8 sets / 16 pairs reprices the "
                               "batch but does not weaken the equivalence claim; "
                               "the deepest proof batches still exercise > 3 "
                               "pass-B chunks and multi-window sets route through "
                               "the same code path",
        },
        "deep_rows": runs["split"]["deep_rows"],
        "deep_row_windows": runs["split"]["deep_row_windows"],
        "deep_batch_pairs": runs["split"]["deep_batch_pairs"],
        "deep_batches": runs["split"]["deep_batches"],
        "noise_floor_variation": runs["refnoise"]["noise_variation"],
        "gpu": runs["split"]["gpu"],
        "masks": {"final_rng_identical": rng_ok,
                  "note": "H160 carries no window dropout (H152-only regularizer), "
                          "so the H152 proof's drop-mask comparison has no analogue "
                          "here; the per-step RNG sequence is exactly one "
                          "dropout([P, hidden]) capture per step in every run"},
        "noise_floor_refnoise_minus_ref": noise_stats,
        "split_minus_ref": split_stats,
        "signed_diffs": {"noise": noise_signed, "split": split_signed},
        "weight_diffs": weight_diffs,
        "memory": memory,
        "verdict": verdict,
        "file_hashes_sha256": hashes,
        "records": {m: runs[m]["records"] for m in runs},
        "run_seconds": {m: runs[m]["run_seconds"] for m in runs},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    out = HERE / "R19-H160_exec_equivalence.json"
    out.write_text(json.dumps(proof, indent=2))

    t_noise, t_split = noise_stats["task_loss"], split_stats["task_loss"]
    print("\n=== EQUIVALENCE PROOF SUMMARY ===", flush=True)
    print(f"steps compared: {n} ({N_REGISTERED_STEPS} registered-pace + "
          f"{N_DEEP_BATCHES} deep)  seed {runs['reference']['seed']}  "
          f"REDUCED geometry {PROOF_SETS_PER_BATCH} sets / "
          f"{PROOF_PAIRS_PER_BATCH} pairs", flush=True)
    print(f"final RNG identical across runs: {rng_ok}  init fingerprints "
          f"identical: {bool(init_ok)}", flush=True)
    print(f"task loss  noise floor max/median |diff|: "
          f"{t_noise['max_abs']:.3e} / {t_noise['median_abs']:.3e}", flush=True)
    print(f"task loss  split vs ref max/median |diff|: "
          f"{t_split['max_abs']:.3e} / {t_split['median_abs']:.3e}  "
          f"mean signed {t_split['mean_signed']:+.3e}  sign consistency "
          f"{t_split['sign_consistency']:.2f}", flush=True)
    print(f"domain loss split vs ref max |diff|: "
          f"{split_stats['domain_loss']['max_abs']:.3e} (floor "
          f"{noise_stats['domain_loss']['max_abs']:.3e})", flush=True)
    print(f"weights raw max |diff| split vs ref: "
          f"{weight_diffs['raw_split_vs_ref']['max_abs']:.3e}  (refnoise vs ref "
          f"{weight_diffs['raw_refnoise_vs_ref']['max_abs']:.3e})", flush=True)
    print(f"split memory: overall peak {memory['split_overall_peak_alloc_gb']:.2f} GB "
          f"alloc / {memory['split_overall_peak_reserved_gb']:.2f} GB reserved; "
          f"deep batches {memory['split_deep_peak_alloc_gb']:.2f} / "
          f"{memory['split_deep_peak_reserved_gb']:.2f} GB "
          f"(deepest batch {memory['deepest_batch_pairs']} pairs, deepest row "
          f"{memory['deepest_single_row_windows']} windows)", flush=True)
    print(f"VERDICT: {'PASS' if verdict['PASS'] else 'FAIL'}  -> {out}", flush=True)
    return verdict["PASS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("reference", "refnoise", "split", "report"))
    args = ap.parse_args()
    if args.mode == "report":
        ok = report()
        raise SystemExit(0 if ok else 1)
    run_mode(args.mode)


if __name__ == "__main__":
    main()
