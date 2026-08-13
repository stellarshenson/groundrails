"""R18-H152 SPLIT-EXECUTOR EQUIVALENCE PROOF - split vs banked, step by step.

Proves that `R18-H152_split_exec.py` (window-chunked, memory-bounded) produces
the SAME training trajectory as the banked H152 path (the wrapper's own
`train_step` over the banked G1 twin trainer) on the H152 draw-2 config -
seed 3152 init (fingerprint asserted against the banked census value), the
seed-3152 permutation replayed and packed by the banked `pack_batches`.

Three runs over the IDENTICAL step sequence - the first 50 registered steps
plus 5 hand-picked DEEP batches carrying the mix's deepest window sets (the
peak-memory path where the split execution actually differs):

    reference   the banked path, verbatim (encode_batch + w152.train_step)
    refnoise    the banked path again under a benign kernel variation (the
                cuDNN sdpa attention backend where the default pick is the
                mem-efficient one) - the reference-vs-reference diff IS the
                noise floor
    split       the window-chunked split executor

Recorded per step: task loss, domain loss, domain accuracy, pre-clip gradient
norm, EMA-vs-raw delta, window-dropout counters and the sorted dropped-window
index list. RNG lock-step is proven by the END-OF-RUN CUDA RNG state being
identical across all three runs (the per-step draw sequence - rand(P) then
dropout([P,256]) - is then identical by construction). The drop masks
themselves are NOT required bitwise: drop ELIGIBILITY compares logits for
exact ties, so bf16 kernel noise flips eligibility on near-tie windows even
between the two reference runs - those flips are part of the measured noise
floor (a dropped window is never the MIL argmax, so a flip changes neither
the task loss nor its gradient routing in material terms). The report
compares per-step mask mismatch counts, split-vs-reference against
refnoise-vs-reference, and requires the same order. End of run: raw and EMA
trunk+task_head states are banked per mode; the report diffs them.

PASS bar (all conditions, evaluated in --mode report):
  * final CUDA RNG state identical across all three runs (per-step RNG draw
    sequence matched exactly)
  * drop-mask mismatches (per-step symmetric differences of the dropped
    index sets), totaled over the run: split-vs-reference within 3x the
    refnoise-vs-reference total (near-tie eligibility flips are noise)
  * split-vs-reference per-step task/domain loss diffs are the SAME ORDER as
    the reference-vs-reference noise floor (max and median within 10x)
  * no systematic offset: sign consistency of the split's signed task-loss
    diffs < 0.8, or |mean signed diff| inside the noise's own offset scale
  * end-of-run raw trunk weight max-abs-diff <= 1e-3 (bf16-noise scale for
    O(0.02-1) weights over 55 steps)

Run (one process per mode, sequential - a clean CUDA heap per run):
    for m in reference refnoise split report; do \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R18-H152_exec_equivalence.py \
        --mode $m; done
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

SEED = 3152  # the H152 draw-2 init - banked census fingerprints:
INIT_FP = "15537f92ad39e09ecb223b1edb347e4f"
PERM_FP = "70d71966b2f7ebcb"
N_STEPS_EXPECT = 14_300
N_REGISTERED_STEPS = 50
N_DEEP_BATCHES = 5

# verdict thresholds (documented in the proof JSON)
SAME_ORDER_FACTOR = 10.0   # split diff vs noise-floor diff, max and median
MASK_MISMATCH_FACTOR = 3.0  # split vs noise total drop-mask mismatches
BIAS_SIGN_CONSISTENCY = 0.8
WEIGHT_MAX_ABS_BAR = 1e-3  # bf16-noise scale over 55 accumulated steps
FLOOR_FALLBACK = 1e-6      # if the noise floor comes out exactly zero

METRICS = ("task_loss", "domain_loss", "grad_norm", "ema_delta")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drop_idx(drop_mask):
    """Sorted indices of the dropped windows - the drop mask, compactly."""
    return sorted(int(i) for i in drop_mask.detach().cpu().nonzero().flatten())


def pick_deep_batches(arm, perm, sizes, n=N_DEEP_BATCHES):
    """The n deepest-window rows (ties: lowest row index), each packed with
    its successors in permutation order under the registered caps - the
    peak-memory batches the split must survive."""
    order = sorted(range(len(sizes)), key=lambda i: (-int(sizes[i]), i))
    deep_rows = [int(r) for r in order[:n]]
    pos = {int(r): p for p, r in enumerate(perm)}
    deep = []
    for r in deep_rows:
        cur, tot, k = [], 0, 0
        p = pos[r]
        while k < len(perm) and len(cur) < arm.SETS_PER_BATCH:
            row = int(perm[(p + k) % len(perm)])
            s = int(sizes[row])
            if cur and tot + s > arm.PAIRS_PER_BATCH:
                break
            cur.append(row)
            tot += s
            k += 1
        deep.append(cur)
    return deep, deep_rows


def run_mode(mode):
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    w152 = _mod("h152", "R18-H152_arm_run.py")
    split_exec = _mod("split_exec", "R18-H152_split_exec.py")

    print(f"=== R18-H152 EXEC EQUIVALENCE - mode {mode}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    split_exec.kernel_self_check()

    if mode == "refnoise":
        # benign kernel variation: force the cuDNN sdpa backend where the
        # default pick is the mem-efficient one (measured: bitwise-identical
        # to default when EFFICIENT is forced, bf16-scale diffs under CUDNN).
        # The MATH backend is unusable here - it materializes [B,H,S,S]
        # attention for backward over all 22 layers and OOMs the 24 GB card.
        print("noise-floor variation: sdpa CUDNN_ATTENTION backend "
              "(default is EFFICIENT)", flush=True)

    # the wrapper train() prelude, verbatim ordering (seed -> mix -> model ->
    # re-issue), so the init and the RNG stream are the registered draw-2 ones
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
    torch.manual_seed(SEED)  # H126 ruling 8: re-issue after construction
    fp, fp_numel = arm.init_fingerprint(model)
    if fp != INIT_FP:
        raise SystemExit(f"INIT ABORT: fingerprint {fp} != banked draw-2 {INIT_FP}")
    if not arm.zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    print(f"init fingerprint {fp} == banked draw-2 value ({fp_numel} params)",
          flush=True)

    perm = np.random.default_rng(SEED).permutation(n_rows)
    if arm.perm_fingerprint(perm) != PERM_FP:
        raise SystemExit("PERM ABORT: permutation != banked draw-2 value")
    batches = arm.pack_batches(perm, sizes)
    n_steps = len(batches)
    if n_steps != N_STEPS_EXPECT:
        raise SystemExit(f"STEP ABORT: {n_steps} != {N_STEPS_EXPECT}")
    print(f"perm fingerprint {PERM_FP}  {n_steps} steps", flush=True)

    old_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": old_params, "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=n_steps, pct_start=arm.WARMUP_FRAC,
        anneal_strategy="linear")
    ema = w152.ema_init(model)

    deep, deep_rows = pick_deep_batches(arm, perm, sizes, n=N_DEEP_BATCHES)
    step_batches = batches[:N_REGISTERED_STEPS] + deep
    kinds = ["registered"] * N_REGISTERED_STEPS + ["deep"] * N_DEEP_BATCHES
    print(f"deep batches: rows {deep_rows} with window counts "
          f"{[int(sizes[r]) for r in deep_rows]}, batch pair totals "
          f"{[sum(int(sizes[i]) for i in b) for b in deep]}", flush=True)

    # capture the pre-clip grad norm inside the banked train_step without
    # touching banked files: spy on the module-level clip function
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
                    arm, w152, model, opt, sched, tok, claims, wsets, batch,
                    y[batch], groups[batch], lam, ema, audit_rng=True)
                rec = {"task_loss": m["task_loss"], "domain_loss": m["domain_loss"],
                       "dom_acc": m["dom_correct"] / m["dom_total"],
                       "grad_norm": m["grad_norm"], "n_pairs": m["n_pairs"],
                       "n_dropped": m["n_dropped"], "n_eligible": m["n_eligible"],
                       "drop_idx": drop_idx(m["drop_mask"])}
            else:
                enc, si = arm.encode_batch(tok, claims, wsets, batch)
                yy = torch.as_tensor(y[batch], device="cuda")
                gg = torch.as_tensor(groups[batch], device="cuda")
                if mode == "refnoise":
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        t_loss, d_loss, domain_logit, drop_mask, n_drop, n_elig = \
                            w152.train_step(arm, model, opt, sched, enc, si, yy,
                                            gg, lam, ema, "cuda")
                else:
                    t_loss, d_loss, domain_logit, drop_mask, n_drop, n_elig = \
                        w152.train_step(arm, model, opt, sched, enc, si, yy,
                                        gg, lam, ema, "cuda")
                rec = {"task_loss": float(t_loss.detach()),
                       "domain_loss": float(d_loss.detach()),
                       "dom_acc": float((domain_logit.argmax(-1) == gg[si])
                                        .float().mean()),
                       "grad_norm": clip_calls[-1], "n_pairs": len(si),
                       "n_dropped": n_drop, "n_eligible": n_elig,
                       "drop_idx": drop_idx(drop_mask)}
            rec.update({
                "step": step, "kind": kinds[step], "n_sets": len(batch),
                "lam": lam, "ema_delta": w152.ema_delta(ema, model),
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "secs": round(time.time() - t0, 2),
            })
            records.append(rec)
            print(f"  [{mode}] step {step:2d} ({kinds[step]:10s}) pairs "
                  f"{rec['n_pairs']:3d} task {rec['task_loss']:.6f} domain "
                  f"{rec['domain_loss']:.6f} gnorm {rec['grad_norm']:.4f} "
                  f"ema-delta {rec['ema_delta']:.4e} drop "
                  f"{rec['n_dropped']}/{rec['n_eligible']} "
                  f"peak {rec['peak_alloc_gb']:.1f}/{rec['peak_reserved_gb']:.1f} GB "
                  f"({rec['secs']:.1f}s)", flush=True)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip

    rng_final = hashlib.blake2b(torch.cuda.get_rng_state().numpy().tobytes(),
                                digest_size=8).hexdigest()
    state = {
        "raw": {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                if n.startswith(w152.EMA_SCOPE)},
        "ema": {k: v.detach().cpu().clone() for k, v in ema.items()},
    }
    torch.save(state, HERE / f"_equiv_state_{mode}.pt")
    payload = {
        "mode": mode, "seed": SEED, "init_fp": fp, "perm_fp": PERM_FP,
        "n_steps_total": n_steps,
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
    (HERE / f"_equiv_records_{mode}.json").write_text(json.dumps(payload, indent=2))
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
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    runs = {}
    for mode in ("reference", "refnoise", "split"):
        p = HERE / f"_equiv_records_{mode}.json"
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

    # drop-mask mismatches: near-tie eligibility flips are noise-floor
    # territory (the two reference runs exhibit them too) - quantify, compare
    mask_mismatch = {"split_vs_ref": [], "refnoise_vs_ref": []}
    for i in range(n):
        ref_set = set(ref[i]["drop_idx"])
        mask_mismatch["split_vs_ref"].append(
            len(set(split[i]["drop_idx"]) ^ ref_set))
        mask_mismatch["refnoise_vs_ref"].append(
            len(set(refnoise[i]["drop_idx"]) ^ ref_set))
    mm_split = sum(mask_mismatch["split_vs_ref"])
    mm_noise = sum(mask_mismatch["refnoise_vs_ref"])
    mask_mismatch_ok = mm_split <= MASK_MISMATCH_FACTOR * max(mm_noise, 5)

    noise_signed, split_signed = {}, {}
    for k in METRICS:
        noise_signed[k] = [refnoise[i][k] - ref[i][k] for i in range(n)]
        split_signed[k] = [split[i][k] - ref[i][k] for i in range(n)]
    noise_stats = {k: diff_stats(v) for k, v in noise_signed.items()}
    split_stats = {k: diff_stats(v) for k, v in split_signed.items()}

    # end-of-run weight diffs over the tracked scope (raw and EMA)
    states = {m: torch.load(HERE / f"_equiv_state_{m}.pt", map_location="cpu",
                            weights_only=False) for m in runs}
    weight_diffs = {}
    for scope in ("raw", "ema"):
        for cmp_name, cmp_mode in (("split_vs_ref", "split"),
                                   ("refnoise_vs_ref", "refnoise")):
            per_tensor = {}
            for name, t in states["reference"][scope].items():
                per_tensor[name] = float((states[cmp_mode][scope][name].float()
                                          - t.float()).abs().max())
            top = sorted(per_tensor.items(), key=lambda kv: -kv[1])[:5]
            weight_diffs[f"{scope}_{cmp_name}"] = {
                "max_abs": max(per_tensor.values()),
                "top_tensors": [{"tensor": k, "max_abs": v} for k, v in top],
            }

    # verdict
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
        "drop_mask_mismatches": {"per_step": mask_mismatch,
                                 "split_vs_ref_total": mm_split,
                                 "refnoise_vs_ref_total": mm_noise,
                                 "same_order_ok": bool(mask_mismatch_ok)},
        "task_loss": verdicts["task_loss"],
        "domain_loss": verdicts["domain_loss"],
        "split_bias_detected": bool(bias_detected),
        "weight_max_abs_ok": bool(weights_ok),
        "thresholds": {"same_order_factor": SAME_ORDER_FACTOR,
                       "mask_mismatch_factor": MASK_MISMATCH_FACTOR,
                       "bias_sign_consistency": BIAS_SIGN_CONSISTENCY,
                       "weight_max_abs_bar": WEIGHT_MAX_ABS_BAR,
                       "floor_fallback": FLOOR_FALLBACK},
    }
    verdict["PASS"] = bool(
        rng_ok and mask_mismatch_ok and verdicts["task_loss"]["magnitude_ok"]
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
        "reference_deep_peak_alloc_gb": max(
            r["peak_alloc_gb"] for r in runs["reference"]["records"]
            if r["kind"] == "deep"),
        "card_bar_gb": 21.81,
    }

    hashes = {}
    for f in ("R16-H142_G1_arm.py", "R18-H152_arm_run.py",
              "R18-H152_split_exec.py", "R18-H152_exec_equivalence.py"):
        hashes[f] = hashlib.sha256((HERE / f).read_bytes()).hexdigest()

    proof = {
        "experiment": "R18-H152 SPLIT-EXECUTOR equivalence proof - "
                      "window-chunked memory-bounded executor vs the banked "
                      "H152 training path, draw-2 config",
        "seed": SEED, "init_fp": runs["reference"]["init_fp"],
        "perm_fp": PERM_FP, "n_steps_total": runs["reference"]["n_steps_total"],
        "steps_compared": n,
        "n_registered_steps": N_REGISTERED_STEPS, "n_deep_batches": N_DEEP_BATCHES,
        "deep_rows": runs["split"]["deep_rows"],
        "deep_row_windows": runs["split"]["deep_row_windows"],
        "deep_batch_pairs": runs["split"]["deep_batch_pairs"],
        "deep_batches": runs["split"]["deep_batches"],
        "noise_floor_variation": runs["refnoise"]["noise_variation"],
        "gpu": runs["split"]["gpu"],
        "masks": {"final_rng_identical": rng_ok,
                  "drop_mask_mismatch_per_step": mask_mismatch,
                  "note": "drop eligibility compares logits for exact ties, so "
                          "bf16 kernel noise flips eligibility on near-tie "
                          "windows even between the two reference runs; a "
                          "dropped window is never the MIL argmax, so a flip "
                          "moves neither the task loss nor its gradient"},
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
    out = HERE / "R18-H152_exec_equivalence.json"
    out.write_text(json.dumps(proof, indent=2))

    t_noise, t_split = noise_stats["task_loss"], split_stats["task_loss"]
    print("\n=== EQUIVALENCE PROOF SUMMARY ===", flush=True)
    print(f"steps compared: {n} (50 registered + 5 deep)  seed {SEED}", flush=True)
    print(f"final RNG identical across runs: {rng_ok}  drop-mask mismatches: "
          f"split {mm_split} vs noise {mm_noise} (same-order "
          f"{bool(mask_mismatch_ok)})", flush=True)
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
