"""R16-H142-T twin promotion HOLDS - deterministic reads on the BANKED draw-1 twin.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R16-H142-T TWIN PROMOTION - registered (2026-08-12)": promotion needs the holds
green on the banked draw-1 checkpoint (`models/R16-H142-G1-twin`) - gold_full
>= 0.84 and the campaign's anti-gaming stage - read deterministically before
draw 2 finishes.

This script covers the gold_full hold plus the checkpoint verification. The
anti-gaming hold is the campaign's own stage (`R14-H133_antigaming.py --arm
R16-H142-T`), run separately.

Loading goes through `R16-H142_G1_arm.load_run` verbatim: the twin's adapter is
frozen at its zero init, `load_run` refuses a twin whose adapter moved off zero,
and the read path is the adapter-aware one, so the adapter's exactly-zero
contribution is carried rather than silently dropped.

Two reads, measurement only, frozen weights:

    gold_full   all 2,752 gold claims, max over the claim's chunk list with
                chunks cut to the serving unit - `R16-H142_G1_arm.evaluate`'s
                own gold_full protocol, unchanged
    hotpotqa    ONE arena subset re-read under the PRIMARY blind windowed
                protocol, purely to reproduce the banked
                `R16-H142_G1_twin_windowed_result.json` value in a fresh
                process - the checkpoint-load fingerprint

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R16-H142_T_holds.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent

GOLD_FULL_BAR = 0.84  # the promotion registration's own number
SPOT_SUBSET = "hotpotqa"  # smallest windowed pair count in the banked read


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


READS = _mod("g1reads", "R16-H142_G1_reads.py")
ARM = READS.ARM
H92 = READS.H92
ARENA = READS.ARENA
M59 = ARENA.M59
H108 = ARM.H108


def windowed_subset(model, tok, sub, claims, chunks, y):
    """The blind windowed read of R16-H142_G1_reads.main, for one subset."""
    flat_s, flat_w, set_index, owner = [], [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
        wlist = READS.evidence_sets("windowed", ks)
        for s in H92.sentences(c):
            sid = len(owner)
            owner.append(i)
            for w in wlist:
                flat_s.append(s)
                flat_w.append(w)
                set_index.append(sid)
    owner = np.array(owner)
    s_sent = ARM.score_sets(model, tok, flat_s, flat_w, set_index, len(owner),
                            tag=f"twin/windowed/{sub}")
    resp = np.array([s_sent[owner == i].min() for i in range(len(y))])
    auc, f1, _ = M59.auc_and_f1(y, resp)
    return auc, f1, len(owner), len(flat_s)


def main():
    ckpt = ARM.ROOT / "models" / ARM.RUNS["twin"]["ckpt"]
    out = HERE / "R16-H142_T_holds_goldfull.json"
    print(f"=== R16-H142-T holds: gold_full + load verification  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"checkpoint {ckpt}", flush=True)

    model, tok = ARM.load_run(ckpt)
    zero_ok = ARM.zero_init_ok(model)
    fp, n_par = ARM.init_fingerprint(model)
    print(f"adapter output layer at zero: {zero_ok}  "
          f"trained trunk+task_head blake2b-128 {fp} over {n_par} params", flush=True)

    t0 = time.time()
    cl_f, ck_f, y_f = H108.gold_full()
    s = ARM.score_claims(model, tok, cl_f, ck_f, tag="gold_full")
    gf_auc, gf_f1, _ = M59.auc_and_f1(y_f, s)
    print(f"\n  gold_full {gf_auc:.4f} (f1 {gf_f1:.4f}, n={len(y_f)}) "
          f"in {time.time() - t0:.0f}s", flush=True)

    subs = ARENA.load_subsets()
    claims, chunks, y = subs[SPOT_SUBSET]
    sp_auc, sp_f1, n_sent, n_pairs = windowed_subset(
        model, tok, SPOT_SUBSET, claims, chunks, y)
    print(f"  {SPOT_SUBSET} windowed {sp_auc:.4f} (n={len(y)}, "
          f"n_sent={n_sent}, n_pairs={n_pairs})", flush=True)

    banked = json.loads(
        (HERE / "R16-H142_G1_twin_result.json").read_text())
    banked_w = json.loads(
        (HERE / "R16-H142_G1_twin_windowed_result.json").read_text())
    payload = {
        "read": "R16-H142-T twin promotion holds - gold_full + windowed spot reproduction",
        "checkpoint": str(ckpt),
        "gold_full": {
            "auc": round(gf_auc, 4), "f1": round(gf_f1, 4), "n": len(y_f),
            "bar": GOLD_FULL_BAR,
            "pass": bool(round(gf_auc, 4) >= GOLD_FULL_BAR),
            "banked_train_time_value": banked["gold_full"]["auc"],
            "reproduces_banked": bool(round(gf_auc, 4) == banked["gold_full"]["auc"]),
        },
        "checkpoint_verification": {
            "adapter_output_layer_all_zero": zero_ok,
            "adapter_active_flag": banked["adapter_active"],
            "trained_trunk_task_head_blake2b_128": fp,
            "n_params_fingerprinted": n_par,
            "spot_subset": SPOT_SUBSET,
            "spot_windowed_auc": round(sp_auc, 4),
            "banked_spot_windowed_auc": banked_w["per_subset"][SPOT_SUBSET]["auc"],
            "reproduces_banked_windowed": bool(
                round(sp_auc, 4) == banked_w["per_subset"][SPOT_SUBSET]["auc"]),
            "banked_windowed_mean": banked_w["mean"],
        },
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  results -> {out}", flush=True)


if __name__ == "__main__":
    main()
