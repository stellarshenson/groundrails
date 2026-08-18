"""R22-H189 mmBERT LAMBDA-0 ABLATION - the DANN adversary switched off, nothing else.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R22-H189
mmBERT LAMBDA-0 ABLATION", author-ordered 2026-08-17 ~23:15.

The campaign has trained with a domain-adversarial head for 22 rounds and has
never ablated it on its own trunk. R12-H123 measured linear domain-group probes
at 0.94-0.997 against a 0.083 chance floor, so at LAMBDA_MAX 0.02 the adversary
erases nothing measurable - yet the non-English holds are still credited to it.
This arm is the separator.

    single variable   LAMBDA_MAX 0.02 -> 0.0 against R18-H150 draw 1.  The domain
                      head still exists and still trains on its own cross-entropy;
                      only the reversed gradient reaching the trunk is removed
    everything else   the flagship's, through R18-H150_arm_run.rebind - mix, seed
                      1150, schedule, LR, objective, 1500/750 windows, MAX_LEN,
                      14 DANN groups, adapter FROZEN
    checkpoint        models/R22-H189-lambda0
    results           R22-H189_lambda0_result.json           (train + in-domain)
                      R22-H189_lambda0_windowed_result.json  (arena, REPORTED only)

PAIRED BY DESIGN: seed 1150 matches R18-H150 draw 1, so the permutation
fingerprint MUST collide with it.  A collision is normally a defect - two draws
sharing a data ordering are not independent - but this is a matched control, not
an independent draw, and R20_perm_guard must not be read as failing here.

Pre-flight aborts, all checked on the CPU census stage before any GPU is spent:
    init fingerprint  MUST equal a8dfdd7275073adf0057cfb40344c45b (trunk+task_head).
                      Group count is unchanged at 14, so RNG consumption is
                      identical and the init must be bit-reproducible.  A mismatch
                      means two variables moved and the run is VOID
    perm fingerprint  MUST equal 7d13f9ac86a79574
    lambda            MUST read exactly 0.0 after the flip
    adapter           MUST remain frozen (twin integrity guard, inside rebind)

PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
.to("cuda") under WSL2 on this box.

Stages:
    census     CPU-only - mix census, window census cross-check, init and
               permutation fingerprints, then exit before any GPU work
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the windowed decomposed-min arena read (REPORTED, not a gate)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R22-H189_lambda0_run.py --stage train
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

CKPT = "R22-H189-lambda0"
TRAIN_OUT = "R22-H189_lambda0_result.json"
READ_OUT = "R22-H189_lambda0_{mode}_result.json"

# The matched control - R18-H150 draw 1, same seed, adversary ON.
PAIR_INIT_FP = "a8dfdd7275073adf0057cfb40344c45b"
PAIR_PERM_FP = "7d13f9ac86a79574"
PAIR = {"gold_full": 0.8659, "ragtruth_en": 0.7967, "ragtruth_nonen": 0.8443,
        "arena_windowed": 0.71436}

# The k=6 population the verdict is taken against, and the floors derived from it.
POP = {"ragtruth_nonen": 0.84472, "gold_full": 0.85862, "arena_windowed": 0.71218}
FLOOR = {"ragtruth_nonen": 0.01106, "gold_full": 0.01849, "arena_windowed": 0.02355}
CHANCE_DOMAIN_ACC = 1 / 14


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H150 = _mod("h150run", "R18-H150_arm_run.py")


def disarm(arm):
    """The ONLY variable. Rebind the flagship first, then remove the adversary."""
    H150.rebind(arm)
    arm.RUNS["twin"]["ckpt"] = CKPT
    arm.RUNS["twin"]["out"] = TRAIN_OUT
    was = arm.LAMBDA_MAX
    arm.LAMBDA_MAX = 0.0
    print(f"[h189] LAMBDA_MAX {was} -> {arm.LAMBDA_MAX} "
          f"(domain head still trains; no reversed gradient reaches the trunk)",
          flush=True)
    if arm.LAMBDA_MAX != 0.0:
        raise SystemExit("H189 ABORT: the adversary was not disabled")
    return arm


def check_fingerprints(res):
    """VOID the run on any drift from the matched control's init or ordering."""
    init = res.get("init_fingerprint")
    perm = res.get("perm_fingerprint")
    if init != PAIR_INIT_FP:
        raise SystemExit(
            f"H189 VOID: init fingerprint {init} != the matched control's "
            f"{PAIR_INIT_FP} - two variables moved, not one")
    if perm != PAIR_PERM_FP:
        raise SystemExit(
            f"H189 VOID: permutation fingerprint {perm} != the matched control's "
            f"{PAIR_PERM_FP} - the paired design requires the same data ordering")
    print(f"[h189] fingerprints match the matched control  init {init}  perm {perm}",
          flush=True)


def relabel_result():
    """Name the arm the record actually ran and attach the pre-registered ladder.
    Every measured number is left untouched; the verdict is the coordinator's."""
    p = HERE / TRAIN_OUT
    res = json.loads(p.read_text())
    check_fingerprints(res)

    ne = res["ragtruth_nonen"]["auc"]
    gf = res["gold_full"]["auc"]

    def rung(read, value):
        bar = POP[read] - FLOOR[read]
        return {
            "read": read, "value": value, "k6_mean": POP[read],
            "floor_2xSE": FLOOR[read], "bar_level": round(bar, 5),
            "delta_vs_k6": round(value - POP[read], 5),
            "delta_vs_paired_draw": round(value - PAIR[read], 5),
            "reading": "LOAD-BEARING" if value <= bar else "BOUNDED SMALL",
        }

    res["arm"] = "h189_lambda0_ablation"
    res["experiment"] = ("R22-H189 mmBERT lambda-0 ablation - the flagship recipe "
                         "with the DANN adversary switched off, nothing else")
    res["single_variable"] = "LAMBDA_MAX 0.02 -> 0.0 against R18-H150 draw 1"
    res["paired_control"] = {"checkpoint": "R18-H150-arm-draw1", "seed": 1150,
                             **PAIR}
    res["h189_primary"] = rung("ragtruth_nonen", ne)
    res["h189_secondary"] = rung("gold_full", gf)
    res["h189_rule"] = (
        "PRIMARY ragtruth_nonen: LOAD-BEARING at or below 0.83366, else BOUNDED "
        "SMALL - which asserts the adversary's contribution is under 0.011, NOT a "
        "null. SECONDARY gold_full at 0.84013 on the same ladder. Arena REPORTED "
        "against 0.71218 with a 0.02355 floor and carries no promotion path at k=1.")
    res["diagnostic_only"] = (
        "no promotion path in either direction; a high arena read licenses a "
        "registered k=2 arm, it does not promote, and no serving change follows")
    res["chance_domain_acc"] = round(CHANCE_DOMAIN_ACC, 4)
    res["ablation_fired_check"] = (
        "final domain-head accuracy must sit materially above the 1/14 = 0.0714 "
        "chance floor - confirms the reversed gradient was removed, not reduced")
    res["note"] = "Numbers recorded, not adjudicated - the coordinator adjudicates."
    p.write_text(json.dumps(res, indent=2))

    print(f"\n  ragtruth_nonen {ne:.4f}  (k=6 {POP['ragtruth_nonen']}, "
          f"bar {res['h189_primary']['bar_level']}, "
          f"paired draw {PAIR['ragtruth_nonen']}) -> {res['h189_primary']['reading']}",
          flush=True)
    print(f"  gold_full      {gf:.4f}  (k=6 {POP['gold_full']}, "
          f"bar {res['h189_secondary']['bar_level']}, "
          f"paired draw {PAIR['gold_full']}) -> {res['h189_secondary']['reading']}",
          flush=True)
    print(f"  ragtruth_en    {res['ragtruth_en']['auc']}  "
          f"(paired draw {PAIR['ragtruth_en']})", flush=True)
    print(f"result relabelled -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    args = ap.parse_args()

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        disarm(reads.ARM)
        reads.out_path = lambda run, mode: HERE / READ_OUT.format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    out = HERE / TRAIN_OUT
    if args.stage == "train" and out.exists() and out.stat().st_size > 0:
        print(f"SKIP (on disk: {out.name})", flush=True)
        return

    arm = disarm(_mod("g1arm", "R16-H142_G1_arm.py"))
    sys.argv = ["arm", "--run", "twin"]
    if args.stage == "census":
        sys.argv.append("--census-only")
    arm.main()
    if args.stage == "train":
        relabel_result()


if __name__ == "__main__":
    main()
