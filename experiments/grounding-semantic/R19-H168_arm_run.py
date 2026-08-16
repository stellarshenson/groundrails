"""R19-H168 EuroBERT-210m TRUNK SWAP - the flagship recipe, one variable changed.

Author-ordered controlled trunk comparison. The claim under test is narrow: hold
EVERYTHING the R18-H150 flagship arm does and change only which encoder the
recipe is applied to.

HOW "EXACT RECIPE" IS GUARANTEED
-------------------------------
This file does NOT restate the recipe. It imports `R18-H150_arm_run` and calls
that module's own `rebind()`, so the mix builder, the 14-group map, the row and
window census aborts, the lane manifests and the seed all come from the
flagship's file rather than from a copy that could drift. Everything below
`rebind()` is the flagship arm verbatim:

    MAX_LEN 512, LR 1e-5, AdamW + OneCycleLR (warmup 0.1), clip 1.0, 1 epoch,
    48 sets / 96 pairs per batch, bf16 trunk encode with fp32 heads and loss,
    MIL max-over-window BCE, 14-group DANN through gradient reversal at
    LAMBDA_MAX 0.02 on the Ganin ramp, WIN/STRIDE 1500/750, adapter OFF.

SEED is deliberately left at the flagship's 1150 so this run is the closest
possible counterpart of banked draw 1 rather than a fresh draw.

THE THREE DEVIATIONS, all forced, all load-time only
----------------------------------------------------
1. `H108.STUDENT` is repointed to `EuroBERT/EuroBERT-210m`. This is the arm.
2. `trust_remote_code=True` is injected into the `AutoModel` / `AutoTokenizer`
   calls, because EuroBERT ships pinned modelling code. The banked arm file is
   NOT edited - it is monkey-patched in this process only, so the flagship stays
   byte-identical and reproducible.
3. The `R19-H168_eurobert_compat` RoPE shim is installed. EuroBERT's pinned code
   targets transformers 4.40 and this project runs 5.14.1, which removed the
   `"default"` key from `ROPE_INIT_FUNCTIONS`. The shim restores the 4.40
   closed form and `verify()` checks it against the analytic definition.

WHAT THE COMPARISON CANNOT HOLD FIXED, measured in Gate B and recorded here
--------------------------------------------------------------------------
The trunks do not share a tokenizer. On identical text EuroBERT spends 11.4%
more tokens on average, and the excess concentrates on the non-English RAGTruth
variants: hu +20.1%, pl +20.0%, it +16.8%, de +16.4%, es +14.9%, cn +12.0%,
while English is 2.6% CHEAPER. At the fixed MAX_LEN of 512 the truncated share
therefore moves against EuroBERT exactly where the non-English hold applies -
de 71.5% -> 93.8%, it 76.5% -> 95.8%, fr 76.2% -> 93.8%.

This is NOT corrected. Tokenizer efficiency is a property of the trunk, so a
trunk that sees less text per token budget is fairly charged for it. But a loss
on the non-English hold must be read against these numbers before it is
attributed to representation quality.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R19-H168_arm_run.py --stage train
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

EUROBERT = "EuroBERT/EuroBERT-210m"
CKPT = "R19-H168-eurobert-draw1"
TRAIN_OUT = "R19-H168_arm_draw1_result.json"
READ_OUT = "R19-H168_arm_draw1_{mode}_result.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


COMPAT = _mod("h168compat", "R19-H168_eurobert_compat.py")
H150 = _mod("h150run", "R18-H150_arm_run.py")


def _wrap(orig):
    """`from_pretrained` with `trust_remote_code` forced on, nothing else changed."""

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **kw):
            kw.setdefault("trust_remote_code", True)
            return orig.from_pretrained(*a, **kw)

    return _Auto


def swap_trunk(mod, tag):
    """The arm's ONLY variable, applied in-process. Never edits a banked file."""
    needed = COMPAT.install()
    if hasattr(mod, "H108"):
        was = mod.H108.STUDENT
        mod.H108.STUDENT = EUROBERT
        print(f"[{tag}] trunk {was} -> {EUROBERT}", flush=True)
    if hasattr(mod, "AutoModel"):
        mod.AutoModel = _wrap(mod.AutoModel)
    if hasattr(mod, "AutoTokenizer"):
        mod.AutoTokenizer = _wrap(mod.AutoTokenizer)
    print(f"[{tag}] rope shim installed={needed}", flush=True)
    return mod


def verify_shim():
    """Fatal pre-flight - a wrong RoPE trains silently, so it is checked here too
    and not only in the gate script."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(EUROBERT, trust_remote_code=True)
    COMPAT.install()
    ctrl = COMPAT.verify(cfg)
    print(f"rope control: theta={ctrl['rope_theta']} head_dim={ctrl['head_dim']} "
          f"n_freqs={ctrl['n_freqs']} err={ctrl['max_abs_err_vs_analytic']:.2e} "
          f"-> {'PASS' if ctrl['pass'] else 'FAIL'}", flush=True)
    if not ctrl["pass"]:
        raise SystemExit("H168 ABORT: the RoPE shim does not reproduce the analytic "
                         "inverse frequencies - refusing to train on a model whose "
                         "positional encoding is wrong")
    return ctrl


def rebind(mod, tag):
    """Flagship recipe first, then the single swap, then this arm's paths."""
    H150.rebind(mod)                      # the ENTIRE flagship recipe, from its own file
    swap_trunk(mod, tag)                  # the one variable
    if hasattr(mod, "RUNS"):
        mod.RUNS["twin"]["ckpt"] = CKPT
        mod.RUNS["twin"]["out"] = TRAIN_OUT
    return mod


def relabel_result():
    p = HERE / TRAIN_OUT
    res = json.loads(p.read_text())
    res["arm"] = "h168_eurobert_draw1"
    res["experiment"] = ("R19-H168 EuroBERT-210m trunk swap draw 1 - the R18-H150 "
                         "flagship recipe applied to EuroBERT/EuroBERT-210m")
    res["trunk"] = EUROBERT
    res["trunk_swapped_from"] = "jhu-clsp/mmBERT-base"
    res["single_variable"] = ("the encoder only - mix, seed, schedule, objective, "
                              "DANN groups, window presentation and MAX_LEN are the "
                              "flagship's, taken from R18-H150_arm_run.rebind")
    res["deviations"] = [
        "trust_remote_code=True injected (EuroBERT ships pinned modelling code)",
        "R19-H168_eurobert_compat RoPE shim - transformers 5.14.1 removed the "
        "'default' ROPE_INIT_FUNCTIONS key EuroBERT's 4.40-era code requires",
    ]
    res["tokenizer_confound"] = ("EuroBERT spends 11.4% more tokens on identical "
                                 "text (hu +20.1%, pl +20.0%, it +16.8%, de +16.4%, "
                                 "en -2.6%); at MAX_LEN 512 it therefore sees less "
                                 "evidence on the non-EN hold subsets. NOT corrected "
                                 "- see R19-H168_trunk_gate_b.json")
    res["comparator"] = {"flagship_2draw_mean": 0.71549,
                         "flagship_draw1_windowed": 0.71436,
                         "bars": {"primary_2draw": 0.72049, "pilot_kill_draw1": 0.71049,
                                  "gold_full": 0.84, "non_en": 0.82, "anti_gaming": 0.7438}}
    res["bars_note"] = ("the `bars`/`control` blocks are the banked G1 twin's; "
                        "H168's registered bars are adjudicated by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    args = ap.parse_args()

    verify_shim()

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        rebind(reads.ARM, "reads")
        reads.out_path = lambda run, mode: HERE / READ_OUT.format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), "arm")
    sys.argv = ["arm", "--run", "twin"]
    if args.stage == "census":
        sys.argv.append("--census-only")
    arm.main()
    if args.stage == "train":
        relabel_result()


if __name__ == "__main__":
    main()
