"""R22-H188 PRIMARY - the FinDVer read of the two derivation draws.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R22-H188 DERIVATION-ENHANCED MIX" (2026-08-17 ~17:32). PRIMARY: FinDVer-numeric
AUROC, mean over the two draws, >= 0.55, against the banked flagship two-draw
mean 0.49585. CONTROL: `ie` and `knowledge` each within 0.02 of the flagship
means 0.66095 / 0.58380. KILL: numeric two-draw mean < 0.52.

MEASUREMENT ONLY - nothing trains, nothing is tuned. The read is the BANKED one,
reused unchanged: `R20-H176_findver_read.py` is loaded and its `main()` called
with only the checkpoint binding and the output path rebound. No part of the
protocol - untruncated evidence windowed 1,500/750, claim scored against every
window, MAX over windows, claim-level rows so no min-over-sentences stage,
frozen trunk + task head through `R15_gate_common.load_ckpt/.score` - is
re-derived here.

Two fields of the emitted payload belong to R20-H176's OWN registration and not
to this arm: `branch` and `branch_note` evaluate H176's CONFIRMED / PARTIAL /
DEPRIORITISE ladder for the FLAGSHIP instrument question. They are left exactly
as the banked reader writes them and are flagged in `banked_branch_scope`;
H188's bars are the coordinator's to adjudicate from the numbers.

FinDVer is an EVALUATION SURFACE and is absent from the training mix by
construction - the H188 mix is the clean public mix (RAGTruth EN + 7, HaluEval,
PsiloQA, VitaminC, TabFact) plus three TabFact/EDGAR-sourced lanes.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R22-H188_findver_read.py
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit("GPU PLACEMENT ABORT: CUDA_VISIBLE_DEVICES is unset - set it "
                     "explicitly (0, 1 or 2)")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R22-H188_findver_read.json"

DRAWS = {"h188d1": "R22-H188-arm-draw1", "h188d2": "R22-H188-arm-draw2"}

FLAGSHIP = {"ie": 0.66095, "numeric": 0.49585, "knowledge": 0.58380}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    read = _mod("h176findver", "R20-H176_findver_read.py")
    read.DRAWS = DRAWS
    read.OUT = OUT
    read.main()

    payload = json.loads(OUT.read_text())
    payload["experiment"] = ("R22-H188 FinDVer PRIMARY read of the derivation-enhanced "
                             "draws (measurement only, zero training)")
    payload["registration"] = ("docs/experiments/semantic-grounding-experiments.md, block "
                               "'R22-H188 DERIVATION-ENHANCED MIX' (2026-08-17 ~17:32)")
    payload["read_path"] = ("R20-H176_findver_read.py main(), reused unchanged with only "
                            "DRAWS and OUT rebound")
    payload["banked_branch_scope"] = (
        "the `branch` / `branch_note` fields are R20-H176's own registered ladder for "
        "the FLAGSHIP instrument question; H188's bars (PRIMARY numeric >= 0.55, "
        "CONTROL ie/knowledge within 0.02, KILL < 0.52) are adjudicated by the "
        "coordinator from the numbers, not by this field")
    payload["flagship_two_draw_mean"] = FLAGSHIP
    payload["delta_vs_flagship"] = {
        k: round(payload["two_draw_mean"][k]["auroc"] - v, 5)
        for k, v in FLAGSHIP.items()}
    payload["note"] = "Numbers recorded, not adjudicated - the coordinator adjudicates."
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n--- H188 vs flagship (two-draw means) ---", flush=True)
    for k, v in FLAGSHIP.items():
        got = payload["two_draw_mean"][k]["auroc"]
        print(f"  {k:<10} {got:.4f}  flagship {v:.5f}  delta {got - v:+.5f}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
