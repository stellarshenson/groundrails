"""R16-H142-T draw 2 - stage-1 recovery driver.

The draw-2 campaign was interrupted AFTER training completed (checkpoint saved
15:17, resume.pt deleted per the trainer's own completion path) but BEFORE the
in-domain suite finished (gold read at 25,600/26,048). A naive relaunch of the
campaign script restarted training from scratch because the trainer leaves no
final-model skip marker - this driver exists so the completed checkpoint does
not get needlessly retrained.

What it does: loads the banked checkpoint with the trainer's OWN load_run and
runs the trainer's OWN evaluate() (the in-domain suite every campaign arm is
read under), then writes the stage-1 result JSON the campaign's stage_unless
checks for. Metadata fields are recovered from the draw-2 log verbatim where
they appear there; the JSON carries a `recovery` block making the provenance
explicit. No training happens in this file.

Run:  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
      uv run python experiments/grounding-semantic/R16-H142_T_draw2_recover.py
"""

import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).parent
CKPT = "models/R16-H142-T-draw2"
OUT = HERE / "R16-H142_T_draw2_result.json"


def main():
    spec = importlib.util.spec_from_file_location("g1arm", HERE / "R16-H142_G1_arm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # the wrapper's rebind, replicated verbatim (R16-H142_T_draw2_run.py)
    m.SEED = 2142
    m.RUNS["twin"]["ckpt"] = "R16-H142-T-draw2"
    m.RUNS["twin"]["out"] = "R16-H142_T_draw2_result.json"

    model, tok = m.load_run(CKPT)
    res = m.evaluate(model, tok)
    res.update({
        "run": "twin", "adapter_active": False, "seed": 2142,
        "experiment": "R16-H142-T twin draw 2 (confirming draw, seed 2142)",
        "init_fingerprint": "9377707d7a926278c850bb5bff8e6b07",
        "perm_fingerprint": "eebe673dabeef46f",
        "mix_rows": 685670, "dann_groups": 12, "n_steps": 14300,
        "recovery": {
            "reason": "container interruption after train, mid in-domain suite",
            "method": "banked load_run + evaluate() on the completed 15:17 checkpoint; "
                      "no retraining. Fingerprints/census from the draw-2 log.",
        },
    })
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("gold", "gold_full", "ragtruth_en", "ragtruth_nonen") if k in res}, indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
