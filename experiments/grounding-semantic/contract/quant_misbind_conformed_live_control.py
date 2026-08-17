"""C4 tiered LIVE positive control, re-run for the CONFORMED member.

The banked control (`quant_misbind_live_control.py`) is imported and run
unchanged; only its output directory is repointed so the conformed pass owns its
own artifact and cites nothing from the original pass.  The control degrades real
arena documents, so its tiers are a property of the gate and the arena; the
member-side baseline it is read against is the conformed census.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_conformed_live_control.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import tempfile

HERE = pathlib.Path(__file__).parent


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    M = _mod("livectl", HERE / "quant_misbind_live_control.py")
    with tempfile.TemporaryDirectory() as td:
        M.HERE = pathlib.Path(td)
        M.main()
        out = json.loads((pathlib.Path(td) / "quant_misbind_c4_live_control.json").read_text())
    census = json.loads((HERE / "quant_misbind_conformed_c4_census.json").read_text())
    out["baseline_lane_reads"] = (
        f"conformed member evidence gate max fraction "
        f"{census['evidence_gate']['max_fraction']}, claim gate "
        f"{census['claims_gate']['max_fraction']} against the same arena"
    )
    (HERE / "quant_misbind_conformed_c4_live_control.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("fires", "monotone_in_degradation", "baseline_lane_reads")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
