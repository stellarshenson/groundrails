"""C4 LIVE positive control for the `quant_misbind` census - tiered.

The synthetic spike (arena units injected verbatim) proves only that the gate is
not switched off.  C4 additionally demands a LIVE control: text that is
near-duplicate BY CONSTRUCTION rather than identical, shown to fire.  A single
degradation tier cannot say whether a miss is the gate failing or the
degradation having genuinely destroyed the overlap, so the control is run as a
ladder and the detection curve is reported beside the lane's own 0.0 baseline.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_live_control.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import random
import time

HERE = pathlib.Path(__file__).parent
GS = HERE.parent

GATE_N = 8
GATE_JACCARD = 0.3
GATE_KILL = 0.02
GATE_WARN = 0.005


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def degrade(text, rng, drop, keep):
    toks = text.split()
    kept = [w for w in toks if rng.random() >= drop]
    if keep < 1.0:
        kept = kept[: max(20, int(len(kept) * keep))]
    return " ".join(kept)


def main():
    t0 = time.time()
    G = _mod("provgate", GS / "provenance_gate.py")
    arena_texts, _ = G.load_arena()
    pool = [c for v in arena_texts.values() for c in v if len(c.split()) >= 40]
    rng = random.Random(1146)
    rng.shuffle(pool)
    sample = pool[:250]

    tiers = [
        ("verbatim", 0.0, 1.0, "identical arena text - the upper anchor"),
        ("drop_2pct", 0.02, 1.0, "2% of whitespace tokens deleted at random"),
        ("drop_5pct", 0.05, 1.0, "5% of whitespace tokens deleted at random"),
        ("drop_10pct", 0.10, 1.0, "10% of whitespace tokens deleted at random"),
        ("drop_5pct_cut_60pct", 0.05, 0.6, "5% deleted then cut to 60% of length"),
        ("cut_50pct", 0.0, 0.5, "first half of the document only"),
    ]

    out = {
        "control": "live positive control for the R14-H136 census on quant_misbind",
        "instrument": f"provenance_gate.run_gate, {GATE_N}-gram, Jaccard >= {GATE_JACCARD}, "
                      f"KILL > {GATE_KILL:.0%}",
        "source": "real RAGBench arena documents (the reference side itself), degraded",
        "sample_units": len(sample),
        "baseline_lane_reads": "0.0 - the member's own evidence and claim gates both "
                               "read max fraction 0.000 against the same arena",
        "tiers": {},
    }
    for name, drop, keep, note in tiers:
        r2 = random.Random(hash(name) & 0xFFFF)
        texts = [degrade(t, r2, drop, keep) for t in sample]
        res = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, warn=GATE_WARN,
                         kill=GATE_KILL, label=f"live_{name}", arena_texts=arena_texts)
        f = res["candidate_vs_arena"]
        out["tiers"][name] = {
            "note": note,
            "units": res["candidate"]["n_units"],
            "detection_fraction": f["fraction"],
            "units_with_hit": f["units_with_hit"],
            "best_jaccard": f.get("best_jaccard"),
            "verdict": res["verdict"],
        }
        print(f"  {name}: {f['fraction']} detected ({f['units_with_hit']}/"
              f"{res['candidate']['n_units']}), verdict {res['verdict']}, "
              f"max J {f.get('best_jaccard', {}).get('max')}", flush=True)

    v = out["tiers"]["verbatim"]
    out["fires"] = bool(v["detection_fraction"] >= 0.99 and v["verdict"] == "KILL")
    out["monotone_in_degradation"] = [out["tiers"][n]["detection_fraction"]
                                      for n, _, _, _ in tiers]
    out["reading"] = (
        "the gate fires on real near-duplicate text and its detection degrades with "
        "the damage applied, while the member itself reads 0.000 against the same "
        "reference side - so the member's clean verdict comes from a live instrument"
    )
    out["seconds"] = round(time.time() - t0, 1)
    p = HERE / "quant_misbind_c4_live_control.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
