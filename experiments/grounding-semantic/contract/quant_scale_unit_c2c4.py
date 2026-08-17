"""C2 (disjointness) and C4 (contamination census) for the `quant_scale_unit` lane.

CPU ONLY.  No GPU is queried or touched.

C2 - three string forms (raw; truncated to CFG.chunk_max_chars = 1500;
whitespace-collapsed case-folded), BOTH directions, against every evaluation
surface: the 10-subset blind arena, `gold_full`, and each held-out mechanism
eval parquet on disk.  Evidence passages are the primary unit; claims are
reported alongside.

C4 - the banked R14-H136 form via `provenance_gate.py`: 8-gram, Jaccard >= 0.3,
bidirectional, KILL above 2%, per-arena-subset attribution, plus
  (a) the synthetic spike control (10 injected arena units, 0 baseline hits)
  (b) a LIVE positive control - the lane's chunks against `R17-H145_scaleunit`,
      built from the same TabFact/FEVEROUS train tables by the same serializers
      and therefore near-duplicate by construction
  (c) coverage: units too short for an 8-gram instrument, counted and covered by
      exact matching

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_c2c4.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import importlib.util
import json
import pathlib
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_c2c4.json"
CHUNK_MAX = 1500

_WS = re.compile(r"\s+")


def forms(t):
    return t, t[:CHUNK_MAX], _WS.sub(" ", t).strip().lower()


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# every held-out mechanism eval parquet on disk, with its evidence column
MECH_EVALS = {
    "R17-H143_evalset": ("R17-H143_evalset.parquet", "chunk", "claim"),
    "R18-H150_unitswap_probe": ("R18-H150_unitswap_probe.parquet", "chunk", "claim"),
    "R17-H148_probe": ("R17-H148_probe.parquet", "chunk", "claim"),
    "R17-H149_probe": ("R17-H149_probe.parquet", "chunk", "claim"),
    "R20-H177_eval_B": ("R20-H177_eval_B.parquet", "chunk", "claim"),
    "R20-H177_eval_C": ("R20-H177_eval_C.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval": ("R20-H175b_qlane_eval.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval_clean": ("R20-H175b_qlane_eval_clean.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval_repaired": ("R20-H175b_qlane_eval_repaired.parquet", "chunk", "claim"),
}


def three_form_overlap(member_texts, surface_texts):
    """Both directions in all three forms.  Symmetric by construction on a set
    intersection, so the two directions differ only in their denominator."""
    m = [forms(t) for t in member_texts if t]
    s = [forms(t) for t in surface_texts if t]
    out = {}
    for i, name in enumerate(("raw", "truncated_1500", "ws_collapsed_casefold")):
        M = {x[i] for x in m}
        S = {x[i] for x in s}
        inter = M & S
        out[name] = {
            "member_units": len(M),
            "surface_units": len(S),
            "shared_units": len(inter),
            "member_to_surface_fraction": round(len(inter) / max(len(M), 1), 6),
            "surface_to_member_fraction": round(len(inter) / max(len(S), 1), 6),
        }
    out["max_shared_over_forms"] = max(v["shared_units"] for v in out.values()
                                       if isinstance(v, dict))
    return out


def main():
    res = {"member": "quant_scale_unit", "artifact": str(LANE)}

    lane = pl.read_parquet(LANE)
    lane_chunks = lane["chunk"].unique().to_list()
    lane_claims = lane["claim"].unique().to_list()
    res["member_units"] = {
        "rows": len(lane),
        "pairs": int(lane["pair_id"].n_unique()),
        "distinct_chunks": len(lane_chunks),
        "distinct_claims": len(lane_claims),
        "documents": int(lane["doc_id"].n_unique()),
    }

    gate = _mod("gate", "provenance_gate.py")

    # ---------------- C2 -----------------------------------------------------
    c2 = {}

    arena_raw, _ = gate.load_arena(None)
    arena_chunks = [c for v in arena_raw.values() for c in v]
    c2["arena_10_subsets"] = {
        "subsets": {k: len(v) for k, v in arena_raw.items()},
        "evidence": three_form_overlap(lane_chunks, arena_chunks),
    }

    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    g_claims, g_chunk_lists, _g_y = arm.H108.gold_full()
    gold_chunks = [c for lst in g_chunk_lists for c in lst]
    c2["gold_full"] = {
        "gold_claims": len(g_claims),
        "gold_chunks": len(gold_chunks),
        "evidence": three_form_overlap(lane_chunks, gold_chunks),
        "claims": three_form_overlap(lane_claims, g_claims),
    }

    c2["mechanism_evals"] = {}
    for name, (fname, kcol, ccol) in MECH_EVALS.items():
        p = SEM / fname
        if not p.exists():
            c2["mechanism_evals"][name] = {"status": "absent"}
            continue
        df = pl.read_parquet(p)
        blk = {"rows": len(df),
               "evidence": three_form_overlap(lane_chunks, df[kcol].to_list())}
        if ccol in df.columns:
            blk["claims"] = three_form_overlap(lane_claims, df[ccol].to_list())
        c2["mechanism_evals"][name] = blk

    worst = 0
    for surf in [c2["arena_10_subsets"], c2["gold_full"], *c2["mechanism_evals"].values()]:
        for key in ("evidence", "claims"):
            if key in surf:
                worst = max(worst, surf[key]["max_shared_over_forms"])
    c2["worst_shared_units_any_surface_any_form"] = worst
    c2["verdict"] = "PASS" if worst == 0 else "FAIL"
    res["c2"] = c2

    # ---------------- C4 -----------------------------------------------------
    c4 = {}
    for unit_name, texts in (("chunks", lane_chunks), ("claims", lane_claims)):
        r = gate.run_gate(texts, n=8, jaccard=0.3, arena_texts=arena_raw,
                          label=f"quant_scale_unit_{unit_name}")
        r.pop("hit_examples", None)
        c4[f"gate_{unit_name}"] = r
        c4[f"spike_{unit_name}"] = gate.spike_control(
            texts, arena_raw, n=8, jaccard=0.3, k=10,
            label=f"quant_scale_unit_{unit_name}_spike")

    # coverage: units too short for an 8-gram instrument
    hasher = gate._TokenHasher()
    short = {}
    for unit_name, texts in (("chunks", lane_chunks), ("claims", lane_claims)):
        sizes = [gate.ngram_hashes(t, 8, hasher).size for t in texts]
        n_short = sum(1 for s in sizes if s == 0)
        short[unit_name] = {
            "n_units": len(texts),
            "n_units_too_short_for_8gram": n_short,
            "share_too_short": round(n_short / max(len(texts), 1), 6),
        }
    # the short units are covered by exact matching against the arena
    arena_forms = {f for c in arena_chunks for f in forms(c)}
    for unit_name, texts in (("chunks", lane_chunks), ("claims", lane_claims)):
        shorts = [t for t in texts if gate.ngram_hashes(t, 8, hasher).size == 0]
        hits = sum(1 for t in shorts if any(f in arena_forms for f in forms(t)))
        short[unit_name]["exact_match_hits_among_short"] = hits
    c4["coverage"] = short

    # LIVE positive control - same source tables, same serializers
    h145 = SEM / "R17-H145_scaleunit.parquet"
    live = {"source": h145.name,
            "rationale": "built from the same TabFact/FEVEROUS train tables by the "
                         "same banked serializers - near-duplicate by construction"}
    if h145.exists():
        h145_chunks = pl.read_parquet(h145)["chunk"].unique().to_list()
        live["control_units"] = len(h145_chunks)
        lr = gate.run_gate(lane_chunks, n=8, jaccard=0.3,
                           arena_texts={"h145_scaleunit": h145_chunks},
                           label="quant_scale_unit_chunks_LIVE")
        lr.pop("hit_examples", None)
        live["result"] = lr
        live["fires"] = lr["candidate_vs_arena"]["units_with_hit"] > 0
        exact_h145 = {f for c in h145_chunks for f in forms(c)}
        live["exact_shared_chunks_any_form"] = sum(
            1 for t in lane_chunks if any(f in exact_h145 for f in forms(t)))
    c4["live_positive_control"] = live

    worst_frac = max(c4["gate_chunks"]["max_fraction"], c4["gate_claims"]["max_fraction"])
    c4["max_fraction_any_unit"] = worst_frac
    c4["kill_threshold"] = 0.02
    c4["spike_passes"] = bool(c4["spike_chunks"]["passes"] and c4["spike_claims"]["passes"])
    c4["spike_baseline_hits"] = {
        "chunks": c4["spike_chunks"]["baseline_hits"],
        "claims": c4["spike_claims"]["baseline_hits"],
    }
    c4["verdict"] = (
        "PASS" if (worst_frac < 0.02 and c4["spike_passes"]
                   and live.get("fires", False)) else "FAIL")
    res["c4"] = c4

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"c2_worst": worst, "c2": c2["verdict"],
                      "c4_max_fraction": worst_frac, "c4": c4["verdict"],
                      "live_fires": live.get("fires"),
                      "spike": c4["spike_baseline_hits"]}, indent=2))


if __name__ == "__main__":
    main()
