"""Contract clause C4 re-verified on the CONFORMED `tabfact` member.

CPU only. The instrument is the banked R14-H136 form, invoked through
`provenance_gate.py` unchanged: 8-gram, Jaccard >= 0.3, BIDIRECTIONAL, WARN
0.5%, KILL 2%, per-arena-subset attribution.

Census on both unit types, coverage stated, then FOUR controls - because the
conforming cut removed exactly the text the first pass used as its live control,
and a clean number from an unproven gate is not evidence:

  1. SYNTHETIC SPIKE - arena units injected into the candidate side; must be
     detected 10/10 with 0 baseline hits.
  2. LIVE control A, REGISTER PROOF - TabFact's own test+validation serialised
     tables offered to the identical gate with the BANKED (pre-conformance)
     member as the reference side. Near-duplicate by construction (one Wikipedia
     table, two csv ids). This is the first pass's control, re-run unchanged: it
     proves the gate fires on pipe-delimited table text, which is the register
     an 8-gram instrument could silently fail on.
  3. LIVE control B, CONFORMED RESIDUAL - the same candidates against the
     CONFORMED member. A fires / B silent pair is the demonstration that the cut
     worked with the gate's power held constant.
  4. LIVE control C, ARENA SIDE - real arena documents sliced to an interior
     1,500-character window (chars 200-1700, only documents longer than 1,800
     chars, so every unit is a STRICT subset of its source and never a copy) fed
     as candidates against the arena itself. The census reads against the arena;
     this shows the gate fires on that reference side on text that is
     near-duplicate rather than identical.

Out: tabfact_conformed_c4.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member_conformed.parquet"
BANKED = HERE / "tabfact_member.parquet"
OUT = HERE / "tabfact_conformed_c4.json"

N = 8
JACCARD = 0.3
KILL = 0.02
WARN = 0.005


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_chunk(cap, tbl):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")


def coverage(G, texts):
    short = [t for t in texts if len(G.normalize(t).split()) < N]
    return {"n_units": len(texts), "n_units_too_short_for_8gram": len(short),
            "share_too_short": round(len(short) / max(len(texts), 1), 6)}


def exact_residual(short_texts, arena_texts):
    pool = {c for v in arena_texts.values() for c in v}
    npool = {" ".join(c.split()).casefold() for c in pool}
    raw = sum(1 for t in short_texts if t in pool)
    nrm = sum(1 for t in short_texts if " ".join(t.split()).casefold() in npool)
    return {"units": len(short_texts), "exact_raw_hits": raw,
            "exact_normalised_hits": nrm}


def main():
    t0 = time.time()
    G = _mod("pgate", SEM / "provenance_gate.py")
    df = pl.read_parquet(MEMBER)

    evidence = sorted(set(df["chunk_untrunc"].to_list()))
    claims = sorted(set(df["claim"].to_list()))
    print(f"candidate units: {len(evidence)} evidence, {len(claims)} claims", flush=True)

    arena_texts, _ = G.load_arena()
    print(f"arena: {sum(len(v) for v in arena_texts.values())} chunks over "
          f"{len(arena_texts)} subsets", flush=True)

    res = {"member": "tabfact_conformed",
           "clause": "C4 - contamination census with a live positive control",
           "instrument": f"provenance_gate.py, R14-H136 form: {N}-gram, "
                         f"Jaccard >= {JACCARD}, bidirectional, WARN {WARN}, KILL {KILL}",
           "census": {}, "coverage": {}, "controls": {}}

    for label, texts in (("evidence_serialised_tables", evidence),
                         ("claims_statements", claims)):
        print(f"--- census: {label} ({len(texts)} units) ---", flush=True)
        t1 = time.time()
        r = G.run_gate(texts, n=N, arena_texts=arena_texts, jaccard=JACCARD,
                       warn=WARN, kill=KILL, label=f"tabfact_conformed_{label}")
        res["census"][label] = r
        cov = coverage(G, texts)
        short = [t for t in texts if len(G.normalize(t).split()) < N]
        cov["exact_match_residual"] = exact_residual(short, arena_texts)
        res["coverage"][label] = cov
        print(f"  verdict {r['verdict']}  max_fraction {r['max_fraction']}  "
              f"({time.time() - t1:.0f}s)", flush=True)

    # ---- 1. synthetic spike ------------------------------------------------ #
    print("--- spike control ---", flush=True)
    res["controls"]["synthetic_spike_on_evidence"] = G.spike_control(
        evidence, arena_texts, n=N, jaccard=JACCARD, k=10, label="tabfact_conformed_spike")
    print(json.dumps(res["controls"]["synthetic_spike_on_evidence"]), flush=True)

    # ---- 2/3. live controls on the TabFact register ------------------------ #
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    held = []
    for split in ("test", "validation"):
        n = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n)))
        held += [build_chunk(c, t) for c, t in
                 zip(d["table_caption"].to_list(), d["table_text"].to_list(), strict=True)]
    held = sorted(set(held))

    banked_ev = sorted(set(pl.read_parquet(BANKED)["chunk_untrunc"].to_list()))
    for key, ref, note in (
        ("live_control_A_register_proof_vs_BANKED_member", banked_ev,
         "the first pass's control re-run UNCHANGED: TabFact's own held-out splits "
         "against the PRE-CONFORMANCE member. It proves the gate fires on "
         "pipe-delimited serialised tables - the register an 8-gram instrument "
         "could silently fail on. Reference side is the banked member, which this "
         "verification does not use for any census"),
        ("live_control_B_conformed_residual", evidence,
         "the same candidates against the CONFORMED member. A fires and B is "
         "silent, with the gate, the units and the threshold all held constant - "
         "that pair is the demonstration that the document cut removed the "
         "near-duplication rather than the gate losing its power"),
    ):
        live = G.run_gate(held, n=N, arena_texts={"tabfact_reference": ref},
                          jaccard=JACCARD, warn=WARN, kill=KILL,
                          label="tabfact_heldout_splits")
        res["controls"][key] = {
            "design": note,
            "candidate_units": len(held),
            "reference_units": len(ref),
            "fires": live["candidate_vs_arena"]["units_with_hit"] > 0,
            "units_with_hit": live["candidate_vs_arena"]["units_with_hit"],
            "fraction": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "reverse_direction": {
                "units_with_hit": live["arena_vs_candidate"]["units_with_hit"],
                "fraction": live["arena_vs_candidate"]["fraction"],
                "best_jaccard": live["arena_vs_candidate"].get("best_jaccard"),
            },
            "hit_examples": live["hit_examples"][:5],
        }
        print(f"{key}: fires={res['controls'][key]['fires']} "
              f"{res['controls'][key]['units_with_hit']}/{len(held)}", flush=True)

    # ---- 4. live control on the ARENA side --------------------------------- #
    windows = [c[200:1700] for v in arena_texts.values() for c in v if len(c) > 1800]
    windows = sorted(set(windows))
    flat_arena = {c for v in arena_texts.values() for c in v}
    identical = sum(1 for w in windows if w in flat_arena)
    live_c = G.run_gate(windows, n=N, arena_texts=arena_texts, jaccard=JACCARD,
                        warn=WARN, kill=KILL, label="arena_interior_windows")
    res["controls"]["live_control_C_arena_side"] = {
        "design": "real arena documents sliced to their interior 1,500-character "
                  "window (chars 200-1700, documents longer than 1,800 chars only), "
                  "fed as candidates against the arena itself. Every unit is a "
                  "STRICT subset of a real arena document, never a copy of one - "
                  "near-duplicate by construction on the reference side the census "
                  "actually reads against",
        "candidate_units": len(windows),
        "candidate_units_identical_to_an_arena_chunk": identical,
        "fires": live_c["candidate_vs_arena"]["units_with_hit"] > 0,
        "units_with_hit": live_c["candidate_vs_arena"]["units_with_hit"],
        "fraction": live_c["candidate_vs_arena"]["fraction"],
        "best_jaccard": live_c["candidate_vs_arena"].get("best_jaccard"),
    }
    print(json.dumps(res["controls"]["live_control_C_arena_side"])[:600], flush=True)

    worst = max(res["census"][k]["max_fraction"] for k in res["census"])
    res["max_fraction_any_unit_type"] = worst
    res["verdict"] = "KILL" if worst >= KILL else ("WARN" if worst >= WARN else "PASS")
    res["margin_to_kill_0.02"] = round(KILL - worst, 6)
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name}  verdict {res['verdict']}  max {worst}  "
          f"({res['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
