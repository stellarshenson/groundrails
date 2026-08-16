"""R20-H174 CONTAMINATION CENSUS - the R14-H136 wall on the BUILT lanes, CPU.

Registration clause (docs/experiments/semantic-grounding-experiments.md, block
"R20-H174 HAGRID/EMANUAL PORTFOLIO ARM"): "L1/L4 are generators (clear by
construction); L2 requires the R14-H136 8-gram census RE-RUN on the built
distractor pairings before any training (MiniCheck/VitaminC hold green today;
the new pairings are new text) ... a census failure kills the lane, not the arm".
The generator lanes are censused anyway - a clause that is only asserted is not
a wall.

INSTRUMENT - reused, not reinvented.  `provenance_gate.py` in the R14-H136
ruling-2 form that `R19_supply_gates.py` runs: 8-gram, Jaccard >= 0.3,
bidirectional, WARN at 0.5%, KILL at 2% of the candidate side, against ALL TEN
walled arena corpora, with the spike control first (arena units injected into
the candidate side must all be detected, so a gate that cannot fire is caught).
Thresholds are imported from that file, never restated here.

UNITS - contamination is a document-overlap property, so each lane is gated on
its deduplicated EVIDENCE units and, separately, on its deduplicated CLAIMS:

  L1  chunks (real MiniCheck / VitaminC evidence) and claims
  L2  the ATOMIC pool passages, not the concatenated pools.  A pool's n-gram set
      is the union of its members' plus the few that cross a join, so gating the
      members is the STRICTER read: Jaccard against a 5,000-char pool is diluted
      by the pool's own size and would understate any overlap a member carries
  L4  generated pages and claims - CLEAR by construction, measured regardless

The arena side is read through `provenance_gate.load_arena()` only, which takes
document n-grams and nothing else.  No arena item text is inspected, printed or
stored by this script.

Run:  uv run python experiments/grounding-semantic/R20-H174_lane_census.py [L1 L2 L4]
"""

import importlib.util
import json
import pathlib
import sys
import time

import polars as pl

HERE = pathlib.Path(__file__).parent

_spec = importlib.util.spec_from_file_location("provgate", HERE / "provenance_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

_spec2 = importlib.util.spec_from_file_location("gates", HERE / "R19_supply_gates.py")
_gates_src = (HERE / "R19_supply_gates.py").read_text()
# the R19 file's thresholds are the registered ones; read them literally rather
# than re-declaring numbers that could drift from the instrument
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])

SEP = "\n\n"          # the L2 pool separator
SPIKE_SAMPLE = 2000   # as in R19_supply_gates.gate_one


def units(lane, df):
    """(evidence units, claim units) - deduplicated, sorted, non-empty."""
    claims = sorted({c for c in df["claim"].to_list() if c.strip()})
    if lane == "L2":
        ev = set()
        for chunk in df["chunk"].to_list():
            ev.update(p for p in chunk.split(SEP) if p.strip())
        note = ("atomic pool passages, deduplicated across the lane - the "
                "stricter unit than the concatenated pool")
    else:
        ev = {c for c in df["chunk"].to_list() if c.strip()}
        note = "lane evidence chunks, deduplicated"
    return sorted(ev), claims, note


def gate(name, texts, arena_texts, spike=False):
    t0 = time.time()
    res = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                     label=name, arena_texts=arena_texts)
    out = {"pass": res["verdict"] != "KILL", "result": res,
           "seconds": round(time.time() - t0, 1)}
    if spike:
        sp = G.spike_control(texts[:SPIKE_SAMPLE], arena_texts, n=GATE_N,
                             jaccard=GATE_JACCARD, k=10, label=f"{name}_spike")
        out["spike_control"] = sp
        out["pass"] = out["pass"] and sp["passes"]
        print(f"  spike control: {sp}", flush=True)
    print(f"  {name}: verdict {res['verdict']} at max fraction "
          f"{res['max_fraction']} (best-Jaccard max "
          f"{res['candidate_vs_arena'].get('best_jaccard', {}).get('max')}) "
          f"in {out['seconds']}s", flush=True)
    return out


def census_one(lane, arena_texts):
    src = HERE / f"R20-H174_lane_{lane}.parquet"
    print(f"\n=== census {lane} ({src.name})", flush=True)
    df = pl.read_parquet(src)
    ev, claims, note = units(lane, df)
    print(f"  gate units: {len(ev)} evidence, {len(claims)} claims "
          f"(rows {df.height})", flush=True)

    res = {
        "lane": lane,
        "parquet": src.name,
        "rows": df.height,
        "instrument": "provenance_gate.py (R14-H136 ruling 2 form: "
                      f"{GATE_N}-gram, Jaccard >= {GATE_JACCARD}, bidirectional, "
                      f"KILL > {GATE_KILL:.0%}), thresholds read from "
                      "R19_supply_gates.py",
        "unit_definition": note,
        "evidence_units": len(ev),
        "claim_units": len(claims),
        "evidence_gate": gate(f"h174_{lane}_evidence", ev, arena_texts, spike=True),
        "claim_gate": gate(f"h174_{lane}_claims", claims, arena_texts),
    }
    res["status"] = ("GREEN" if res["evidence_gate"]["pass"] and res["claim_gate"]["pass"]
                     else "RED")
    out = HERE / f"R20-H174_lane_{lane}_census.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  === {lane} CENSUS {res['status']} -> {out.name}", flush=True)
    return res["status"]


def main():
    lanes = sys.argv[1:] or ["L1", "L2", "L4"]
    arena_texts, _ = G.load_arena()   # n-grams only; no item text is read here
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    summary = {}
    for lane in lanes:
        summary[lane] = census_one(lane, arena_texts)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if all(v == "GREEN" for v in summary.values()) else 1)


if __name__ == "__main__":
    main()
