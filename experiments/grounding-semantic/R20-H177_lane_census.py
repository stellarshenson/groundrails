"""R20-H177 CONTAMINATION CENSUS - the R14-H136 wall on the built lanes, CPU.

Registration clause (docs/experiments/semantic-grounding-experiments.md, block
"R20-H177 NUMERIC-VERIFICATION PORTFOLIO ARM", Stage 0): "R14-H136 8-gram census
on all new pairings, contamination: FinQA/TAT-QA source corpora WALLED -
untouched; EDGAR restricted slice only".  Both training lanes AND both held-out
mechanism evals are censused - an eval that leaks the arena would invalidate the
PRIMARY gate it exists to serve, so it carries the same wall as the lane.

INSTRUMENT - reused, not reinvented.  `provenance_gate.py` in the R14-H136
ruling-2 form that `R19_supply_gates.py` and `R20-H174_lane_census.py` run:
8-gram, Jaccard >= 0.3, bidirectional, WARN at 0.5%, KILL at 2% of the candidate
side, against ALL TEN walled arena corpora, with the spike control (arena units
injected into the candidate side must all be detected, so a gate that cannot fire
is caught).  Thresholds are read from `R19_supply_gates.py`, never restated here.

UNITS - contamination is a document-overlap property, so each artifact is gated
on its deduplicated EVIDENCE chunks and, separately, on its deduplicated CLAIMS.
Lane B mixes two registers in one lane (TabFact serialisations and EDGAR prose);
they are gated together, because the wall is a property of the lane the model
sees, and the per-arena-subset breakdown in the result localises any hit.

The arena side is read through `provenance_gate.load_arena()` only, which takes
document n-grams and nothing else.  No arena item text is inspected, printed or
stored by this script.

Run:  uv run python experiments/grounding-semantic/R20-H177_lane_census.py
      [lane_B lane_C eval_B eval_C]
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

_gates_src = (HERE / "R19_supply_gates.py").read_text()
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])

SPIKE_SAMPLE = 2000
ARTIFACTS = ("lane_B", "lane_C", "eval_B", "eval_C")


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


def census_one(artifact, arena_texts):
    src = HERE / f"R20-H177_{artifact}.parquet"
    print(f"\n=== census {artifact} ({src.name})", flush=True)
    df = pl.read_parquet(src)
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    claims = sorted({c for c in df["claim"].to_list() if c.strip()})
    print(f"  gate units: {len(ev)} evidence, {len(claims)} claims "
          f"(rows {df.height})", flush=True)

    res = {
        "artifact": artifact,
        "parquet": src.name,
        "rows": df.height,
        "pairs": int(df["pair_id"].n_unique()),
        "instrument": "provenance_gate.py (R14-H136 ruling 2 form: "
                      f"{GATE_N}-gram, Jaccard >= {GATE_JACCARD}, bidirectional, "
                      f"KILL > {GATE_KILL:.0%}), thresholds read from "
                      "R19_supply_gates.py",
        "unit_definition": "deduplicated lane evidence chunks; deduplicated claims",
        "source_rows": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "evidence_units": len(ev),
        "claim_units": len(claims),
        "evidence_gate": gate(f"h177_{artifact}_evidence", ev, arena_texts, spike=True),
        "claim_gate": gate(f"h177_{artifact}_claims", claims, arena_texts),
    }
    res["status"] = ("GREEN" if res["evidence_gate"]["pass"] and res["claim_gate"]["pass"]
                     else "RED")
    out = HERE / f"R20-H177_{artifact}_census.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  === {artifact} CENSUS {res['status']} -> {out.name}", flush=True)
    return res["status"]


def census_summary(artifact):
    d = json.loads((HERE / f"R20-H177_{artifact}_census.json").read_text())
    eg, cg = d["evidence_gate"], d["claim_gate"]
    return {
        "artifact": artifact, "status": d["status"],
        "json": f"R20-H177_{artifact}_census.json",
        "instrument": d["instrument"],
        "evidence_units": d["evidence_units"], "claim_units": d["claim_units"],
        "evidence_max_fraction": eg["result"]["max_fraction"],
        "evidence_verdict": eg["result"]["verdict"],
        "evidence_best_jaccard_max":
            eg["result"]["candidate_vs_arena"].get("best_jaccard", {}).get("max"),
        "claim_max_fraction": cg["result"]["max_fraction"],
        "claim_verdict": cg["result"]["verdict"],
        "spike_control": eg["spike_control"],
    }


def merge_into_manifests():
    """Fold the census verdicts back into the lane manifests.

    A rebuilt lane rewrites its manifest without a census block - so the census
    is what re-attaches it, and the manifest stays the single self-contained
    record (rows / pairs / families / leak suite / census / sources)."""
    for lane, ev in (("B", "eval_B"), ("C", "eval_C")):
        p = HERE / f"R20-H177_lane_{lane}_manifest.json"
        cj = HERE / f"R20-H177_lane_{lane}_census.json"
        if not (p.exists() and cj.exists() and (HERE / f"R20-H177_{ev}_census.json").exists()):
            continue
        m = json.loads(p.read_text())
        m["census"] = census_summary(f"lane_{lane}")
        m["held_out_eval"]["census"] = census_summary(ev)
        p.write_text(json.dumps(m, indent=2))
        print(f"  {p.name}: census {m['census']['status']} (lane), "
              f"{m['held_out_eval']['census']['status']} (eval)", flush=True)


def main():
    which = sys.argv[1:] or list(ARTIFACTS)
    arena_texts, _ = G.load_arena()
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    summary = {a: census_one(a, arena_texts) for a in which}
    merge_into_manifests()
    print("\n" + json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if all(v == "GREEN" for v in summary.values()) else 1)


if __name__ == "__main__":
    main()
