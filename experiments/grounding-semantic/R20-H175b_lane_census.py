"""R20-H175b CONTAMINATION CENSUS - the R14-H136 wall on the question lane, CPU.

Registration clause (docs/experiments/semantic-grounding-experiments.md, block
"R20-H175b QUESTION CONDITIONING (measurement only)", STAGE 0): "R14-H136 8-gram
census against all ten walled corpora".  The training lane AND the held-out eval
are both censused - an eval that leaks the arena would invalidate the PRIMARY
mechanism gate it exists to serve, so it carries the same wall as the lane.

INSTRUMENT - reused, not reinvented.  `provenance_gate.py` in the R14-H136
ruling-2 form that `R19_supply_gates.py`, `R20-H174_lane_census.py` and
`R20-H177_lane_census.py` run: 8-gram, Jaccard >= 0.3, bidirectional, KILL at 2%
of the candidate side, against ALL TEN walled arena corpora, with the spike
control.  Thresholds are read from `R19_supply_gates.py`, never restated here.

UNITS - three, not two.  Contamination is a document-overlap property, so the
lane is gated on its deduplicated EVIDENCE passages and on its deduplicated
CLAIMS as the banked lanes are; this lane also gates its deduplicated QUESTIONS,
because the question is new text entering the model that no earlier census has
seen.

PsiloQA is a Wikipedia corpus and three walled arena subsets (hagrid, hotpotqa,
covidqa) are Wikipedia-adjacent, so this census is the load-bearing wall for the
lane rather than a formality - and it is PsiloQA's first 8-gram census at all,
the corpus having entered the mix at R8-H84, before the R14-H136 instrument
existed.

The arena side is read through `provenance_gate.load_arena()` only, which takes
document n-grams and nothing else.  No arena item text is inspected, printed or
stored by this script.

Run:  uv run python experiments/grounding-semantic/R20-H175b_lane_census.py
      [qlane qlane_eval qlane_repaired qlane_eval_repaired]
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
ARTIFACTS = ("qlane", "qlane_eval", "qlane_repaired", "qlane_eval_repaired")


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
    src = HERE / f"R20-H175b_{artifact}.parquet"
    if not src.exists():
        print(f"\n=== census {artifact}: {src.name} absent - skipped", flush=True)
        return None
    print(f"\n=== census {artifact} ({src.name})", flush=True)
    df = pl.read_parquet(src)
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    claims = sorted({c for c in df["claim"].to_list() if c.strip()})
    questions = sorted({q for q in df["question"].to_list() if q.strip()})
    print(f"  gate units: {len(ev)} evidence, {len(claims)} claims, "
          f"{len(questions)} questions (rows {df.height})", flush=True)

    res = {
        "artifact": artifact,
        "parquet": src.name,
        "rows": df.height,
        "pairs": int(df["pair_id"].n_unique()),
        "instrument": "provenance_gate.py (R14-H136 ruling 2 form: "
                      f"{GATE_N}-gram, Jaccard >= {GATE_JACCARD}, bidirectional, "
                      f"KILL > {GATE_KILL:.0%}), thresholds read from "
                      "R19_supply_gates.py",
        "unit_definition": "deduplicated lane evidence passages; deduplicated "
                           "claims; deduplicated questions",
        "source_rows": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "evidence_units": len(ev),
        "claim_units": len(claims),
        "question_units": len(questions),
        "evidence_gate": gate(f"h175b_{artifact}_evidence", ev, arena_texts, spike=True),
        "claim_gate": gate(f"h175b_{artifact}_claims", claims, arena_texts),
        "question_gate": gate(f"h175b_{artifact}_questions", questions, arena_texts),
    }
    res["status"] = ("GREEN" if all(res[k]["pass"] for k in
                                    ("evidence_gate", "claim_gate", "question_gate"))
                     else "RED")
    out = HERE / f"R20-H175b_{artifact}_census.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  === {artifact} CENSUS {res['status']} -> {out.name}", flush=True)
    return res["status"]


def census_summary(artifact):
    p = HERE / f"R20-H175b_{artifact}_census.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    eg, cg, qg = d["evidence_gate"], d["claim_gate"], d["question_gate"]
    return {
        "artifact": artifact, "status": d["status"],
        "json": p.name, "instrument": d["instrument"],
        "evidence_units": d["evidence_units"], "claim_units": d["claim_units"],
        "question_units": d["question_units"],
        "evidence_max_fraction": eg["result"]["max_fraction"],
        "evidence_verdict": eg["result"]["verdict"],
        "evidence_best_jaccard_max":
            eg["result"]["candidate_vs_arena"].get("best_jaccard", {}).get("max"),
        "claim_max_fraction": cg["result"]["max_fraction"],
        "claim_verdict": cg["result"]["verdict"],
        "question_max_fraction": qg["result"]["max_fraction"],
        "question_verdict": qg["result"]["verdict"],
        "spike_control": eg["spike_control"],
    }


def merge_into_manifests():
    """Fold the census verdicts back into the lane manifests, so each manifest
    stays the single self-contained record of its lane."""
    for suffix, lane, ev in (("", "qlane", "qlane_eval"),
                             ("_repaired", "qlane_repaired", "qlane_eval_repaired")):
        p = HERE / f"R20-H175b_qlane{suffix}_manifest.json"
        ls, es = census_summary(lane), census_summary(ev)
        if not (p.exists() and ls and es):
            continue
        m = json.loads(p.read_text())
        m["census"] = ls
        m["held_out_eval"]["census"] = es
        p.write_text(json.dumps(m, indent=2))
        print(f"  {p.name}: census {ls['status']} (lane), {es['status']} (eval)",
              flush=True)


def main():
    which = sys.argv[1:] or list(ARTIFACTS)
    arena_texts, _ = G.load_arena()
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    summary = {a: census_one(a, arena_texts) for a in which}
    merge_into_manifests()
    print("\n" + json.dumps(summary, indent=2), flush=True)
    done = [v for v in summary.values() if v is not None]
    raise SystemExit(0 if done and all(v == "GREEN" for v in done) else 1)


if __name__ == "__main__":
    main()
