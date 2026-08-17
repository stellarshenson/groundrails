"""Dataset contract C4 - contamination census for the `vitaminc` mix member.

Instrument REUSED, not rewritten: `provenance_gate.run_gate` in the R14-H136
ruling-2 form (8-gram, Jaccard >= 0.3, bidirectional, WARN 0.5%, KILL 2% of the
candidate side), thresholds read literally from `R19_supply_gates.py`, arena
reference n-grams from `provenance_gate.load_arena()` (document text only - no
arena item is inspected, printed or stored).

UNITS.  The member is `tals__vitaminc__train.parquet` read with the loader's own
selection predicate (`R10-H108_lane.public_train`, lines 150-165: the single
`endswith("__train.parquet")` member, ALL rows, no filter).  Contamination is a
document-overlap property, so the gate runs on the deduplicated EVIDENCE strings
and, separately, on the deduplicated CLAIM strings.  Evidence is taken
UNTRUNCATED - the R18-H150 / R20-H174 serving protocol reads it untruncated and
windows it, and untruncated text is the stricter census unit.

CONTROLS.
  spike (synthetic)  arena units injected into the candidate side must all be
                     detected - guards a gate that cannot fire.
  LIVE POSITIVE      VitaminC's own official TEST split offered to the IDENTICAL
                     instrument against the VitaminC TRAIN side.  Those strings
                     are genuine near-duplicates by construction (the official
                     split is not text-disjoint), so a gate that fires there and
                     reads zero against the arena is a clean read rather than an
                     instrument that failed to fire.  This is the control
                     pattern banked by `R20_goldfull_split_audit_control.py`.

CPU ONLY - CUDA_VISIBLE_DEVICES is forced empty before any import.

Run:  uv run python experiments/grounding-semantic/contract/vitaminc_census.py \
          2>&1 | tee logs/vitaminc_contract_census.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util
import io
import json
import pathlib
import time
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "vitaminc_census.json"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _mod("provgate", "provenance_gate.py")

_gates_src = (SEM / "R19_supply_gates.py").read_text()
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])
SPIKE_SAMPLE = 2000


def split_frame(split):
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    name = next(n for n in z.namelist() if n.endswith(f"__{split}.parquet"))
    return pl.read_parquet(io.BytesIO(z.read(name))), name


def units(df):
    ev = sorted({t for t in df["evidence"].to_list() if t and t.strip()})
    cl = sorted({t for t in df["claim"].to_list() if t and t.strip()})
    return ev, cl


def gate(name, texts, ref_texts, spike=False):
    t0 = time.time()
    res = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                     label=name, arena_texts=ref_texts)
    out = {"pass": res["verdict"] != "KILL", "result": res,
           "seconds": round(time.time() - t0, 1)}
    if spike:
        sp = G.spike_control(texts[:SPIKE_SAMPLE], ref_texts, n=GATE_N,
                             jaccard=GATE_JACCARD, k=10, label=f"{name}_spike")
        out["spike_control"] = sp
        out["pass"] = out["pass"] and sp["passes"]
        log(f"  spike control: {sp}")
    log(f"  {name}: verdict {res['verdict']} max fraction {res['max_fraction']} "
        f"(best-Jaccard max "
        f"{res['candidate_vs_arena'].get('best_jaccard', {}).get('max')}) "
        f"in {out['seconds']}s")
    return out


def main():
    train, train_name = split_frame("train")
    test, test_name = split_frame("test")
    log(f"member {train_name}: {train.height} rows")
    ev, cl = units(train)
    log(f"gate units: {len(ev)} unique evidence, {len(cl)} unique claims")

    arena_texts, _ = G.load_arena()
    log(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
        f"{len(arena_texts)} subsets")

    res = {
        "member": "vitaminc",
        "clause": "C4",
        "instrument": (
            f"provenance_gate.py (R14-H136 ruling-2 form: {GATE_N}-gram, "
            f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL > {GATE_KILL:.0%}); "
            "thresholds read from R19_supply_gates.py"),
        "selection_predicate": (
            "dataset-vitaminc.zip :: tals__vitaminc__train.parquet, ALL rows "
            "(R10-H108_lane.public_train lines 150-165), evidence UNTRUNCATED"),
        "rows": train.height,
        "evidence_units": len(ev),
        "claim_units": len(cl),
        "arena_units": sum(len(v) for v in arena_texts.values()),
    }

    log("=== evidence vs arena")
    res["evidence_gate"] = gate("vitaminc_train_evidence", ev, arena_texts, spike=True)
    log("=== claims vs arena")
    res["claim_gate"] = gate("vitaminc_train_claims", cl, arena_texts, spike=True)

    # LIVE POSITIVE CONTROL - the official TEST split against the TRAIN side.
    ev_t, cl_t = units(test)
    log(f"=== LIVE positive control: {test_name} ({test.height} rows, "
        f"{len(ev_t)} evidence, {len(cl_t)} claims) vs the TRAIN side")
    res["live_positive_control"] = {
        "why": ("VitaminC's official split is disjoint by unique_id/case_id but "
                "NOT by page, claim or evidence text; its test split is therefore "
                "genuinely near-duplicate to train by construction and must make "
                "this instrument fire"),
        "candidate": test_name,
        "candidate_rows": test.height,
        "claims": gate("vitaminc_test_claims_vs_train",
                       cl_t, {"vitaminc_train_claims": cl}),
        "evidence": gate("vitaminc_test_evidence_vs_train",
                         ev_t, {"vitaminc_train_evidence": ev}),
    }

    res["coverage"] = {
        "evidence_scorable": res["evidence_gate"]["result"]["candidate"]["n_units_scorable"],
        "evidence_units": len(ev),
        "claims_scorable": res["claim_gate"]["result"]["candidate"]["n_units_scorable"],
        "claim_units": len(cl),
        "note": ("units shorter than the 8-gram window carry no n-gram and are "
                 "unscorable by this instrument; they are covered by the C2 "
                 "exact-match channel"),
    }
    res["status"] = ("GREEN" if res["evidence_gate"]["pass"] and res["claim_gate"]["pass"]
                     else "RED")
    res["seconds"] = round(time.time() - T0, 1)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    log(f"census {res['status']} -> {OUT}")


if __name__ == "__main__":
    main()
