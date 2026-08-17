"""Dataset contract C4 - contamination census for the CONFORMED `vitaminc` member.

Identical instrument, thresholds, controls and unit definition to
`vitaminc_census.py`; the only change is the candidate side, which is
`vitaminc_conformed.parquet` (the member after the C2 collision filter and the
C1 structural filter) instead of the raw archive train split.

Instrument REUSED, not rewritten: `provenance_gate.run_gate` in the R14-H136
ruling-2 form (8-gram, Jaccard >= 0.3, bidirectional, WARN 0.5%, KILL 2% of the
candidate side), thresholds read literally from `R19_supply_gates.py`, arena
reference n-grams from `provenance_gate.load_arena()`.

CONTROLS.
  spike (synthetic)  arena units injected into the candidate side must all be
                     detected - guards a gate that cannot fire.
  LIVE POSITIVE      VitaminC's own official TEST split offered to the IDENTICAL
                     instrument against the CONFORMED train side.  Those strings
                     are genuine near-duplicates by construction, so a gate that
                     fires there and reads clean against the arena is a clean
                     read rather than an instrument that failed to fire.

CPU ONLY - CUDA_VISIBLE_DEVICES is forced empty before any import.

Run:  uv run python experiments/grounding-semantic/contract/vitaminc_conformed_census.py \
          2>&1 | tee logs/vitaminc_conformed_census.log
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
MEMBER = HERE / "vitaminc_conformed.parquet"
OUT = HERE / "vitaminc_conformed_census.json"

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


def units(df, ev_col="evidence", cl_col="claim"):
    ev = sorted({t for t in df[ev_col].to_list() if t and t.strip()})
    cl = sorted({t for t in df[cl_col].to_list() if t and t.strip()})
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
    m = pl.read_parquet(MEMBER)
    test, test_name = split_frame("test")
    log(f"conformed member: {m.height} rows")
    ev, cl = units(m, ev_col="chunk", cl_col="claim")
    log(f"gate units: {len(ev)} unique evidence, {len(cl)} unique claims")

    arena_texts, _ = G.load_arena()
    log(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
        f"{len(arena_texts)} subsets")

    res = {
        "member": "vitaminc_conformed",
        "clause": "C4",
        "instrument": (
            f"provenance_gate.py (R14-H136 ruling-2 form: {GATE_N}-gram, "
            f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL > {GATE_KILL:.0%}); "
            "thresholds read from R19_supply_gates.py"),
        "selection_predicate": (
            "vitaminc_conformed.parquet - the archive train split under the loader's "
            "own predicate, then the F1 evaluation-surface collision filter and the "
            "F2 structural label-conflict filter; evidence UNTRUNCATED"),
        "rows": m.height,
        "evidence_units": len(ev),
        "claim_units": len(cl),
        "arena_units": sum(len(v) for v in arena_texts.values()),
    }

    log("=== evidence vs arena")
    res["evidence_gate"] = gate("vitaminc_conformed_evidence", ev, arena_texts, spike=True)
    log("=== claims vs arena")
    res["claim_gate"] = gate("vitaminc_conformed_claims", cl, arena_texts, spike=True)

    ev_t, cl_t = units(test)
    log(f"=== LIVE positive control: {test_name} ({test.height} rows, "
        f"{len(ev_t)} evidence, {len(cl_t)} claims) vs the CONFORMED side")
    res["live_positive_control"] = {
        "why": ("VitaminC's official split is disjoint by unique_id/case_id but NOT "
                "by page, claim or evidence text; its test split is therefore "
                "genuinely near-duplicate to train by construction and must make "
                "this instrument fire"),
        "candidate": test_name,
        "candidate_rows": test.height,
        "claims": gate("vitaminc_test_claims_vs_conformed",
                       cl_t, {"vitaminc_conformed_claims": cl}),
        "evidence": gate("vitaminc_test_evidence_vs_conformed",
                         ev_t, {"vitaminc_conformed_evidence": ev}),
    }
    lpc = res["live_positive_control"]
    res["live_positive_control"]["fires"] = bool(
        lpc["claims"]["result"]["candidate_vs_arena"]["units_with_hit"] > 0
        and lpc["evidence"]["result"]["candidate_vs_arena"]["units_with_hit"] > 0)
    res["live_positive_control"]["how_to_read_the_verdict_string"] = (
        "the control's own gate prints KILL on the evidence leg. That is the "
        "control WORKING: it says the VitaminC test split is near-duplicate to the "
        "VitaminC train side, which is the overlap C3 measures and the reason the "
        "H166-A1 holdout is key-filtered. It is NOT a verdict on the member's arena "
        "census, which is the `evidence_gate` / `claim_gate` pair above")

    res["coverage"] = {
        "evidence_scorable": res["evidence_gate"]["result"]["candidate"]["n_units_scorable"],
        "evidence_units": len(ev),
        "claims_scorable": res["claim_gate"]["result"]["candidate"]["n_units_scorable"],
        "claim_units": len(cl),
        "note": ("units shorter than the 8-gram window carry no n-gram and are "
                 "unscorable by this instrument; they are covered by the C2 "
                 "exact-match channel, which reads zero against every registered "
                 "evaluation surface on all three string forms"),
    }
    res["status"] = ("GREEN" if res["evidence_gate"]["pass"] and res["claim_gate"]["pass"]
                     else "RED")
    res["seconds"] = round(time.time() - T0, 1)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    log(f"census {res['status']} -> {OUT}")


if __name__ == "__main__":
    main()
