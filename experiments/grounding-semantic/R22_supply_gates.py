"""R22 derivation-supply wave - per-corpus contamination gates, CPU only, no GPU.

The wave answers the R22 audit finding (`R22-H186_numeric_lane_supply_audit.json`):
no member of the mix carries a claim stating a value DERIVED from evidence values
with the operands left correct.  Every corpus the fetcher
(`scripts/fetch_grounding_datasets.py`) landed for it is checked here before
banking, on the same three checks the R19 wave used:

  1. LICENCE SIDECAR - the tracked `data/external/datasets/dataset-<name>.md`
     exists and records the licence tag re-read from the source at pull time
  2. DERIVATION RE-VERIFICATION - the admission claim is re-derived FROM THE
     DATA, never trusted from the survey: EQUATE's AWPNLI pairs must share a
     premise and differ only in a numeric token, HiTab's aggregation operators
     and answer formulas must be present on the derived slice, NumGLUE's
     Type_5/6 DROP overlap is measured and reported, SciTab's three-way labels
     and `[BOLD]` markup rate are measured, DROP's answer types are counted
  3. R14-H136 PROVENANCE GATE - the registered instrument (`provenance_gate.py`,
     ruling 2 form: 8-gram, Jaccard >= 0.3, bidirectional) against ALL TEN
     walled arena corpora, KILL above 2% max fraction, spike control first.
     The arena reference n-grams are the ONLY thing read from RAGBench

Gate units are the corpus's EVIDENCE texts (premises, passages, linearized
tables), deduplicated - contamination is a document-overlap property.

Any corpus over bar QUARANTINES: the gate JSON says RED and no lane may draw on
it.  A corpus that cannot be read at all is recorded as a defect and the wave
continues.

Run:  uv run python experiments/grounding-semantic/R22_supply_gates.py [name ...]
"""

import collections
import importlib.util
import io
import json
import pathlib
import re
import sys
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

GATE_N = 8
GATE_JACCARD = 0.3
GATE_KILL = 0.02

# Deduped evidence sides run to hundreds of thousands of units on DROP; the
# Jaccard mode is O(candidate x arena) so the gate samples deterministically
# above this ceiling and records the coverage it achieved.
MAX_GATE_UNITS = 60000
MAX_SPIKE_UNITS = 2000

_spec = importlib.util.spec_from_file_location("provgate", HERE / "provenance_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


def zip_parquets(name):
    """split -> polars frame for every parquet inside dataset-<name>.zip."""
    z = zipfile.ZipFile(DATA / f"dataset-{name}.zip")
    out = {}
    for m in z.namelist():
        if m.endswith(".parquet"):
            split = m[: -len(".parquet")].split("__")[-1]
            out[split] = pl.read_parquet(io.BytesIO(z.read(m)))
    return out


def _dedupe(texts):
    return sorted({t.strip() for t in texts if t and t.strip()})


def _sample(texts, cap, seed=0):
    """Deterministic subsample of the gate units, with the coverage recorded."""
    if len(texts) <= cap:
        return texts, 1.0
    step = len(texts) / cap
    return [texts[int(i * step)] for i in range(cap)], round(cap / len(texts), 6)


# --------------------------------------------------------------------------- #
# per-corpus readers: (gate_texts, derivation_evidence)
# --------------------------------------------------------------------------- #
def read_equate():
    """AWPNLI is the admission claim: the same premise carries one hypothesis
    stating the correct arithmetic result and one stating a wrong one.  That is
    re-derived here - premises shared across both labels, and the two
    hypotheses differing ONLY in numeric tokens."""
    parts = zip_parquets("equate")
    ev = {"splits": {k: v.height for k, v in parts.items()},
          "labels_by_split": {k: dict(collections.Counter(v["gold_label"].to_list()))
                              for k, v in parts.items()}}

    awp = parts["awpnli"]
    by_premise = collections.defaultdict(dict)
    for p, h, lab in awp.select("sentence1", "sentence2", "gold_label").iter_rows():
        by_premise[p][lab] = h
    both = {p: d for p, d in by_premise.items() if len(d) == 2}
    numeric_only = 0
    for d in both.values():
        a, b = d["entailment"], d["contradiction"]
        if _NUM.sub("#", a) == _NUM.sub("#", b) and a != b:
            numeric_only += 1
    ev["awpnli"] = {
        "rows": awp.height,
        "distinct_premises": len(by_premise),
        "premises_carrying_both_legs": len(both),
        "both_legs_differing_only_in_numbers": numeric_only,
        "derivation_rate": round(numeric_only / max(len(both), 1), 6),
        "example": next(
            ({"premise": p, "entailed": d["entailment"], "contradicted": d["contradiction"]}
             for p, d in both.items()), None),
    }
    # the binding re-verification: AWPNLI must actually be a result-only
    # corruption set, not an operand-substitution one
    ev["pass"] = ev["awpnli"]["derivation_rate"] >= 0.90
    texts = _dedupe(t for v in parts.values() for t in v["sentence1"].to_list())
    return texts, ev


def read_numglue():
    """Type_7 is the labeled quantitative-NLI slice; Types 5 and 6 are DROP
    items re-served and are counted so the two members are never mixed."""
    parts = zip_parquets("numglue")
    df = pl.concat(list(parts.values()))
    types = collections.Counter(df["type"].to_list())
    t7 = df.filter(pl.col("type") == "Type_7")
    ev = {"rows": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "type_distribution": dict(types),
          "type_7_quantitative_nli_rows": t7.height,
          "type_7_labels": dict(collections.Counter(t7["answer"].to_list())),
          "drop_derived_rows_type_5_6": types.get("Type_5", 0) + types.get("Type_6", 0),
          "drop_derived_fraction": round(
              (types.get("Type_5", 0) + types.get("Type_6", 0)) / max(df.height, 1), 6),
          "word_problem_rows_types_1_2_4_8": sum(
              types.get(f"Type_{i}", 0) for i in (1, 2, 4, 8)),
          "note": "Type_5/6 are DROP re-served - mixing this member with `drop` "
                  "double-counts them and breaks the split axis of both",
          "pass": t7.height > 0 and len(types) == 8}
    # the evidence side is the passage where the task has one (Types 5/6),
    # otherwise the word problem or the NLI premise carrying the operands
    texts = _dedupe(p or q or s1 for p, q, s1 in zip(
        df["passage"].to_list(), df["question"].to_list(),
        df["statement1"].to_list(), strict=True))
    return texts, ev


def read_hitab():
    """The derived slice is the rows whose aggregation is not ['none']; the
    admission claim is that those rows ship the derivation itself, so the
    answer-formula coverage on that slice is what is measured."""
    parts = zip_parquets("hitab")
    tables = parts.pop("tables")
    df = pl.concat(list(parts.values()))
    aggs = collections.Counter()
    derived = with_formula = 0
    for a, f in zip(df["aggregation"].to_list(), df["answer_formulas"].to_list(),
                    strict=True):
        ops = json.loads(a) or []
        aggs["+".join(ops) or "empty"] += 1
        if ops != ["none"]:
            derived += 1
            if json.loads(f):
                with_formula += 1
    ev = {"rows": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "tables": tables.height,
          "aggregation_distribution": dict(aggs.most_common()),
          "derived_rows": derived,
          "derived_fraction": round(derived / max(df.height, 1), 6),
          "lookup_rows_aggregation_none": aggs.get("none", 0),
          "derived_rows_carrying_answer_formula": with_formula,
          "formula_coverage_on_derived": round(with_formula / max(derived, 1), 6),
          "note": "lookup-dominant - the derived slice is the minority and is the "
                  "only part answering the R22 supply gap",
          "pass": derived > 0 and with_formula / max(derived, 1) >= 0.90}
    texts = []
    for raw in tables["table_json"].to_list():
        t = json.loads(raw)
        cells = [str(c.get("value", "")) for row in t.get("data", []) for c in row]
        texts.append(" ".join([t.get("title", "")] + cells))
    return _dedupe(texts), ev


def read_scitab():
    """Three-way claim verification; the `[BOLD]` markup inherited from the
    SciGen table extraction is counted because a lane must strip it."""
    parts = zip_parquets("scitab")
    df = parts["all"]
    bold = sum(1 for v in df["table_content_values"].to_list() if "[BOLD]" in v)
    ev = {"rows": df.height,
          "label_distribution": dict(collections.Counter(df["label"].to_list())),
          "distinct_tables": df["table_id"].n_unique(),
          "distinct_papers": df["paper_id"].n_unique(),
          "rows_with_BOLD_markup": bold,
          "bold_rate": round(bold / max(df.height, 1), 6),
          "pass": set(df["label"].to_list())
                  <= {"supports", "refutes", "not enough info"}}
    texts = _dedupe(
        f"{cap} {' '.join(str(c) for row in json.loads(vals) for c in row)}"
        for cap, vals in zip(df["table_caption"].to_list(),
                             df["table_content_values"].to_list(), strict=True))
    return texts, ev


def read_drop():
    """Only the number-typed answers carry a derivation; span and date answers
    are lookup and that split is what is measured here."""
    parts = zip_parquets("drop")
    df = pl.concat(list(parts.values()), how="diagonal_relaxed")
    types = collections.Counter()
    for a in df["answers_spans"].to_list():
        for t in (a.get("types") or []):
            types[t] += 1
    ev = {"rows": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "distinct_passages": df["passage"].n_unique(),
          "answer_type_distribution": dict(types),
          "note": "answers ship correct-only; a result-perturbation negative is "
                  "constructed at lane build, not supplied here",
          "pass": df.height > 0 and df["passage"].n_unique() > 0}
    return _dedupe(df["passage"].to_list()), ev


READERS = {
    "equate": read_equate,
    "numglue": read_numglue,
    "hitab": read_hitab,
    "scitab": read_scitab,
    "drop": read_drop,
}

LICENCE_TOKEN = {  # the tag re-read from the source at pull time
    "equate": "MIT",
    "numglue": "ODC-By",
    "hitab": "Computational Use of Data Agreement",
    "scitab": "MIT",
    "drop": "CC-BY-SA-4.0",
}


def check_sidecar(name):
    sc = DATA / f"dataset-{name}.md"
    ev = {"path": str(sc), "present": sc.exists()}
    ok = sc.exists()
    if ok:
        txt = sc.read_text()
        ev["records_licence_token"] = LICENCE_TOKEN[name] in txt
        ev["records_r22_registration"] = "R22" in txt
        ok &= ev["records_licence_token"] and ev["records_r22_registration"]
    return ok, ev


def gate_one(name, arena_texts):
    print(f"\n=== {name}", flush=True)
    res = {"corpus": name,
           "wave": "R22 derivation supply",
           "instrument": "provenance_gate.py (R14-H136 ruling 2 form: 8-gram, "
                         "Jaccard >= 0.3, bidirectional, KILL > 2%)"}
    ok_side, ev_side = check_sidecar(name)
    res["licence_sidecar"] = {"pass": ok_side, **ev_side}

    texts, ev_derive = READERS[name]()
    gate_texts, coverage = _sample(texts, MAX_GATE_UNITS)
    print(f"  evidence units deduped: {len(texts)}; gated: {len(gate_texts)} "
          f"(coverage {coverage})", flush=True)
    res["derivation_reverification"] = {"pass": ev_derive.pop("pass"), **ev_derive}

    spike = G.spike_control(gate_texts[:MAX_SPIKE_UNITS], arena_texts, n=GATE_N,
                            jaccard=GATE_JACCARD, k=10, label=f"{name}_spike")
    print(f"  spike control: {spike}", flush=True)
    gate = G.run_gate(gate_texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label=f"r22_{name}", arena_texts=arena_texts)
    gate["gate_unit_coverage"] = coverage
    gate["gate_units_deduped_total"] = len(texts)
    print(f"  gate verdict {gate['verdict']} at max fraction {gate['max_fraction']} "
          f"(best-Jaccard max "
          f"{gate['candidate_vs_arena'].get('best_jaccard', {}).get('max')})", flush=True)
    res["provenance_gate"] = {"pass": gate["verdict"] != "KILL" and spike["passes"],
                              "spike_control": spike, "result": gate}

    res["status"] = ("GREEN" if all(res[k]["pass"] for k in (
        "licence_sidecar", "derivation_reverification", "provenance_gate")) else "RED")
    out = HERE / f"R22_{name}_gate.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  === {name} GATE {res['status']} -> {out.name}", flush=True)
    return res["status"]


def main():
    names = sys.argv[1:] or list(READERS)
    arena_texts, _ = G.load_arena()  # all ten walled subsets; n-grams only
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    summary = {}
    for name in names:
        try:
            summary[name] = gate_one(name, arena_texts)
        except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
            summary[name] = f"DEFECT: {type(e).__name__}: {str(e)[:200]}"
            print(f"  === {name} GATE DEFECT: {summary[name]}", flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
