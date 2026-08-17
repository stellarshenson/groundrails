"""R20 split audit - POSITIVE CONTROL on real overlap, and the VitaminC premise.

The main audit (`R20_goldfull_split_audit.py`) returned zero on every channel.
A clean number is only worth what its controls are worth, so this adds two
that the audit itself cannot supply:

  1. REAL-OVERLAP CONTROL.  The spike control inside `provenance_gate` proves the
     gate fires on units injected from the other side.  It does not prove the
     gate fires on genuine, unmanufactured overlap at mix scale.  VitaminC's own
     TEST split is that case: it is a revision-family corpus whose train split
     supplies 370,653 of the mix's 721,210 rows, so its test claims are
     near-duplicates of training claims by construction.  Run through the SAME
     instrument, the SAME arena (the assembled mix), the SAME thresholds - the
     only change is which candidate is offered.  A large fraction here plus zero
     for gold_full separates "clean" from "instrument cannot fire".

  2. PREMISE REPRODUCTION.  The audit was opened on a reported VitaminC finding:
     official split disjoint by `unique_id` / `case_id` but not by page, claim
     text, evidence text or `wiki_revision_id` - 1,214 / 110 / 221 / 41,488.
     Those counts are recomputed here from the shipped archive so the audit's
     motivating evidence is itself checkable, together with the distinct-value
     counts behind them.

CPU ONLY.  Polars.  Nothing trains; no banked number is recomputed.

Run:  uv run python experiments/grounding-semantic/R20_goldfull_split_audit_control.py \
          2>&1 | tee -a logs/R20_goldfull_split_audit.log
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
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "R20_goldfull_split_audit.json"
VITC = DATA / "dataset-vitaminc.zip"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading banked modules (CPU, CUDA_VISIBLE_DEVICES='')")
AUD = _mod("audit", "R20_goldfull_split_audit.py")
G = AUD.G


def vitaminc_split(suffix):
    z = zipfile.ZipFile(VITC)
    return pl.read_parquet(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(suffix)))))


def premise():
    """Recompute the reported VitaminC official-split collisions."""
    tr = vitaminc_split("__train.parquet")
    out = {"train_rows": tr.height, "splits": {}}
    for split in ("test", "validation"):
        d = vitaminc_split(f"__{split}.parquet")
        blk = {"rows": d.height}
        for col in ("claim", "evidence", "page", "wiki_revision_id", "unique_id", "case_id"):
            tv = set(tr[col].cast(pl.Utf8).to_list())
            sv = d[col].cast(pl.Utf8)
            blk[col] = {
                "distinct_values_shared_with_train": len(set(sv.to_list()) & tv),
                "rows_colliding_with_train": int(sv.is_in(list(tv)).sum()),
            }
        out["splits"][split] = blk
        log(f"vitaminc {split}: " + ", ".join(
            f"{c} {blk[c]['rows_colliding_with_train']} rows / "
            f"{blk[c]['distinct_values_shared_with_train']} values"
            for c in ("claim", "evidence", "page", "wiki_revision_id", "unique_id")))

    combined = {c: sum(out["splits"][s][c]["rows_colliding_with_train"] for s in ("test", "validation"))
                for c in ("claim", "evidence", "page", "wiki_revision_id", "unique_id", "case_id")}
    out["test_plus_validation_rows_colliding_with_train"] = combined

    # what actually drives the revision-id figure
    tv = set(tr["wiki_revision_id"].cast(pl.Utf8).to_list())
    shared = sorted({v for s in ("test", "validation")
                     for v in set(vitaminc_split(f"__{s}.parquet")["wiki_revision_id"].cast(pl.Utf8).to_list())} & tv)
    empties = sum(
        int((vitaminc_split(f"__{s}.parquet")["wiki_revision_id"].cast(pl.Utf8) == "").sum())
        for s in ("test", "validation"))
    out["wiki_revision_id_detail"] = {
        "distinct_shared_values": shared,
        "rows_with_empty_revision_id_in_test_plus_validation": empties,
        "note": ("the combined revision-id figure is carried by the EMPTY-STRING sentinel; "
                 "excluding it leaves only the genuinely shared revision ids listed above"),
    }
    log(f"vitaminc combined rows-colliding: {combined}")
    log(f"revision-id shared values {shared}, empty-sentinel rows {empties}")
    return out


def real_overlap_control(arena_texts, n_units):
    """VitaminC TEST claims through the same gate against the same mix."""
    d = vitaminc_split("__test.parquet")
    cand = sorted({c for c in d["claim"].to_list() if c and c.strip()})
    log(f"real-overlap control: {len(cand)} VitaminC test claims vs {n_units} mix units")
    t = time.time()
    res = G.run_gate(cand, n=AUD.GATE_N, jaccard=AUD.GATE_JACCARD, kill=AUD.GATE_KILL,
                     label="vitaminc_test_claims", arena_texts=arena_texts)
    log(f"real-overlap control: verdict {res['verdict']}  max_fraction {res['max_fraction']}  "
        f"best_jaccard {res['candidate_vs_arena'].get('best_jaccard')}  "
        f"({time.time() - t:.1f}s)")
    return {"candidate": "VitaminC official TEST split, deduplicated claims",
            "candidate_units": len(cand), "result": res, "seconds": round(time.time() - t, 1)}


def main():
    res = json.loads(OUT.read_text())

    mix, n_mix = AUD.assemble_mix()
    arena_claims = {t[0]: sorted({c for c in g["claim"].to_list() if c and c.strip()})
                    for t, g in mix.group_by("tag")}
    n_units = sum(len(v) for v in arena_claims.values())

    ctrl = {
        "why": ("the audit returned zero on every channel; these controls establish that the "
                "instrument fires on genuine overlap at mix scale, and that the finding which "
                "motivated the audit reproduces"),
        "real_overlap_control": real_overlap_control(arena_claims, n_units),
        "vitaminc_official_split_premise": premise(),
    }
    ctrl["real_overlap_control"]["reference_gold_full_claims_max_fraction"] = \
        res["near_duplicate"]["claims"]["result"]["max_fraction"]
    res["positive_controls"] = ctrl
    res["summary"]["real_overlap_control_max_fraction"] = \
        ctrl["real_overlap_control"]["result"]["max_fraction"]
    res["summary"]["real_overlap_control_verdict"] = \
        ctrl["real_overlap_control"]["result"]["verdict"]
    OUT.write_text(json.dumps(res, indent=2))
    log(f"mix re-assembled at {n_mix} rows; controls banked -> {OUT}")
    log("=== CONTROL DONE ===")


if __name__ == "__main__":
    main()
