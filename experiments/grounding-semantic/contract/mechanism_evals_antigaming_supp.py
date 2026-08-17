"""Supplement to the mechanism-eval contract pass - the ANTI-GAMING instrument.

CPU only.  Measurement only; no verdict is adjudicated here.

The main pass found 14 distinct content fingerprints across the 14 banked
`*_antigaming_set.parquet` files, so the hold every arm was read against is not
one instrument.  This supplement measures two things the fingerprint alone does
not answer:

  1. HOW different - pairwise item overlap between every banked set and the
     flagship's (`R18-H150_antigaming_set.parquet`), split by `kind`, plus the
     per-arm claim and table counts
  2. WHY - whether the builder's own supply is order-stable.  `R15_gate_common.
     held_tabfact()` de-duplicates with `unique(subset=["table_id"],
     keep="first")` and Polars does not preserve row order through `unique`
     unless asked to, so a seeded `rng.permutation` indexes a differently
     ordered list on each run.  The check runs the banked function twice in one
     process and compares membership and order

Out: contract/mechanism_evals_antigaming_supp.json
Run: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
     experiments/grounding-semantic/contract/mechanism_evals_antigaming_supp.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util as _ilu
import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
OUT = HERE / "mechanism_evals_antigaming_supp.json"
REF = "R18-H150_antigaming_set.parquet"  # the flagship draw-1 read
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def items(p, kind=None):
    d = pl.read_parquet(p)
    if kind:
        d = d.filter(pl.col("kind") == kind)
    return {(a, b, c) for a, b, c in zip(d["table_id"].to_list(),
                                         d["claim_pos"].to_list(),
                                         d["claim_neg"].to_list())}


def jac(a, b):
    return round(len(a & b) / max(len(a | b), 1), 4)


def main():
    files = sorted(SEM.glob("*antigaming_set.parquet"))
    ref_all = items(SEM / REF)
    ref_nm = items(SEM / REF, "nearmiss")
    ref_br = items(SEM / REF, "bind_row")

    per = {}
    for p in files:
        a, nm, br = items(p), items(p, "nearmiss"), items(p, "bind_row")
        d = pl.read_parquet(p)
        per[p.name] = {
            "rows": d.height,
            "nearmiss_rows": int((d["kind"] == "nearmiss").sum()),
            "bind_row_rows": int((d["kind"] == "bind_row").sum()),
            "distinct_tables": int(d["table_id"].n_unique()),
            "shared_items_with_flagship": len(a & ref_all),
            "jaccard_with_flagship_all": jac(a, ref_all),
            "jaccard_with_flagship_nearmiss": jac(nm, ref_nm),
            "jaccard_with_flagship_bind_row": jac(br, ref_br),
        }
        print(f"{p.name:<42} rows {d.height} nm {per[p.name]['nearmiss_rows']:>4} "
              f"br {per[p.name]['bind_row_rows']:>4} "
              f"J(all) {per[p.name]['jaccard_with_flagship_all']} "
              f"J(nm) {per[p.name]['jaccard_with_flagship_nearmiss']} "
              f"J(br) {per[p.name]['jaccard_with_flagship_bind_row']}", flush=True)

    # ---- where the drift comes from ------------------------------------- #
    C = _mod("c", SEM / "R15_gate_common.py")
    runs = [C.held_tabfact() for _ in range(3)]
    ids = [r[2] for r in runs]
    same_membership = all(set(x) == set(ids[0]) for x in ids)
    same_order = all(x == ids[0] for x in ids)
    first_div = None
    for i, (a, b) in enumerate(zip(ids[0], ids[1])):
        if a != b:
            first_div = {"position": i, "run1": a, "run2": b}
            break

    det = {
        "function": "R15_gate_common.held_tabfact() - the supply both anti-gaming "
                    "arms and the R15 probe bank draw from",
        "runs": len(ids),
        "rows_each": [len(x) for x in ids],
        "membership_identical_across_runs": same_membership,
        "ORDER_identical_across_runs": same_order,
        "first_divergence": first_div,
        "mechanism": "the function ends in `unique(subset=['table_id'], "
                     "keep='first')`. Polars does not preserve row order through "
                     "`unique` unless `maintain_order=True` is passed, so the "
                     "seeded `rng.permutation(len(tbls))` in "
                     "`R14-H133_antigaming.build_bindrow` indexes a differently "
                     "ordered list on each run and selects different tables. The "
                     "seed is constant; the SUPPLY ORDER is not",
        "reading": "membership identical with order unstable means the sets are "
                   "drawn from the same pool by a stable seed over an unstable "
                   "ordering - the instrument re-rolls itself per run",
    }
    print(f"\nheld_tabfact determinism: membership identical {same_membership}, "
          f"order identical {same_order}", flush=True)

    OUT.write_text(json.dumps({
        "artifact": OUT.name,
        "scope": "supplement to contract/mechanism_evals_report.json - the "
                 "anti-gaming instrument's cross-arm comparability",
        "reference_set": REF,
        "per_file": per,
        "supply_determinism": det,
        "note": NOTE,
    }, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
