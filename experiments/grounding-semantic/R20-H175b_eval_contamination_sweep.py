"""R20 BLAST RADIUS - which held-out evaluation sets carry mix passages? CPU only.

Scoping, not a rebuild.  The R20-H175b eval was found contaminated against the
TRAINING MIX after being verified disjoint only from its own contrast lane.  The
question this answers is how far that pattern reaches: every held-out eval
parquet in the campaign is checked the same way, against the same assembled mix,
in the same six string forms.

`gold_full` is excluded - it comes from a different source and was already
audited by `R20_goldfull_split_audit.py`.

Run:  uv run python experiments/grounding-semantic/R20-H175b_eval_contamination_sweep.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util as _ilu
import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
OUT = HERE / "R20-H175b_eval_contamination_sweep.json"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CL = _mod("cleanbuild", HERE / "R20-H175b_qlane_eval_clean.py")

EVALS = (
    "R20-H175b_qlane_eval.parquet",
    "R20-H175b_qlane_eval_repaired.parquet",
    "R20-H175b_qlane_eval_clean_prefix.parquet",
    "R20-H177_eval_B.parquet",
    "R20-H177_eval_C.parquet",
    "R17-H143_evalset.parquet",
)


def check(path, mix):
    d = pl.read_parquet(path)
    col = next((c for c in ("chunk", "evidence", "context") if c in d.columns), None)
    if col is None:
        return {"parquet": path.name, "error": "no evidence column"}
    ev = sorted({c for c in d[col].to_list() if c and c.strip()})
    cut = mix["chunk_max"]
    hit = set()
    forms = {}
    for name, test in (
        ("raw_in_mix_raw", lambda p: p in mix["raw"]),
        ("raw_in_mix_truncated", lambda p: p in mix["trunc"]),
        ("truncated_in_mix_raw", lambda p: p[:cut] in mix["raw"]),
        ("truncated_in_mix_truncated", lambda p: p[:cut] in mix["trunc"]),
        ("normalised_in_mix_normalised_raw", lambda p: CL.norm(p) in mix["nraw"]),
        ("normalised_in_mix_normalised_truncated",
         lambda p: CL.norm(p) in mix["ntrunc"]),
    ):
        h = {p for p in ev if test(p)}
        forms[name] = len(h)
        hit |= h
    res = {
        "parquet": path.name, "rows": d.height,
        "evidence_column": col,
        "distinct_passages": len(ev),
        "passages_in_the_mix": len(hit),
        "share_in_the_mix": round(len(hit) / len(ev), 4) if ev else 0.0,
        "by_form": forms,
        "sources": (dict(d.group_by("source").len().iter_rows())
                    if "source" in d.columns else None),
        "status": "CONTAMINATED" if hit else "CLEAN",
    }
    print(f"  {path.name}: {len(hit)}/{len(ev)} passages in the mix "
          f"({res['share_in_the_mix']:.1%}) -> {res['status']}", flush=True)
    return res


def main():
    print("=== R20 held-out eval contamination sweep (CPU only)", flush=True)
    mix = CL.assemble_mix()
    out = {}
    for name in EVALS:
        p = HERE / name
        if not p.exists():
            print(f"  {name}: absent - skipped", flush=True)
            continue
        out[name] = check(p, mix)
    summary = {
        "method": "exact string membership of the assembled R20-H175b mix chunk "
                  "set - R10-H108_lane.public_train() with the evidence cut "
                  "lifted, plus all three arm lanes - in raw, truncated and "
                  "whitespace-collapsed case-folded forms",
        "excluded": {"gold_full": "different source; already audited by "
                                  "R20_goldfull_split_audit.py"},
        "mix_distinct_raw_chunks": len(mix["raw"]),
        "results": out,
        "contaminated": sorted(k for k, v in out.items()
                               if v.get("status") == "CONTAMINATED"),
        "clean": sorted(k for k, v in out.items() if v.get("status") == "CLEAN"),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("contaminated", "clean")}, indent=2),
          flush=True)
    print("=== SWEEP DONE ===", flush=True)


if __name__ == "__main__":
    main()
