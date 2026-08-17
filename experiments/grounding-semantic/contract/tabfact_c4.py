"""Contract clause C4 - contamination census for `tabfact`, with a LIVE control.

CPU only. The instrument is the banked R14-H136 form, invoked through
`provenance_gate.py` unchanged: 8-gram, Jaccard >= 0.3, BIDIRECTIONAL, WARN
0.5%, KILL 2%, per-arena-subset attribution.

Three things are produced, because a clean number from an unproven gate is not
evidence:

  1. the census itself, on both member unit types - EVIDENCE (distinct
     serialised tables) and CLAIM (distinct statements)
  2. the SYNTHETIC SPIKE control - arena units injected into the candidate side,
     which must be detected 10/10 with 0 baseline hits
  3. the LIVE POSITIVE control, the `gold_full` audit pattern: TabFact's OWN
     test+validation splits - text that is near-duplicate to the member by
     construction, since one Wikipedia table is serialised under both a `1-` and
     a `2-` csv id - offered to the IDENTICAL gate with the MEMBER as the
     reference side. An 8-gram instrument over pipe-delimited table rows could
     silently fail to fire; this shows whether it does.

Coverage is stated: units shorter than 8 normalised tokens cannot be scored by
the n-gram instrument and are covered by exact matching instead.

Out: tabfact_c4.json
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
MEMBER = HERE / "tabfact_member.parquet"
OUT = HERE / "tabfact_c4.json"

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
    """Units too short for an 8-gram, counted rather than assumed away."""
    short = [t for t in texts if len(G.normalize(t).split()) < N]
    return {"n_units": len(texts), "n_units_too_short_for_8gram": len(short),
            "share_too_short": round(len(short) / max(len(texts), 1), 6)}


def exact_residual(short_texts, arena_texts):
    """Exact matching over the units the n-gram gate cannot score."""
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

    res = {"member": "tabfact",
           "clause": "C4 - contamination census with a live positive control",
           "instrument": f"provenance_gate.py, R14-H136 form: {N}-gram, "
                         f"Jaccard >= {JACCARD}, bidirectional, WARN {WARN}, KILL {KILL}",
           "census": {}, "coverage": {}, "controls": {}}

    for label, texts in (("evidence_serialised_tables", evidence), ("claims_statements", claims)):
        print(f"--- census: {label} ({len(texts)} units) ---", flush=True)
        t1 = time.time()
        r = G.run_gate(texts, n=N, arena_texts=arena_texts, jaccard=JACCARD,
                       warn=WARN, kill=KILL, label=f"tabfact_{label}")
        res["census"][label] = r
        cov = coverage(G, texts)
        short = [t for t in texts if len(G.normalize(t).split()) < N]
        cov["exact_match_residual"] = exact_residual(short, arena_texts)
        res["coverage"][label] = cov
        print(f"  verdict {r['verdict']}  max_fraction {r['max_fraction']}  "
              f"({time.time() - t1:.0f}s)", flush=True)

    # ---- 2. synthetic spike control -------------------------------------- #
    print("--- spike control ---", flush=True)
    res["controls"]["synthetic_spike_on_evidence"] = G.spike_control(
        evidence, arena_texts, n=N, jaccard=JACCARD, k=10, label="tabfact_spike")
    print(json.dumps(res["controls"]["synthetic_spike_on_evidence"]), flush=True)

    # ---- 3. LIVE positive control ---------------------------------------- #
    print("--- live positive control ---", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    held = []
    for split in ("test", "validation"):
        n = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n)))
        held += [build_chunk(c, t) for c, t in
                 zip(d["table_caption"].to_list(), d["table_text"].to_list(), strict=True)]
    held = sorted(set(held))
    # the member is the REFERENCE side, bucketed as one corpus
    live = G.run_gate(held, n=N, arena_texts={"tabfact_member": evidence},
                      jaccard=JACCARD, warn=WARN, kill=KILL,
                      label="tabfact_heldout_splits")
    res["controls"]["live_positive_control"] = {
        "design": "the gold_full audit pattern - TabFact's OWN test+validation "
                  "serialised tables offered to the IDENTICAL gate with the "
                  "`tabfact` member as the reference side. Near-duplicates exist "
                  "by construction (one table, two csv ids), so a gate that cannot "
                  "fire on this register shows up here",
        "candidate_units": len(held),
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
    print(json.dumps(res["controls"]["live_positive_control"], indent=1)[:1500], flush=True)

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
