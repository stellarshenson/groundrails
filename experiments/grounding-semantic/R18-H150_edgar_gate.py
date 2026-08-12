"""R18-H150 amendment A2 - EDGAR provenance + contamination gate, CPU only.

Stage 2, and a hard precondition of the scale_word lane build: no row enters the
lane without the green sidecar this script writes.

Four checks, all of them blocking:

  1. LICENCE / PROVENANCE SIDECAR - the tracked `dataset-edgar-restricted.md`
     sidecar exists and records the Apache-2.0 upstream tag, the R14-H136
     restriction clauses are recorded, and the fetch state (`_state.json`,
     `_counts.json`) reports a complete, restriction-filtered fetch.  Provenance
     is recorded per document by CIK, filing year and filename
  2. RESTRICTION RE-VERIFICATION - the restriction is re-derived from the data
     rather than trusted: every filing year >= 2020, and no CIK on the
     1999-2019 S&P 500 constituent list (the clause that makes the slice
     document-disjoint from FinQA's source population)
  3. R14-H136 PROVENANCE GATE - the registered instrument
     (`provenance_gate.py`, ruling 2 form: 8-gram, Jaccard >= 0.3, bidirectional)
     against the finqa and tatqa arena documents, KILL at > 2%.  A spike control
     runs first: arena units are injected into the candidate side and must be
     detected, so a gate that cannot fire cannot pass
  4. CONTAMINATION WALL - EDGAR is not a RAGBench source corpus, asserted and
     then MEASURED anyway: the ruling-4 default gate (13-gram containment) over
     ALL ten arena subsets, plus exact-chunk and content-fingerprint overlap
     against the banked R17-H143 eval set and the H144/H145/H146 lanes

ADMISSION.  Checks 3 and 4 are evaluated on the ADMITTED set, not the raw one,
and the two filters that produce it are recorded rather than argued away:

  * FILER CLAUSE - the fetch-time S&P 500 exclusion left 431 tickers unresolved
    and two constituents survived into the slice (CIK 14707 Caleres, CIK 78890
    Brink's).  Both are re-derived here from the constituent CSV and their
    filings are dropped whole
  * BOILERPLATE COLLISION - on the RAW set the registered instrument reads 0.0
    (8-gram Jaccard, max observed similarity 0.157 against a 0.3 bar) while the
    13-gram containment wall reads 4.04%, ALL of it on the two financial arena
    subsets (tatqa 3.63%, finqa 0.58%) and exactly 0.0 on the other eight.  That
    is shared accounting boilerplate, not document reuse - the Jaccard reading
    would be high if documents were shared.  The 4% is dropped anyway rather
    than defended: every chunk carrying a single 13-gram arena hit is excluded,
    so the wall reads 0 on the admitted set by construction

Run:  uv run python experiments/grounding-semantic/R18-H150_edgar_gate.py
"""

import csv
import hashlib
import importlib.util
import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
EDGAR = DATA / "edgar-restricted"
SIDECAR = DATA / "dataset-edgar-restricted.md"
CHUNKS = HERE / "R18-H150_edgar_chunks.parquet"
EVALSET = HERE / "R17-H143_evalset.parquet"
ADMITTED = HERE / "R18-H150_edgar_admitted.parquet"
OUT = HERE / "R18-H150_edgar_gate.json"

GATE_N = 8
GATE_JACCARD = 0.3
GATE_KILL = 0.02
GATE_SUBSETS = ["finqa", "tatqa"]
WALL_N = 13

_spec = importlib.util.spec_from_file_location("provgate", HERE / "provenance_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def fingerprint(text):
    return hashlib.blake2b(G.normalize(text).encode(), digest_size=16).hexdigest()


def check_sidecar():
    ok, ev = True, {}
    ev["sidecar_path"] = str(SIDECAR)
    ev["sidecar_present"] = SIDECAR.exists()
    ok &= ev["sidecar_present"]
    if ev["sidecar_present"]:
        txt = SIDECAR.read_text()
        ev["licence_line"] = next((ln.strip() for ln in txt.splitlines()
                                   if ln.lower().lstrip("- *").startswith("**licence**")
                                   or "licence" in ln.lower()[:20]), None)
        ev["records_apache_2_0"] = "Apache-2.0" in txt
        ev["records_restriction_clauses"] = ("non-S&P-500" in txt and "2020" in txt)
        ev["records_provenance_gate_caveat"] = "provenance_gate.py" in txt
        ok &= ev["records_apache_2_0"] and ev["records_restriction_clauses"]
    for name in ("_state.json", "_counts.json"):
        p = EDGAR / name
        ev[name] = json.loads(p.read_text()) if p.exists() else None
        ok &= p.exists()
    ev["fetch_complete"] = bool((ev.get("_state.json") or {}).get("complete"))
    ok &= ev["fetch_complete"]
    return ok, ev


def sp500_ciks():
    """CIKs of S&P 500 constituents over 1999-2019, re-derived from the CSV.

    A ticker carrying a date suffix (`CAL-200612`) names a removal, so only the
    BARE tickers on a row were constituents at that row's date; scanning every
    row therefore recovers the union over the window."""
    sp = EDGAR / "sp500_historical_components.csv"
    tick = EDGAR / "sec_company_tickers.json"
    if not (tick.exists() and sp.exists()):
        return set()
    cik_by_ticker = {}
    for rec in json.loads(tick.read_text()).values():
        cik_by_ticker[str(rec["ticker"]).upper()] = int(rec["cik_str"])
    tickers = set()
    with sp.open() as fh:
        for row in csv.reader(fh):
            for cell in row:
                for t in str(cell).replace(";", ",").split(","):
                    t = t.strip().strip("'\" []").upper()
                    if t and t.replace(".", "").replace("-", "").isalpha() and len(t) <= 5:
                        tickers.add(t)
    return {cik_by_ticker[t] for t in tickers if t in cik_by_ticker}


def check_restriction(chunks):
    """Re-derive the R14-H136 restriction from the data, do not trust it."""
    excluded = sp500_ciks()
    years = chunks["year"].unique().to_list()
    ciks = set(int(c) for c in chunks["cik"].unique().to_list())
    breach = sorted(ciks & excluded)
    ev = {"years_present": sorted(years),
          "year_clause_ok": all(int(y) >= 2020 for y in years),
          "sp500_ciks_resolved": len(excluded),
          "filer_clause_breaches": len(breach),
          "breach_examples": breach[:5],
          "documents": chunks["doc_id"].n_unique(),
          "provenance_recorded_per_document": "cik + year + filename in doc_id"}
    return ev["year_clause_ok"] and ev["filer_clause_breaches"] == 0, ev


def main():
    chunks = pl.read_parquet(CHUNKS)
    texts = chunks["chunk"].to_list()
    print(f"candidate units: {len(texts)} chunks / {chunks['doc_id'].n_unique()} filings",
          flush=True)

    res = {"candidate": {"path": str(CHUNKS), "units": len(texts),
                         "documents": chunks["doc_id"].n_unique()}}

    print("check 1: licence / provenance sidecar", flush=True)
    ok_sidecar, ev_sidecar = check_sidecar()
    res["licence_sidecar"] = {"pass": ok_sidecar, **ev_sidecar}

    print("check 2: restriction re-verification (raw slice)", flush=True)
    ok_restrict, ev_restrict = check_restriction(chunks)
    res["restriction_reverification_raw"] = {"pass": ok_restrict, **ev_restrict}

    # ---- admission ---------------------------------------------------------
    print("admission: filer clause + per-chunk boilerplate exclusion", flush=True)
    excluded_ciks = sp500_ciks()
    keep_filer = ~chunks["cik"].cast(pl.Int64).is_in(sorted(excluded_ciks))
    dropped_filer = int((~keep_filer).sum())
    adm = chunks.filter(keep_filer)

    hasher = G._TokenHasher()
    all_arena, _ = G.load_arena()
    wall_side = G._Side("arena")
    for sub, cs in all_arena.items():
        for c in cs:
            wall_side.add(sub, G.ngram_hashes(c, WALL_N, hasher))
    wall_idx = wall_side.index()
    hit = []
    for t in adm["chunk"].to_list():
        q = G.ngram_hashes(t, WALL_N, hasher)
        hit.append(any(G._hit_mask(q, h) for h, _, _ in wall_idx.values()))
    dropped_boiler = int(sum(hit))
    adm = adm.filter(~pl.Series("wall_hit", hit))
    adm.write_parquet(ADMITTED)
    print(f"  dropped {dropped_filer} chunks on the filer clause, "
          f"{dropped_boiler} on 13-gram arena collision -> {adm.height} admitted",
          flush=True)
    res["admission"] = {
        "raw_chunks": chunks.height, "raw_documents": chunks["doc_id"].n_unique(),
        "dropped_filer_clause_chunks": dropped_filer,
        "dropped_filer_clause_ciks": sorted(
            set(int(c) for c in chunks["cik"].unique().to_list()) & excluded_ciks),
        "dropped_boilerplate_collision_chunks": dropped_boiler,
        "admitted_chunks": adm.height, "admitted_documents": adm["doc_id"].n_unique(),
        "admitted_path": str(ADMITTED)}
    texts = adm["chunk"].to_list()

    ok_restrict2, ev_restrict2 = check_restriction(adm)
    res["restriction_reverification"] = {"pass": ok_restrict2, **ev_restrict2}

    print(f"check 3: R14-H136 provenance gate ({GATE_N}-gram, Jaccard {GATE_JACCARD}, "
          f"subsets {GATE_SUBSETS}, KILL > {GATE_KILL})", flush=True)
    arena_texts, _ = G.load_arena(GATE_SUBSETS)
    spike = G.spike_control(texts[:2000], arena_texts, n=GATE_N, jaccard=GATE_JACCARD,
                            k=10, label="edgar_spike")
    print(f"  spike control: {spike}", flush=True)
    gate = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label="edgar_mda", arena_texts=arena_texts)
    print(f"  verdict {gate['verdict']} at max fraction {gate['max_fraction']}", flush=True)
    res["provenance_gate"] = {"pass": gate["verdict"] != "KILL" and spike["passes"],
                              "spike_control": spike, "result": gate}

    print(f"check 4: contamination wall ({WALL_N}-gram containment, all arena subsets)",
          flush=True)
    wall = G.run_gate(texts, n=WALL_N, kill=GATE_KILL, label="edgar_mda_wall",
                      arena_texts=all_arena)
    print(f"  verdict {wall['verdict']} at max fraction {wall['max_fraction']}", flush=True)

    ev = pl.read_parquet(EVALSET, columns=["chunk"])
    ev_chunks = set(ev["chunk"].to_list())
    ev_prints = {fingerprint(c) for c in ev_chunks}
    cand_prints = {fingerprint(c) for c in texts}
    lane_overlap = {}
    for name in ("R17-H144_pairs", "R17-H145_scaleunit", "R17-H146_lane",
                 "R18-H150_scaleunit_lane"):
        p = HERE / f"{name}.parquet"
        if not p.exists():
            continue
        other = set(pl.read_parquet(p, columns=["chunk"])["chunk"].to_list())
        lane_overlap[name] = len(other & set(texts))
    res["contamination_wall"] = {
        "edgar_is_a_ragbench_source_corpus": False,
        "assertion": "EDGAR-CORPUS is not one of the ten RAGBench source corpora; "
                     "the overlap below is measured anyway, not assumed",
        "ngram_containment_all_subsets": wall,
        "evalset_shared_chunks": len(ev_chunks & set(texts)),
        "evalset_shared_content_fingerprints": len(ev_prints & cand_prints),
        "lane_shared_chunks": lane_overlap,
        "pass": (wall["verdict"] != "KILL" and not (ev_chunks & set(texts))
                 and not (ev_prints & cand_prints) and not any(lane_overlap.values()))}

    res["status"] = ("GREEN" if all(res[k]["pass"] for k in
                                    ("licence_sidecar", "restriction_reverification",
                                     "provenance_gate", "contamination_wall"))
                     else "RED")
    res["binding_instrument"] = (
        "R14-H136 ruling 2: 8-gram Jaccard >= 0.3 vs finqa+tatqa, KILL > 2%. The "
        "13-gram containment wall is carried as a second, stricter screen and its "
        "hits are EXCLUDED at admission rather than adjudicated.")
    OUT.write_text(json.dumps(res, indent=2))
    summary = {k: ({a: b for a, b in v.items() if a != "result"}
                   if isinstance(v, dict) else v)
               for k, v in res.items() if k != "candidate"}
    print(json.dumps(summary, indent=2)[:6000], flush=True)
    print(f"=== EDGAR GATE {res['status']} ===", flush=True)
    raise SystemExit(0 if res["status"] == "GREEN" else 1)


if __name__ == "__main__":
    main()
