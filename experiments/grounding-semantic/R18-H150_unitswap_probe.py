"""R18-H150 unit_swap PROBE - held-out measurement set, CPU only, never trains.

A document-disjoint companion to `R18-H150_scaleunit_lane.parquet` for the H150
arm's REPORTED-SECONDARY read.  The lane drew 2,770 pairs from 410 of the 1,048
unit-bearing tables; this probe is built from the UNUSED supply and from nothing
else, so a checkpoint trained on the lane meets no probe document twice.

Disjointness, all enforced at build time:
  * every lane document is excluded by doc_id - the probe and the lane share no
    table, therefore no chunk and no row key
  * the R17-H143 eval set is excluded on CONTENT fingerprints, the same
    machinery the lane used (R17-H144 method), not merely by id
  * overlap with the H144 / H145 / H146 corpora is measured and recorded

Construction and verification are IMPORTED from the lane builder rather than
restated - same templates, same unit vocabulary, same surface-disjointness rule,
same value-surface bucket matching, same hub families, same six bars plus the
H148 literal-presence audit and the re-derivation audit.  A probe built by
different code would measure a different thing.

Seed 11500 (the lane's 1150 with a probe suffix), so the draw is independent of
the lane's.

Run:  uv run python experiments/grounding-semantic/R18-H150_unitswap_probe.py
"""

import collections
import importlib.util
import json
import pathlib
import random

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
LANE = HERE / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "R18-H150_unitswap_probe.parquet"
MANIFEST = HERE / "R18-H150_unitswap_probe_manifest.json"
VERIFY = HERE / "R18-H150_unitswap_probe_verify.json"

SEED = 11500
TARGET_PAIRS = 500
AUDIT_N = 300

_spec = importlib.util.spec_from_file_location("h150lane", HERE / "R18-H150_scaleunit_lane.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
P = L.P

# The probe is one sixth of the lane's size, so the lane's 100-pair family floor
# would admit at most five families.  The floor is scaled with the target and the
# per-family standard error is reported beside every read, so an underpowered
# family is visible rather than silently decisive.
L.MIN_FAMILY_PAIRS = 32
L.AUDIT_N = AUDIT_N


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    lane = pl.read_parquet(LANE, columns=["doc_id", "chunk"])
    lane_docs = set(lane["doc_id"].to_list())
    lane_chunks = set(lane["chunk"].to_list())
    print(f"lane: {len(lane_docs)} documents to exclude", flush=True)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    raw = P.tabfact_tables() + P.feverous_tables()
    drop_idx = P.excluded_tables(raw, prints)
    tables, dropped_eval, dropped_lane = [], 0, 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped_eval += 1
            continue
        if t["doc_id"] in lane_docs:
            dropped_lane += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval content; "
          f"{dropped_eval} dropped on eval, {dropped_lane} dropped as lane documents; "
          f"{len(tables)} admitted", flush=True)

    forms = list(L.FORM_WEIGHTS)
    w = np.array([L.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]
    by_doc = {t["doc_id"]: t for t in tables}

    # ---- enumerate positives (lane machinery, lane rules) -------------------
    census = collections.Counter()
    positives, seen_key, per_doc = [], set(), collections.Counter()
    order = list(range(len(tables)))
    rng.shuffle(order)
    for cap in L.DOC_CAP_LADDER:
        for ti in order:
            t = tables[ti]
            if per_doc[t["doc_id"]] >= cap:
                continue
            if "cands" not in t:
                t["cands"], t["attested"] = L.table_candidates(t)
                if t["cands"]:
                    census["unit_bearing_tables"] += 1
                rng.shuffle(t["cands"])
            for c in t["cands"]:
                if per_doc[t["doc_id"]] >= cap:
                    break
                key = (t["doc_id"], c["ci"], c["ri"])
                if key in seen_key:
                    continue
                row_key = t["body"][c["ri"]][t["lab_ci"]].strip()
                if not row_key or len(row_key) > 60 or P.as_num(row_key) is not None:
                    continue
                chunk = L.make_chunk(t, c["ci"], c["ri"], c["cell"], rng)
                if chunk is None or row_key not in chunk:
                    census["dropped_no_chunk"] += 1
                    continue
                if chunk in lane_chunks:
                    census["dropped_lane_chunk"] += 1
                    continue
                low = chunk.lower()
                toks = set(L.WORD.findall(low))
                if not L.phrase_absent(c["unit"], low):
                    census["dropped_claim_phrase_present"] += 1
                    continue
                carrier_text = (P.clean(t["hdr"][c["ci"]]) if c["carrier"] == "header"
                                else c["cell"])
                if carrier_text not in chunk:
                    census["dropped_carrier_truncated"] += 1
                    continue
                seen_key.add(key)
                per_doc[t["doc_id"]] += 1
                positives.append({**c, "doc_id": t["doc_id"], "source": t["source"],
                                  "chunk": chunk, "chunk_low": low, "chunk_toks": toks,
                                  "row_key": row_key, "form": t["form"],
                                  "lab_ci": t["lab_ci"], "tab": t})
        print(f"  cap {cap}: {len(positives)} positives over {len(per_doc)} documents",
              flush=True)

    hist = collections.Counter(p["unit"] for p in positives)
    print(f"positives: {len(positives)}", flush=True)
    print("  by unit: " + json.dumps(dict(hist.most_common())), flush=True)
    bydim = collections.defaultdict(collections.Counter)
    for p in positives:
        bydim[L.DIM[p["unit"]]][p["unit"]] += 1
    print("  by dimension: " + json.dumps({d: dict(c) for d, c in bydim.items()}), flush=True)

    # ---- families, twin-falsity guard, rebalance (lane machinery) -----------
    # Assemble at FULL supply and trim afterwards.  Passing the probe's target as
    # the builder's budget shrinks every family below the family floor first, and
    # the floor then deletes the lot.
    fam_pairs, fam_report = L.build_families(positives, rng, None)

    def negative_is_false(p, neg_unit):
        t, row = p["tab"], p["tab"]["body"][p["ri"]]
        for ci, hdr in enumerate(t["hdr"]):
            if ci in (p["ci"], t["lab_ci"]):
                continue
            s = row[ci].strip()
            hu, _ = L.header_unit(hdr)
            cu, num = L.cell_unit(s)
            if (hu == neg_unit and s == p["val"]) or (cu == neg_unit and num == p["val"]):
                return False
        return True

    print(f"  build_families -> {len(fam_pairs)} pairs; report: "
          + json.dumps(fam_report)[:1500], flush=True)
    before = len(fam_pairs)
    fam_pairs = [(p, n) for p, n in fam_pairs if negative_is_false(p, n)]
    dropped_true_twin = before - len(fam_pairs)

    by_fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, n in fam_pairs:
        by_fam["<->".join(sorted((p["unit"], n)))][p["unit"]].append((p, n))
    kept_by_fam, dropped_small = {}, {}
    for fam, dirs in sorted(by_fam.items()):
        kept = L.rebalance_family([x for v in dirs.values() for x in v], rng)
        if len(kept) < L.MIN_FAMILY_PAIRS:
            dropped_small[fam] = {"kept": 0, "offered": len(kept),
                                  "reason": f"below MIN_FAMILY_PAIRS={L.MIN_FAMILY_PAIRS}"}
            continue
        kept_by_fam[fam] = kept
    full = sum(len(v) for v in kept_by_fam.values())

    # trim to the probe target, proportionally, whole bucket groups only, and
    # never below the family floor
    fam_pairs = []
    if full > TARGET_PAIRS:
        scale = TARGET_PAIRS / full
        for fam, lst in sorted(kept_by_fam.items()):
            want = max(L.MIN_FAMILY_PAIRS, int(len(lst) * scale))
            fam_pairs += L.rebalance_family(lst, rng, want)
    else:
        fam_pairs = [x for v in kept_by_fam.values() for x in v]
    print(f"  {full} pairs at full supply -> {len(fam_pairs)} after trim to "
          f"target {TARGET_PAIRS}; dropped {dropped_true_twin} satisfiable twins; "
          f"dropped families {sorted(dropped_small)}", flush=True)

    rows = []
    for pid, (p, neg_unit) in enumerate(sorted(
            fam_pairs, key=lambda x: (x[0]["doc_id"], x[0]["ci"], x[0]["ri"], x[1]))):
        tpl = L.CLAIM_TEMPLATES[pid % len(L.CLAIM_TEMPLATES)]
        pos_claim = L.build_claim(tpl, p["col"], p["row_key"], p["val"], p["unit"])
        neg_claim = L.build_claim(tpl, p["col"], p["row_key"], p["val"], neg_unit)
        dim = L.DIM[p["unit"]]
        base = dict(chunk=p["chunk"], doc_id=p["doc_id"], source=p["source"],
                    column=p["col"], column_index=p["ci"], row_key=p["row_key"],
                    row_index=p["ri"], cited_value=p["val"],
                    correct_unit=p["unit"], wrong_unit=neg_unit, dimension=dim,
                    neg_family=L.FAMILY_OF_DIM.get(dim, "unit_swap"),
                    swap_family="<->".join(sorted((p["unit"], neg_unit))),
                    direction=f"{p['unit']}->{neg_unit}",
                    unit_carrier=p["carrier"],
                    distractor_in_chunk=L.attested_in_chunk(p, neg_unit),
                    surface_parity=bool(abs(len(pos_claim) - len(neg_claim)) <= 2),
                    serial_form=p["form"], template_id=pid % len(L.CLAIM_TEMPLATES),
                    tag=L.TAG, split="probe")
        rows.append(dict(pair_id=pid, label=1, claim=pos_claim,
                         cited_unit=p["unit"], **base))
        rows.append(dict(pair_id=pid, label=0, claim=neg_claim,
                         cited_unit=neg_unit, **base))

    if not rows:
        report = {"pairs": 0, "documents": 0, "supply_at_full_ladder": len(positives),
                  "family_construction": fam_report,
                  "verdict": "NO FAMILY REACHED THE MEASURABLE FLOOR - the lane "
                             "already consumed the rich documents and the residual "
                             "supply is the tail"}
        MANIFEST.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2)[:4000], flush=True)
        print("=== R18-H150 UNIT_SWAP PROBE SUPPLY-BLOCKED ===", flush=True)
        raise SystemExit(1)

    df = pl.DataFrame(rows).unique(subset=["claim", "chunk", "label"],
                                   keep="first", maintain_order=True)
    keep_pairs = df.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"]
    df = df.filter(pl.col("pair_id").is_in(keep_pairs)).sort(
        ["pair_id", "label"], descending=[False, True])
    n_pairs = df["pair_id"].n_unique()
    df.write_parquet(OUT)
    print(f"\n{df.height} rows / {n_pairs} pairs over {df['doc_id'].n_unique()} documents",
          flush=True)

    res = L.verify(df, rng, by_doc)

    shared_docs = len(set(df["doc_id"].to_list()) & lane_docs)
    shared_chunks = len(set(df["chunk"].to_list()) & lane_chunks)
    man = dict(
        experiment="R18-H150 unit_swap held-out PROBE (measurement only, never trains)",
        seed=SEED, target_pairs=TARGET_PAIRS, rows=df.height, pairs=n_pairs,
        documents=df["doc_id"].n_unique(),
        pairs_per_document=round(n_pairs / max(df["doc_id"].n_unique(), 1), 3),
        min_family_pairs=L.MIN_FAMILY_PAIRS, doc_cap=L.DOC_CAP_LADDER[-1], tag=L.TAG,
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        swap_families={k: v for k, v in df.group_by("swap_family").len().iter_rows()},
        dimensions={k: v for k, v in df.group_by("dimension").len().iter_rows()},
        unit_carrier={k: v for k, v in df.group_by("unit_carrier").len().iter_rows()},
        distractor_in_chunk={str(k): v for k, v in
                             df.group_by("distractor_in_chunk").len().iter_rows()},
        diversity=dict(
            serial_forms={k: v for k, v in df.group_by("serial_form").len().iter_rows()},
            templates={str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
            sources={k: v for k, v in df.group_by("source").len().iter_rows()},
            distinct_claims=df["claim"].n_unique(),
            distinct_chunks=df["chunk"].n_unique(),
            distinct_columns=df["column"].n_unique()),
        lane_disjointness=dict(
            lane_file=LANE.name, lane_documents=len(lane_docs),
            lane_documents_dropped_from_supply=dropped_lane,
            shared_documents=shared_docs, shared_chunks=shared_chunks,
            method="doc_id exclusion, enforced; a shared table is impossible so a "
                   "shared chunk or row key is impossible"),
        eval_disjointness=dict(evalset_rows=eval_rows, excluded_doc_ids=len(excluded_ids),
                               content_matched_tables=len(drop_idx),
                               tables_dropped=dropped_eval, tables_admitted=len(tables),
                               shared_documents=0, shared_chunks=0,
                               method="content-based (R17-H144 method), enforced"),
        overlap_permitted=[L.overlap(df, HERE / "R17-H144_pairs.parquet", "H144 pair corpus"),
                           L.overlap(df, HERE / "R17-H145_scaleunit.parquet", "H145 lane"),
                           L.overlap(df, HERE / "R17-H146_lane.parquet", "H146 lane")],
        supply=dict(pairs_at_full_supply=full, trimmed_to=len(fam_pairs),
                    target_pairs=TARGET_PAIRS),
        census=dict(unit_bearing_tables=int(census["unit_bearing_tables"]),
                    positives_built=len(positives),
                    dropped_claim_phrase_present=int(census["dropped_claim_phrase_present"]),
                    dropped_carrier_truncated=int(census["dropped_carrier_truncated"]),
                    dropped_no_chunk=int(census["dropped_no_chunk"]),
                    dropped_lane_chunk=int(census["dropped_lane_chunk"]),
                    dropped_satisfiable_negative=dropped_true_twin,
                    dropped_small_families=dropped_small),
        family_construction=fam_report,
        usage="REPORTED-SECONDARY read for the H150 arm. Measurement infrastructure: "
              "it is never added to a training mix and never selected on.",
        verify=res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    VERIFY.write_text(json.dumps({"rows": df.height, "pairs": n_pairs,
                                  "documents": df["doc_id"].n_unique(),
                                  "verify": res}, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "swap_families",
                       "lane_disjointness", "verify")}, indent=2)[:9000], flush=True)
    ok = res["all_bars_pass"] and res["h148_and_audit_pass"] and shared_docs == 0
    print(f"=== R18-H150 UNIT_SWAP PROBE {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
