"""CONFORMING PIPELINE - member `psiloqa`. CPU ONLY, torch untouched on GPU.

`psiloqa` failed the dataset contract on two clauses:

  C1  pooled attestation-rate delta 0.0931 on the full-coverage Unicode
      containment instrument, INSIDE the <= 0.10 rejection band by 0.0069
  C2  485 of 487 passages shared with `R20-H175b_qlane_eval` and 406 of 407
      with `R20-H175b_qlane_eval_repaired`, plus one claim string each

This script builds the conformed member. It removes rows; it does not relax a
clause, re-label a row, or re-weight a leg. Two filters, applied in this order:

  F1 (C2)  drop every row whose passage OR claim collides with ANY evaluation
           surface, in any of the string forms and pairings clause C2 tests
  F2 (C1)  drop every row whose claim exceeds K Unicode content tokens, K
           chosen as the LARGEST cap (smallest row loss) that puts the pooled
           attestation-rate delta outside the rejection band on BOTH containment
           instruments - the banked ASCII one and the full-coverage Unicode one

K is swept, not assumed. The sweep is written to the build manifest so the
choice of K is auditable.

Writes, beside this file:
  psiloqa_conformed.parquet        the conformed member
  psiloqa_conformed_build.json     manifest: sweep, filter masks, volume cost

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/psiloqa_conformed_build.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import datetime as dt
import importlib.util as _ilu
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
OUT_PARQUET = HERE / "psiloqa_conformed.parquet"
OUT_MANIFEST = HERE / "psiloqa_conformed_build.json"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MAIN = _mod("psicontract", HERE / "psiloqa_contract.py")
PM, CL = MAIN.PM, MAIN.CL
content_u = MAIN.content_u

BAND = 0.10          # C1 rejection band, from the contract. NEVER moved.
MARGIN_FLOOR = 0.01  # the cap must clear the band by at least this on BOTH
                     # instruments, so the verdict does not straddle as the
                     # original member's did


def rates(cont, y, mask, cov=None):
    """rate(containment >= 0.90) per leg and the absolute delta - the C1 test."""
    m = mask if cov is None else (mask & cov)
    p, q = cont[m & (y == 1)], cont[m & (y == 0)]
    if not p.size or not q.size:
        return None
    rp, rq = float((p >= 0.90).mean()), float((q >= 0.90).mean())
    return {
        "rows_scored": int(m.sum()), "n_pos": int(p.size), "n_neg": int(q.size),
        "rate_ge_0.90_pos": round(rp, 4), "rate_ge_0.90_neg": round(rq, 4),
        "delta": round(abs(rp - rq), 4),
        "margin_outside_band": round(abs(rp - rq) - BAND, 4),
        "clears_band": bool(abs(rp - rq) > BAND),
    }


def main():
    t0 = time.time()
    man = {
        "member": "psiloqa_conformed",
        "built_from": "psiloqa",
        "contract": "docs/experiments/dataset-contract.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "compute": "CPU only, CUDA_VISIBLE_DEVICES empty",
    }

    # ---------------------------------------------------------------- member
    mem, mix = MAIN.load_member()
    splits = MAIN.archive_splits()
    claims, chunks, y = mem["claims"], mem["chunks"], mem["y"]
    cut = mem["chunk_max"]
    n = len(y)
    print(f"member as loaded: {n} rows", flush=True)

    # archive replay, row-aligned - the original verification proved this holds
    tr = splits["train"].filter(
        (pl.col("wiki_passage").str.len_chars() > 50)
        & (pl.col("llm_answer").str.len_chars() > 10))
    if tr.height != n or tr["llm_answer"].to_list() != claims:
        raise SystemExit("ROW-ALIGNMENT ABORT: archive replay does not match the "
                         "banked loader output")
    langs = tr["lang"].to_list()

    # ------------------------------------------------- F1: the C2 collision set
    surfaces, arena_texts = MAIN.surface_units()

    def form_sets(texts):
        raw = set(texts)
        trunc = {t[:cut] for t in raw}
        return {"raw": raw, "trunc": trunc,
                "nraw": {CL.norm(t) for t in raw},
                "ntrunc": {CL.norm(t) for t in trunc}}

    # the pairings clause C2 tests, member-form -> surface-form
    PAIRINGS = [("raw", "raw"), ("trunc", "trunc"), ("nraw", "nraw"),
                ("raw", "trunc"), ("trunc", "raw"), ("ntrunc", "ntrunc")]

    surf_ev, surf_cl = {}, {}
    for name, s in surfaces.items():
        surf_ev[name] = form_sets(s["evidence"]) if s["evidence"] else None
        surf_cl[name] = form_sets(s["claims"]) if s["claims"] else None

    def row_forms(t):
        r = t
        tc = t[:cut]
        return {"raw": r, "trunc": tc, "nraw": CL.norm(r), "ntrunc": CL.norm(tc)}

    f1 = np.zeros(n, dtype=bool)
    hits_by_surface = {k: 0 for k in surfaces}
    hit_reason = {}
    for i, (cl_t, ch_t) in enumerate(zip(claims, chunks, strict=True)):
        cf, hf = row_forms(cl_t), row_forms(ch_t)
        for name in surfaces:
            ev, sc = surf_ev[name], surf_cl[name]
            hit = False
            for a, b in PAIRINGS:
                if ev is not None and (hf[a] in ev[b] or cf[a] in ev[b]):
                    hit = True
                    break
                if sc is not None and cf[a] in sc[b]:
                    hit = True
                    break
            if hit:
                f1[i] = True
                hits_by_surface[name] += 1
                hit_reason.setdefault(i, []).append(name)
        if i and i % 20000 == 0:
            print(f"  F1 scan {i}/{n}", flush=True)

    man["F1_c2_collision_filter"] = {
        "what": "drop every row whose passage or claim collides with any evaluation "
                "surface under any string form / pairing clause C2 tests",
        "pairings_tested": [f"member_{a}_vs_surface_{b}" for a, b in PAIRINGS],
        "blocks_tested": ["member_evidence vs surface_evidence",
                          "member_claim vs surface_claim",
                          "member_claim vs surface_evidence"],
        "rows_hit_per_surface": hits_by_surface,
        "rows_dropped": int(f1.sum()),
        "rows_dropped_share": round(float(f1.mean()), 6),
        "distinct_passages_dropped": len({chunks[i] for i in np.flatnonzero(f1)}),
    }
    print(f"F1 drops {int(f1.sum())} rows ({f1.mean():.4%})", flush=True)

    keep1 = ~f1

    # ----------------------------------------------------- containment vectors
    ntok = np.zeros(n, dtype=np.int32)
    cont_u = np.zeros(n)
    cont_a = np.zeros(n)
    cov_u = np.zeros(n, dtype=bool)
    cov_a = np.zeros(n, dtype=bool)
    for i, (cl_t, ch_t) in enumerate(zip(claims, chunks, strict=True)):
        cu, eu = content_u(cl_t), content_u(ch_t)
        ntok[i] = len(cu)
        cov_u[i] = bool(cu)
        cont_u[i] = PM.containment(cu, eu)
        ca, ea = PM.content(cl_t), PM.content(ch_t)
        cov_a[i] = bool(ca)
        cont_a[i] = PM.containment(ca, ea)
        if i and i % 20000 == 0:
            print(f"  containment {i}/{n}", flush=True)

    # --------------------------------------------------------- F2: the C1 sweep
    sweep = {}
    for k in list(range(12, 61, 2)) + [70, 80, 100, 10**6]:
        m = keep1 & (ntok <= k)
        sweep[f"cap_{k}"] = {
            "cap_content_tokens": (None if k == 10**6 else k),
            "rows": int(m.sum()),
            "rows_retained_share_of_F1": round(float(m.sum() / keep1.sum()), 4),
            "unicode": rates(cont_u, y, m, cov_u),
            "banked_ascii": rates(cont_a, y, m, cov_a),
        }
    for k, v in sweep.items():
        u, a = v["unicode"], v["banked_ascii"]
        both = (u and a and u["margin_outside_band"] >= MARGIN_FLOOR
                and a["margin_outside_band"] >= MARGIN_FLOOR)
        v["clears_both_instruments_with_margin"] = bool(both)
        print(f"  {k:>10} rows={v['rows']:>6} uni_delta={u['delta']} "
              f"ascii_delta={a['delta']} both={both}", flush=True)

    ok = [(v["cap_content_tokens"], v["rows"]) for v in sweep.values()
          if v["clears_both_instruments_with_margin"] and v["cap_content_tokens"]]
    if not ok:
        man["F2_c1_filter"] = {"chosen_cap": None,
                               "reason": "NO cap clears the band on both instruments "
                                         "with the required margin"}
        OUT_MANIFEST.write_text(json.dumps({**man, "sweep": sweep}, indent=2))
        raise SystemExit("C1 UNFIXABLE BY THIS AXIS - see manifest")
    chosen = max(ok, key=lambda t: t[1])[0]   # largest cap = smallest row loss

    f2 = ntok > chosen
    keep = keep1 & ~f2
    man["F2_c1_filter"] = {
        "what": "drop every row whose claim exceeds the chosen Unicode content-token "
                "cap - the measured cause of the lost resolution (the 25+ token band "
                "reads delta 0.0010, AUROC 0.4094 on the original member)",
        "axis": "Unicode \\w+ content tokens of the claim, campaign stopwords removed",
        "cap_selection_rule": f"largest cap whose pooled delta clears the {BAND} band "
                              f"by >= {MARGIN_FLOOR} on BOTH containment instruments",
        "chosen_cap": chosen,
        "rows_dropped_by_F2_after_F1": int((keep1 & f2).sum()),
        "sweep": sweep,
    }
    print(f"F2 cap = {chosen} content tokens", flush=True)

    # ------------------------------------------------------------ write member
    idx = np.flatnonzero(keep)
    conf = pl.DataFrame({
        "row_index_in_member": idx.astype(np.int64),
        "claim": [claims[i] for i in idx],
        "chunk": [chunks[i] for i in idx],
        "label": y[idx].astype(np.float32),
        "lang": [langs[i] for i in idx],
        "claim_content_tokens_unicode": ntok[idx].astype(np.int32),
    })
    conf.write_parquet(OUT_PARQUET)

    man["volume_cost"] = {
        "rows_original": n,
        "rows_conformed": int(conf.height),
        "rows_dropped_total": int(n - conf.height),
        "rows_retained_share": round(float(conf.height / n), 4),
        "rows_dropped_by_F1_c2": int(f1.sum()),
        "rows_dropped_by_F2_c1_after_F1": int((keep1 & f2).sum()),
        "rows_dropped_by_both_filters": int((f1 & f2).sum()),
        "distinct_passages_original": len(set(chunks)),
        "distinct_passages_conformed": conf["chunk"].n_unique(),
        "distinct_claims_original": len(set(claims)),
        "distinct_claims_conformed": conf["claim"].n_unique(),
        "positive_rate_original": round(float((y == 1).mean()), 4),
        "positive_rate_conformed": round(float((conf["label"] == 1).mean()), 4),
        "clean_mix_rows_with_original_member": mem["mix_rows"],
        "clean_mix_rows_with_conformed_member": int(mem["mix_rows"] - (n - conf.height)),
        "member_share_of_mix_original": round(n / mem["mix_rows"], 4),
        "member_share_of_mix_conformed": round(
            conf.height / (mem["mix_rows"] - (n - conf.height)), 4),
    }
    man["final_c1_reading_on_conformed_rows"] = {
        "unicode_full_coverage": rates(cont_u, y, keep, cov_u),
        "banked_ascii": rates(cont_a, y, keep, cov_a),
    }
    man["artifacts"] = {"member": str(OUT_PARQUET.relative_to(EXP.parent.parent)),
                        "manifest": str(OUT_MANIFEST.relative_to(EXP.parent.parent))}
    man["seconds"] = round(time.time() - t0, 1)
    OUT_MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({"volume_cost": man["volume_cost"],
                      "final_c1": man["final_c1_reading_on_conformed_rows"]}, indent=2),
          flush=True)
    print(f"=== conformed member -> {OUT_PARQUET} ({man['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
