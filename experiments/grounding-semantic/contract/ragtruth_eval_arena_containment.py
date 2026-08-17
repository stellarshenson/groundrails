"""R21-H180 eval - the ARENA containment channel, and the whole-reference-union
containment reading. CPU ONLY. Supplement to `ragtruth_eval_report.json`.

Why this exists. The banked arena surface audit found 17 arena documents
byte-for-byte inside `halueval` training chunks - all in `hotpotqa`, exposing 102
of 250 hotpotqa responses at 8-gram containment >= 0.10 - and exact string
matching saw none of it. Any arena disjointness claim made on exact or
whitespace-normalised matching alone is therefore not comparable evidence. This
runs the SAME instrument against the R21-H180 eval, in both directions.

Two readings the main pass did not carry:

  1. ARENA channel - eval units against the arena's own documents and responses,
     under `arena_surface_verify._max_pair` (Jaccard AND containment from one
     search, reused verbatim), rolled up to the arena's response-level exposure
     at the audit's own thresholds 0.10 / 0.25 / 0.50 / 0.90 / 1.00
  2. WHOLE-UNION containment - the fraction of an eval unit's 8-grams present
     ANYWHERE in a reference side, not merely inside one single unit. The
     best-single-unit reading the main pass banked is a lower bound: a source
     document re-chunked across several mix rows is invisible to it

Both directions carry a LIVE positive control - reference text re-chunked at a
different offset, which the audit's own control showed reads containment
0.9891-1.0000 while Jaccard drops as low as 0.2796.

Merges its output into `ragtruth_eval_report.json` under
`clauses.C2.document_channel`; the existing best-single-unit block is left intact.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/contract/ragtruth_eval_arena_containment.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
EVAL = EXP / "R21-H180_ragtruth_eval.parquet"
REPORT = HERE / "ragtruth_eval_report.json"

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
THRESHOLDS = (0.10, 0.25, 0.50, 0.90, 1.00)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading the banked arena instrument")
ASV = _mod("asv", HERE / "arena_surface_verify.py")   # load_arena, arena_units, _max_pair, _index
PG = ASV.PG
ARM = ASV.ARM
GATE_N = ASV.GATE_N
GATE_JACCARD = ASV.GATE_JACCARD
GATE_KILL = ASV.GATE_KILL

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 721_210
LANES = (("R17-H146_lane.parquet", "quant_misbind", 30_000),
         ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540))


def flagship_mix():
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = ARM.H108.public_train()
    if len(claims) != EXPECTED_CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: {len(claims)} clean rows")
    for fname, group, n_rows in LANES:
        d = pl.read_parquet(EXP / fname)
        if d.height != n_rows:
            raise SystemExit(f"LANE ABORT ({group}): {d.height} rows")
        ch = "chunk" if "chunk" in d.columns else "evidence"
        claims += d["claim"].to_list()
        chunks += d[ch].to_list()
        tags += [group] * d.height
    if len(claims) != EXPECTED_MIX_ROWS:
        raise SystemExit(f"MIX ABORT: {len(claims)} rows")
    log(f"mix assembled: {len(claims)} rows over {len(set(tags))} groups")
    return claims, chunks, tags


def rechunk(text, offset=137):
    """The audit's live control shape: the same bytes re-wrapped at a different
    window offset, so the text is near-duplicate by construction but no chunk
    boundary is shared."""
    return text[offset:] + " " + text[:offset]


# --------------------------------------------------------------------------- #
def main():
    if not REPORT.exists():
        raise SystemExit(f"ABORT: {REPORT} absent - run ragtruth_eval_contract.py first")
    ev = pl.read_parquet(EVAL)
    hasher = PG._TokenHasher()

    contexts = sorted({c for c in ev["chunk"].to_list() if c.strip()})
    subs = sorted({p.strip() for c in contexts for p in c.split("\n\n") if p.strip()})
    claims = sorted({c for c in ev["claim"].to_list() if c.strip()})
    units = {"eval_contexts": contexts, "eval_context_subpassages": subs,
             "eval_claims": claims}
    qh = {k: [PG.ngram_hashes(t, GATE_N, hasher) for t in v] for k, v in units.items()}
    for k, v in qh.items():
        log(f"eval {k}: {len(v)} units, {sum(1 for a in v if a.size)} scorable at "
            f"{GATE_N}-grams")

    # ----------------------------------------------------------------- #
    # 1. ARENA channel - the banked instrument, both directions
    # ----------------------------------------------------------------- #
    subsets = ASV.load_arena()
    arena_ch, arena_owner = ASV.arena_units(subsets)
    arena_units_uniq = {}
    arena_unit_sub = {}
    for chn in ("documents", "responses"):
        uniq = sorted(set(arena_ch[chn]))
        sub_of = {}
        for u, s in zip(arena_ch[chn], arena_owner[chn], strict=True):
            sub_of.setdefault(u, s)
        arena_units_uniq[chn] = uniq
        arena_unit_sub[chn] = [sub_of[u] for u in uniq]
        log(f"arena {chn}: {len(uniq)} distinct units")

    arena_block = {}
    for chn in ("documents", "responses"):
        ah = [PG.ngram_hashes(t, GATE_N, hasher) for t in arena_units_uniq[chn]]
        a_flat, a_owner, a_sizes = ASV._index(ah)
        a_pool = np.unique(a_flat)          # whole-arena-channel union
        block = {}
        for uname, qs in qh.items():
            jac = np.zeros(len(qs))
            cov = np.zeros(len(qs))
            uni = np.zeros(len(qs))
            who = [None] * len(qs)
            for i, q in enumerate(qs):
                if q.size == 0:
                    continue
                j, c, _uj, uc = ASV._max_pair(q, a_flat, a_owner, a_sizes)
                jac[i], cov[i] = j, c
                if uc >= 0:
                    who[i] = arena_unit_sub[chn][uc]
                lo = np.searchsorted(a_pool, q, side="left")
                hi = np.searchsorted(a_pool, q, side="right")
                uni[i] = float((hi > lo).sum()) / q.size
            block[uname] = {
                "eval_units": len(qs),
                "best_single_arena_unit": {
                    "max_jaccard": round(float(jac.max()), 6),
                    "max_containment": round(float(cov.max()), 6),
                    "mean_containment": round(float(cov.mean()), 6),
                    "units_at_jaccard_ge_0.30": int((jac >= GATE_JACCARD).sum()),
                    "units_by_containment_threshold": {
                        str(t): int((cov >= t).sum()) for t in THRESHOLDS},
                    "attributed_subsets_above_0.10_containment": dict(
                        collections.Counter(w for w, c in zip(who, cov)
                                            if c >= 0.10 and w)),
                },
                "whole_arena_channel_union": {
                    "max_containment": round(float(uni.max()), 6),
                    "mean_containment": round(float(uni.mean()), 6),
                    "units_by_containment_threshold": {
                        str(t): int((uni >= t).sum()) for t in THRESHOLDS},
                },
            }
            log(f"  arena {chn} <- {uname}: best-single max_cont "
                f"{cov.max():.4f}, union max_cont {uni.max():.4f}, "
                f"J>=0.3 {int((jac >= GATE_JACCARD).sum())}")
        arena_block[f"eval_into_arena_{chn}"] = block
        del a_flat, a_owner, a_pool, ah

    # reverse direction, rolled up the way the banked audit rolls it up:
    # an arena RESPONSE is "exposed" when any of its documents is contained in
    # the eval side above the threshold
    e_flat, e_owner, e_sizes = ASV._index(qh["eval_contexts"])
    e_pool = np.unique(e_flat)
    doc_uniq = arena_units_uniq["documents"]
    dcov = np.zeros(len(doc_uniq))
    duni = np.zeros(len(doc_uniq))
    for i, t in enumerate(doc_uniq):
        q = PG.ngram_hashes(t, GATE_N, hasher)
        if q.size == 0:
            continue
        _j, c, _uj, _uc = ASV._max_pair(q, e_flat, e_owner, e_sizes)
        dcov[i] = c
        lo = np.searchsorted(e_pool, q, side="left")
        hi = np.searchsorted(e_pool, q, side="right")
        duni[i] = float((hi > lo).sum()) / q.size
    doc_pos = {d: i for i, d in enumerate(doc_uniq)}
    exposure = {}
    for t in THRESHOLDS:
        hit_single = {doc_uniq[i] for i in range(len(doc_uniq)) if dcov[i] >= t}
        hit_union = {doc_uniq[i] for i in range(len(doc_uniq)) if duni[i] >= t}
        per_sub, tot_s, tot_u = {}, 0, 0
        for sub, v in subsets.items():
            ns = sum(1 for docs in v["documents"] if any(c in hit_single for c in docs))
            nu = sum(1 for docs in v["documents"] if any(c in hit_union for c in docs))
            tot_s += ns
            tot_u += nu
            per_sub[sub] = {"responses": len(v["responses"]),
                            "responses_touched_single_eval_unit": ns,
                            "responses_touched_eval_union": nu}
        exposure[str(t)] = {
            "arena_documents_single_eval_unit": len(hit_single),
            "arena_documents_eval_union": len(hit_union),
            "arena_responses_touched_single_eval_unit": tot_s,
            "arena_responses_touched_eval_union": tot_u,
            "per_subset": per_sub,
        }
        log(f"  arena reverse @{t}: {len(hit_single)} docs / {tot_s} responses "
            f"(single), {len(hit_union)} docs / {tot_u} responses (union)")

    # LIVE positive control on the arena channel: 10 arena documents re-chunked
    # at a different offset, offered to the SAME instrument against an index
    # built from the untouched arena documents. A gate that cannot fire dies here.
    a_h = [PG.ngram_hashes(t, GATE_N, hasher) for t in doc_uniq]
    a_flat, a_owner, a_sizes = ASV._index(a_h)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(doc_uniq), size=10, replace=False)
    ctrl = []
    for i in pick:
        q = PG.ngram_hashes(rechunk(doc_uniq[int(i)]), GATE_N, hasher)
        j, c, _a, _b = ASV._max_pair(q, a_flat, a_owner, a_sizes)
        ctrl.append({"jaccard": round(j, 4), "containment": round(c, 4)})
    live_arena = {
        "construction": "10 arena documents re-wrapped at a 137-character offset - "
                        "near-duplicate by construction, no shared chunk boundary - "
                        "offered to the same instrument against the untouched arena "
                        "document index",
        "per_unit": ctrl,
        "min_containment": round(min(c["containment"] for c in ctrl), 4),
        "min_jaccard": round(min(c["jaccard"] for c in ctrl), 4),
        "units_below_jaccard_bar": sum(1 for c in ctrl if c["jaccard"] < GATE_JACCARD),
        "fires": all(c["containment"] >= 0.90 for c in ctrl),
        "reading": "the containment channel is what detects a re-chunked document; "
                   "Jaccard's union denominator drops some of the same units below "
                   "its bar, which is why a Jaccard-only count is a lower bound",
    }
    log(f"  arena live control: min containment {live_arena['min_containment']}, "
        f"min Jaccard {live_arena['min_jaccard']}, "
        f"{live_arena['units_below_jaccard_bar']} below the Jaccard bar")
    del a_flat, a_owner, a_h, e_flat, e_owner, e_pool

    # ----------------------------------------------------------------- #
    # 2. WHOLE-UNION containment against the training mix, per group
    # ----------------------------------------------------------------- #
    mix_claims, mix_chunks, mix_tags = flagship_mix()
    by_group = collections.defaultdict(set)
    for c, k, t in zip(mix_claims, mix_chunks, mix_tags, strict=True):
        if k and k.strip():
            by_group[t].add(k)
        if c and c.strip():
            by_group[t].add(c)
    del mix_claims, mix_chunks, mix_tags

    seen = {k: [np.zeros(a.size, dtype=bool) for a in v] for k, v in qh.items()}
    per_group_union = {}
    ctx_cov_by_group = {}
    for grp in sorted(by_group):
        t0 = time.time()
        pool = np.unique(np.concatenate(
            [h for h in (PG.ngram_hashes(u, GATE_N, hasher) for u in sorted(by_group[grp]))
             if h.size]))
        grp_union = {}
        for uname, qs in qh.items():
            cov = np.zeros(len(qs))
            for i, q in enumerate(qs):
                if q.size == 0:
                    continue
                lo = np.searchsorted(pool, q, side="left")
                hi = np.searchsorted(pool, q, side="right")
                m = hi > lo
                seen[uname][i] |= m
                cov[i] = float(m.sum()) / q.size
            grp_union[uname] = {
                "max_containment": round(float(cov.max()), 6),
                "mean_containment": round(float(cov.mean()), 6),
                "units_by_containment_threshold": {
                    str(t): int((cov >= t).sum()) for t in THRESHOLDS},
            }
            if uname == "eval_contexts":
                ctx_cov_by_group[grp] = cov.copy()
        per_group_union[grp] = grp_union
        log(f"  union {grp}: {len(by_group[grp])} units, ctx max "
            f"{grp_union['eval_contexts']['max_containment']:.4f}, sub max "
            f"{grp_union['eval_context_subpassages']['max_containment']:.4f}, "
            f"claim max {grp_union['eval_claims']['max_containment']:.4f} "
            f"({time.time() - t0:.0f}s)")
        del pool

    whole_mix = {}
    for uname, qs in qh.items():
        cov = np.array([(s.sum() / a.size) if a.size else 0.0
                        for s, a in zip(seen[uname], qs, strict=True)])
        whole_mix[uname] = {
            "eval_units": len(qs),
            "max_containment": round(float(cov.max()), 6),
            "mean_containment": round(float(cov.mean()), 6),
            "median_containment": round(float(np.median(cov)), 6),
            "p90_containment": round(float(np.percentile(cov, 90)), 6),
            "units_by_containment_threshold": {
                str(t): int((cov >= t).sum()) for t in THRESHOLDS},
        }
        log(f"  whole-mix union {uname}: max {cov.max():.4f} mean {cov.mean():.4f} "
            f">=0.90 {int((cov >= 0.90).sum())} >=0.10 {int((cov >= 0.10).sum())}")

    # LIVE positive control on the union channel: eval contexts re-chunked and
    # offered to an index built from the eval contexts themselves
    ep = np.unique(np.concatenate([a for a in qh["eval_contexts"] if a.size]))
    ctrl2 = []
    for i in rng.choice(len(contexts), size=10, replace=False):
        q = PG.ngram_hashes(rechunk(contexts[int(i)]), GATE_N, hasher)
        lo = np.searchsorted(ep, q, side="left")
        hi = np.searchsorted(ep, q, side="right")
        ctrl2.append(round(float((hi > lo).sum()) / q.size, 4))
    live_union = {
        "construction": "10 eval contexts re-wrapped at a 137-character offset, "
                        "offered to the union index built from the untouched eval "
                        "contexts",
        "containments": ctrl2,
        "min": min(ctrl2),
        "fires": min(ctrl2) >= 0.90,
    }
    log(f"  union live control: min {live_union['min']}")

    # which eval contexts the mix already carries, and what they are - the
    # exposure is worthless to a reader without its composition
    tt_of = dict(zip(ev["chunk"].to_list(), ev["task_type"].to_list(), strict=True))
    ctx_tt = [tt_of[c] for c in contexts]
    rows_per_ctx = collections.Counter(ev["chunk"].to_list())
    lab_by_ctx = ev.group_by("chunk").agg(pl.col("label").mean().alias("p"))
    p_of = dict(zip(lab_by_ctx["chunk"].to_list(), lab_by_ctx["p"].to_list(), strict=True))
    charac = {}
    for grp, cov in ctx_cov_by_group.items():
        if float(cov.max()) < 0.10:
            continue
        blk = {}
        for t in (0.10, 0.50, 0.90):
            hit = [i for i in range(len(contexts)) if cov[i] >= t]
            blk[str(t)] = {
                "eval_contexts": len(hit),
                "eval_rows_behind_them": int(sum(rows_per_ctx[contexts[i]] for i in hit)),
                "by_task_type": dict(collections.Counter(ctx_tt[i] for i in hit)),
                "positive_rate_of_those_rows": (
                    round(float(np.mean([p_of[contexts[i]] for i in hit])), 4)
                    if hit else None),
            }
        charac[grp] = blk
        log(f"  exposure composition {grp}: {json.dumps(blk['0.9'])}")
    baseline = {
        "all_450_contexts_by_task_type": dict(collections.Counter(ctx_tt)),
        "all_2700_rows_positive_rate": round(float(ev["label"].mean()), 4),
    }

    # ----------------------------------------------------------------- #
    # merge into the banked report
    # ----------------------------------------------------------------- #
    rep = json.loads(REPORT.read_text())
    dc = rep["clauses"]["C2"]["document_channel"]
    dc["arena_containment_channel"] = {
        "why": "the banked arena surface audit found 17 arena documents byte-for-byte "
               "inside halueval training chunks - all hotpotqa, 102 of 250 hotpotqa "
               "responses exposed at 8-gram containment >= 0.10 - and exact string "
               "matching saw none of it. A zero here is only comparable evidence if it "
               "is measured on the same channel",
        "instrument": "arena_surface_verify._max_pair (Jaccard AND containment from one "
                      "search) and its whole-channel union pool, reused verbatim; arena "
                      "built by arena_surface_verify.load_arena - the frozen R8-H77 "
                      "gate, 10 subsets / 2,264 responses",
        "expectation_stated_before_the_read": "RAGTruth's contexts derive from MS MARCO "
                                              "(QA), CNN/DailyMail (Summary) and Yelp "
                                              "(Data2txt); none has an arena subset, so "
                                              "zero is the expected reading",
        "thresholds": list(THRESHOLDS),
        **arena_block,
        "arena_response_level_exposure_reverse_direction": exposure,
        "live_positive_control_rechunk": live_arena,
        "note": NOTE,
    }
    dc["whole_reference_union_containment"] = {
        "why": "the best-single-unit reading banked in this block is a LOWER BOUND - a "
               "source document re-chunked across several mix rows is invisible to it. "
               "This is the fraction of an eval unit's 8-grams present ANYWHERE in a "
               "reference side",
        "instrument": f"{GATE_N}-gram union pool per training-mix DANN group, "
                      "accumulated across all 14 groups; claims and evidence pooled per "
                      "group",
        "thresholds": list(THRESHOLDS),
        "per_training_mix_group": per_group_union,
        "whole_training_mix": whole_mix,
        "live_positive_control_rechunk": live_union,
        "exposure_composition_per_group": charac,
        "eval_baseline_for_comparison": baseline,
        "note": NOTE,
    }

    m = rep["clauses"]["C2"]["measured"]
    m["arena_channel_max_containment_eval_into_arena_documents"] = arena_block[
        "eval_into_arena_documents"]["eval_contexts"]["best_single_arena_unit"][
        "max_containment"]
    m["arena_channel_units_at_jaccard_ge_0.30"] = arena_block[
        "eval_into_arena_documents"]["eval_contexts"]["best_single_arena_unit"][
        "units_at_jaccard_ge_0.30"]
    m["arena_response_level_exposure_at_0.10"] = exposure["0.1"][
        "arena_responses_touched_eval_union"]
    m["whole_training_mix_union_containment_max"] = {
        k: v["max_containment"] for k, v in whole_mix.items()}
    m["whole_training_mix_union_units_ge_0.90"] = {
        k: v["units_by_containment_threshold"]["0.9"] for k, v in whole_mix.items()}
    REPORT.write_text(json.dumps(rep, indent=2))
    log(f"merged -> {REPORT}")


if __name__ == "__main__":
    main()
