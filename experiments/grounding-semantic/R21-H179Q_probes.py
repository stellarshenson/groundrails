"""R21-H179Q - the three probes over the blind-arena per-item scores.

MEASUREMENT ONLY: nothing here trains, tunes, selects, promotes, kills or
adjudicates. Every artifact carries the note verbatim.

Companion to `R21-H179_arena_scores.py`, which banks the per-item response
scores of the six banked flagship draws under the shipped windowed
decomposed-min read. This script adds the three questions the coordinator
attached to that pass:

    q1        IS THE HEADLINE INFLATED BY CONTAMINATION? The arena mean as
              banked, with the verbatim-exposed responses dropped, and with the
              containment-exposed responses dropped; per subset and per draw,
              plus the 6-draw mean and its delta against the 0.71218 headline.
              CPU, from the banked npz - no GPU, no re-read.
    ablate    DOES THE MODEL USE THE EVIDENCE? The SAME six checkpoints and the
              SAME shipped read, with the evidence PERMUTED across items inside
              each subset: every surface statistic of the evidence side is
              preserved (length, register, token distribution, the multiset of
              per-item document counts) and only the claim-evidence relation is
              destroyed. GPU, one checkpoint per invocation.
    q2        the ablated arena mean rolled up against the true-evidence mean.
              CPU.
    q3        the contamination-exposure column for the failure-mode autopsy, so
              its error classes cannot be confounded by q1. CPU.

WHERE THE EXPOSED-RESPONSE IDS COME FROM. They are NOT re-derived. Both lists
are taken from the banked surface audit:

  * verbatim - `contract/arena_surface_report.json ::
    verbatim_substring_proof.documents[*].doc_blake2b_64`, the 17 arena
    documents proven byte-for-byte inside a HaluEval training chunk. A response
    is exposed when any of ITS OWN retained documents hashes into that set,
    which is the report's own attribution rule
    (`arena_surface_substring_proof.py`, `hit_docs` / `per_sub`).
  * containment >= 0.10 - the report banks only the COUNTS at that threshold,
    not the document ids, so the per-document containment array is read from the
    checkpointed census the report itself was computed from,
    `tmp/arena_surface/census.npz :: best_c|documents`, indexed by the report's
    own `sorted(set(arena documents))` ordering
    (`arena_surface_containment_exposure.py`).

Both mappings are CONTROLLED before use: the recomputed exposed-response counts
must reproduce the banked per-subset counts in the report exactly, or the run
aborts. The document universe is the read's own `documents[:8]` (MAX_CHUNKS = 8
in both `R8-H77_unseen_arena.py` and the audit's reproduction of it), so the
exposure attributed to a response is exactly the text the model was shown.

Run (CPU stages):
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/R21-H179Q_probes.py --stage q1
Run (GPU stage, one checkpoint per invocation, card bound explicitly):
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/R21-H179Q_probes.py --stage ablate --draw d1
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit(
        "GPU PLACEMENT ABORT: set CUDA_VISIBLE_DEVICES explicitly (a card index "
        "for --stage ablate, empty for the CPU stages) - the other cards carry "
        "R20-H174 training draws and must not be touched")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
CACHE = ROOT / "tmp" / "arena_surface"
REPORT = HERE / "contract" / "arena_surface_report.json"

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
HEADLINE = 0.71218          # the banked 6-draw flagship arena mean
CONTAINMENT_THR = 0.10      # the coordinator's named exposure threshold
ABLATION_SEED = 179         # fixed; the permutation is banked
FIDELITY_TOL = 1e-4

Q1_JSON = HERE / "R21-H179Q_contamination_drop.json"
Q2_JSON = HERE / "R21-H179Q_evidence_ablation.json"
Q3_JSON = HERE / "R21-H179Q_autopsy_exposure.json"
EXPOSURE_PARQUET = HERE / "R21-H179Q_exposed_responses.parquet"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _mod("r21scores", "R21-H179_arena_scores.py")
DRAWS = S.DRAWS


def ablated_npz(tag):
    return HERE / f"R21-H179Q_ablated_scores_{DRAWS[tag]['ckpt']}.npz"


def ablated_json(tag):
    return HERE / f"R21-H179Q_ablated_{tag}.json"


def ablated_read_json(tag):
    return HERE / f"R21-H179Q_ablated_read_{DRAWS[tag]['ckpt']}_windowed_result.json"


# --- the exposed-response lists ----------------------------------------------------


def doc_key(d):
    """The surface audit's own document key (arena_surface_substring_proof.py)."""
    return hashlib.blake2b(d.encode("utf-8"), digest_size=8).hexdigest()


def exposure_table():
    """One row per arena response, carrying the two exposure flags.

    Built over the READ'S OWN item table (`R21-H179_arena_items.parquet`, the
    documents[:8] the model was shown), so the flags align row-for-row with the
    banked per-item scores. Controlled against the report's banked counts.
    """
    rep = json.loads(REPORT.read_text())
    items = pl.read_parquet(S.ITEMS_PARQUET)

    # --- verbatim: the report's own 17 document ids ---------------------------
    verbatim_ids = {h["doc_blake2b_64"] for h in rep["verbatim_substring_proof"]["documents"]}

    # --- containment: the banked census, in the report's own uniq ordering ----
    z = np.load(CACHE / "census.npz")
    best_c = z["best_c|documents"]
    uniq = sorted({d for docs in items["documents"].to_list() for d in docs})
    if len(uniq) != best_c.size:
        raise SystemExit(
            f"EXPOSURE ABORT: {len(uniq)} distinct arena documents in the read's "
            f"item table but {best_c.size} in the banked census - the two are not "
            "the same document universe and the containment array cannot be indexed")
    contained = {uniq[i] for i in range(len(uniq)) if best_c[i] >= CONTAINMENT_THR}

    rows = []
    for r in items.iter_rows(named=True):
        keys = {doc_key(d) for d in r["documents"]}
        rows.append({
            "subset": r["subset"], "item": r["item"], "row_id": r["row_id"],
            "label": r["label"],
            "exposed_verbatim": bool(keys & verbatim_ids),
            "exposed_containment": bool(any(d in contained for d in r["documents"])),
        })
    tbl = pl.DataFrame(rows)

    # --- control: reproduce the report's banked per-subset counts exactly -----
    want_v = {s: v["responses_retrieving_a_verbatim_document"] for s, v in
              rep["verbatim_substring_proof"]["per_subset_response_exposure"].items()}
    want_c = {s: v["responses_touched_single_chunk"] for s, v in
              rep["containment_exposure"]["per_threshold"]
              [f"containment_ge_{CONTAINMENT_THR:.2f}"]["per_subset"].items()}
    got_v = dict(tbl.group_by("subset").agg(pl.col("exposed_verbatim").sum())
                 .iter_rows())
    got_c = dict(tbl.group_by("subset").agg(pl.col("exposed_containment").sum())
                 .iter_rows())
    control = {
        "verbatim_banked": want_v, "verbatim_recomputed": got_v,
        "containment_banked": want_c, "containment_recomputed": got_c,
        "verbatim_ok": got_v == want_v, "containment_ok": got_c == want_c,
    }
    if not (control["verbatim_ok"] and control["containment_ok"]):
        raise SystemExit(
            "EXPOSURE ABORT: the exposed-response sets rebuilt over the read's "
            f"item table do not reproduce the banked report counts.\n{control}")
    print(f"  exposure control PASS: verbatim {sum(want_v.values())} responses, "
          f"containment>={CONTAINMENT_THR:.2f} {sum(want_c.values())} responses",
          flush=True)
    tbl.write_parquet(EXPOSURE_PARQUET)
    return tbl, control


# --- q1: the contamination-drop re-read --------------------------------------------


def subset_auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def stage_q1():
    rng_seeds = tuple(range(20))
    tbl, control = exposure_table()
    zs = {t: np.load(S.npz_path(t), allow_pickle=False) for t in DRAWS}
    subsets = sorted({k.split("__")[1] for k in zs["d1"].files if k.startswith("y__")})

    keep = {}
    for sub in subsets:
        m = tbl.filter(pl.col("subset") == sub).sort("item")
        keep[sub] = {
            "all": np.ones(len(m), dtype=bool),
            "drop_verbatim": ~m["exposed_verbatim"].to_numpy(),
            "drop_containment": ~m["exposed_containment"].to_numpy(),
        }

    variants = ("all", "drop_verbatim", "drop_containment")
    per_draw, per_subset = {}, {}
    for t in DRAWS:
        z = zs[t]
        rows = {}
        for sub in subsets:
            y, s = z[f"y__{sub}"], z[f"resp__{sub}"]
            if len(y) != keep[sub]["all"].size:
                raise SystemExit(f"ALIGN ABORT {t}/{sub}: {len(y)} scores vs "
                                 f"{keep[sub]['all'].size} exposure rows")
            rows[sub] = {v: subset_auc(y[keep[sub][v]], s[keep[sub][v]]) for v in variants}
            rows[sub]["n_dropped_verbatim"] = int((~keep[sub]["drop_verbatim"]).sum())
            rows[sub]["n_dropped_containment"] = int((~keep[sub]["drop_containment"]).sum())
        per_draw[t] = {
            "checkpoint": DRAWS[t]["ckpt"],
            "banked_mean": json.loads((HERE / DRAWS[t]["banked"]).read_text())["mean"],
            "per_subset": {s: {v: (None if rows[s][v] is None else round(rows[s][v], 6))
                               for v in variants} for s in subsets},
            "mean": {v: round(float(np.mean([rows[s][v] for s in subsets])), 6)
                     for v in variants},
        }
        per_draw[t]["delta_vs_all"] = {
            v: round(per_draw[t]["mean"][v] - per_draw[t]["mean"]["all"], 6)
            for v in ("drop_verbatim", "drop_containment")}
        per_subset[t] = rows
        print(f"  {t} {DRAWS[t]['ckpt']:22s} all {per_draw[t]['mean']['all']:.5f}  "
              f"-verbatim {per_draw[t]['mean']['drop_verbatim']:.5f}  "
              f"-containment {per_draw[t]['mean']['drop_containment']:.5f}", flush=True)

    k6 = {v: round(float(np.mean([per_draw[t]["mean"][v] for t in DRAWS])), 6)
          for v in variants}
    k6_delta = {v: round(k6[v] - k6["all"], 6) for v in ("drop_verbatim", "drop_containment")}

    sub_k6 = {}
    for sub in subsets:
        a = {v: float(np.mean([per_subset[t][sub][v] for t in DRAWS])) for v in variants}
        sub_k6[sub] = {
            **{v: round(a[v], 6) for v in variants},
            "delta_drop_verbatim": round(a["drop_verbatim"] - a["all"], 6),
            "delta_drop_containment": round(a["drop_containment"] - a["all"], 6),
            "n": int(keep[sub]["all"].size),
            "n_dropped_verbatim": int((~keep[sub]["drop_verbatim"]).sum()),
            "n_dropped_containment": int((~keep[sub]["drop_containment"]).sum()),
        }

    # --- executor-added control: the matched RANDOM drop -----------------------
    # Dropping items moves an AUROC on its own, through sampling alone. The same
    # count of responses is dropped at random from the same subset, over 20
    # seeds, to show the spread a null drop produces. A DIAGNOSTIC, not a bar.
    null = {}
    for name, flagcol in (("verbatim", "exposed_verbatim"),
                          ("containment", "exposed_containment")):
        counts = {sub: int(tbl.filter((pl.col("subset") == sub) & pl.col(flagcol)).height)
                  for sub in subsets}
        deltas = []
        for seed in rng_seeds:
            rng = np.random.default_rng(seed)
            per = []
            for t in DRAWS:
                z = zs[t]
                accum = []
                for sub in subsets:
                    y, s = z[f"y__{sub}"], z[f"resp__{sub}"]
                    k = counts[sub]
                    mask = np.ones(len(y), dtype=bool)
                    if k:
                        mask[rng.choice(len(y), size=k, replace=False)] = False
                    accum.append(subset_auc(y[mask], s[mask]))
                per.append(float(np.mean(accum)))
            deltas.append(float(np.mean(per)) - k6["all"])
        null[name] = {
            "n_dropped_per_subset": counts,
            "seeds": len(rng_seeds),
            "mean_delta": round(float(np.mean(deltas)), 6),
            "sd_delta": round(float(np.std(deltas, ddof=1)), 6),
            "min_delta": round(float(np.min(deltas)), 6),
            "max_delta": round(float(np.max(deltas)), 6),
            "observed_delta": k6_delta[f"drop_{name}"],
        }
        print(f"  null-drop control ({name}): observed {null[name]['observed_delta']:+.6f} "
              f"vs random {null[name]['mean_delta']:+.6f} "
              f"+/- {null[name]['sd_delta']:.6f} "
              f"[{null[name]['min_delta']:+.6f}, {null[name]['max_delta']:+.6f}]",
              flush=True)

    payload = {
        "arm": "R21-H179Q q1 - is the arena headline inflated by contamination",
        "licence": ("MEASUREMENT ONLY - no bar, no promotion, no kill, no "
                    "adjudication"),
        "question": ("the arena headline is the UNWEIGHTED MEAN over ten subsets, "
                     "so hotpotqa carries 10% of it. What does the headline read "
                     "with the contamination-exposed responses removed?"),
        "read": ("the banked per-item response scores of R21-H179 stage 1 - the "
                 "shipped windowed decomposed-min read, AUROC unit = the RESPONSE. "
                 "NOTHING is re-scored here; the subsets are re-aggregated over a "
                 "subset of their own items"),
        "exposure_source": {
            "verbatim": ("arena_surface_report.json :: "
                         "verbatim_substring_proof.documents[*].doc_blake2b_64 - "
                         "the report's OWN 17 document ids, not re-derived; a "
                         "response is exposed when one of its retained "
                         "documents[:8] hashes into that set, the report's own "
                         "attribution rule"),
            "containment": (f"the report banks only COUNTS at containment >= "
                            f"{CONTAINMENT_THR:.2f}, so the per-document "
                            "containment array is read from the checkpointed "
                            "census the report was computed from, "
                            "tmp/arena_surface/census.npz :: best_c|documents, "
                            "indexed by the report's own sorted(set(documents)) "
                            "ordering"),
            "control": ("the rebuilt sets must reproduce the report's banked "
                        "per-subset exposed-response counts EXACTLY or the run "
                        "aborts"),
        },
        "exposure_control": control,
        "headline_reference": HEADLINE,
        "k6_mean": k6,
        "k6_delta_vs_all": k6_delta,
        "k6_vs_headline": {
            "banked_headline": HEADLINE,
            "recomputed_all": k6["all"],
            "recomputed_minus_headline": round(k6["all"] - HEADLINE, 6),
            "drop_verbatim": k6["drop_verbatim"],
            "drop_verbatim_minus_headline": round(k6["drop_verbatim"] - HEADLINE, 6),
            "drop_containment": k6["drop_containment"],
            "drop_containment_minus_headline": round(k6["drop_containment"] - HEADLINE, 6),
        },
        "per_subset_k6": sub_k6,
        "per_draw": per_draw,
        "null_drop_control": {
            "what": ("executor-added DIAGNOSTIC: the same number of responses "
                     "dropped at RANDOM from the same subsets, 20 seeds, so the "
                     "observed delta can be read against the spread a null drop "
                     "produces. Not a bar and not a verdict."),
            **null,
        },
        "artifacts": {"exposure_table": EXPOSURE_PARQUET.name},
        "note": NOTE,
        "written": time.strftime("%F %T"),
    }
    Q1_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nq1 -> {Q1_JSON.name}", flush=True)
    print(f"  6-draw arena mean   as banked      {k6['all']:.5f}  "
          f"(headline {HEADLINE:.5f}, delta {k6['all'] - HEADLINE:+.5f})", flush=True)
    print(f"  6-draw arena mean   -16 verbatim   {k6['drop_verbatim']:.5f}  "
          f"(delta vs headline {k6['drop_verbatim'] - HEADLINE:+.5f})", flush=True)
    print(f"  6-draw arena mean   -117 contain.  {k6['drop_containment']:.5f}  "
          f"(delta vs headline {k6['drop_containment'] - HEADLINE:+.5f})", flush=True)
    for sub in subsets:
        r = sub_k6[sub]
        print(f"    {sub:12s} n={r['n']:>4} all {r['all']:.5f}  "
              f"-verb({r['n_dropped_verbatim']:>3}) {r['delta_drop_verbatim']:+.5f}  "
              f"-cont({r['n_dropped_containment']:>3}) "
              f"{r['delta_drop_containment']:+.5f}", flush=True)


# --- ablate: the evidence-ablated arena read (GPU) ---------------------------------


PERM_MAX_TRIES = 1000


def permutations_for(subs, seed=ABLATION_SEED):
    """A fixed-seed derangement of the evidence, inside each subset.

    Item i is given item p[i]'s document list. Any permutation preserves the
    subset's evidence multiset exactly - register, length distribution, token
    distribution, the multiset of per-item document counts and the total window
    count are all unchanged - so the permutation destroys the claim-evidence
    relation and nothing else.

    The rejection is on TEXT, not on the index. A derangement alone only
    guarantees no item keeps its own POSITION; a subset holding two responses
    with identical document lists could still hand an item its own evidence
    back. A candidate is accepted only when ZERO items retain their original
    evidence text, which is the control this arm is asserted on.
    """
    rng = np.random.default_rng(seed)
    perms, tries = {}, {}
    for sub, (_claims, chunks, _y) in subs.items():
        n = len(chunks)
        for k in range(1, PERM_MAX_TRIES + 1):
            p = rng.permutation(n)
            if not any(chunks[i] == chunks[p[i]] for i in range(n)):
                perms[sub], tries[sub] = p, k
                break
        else:
            raise SystemExit(
                f"PERMUTATION ABORT ({sub}): no evidence-distinct permutation "
                f"found in {PERM_MAX_TRIES} draws - some document list occupies "
                "too large a share of the subset for the ablation to be clean")
    return perms, tries


def retained_evidence(subs, abl):
    """Items that still hold their ORIGINAL evidence text after the shuffle."""
    return {s: int(sum(1 for i in range(len(subs[s][1]))
                       if subs[s][1][i] == abl[s][1][i])) for s in abl}


def stage_ablate(tag):
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"GPU PLACEMENT ABORT: {torch.cuda.device_count()} visible devices - "
            "bind exactly one card")
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})  free "
          f"{free / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    if free < 8e9:
        raise SystemExit(
            f"CARD-BUSY ABORT: only {free / 1e9:.1f} GB free on the bound card")

    out_npz = ablated_npz(tag)
    if out_npz.exists() and out_npz.stat().st_size > 0:
        print(f"  SKIP ablate {tag} (on disk: {out_npz.name})", flush=True)
        return

    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    H92 = reads.H92
    true_subs = reads.ARENA.load_subsets()
    perms, tries = permutations_for(true_subs)
    abl = {s: (c, [k[i] for i in perms[s]], y) for s, (c, k, y) in true_subs.items()}
    reads.ARENA.load_subsets = lambda: abl          # the read sees only this
    owners = {s: S.owners_for(H92, abl[s][0]) for s in abl}

    perm_sha = hashlib.sha256(
        b"".join(perms[s].astype(np.int64).tobytes() for s in sorted(perms))
    ).hexdigest()[:16]
    # THE ABLATION CONTROL: zero items may retain their original evidence. The
    # permutation is rejected on TEXT, not on index, so this is an assertion the
    # run dies on rather than a number to be read later.
    retained = retained_evidence(true_subs, abl)
    n_items = sum(len(v[2]) for v in abl.values())
    if sum(retained.values()) != 0:
        raise SystemExit(
            f"ABLATION CONTROL ABORT: {sum(retained.values())} of {n_items} items "
            f"retain their original evidence after the shuffle - {retained}")
    print(f"evidence permuted inside each subset, seed {ABLATION_SEED}, sha "
          f"{perm_sha}; ABLATION CONTROL PASS: 0/{n_items} items retain their "
          f"original evidence (draws per subset: {tries})", flush=True)

    spec = DRAWS[tag]
    print(f"\n--- {tag} {spec['ckpt']} ABLATED windowed arena read  "
          f"{time.strftime('%F %T')} ---", flush=True)
    S._CAPTURE.clear()
    reads.ARM.RUNS["twin"]["ckpt"] = spec["ckpt"]
    reads.out_path = lambda run, mode: ablated_read_json(tag)
    orig = reads.ARM.score_sets
    reads.ARM.score_sets = S._wrap_score_sets(orig)
    argv = sys.argv
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    try:
        reads.main()
    finally:
        sys.argv = argv
        reads.ARM.score_sets = orig
    torch.cuda.empty_cache()
    cap = dict(S._CAPTURE)
    resp = {s: np.array([cap[s]["s_sent"][owners[s] == i].min()
                         for i in range(len(abl[s][2]))]) for s in abl}

    # --- harness control: the per-item scores must reproduce the read's own
    # per-subset AUROC. There is no banked aggregate for an ablated read, so the
    # control here is self-consistency of the capture, not agreement with a bank.
    read_out = json.loads(ablated_read_json(tag).read_text())
    per_sub, worst = {}, 0.0
    for s in abl:
        y = abl[s][2]
        b = read_out["per_subset"][s]
        auc, _, _ = reads.M59.auc_and_f1(y, resp[s])
        d = abs(auc - b["auc"])
        worst = max(worst, d)
        fp_got = {"n": len(y), "n_sent": cap[s]["n_sets"], "n_pairs": cap[s]["n_pairs"]}
        fp_want = {"n": b["n"], "n_sent": b["n_sent"], "n_pairs": b["n_pairs"]}
        per_sub[s] = {
            "ablated_auc": b["auc"], "recomputed_auc": round(float(auc), 6),
            "abs_delta": round(float(d), 8), "fingerprint": fp_got,
            "read_fingerprint": fp_want, "fingerprint_ok": fp_got == fp_want,
            "passes": bool(d <= FIDELITY_TOL and fp_got == fp_want),
        }
        print(f"    {s:12s} ablated {b['auc']:.4f}  from-items {auc:.6f}  "
              f"|d| {d:.2e}  {'PASS' if per_sub[s]['passes'] else 'FAIL'}", flush=True)
    if not all(v["passes"] for v in per_sub.values()):
        raise SystemExit(f"ABLATION HARNESS ABORT ({tag}): worst |delta| {worst:.2e}")

    payload = {tag: np.array([0])}
    for s in abl:
        payload[f"resp__{s}"] = resp[s].astype(np.float64)
        payload[f"sent__{s}"] = cap[s]["s_sent"].astype(np.float64)
        payload[f"owner__{s}"] = owners[s]
        payload[f"y__{s}"] = abl[s][2].astype(np.int64)
        payload[f"perm__{s}"] = perms[s].astype(np.int64)
        payload[f"npairs__{s}"] = np.array([cap[s]["n_pairs"]], dtype=np.int64)
    np.savez_compressed(out_npz, **payload)

    ablated_json(tag).write_text(json.dumps({
        "arm": "R21-H179Q ablate - the arena read with the evidence permuted",
        "licence": "MEASUREMENT ONLY - nothing is adjudicated here",
        "draw": tag, "checkpoint": spec["ckpt"], "label": spec["label"],
        "ablation": ("the evidence document list is permuted ACROSS ITEMS inside "
                     "each subset by a fixed-seed derangement; claims, sentences, "
                     "labels and the read itself are untouched, so every surface "
                     "statistic of the evidence side is preserved and only the "
                     "claim-evidence relation is destroyed"),
        "seed": ABLATION_SEED, "permutation_sha256_16": perm_sha,
        "permutation_banked_in": out_npz.name,
        "ablation_control": {
            "assertion": ("ZERO items retain their original evidence text after "
                          "the shuffle - the run aborts otherwise"),
            "items_retaining_original_evidence": int(sum(retained.values())),
            "items_retaining_original_evidence_per_subset": retained,
            "n_items": int(n_items),
            "rejection_is_on_text_not_index": (
                "a derangement alone only guarantees no item keeps its own "
                "POSITION; a subset holding two responses with identical "
                "document lists could still hand an item its own evidence back, "
                "so candidate permutations are rejected on the evidence TEXT"),
            "draws_until_accepted_per_subset": tries,
            "verdict": "PASS",
        },
        "ablated_mean": read_out["mean"],
        "ablated_per_subset": {s: read_out["per_subset"][s]["auc"] for s in abl},
        "harness_control": {"tolerance": FIDELITY_TOL,
                            "worst_abs_delta": round(float(worst), 8),
                            "per_subset": per_sub, "verdict": "PASS"},
        "note": NOTE, "written": time.strftime("%F %T"),
    }, indent=2))
    print(f"\n  ablated per-item scores -> {out_npz.name} "
          f"({out_npz.stat().st_size / 1e6:.1f} MB)  mean {read_out['mean']:.5f}",
          flush=True)


# --- q2: the ablation roll-up ------------------------------------------------------


def stage_q2():
    true_z = {t: np.load(S.npz_path(t), allow_pickle=False) for t in DRAWS}
    abl_z = {t: np.load(ablated_npz(t), allow_pickle=False) for t in DRAWS}
    subsets = sorted({k.split("__")[1] for k in true_z["d1"].files if k.startswith("y__")})

    per_draw = {}
    for t in DRAWS:
        rows = {}
        for sub in subsets:
            y = true_z[t][f"y__{sub}"]
            if not np.array_equal(abl_z[t][f"y__{sub}"], y):
                raise SystemExit(f"LABEL ABORT {t}/{sub}: ablated labels differ")
            rows[sub] = {
                "true": round(subset_auc(y, true_z[t][f"resp__{sub}"]), 6),
                "ablated": round(subset_auc(y, abl_z[t][f"resp__{sub}"]), 6),
            }
            rows[sub]["delta"] = round(rows[sub]["ablated"] - rows[sub]["true"], 6)
        per_draw[t] = {
            "checkpoint": DRAWS[t]["ckpt"], "per_subset": rows,
            "mean_true": round(float(np.mean([rows[s]["true"] for s in subsets])), 6),
            "mean_ablated": round(float(np.mean([rows[s]["ablated"] for s in subsets])), 6),
        }
        per_draw[t]["mean_delta"] = round(
            per_draw[t]["mean_ablated"] - per_draw[t]["mean_true"], 6)
        print(f"  {t} {DRAWS[t]['ckpt']:22s} true {per_draw[t]['mean_true']:.5f}  "
              f"ablated {per_draw[t]['mean_ablated']:.5f}  "
              f"delta {per_draw[t]['mean_delta']:+.5f}", flush=True)

    sub_k6 = {}
    for sub in subsets:
        tr = float(np.mean([per_draw[t]["per_subset"][sub]["true"] for t in DRAWS]))
        ab = float(np.mean([per_draw[t]["per_subset"][sub]["ablated"] for t in DRAWS]))
        sub_k6[sub] = {"true_k6": round(tr, 6), "ablated_k6": round(ab, 6),
                       "delta": round(ab - tr, 6)}

    k6_true = round(float(np.mean([per_draw[t]["mean_true"] for t in DRAWS])), 6)
    k6_abl = round(float(np.mean([per_draw[t]["mean_ablated"] for t in DRAWS])), 6)

    payload = {
        "arm": "R21-H179Q q2 - does the model actually use the evidence",
        "licence": ("MEASUREMENT ONLY - the branches on this number were fixed "
                    "before it existed and are the coordinator's; this arm "
                    "produces the number and nothing else"),
        "ablation": ("the SAME six checkpoints and the SAME shipped windowed "
                     "decomposed-min read, with the evidence permuted across "
                     "items inside each subset by a fixed-seed derangement"),
        "seed": ABLATION_SEED,
        "context_not_a_comparator": (
            "a claim-only probe that never reads evidence scores subset-mean "
            "0.5683 on this arena (R20_claimonly_sweep.json, fit_on_whole_mix). "
            "That number says a shortcut is AVAILABLE; this arm measures whether "
            "this model TAKES it. The two are different instruments and are not "
            "differenced here."),
        "k6_mean_true": k6_true,
        "k6_mean_ablated": k6_abl,
        "k6_mean_delta": round(k6_abl - k6_true, 6),
        "headline_reference": HEADLINE,
        "per_subset_k6": sub_k6,
        "per_draw": per_draw,
        "artifacts": {"ablated_scores": [ablated_npz(t).name for t in DRAWS],
                      "ablated_records": [ablated_json(t).name for t in DRAWS]},
        "note": NOTE,
        "written": time.strftime("%F %T"),
    }
    Q2_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nq2 -> {Q2_JSON.name}", flush=True)
    print(f"  6-draw arena mean  true evidence     {k6_true:.5f}", flush=True)
    print(f"  6-draw arena mean  ablated evidence  {k6_abl:.5f}  "
          f"(delta {k6_abl - k6_true:+.5f})", flush=True)
    for sub in subsets:
        r = sub_k6[sub]
        print(f"    {sub:12s} true {r['true_k6']:.5f}  ablated {r['ablated_k6']:.5f}  "
              f"delta {r['delta']:+.5f}", flush=True)


# --- q3: the exposure column for the autopsy ---------------------------------------


def stage_q3():
    tbl, control = exposure_table()
    df = pl.read_parquet(S.CONSENSUS_PARQUET)
    before = df.height
    df = df.drop([c for c in ("exposed_verbatim", "exposed_containment")
                  if c in df.columns])
    # `row_id` is the RAGBench SOURCE row and is not unique - 292 of them supply
    # two arena responses each. `item` is the arena's own per-response index and
    # is unique within a subset, so it is the only safe join key.
    df = df.join(tbl.select("subset", "item", "exposed_verbatim",
                            "exposed_containment"),
                 on=("subset", "item"), how="left")
    if df.height != before or df["exposed_verbatim"].null_count():
        raise SystemExit(
            f"JOIN ABORT: {before} consensus rows -> {df.height}, "
            f"{df['exposed_verbatim'].null_count()} unmatched")
    df = df.sort(("subset", "item"))
    df.write_parquet(S.CONSENSUS_PARQUET)
    print(f"  exposure columns added to {S.CONSENSUS_PARQUET.name} "
          f"({df.height} rows, no row added or lost)", flush=True)

    def block(flag):
        out = {}
        for sub in sorted(df["subset"].unique().to_list()):
            d = df.filter(pl.col("subset") == sub)
            ce = d.filter(pl.col("consensus_error_threshold"))
            ce_r = d.filter(pl.col("consensus_error_rank"))
            top = d.filter(pl.col("top40"))
            n_exp = int(d[flag].sum())
            out[sub] = {
                "n": d.height, "n_exposed": n_exp,
                "exposed_share_of_subset": round(n_exp / max(d.height, 1), 4),
                "consensus_errors_threshold": ce.height,
                "consensus_errors_threshold_exposed": int(ce[flag].sum()),
                "consensus_errors_rank": ce_r.height,
                "consensus_errors_rank_exposed": int(ce_r[flag].sum()),
                "top40_exposed": int(top[flag].sum()),
                # both sides of the subset's total deficit mass; the same
                # denominator R21-H179's own `share_of_total_mass` uses
                "deficit_mass_share_carried_by_exposed": round(float(
                    d.filter(pl.col(flag))["deficit_contrib_mean"].sum()
                    / max(float(d["deficit_contrib_mean"].sum()), 1e-12)), 4),
            }
        tot = {
            "n": df.height, "n_exposed": int(df[flag].sum()),
            "consensus_errors_threshold": int(df["consensus_error_threshold"].sum()),
            "consensus_errors_threshold_exposed": int(
                df.filter(pl.col("consensus_error_threshold"))[flag].sum()),
            "consensus_errors_rank": int(df["consensus_error_rank"].sum()),
            "consensus_errors_rank_exposed": int(
                df.filter(pl.col("consensus_error_rank"))[flag].sum()),
            "top40_marked": int(df["top40"].sum()),
            "top40_exposed": int(df.filter(pl.col("top40"))[flag].sum()),
        }
        return {"per_subset": out, "totals": tot}

    payload = {
        "arm": ("R21-H179Q q3 - the contamination-exposure column for the "
                "blind-arena failure-mode autopsy"),
        "licence": ("MEASUREMENT ONLY - no annotation, no classification, no "
                    "adjudication; later stages own those"),
        "why": ("so the autopsy's error classes cannot be confounded by the "
                "contamination q1 measures: every consensus-error item now "
                "carries whether it retrieved a contamination-exposed document"),
        "column_added_to": S.CONSENSUS_PARQUET.name,
        "columns": ["exposed_verbatim", "exposed_containment"],
        "exposure_control": control,
        "verbatim": block("exposed_verbatim"),
        "containment": block("exposed_containment"),
        "note": NOTE,
        "written": time.strftime("%F %T"),
    }
    Q3_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nq3 -> {Q3_JSON.name}", flush=True)
    for name in ("verbatim", "containment"):
        t = payload[name]["totals"]
        print(f"  {name:12s} exposed {t['n_exposed']:>4}/{t['n']}  "
              f"consensus-error(threshold) {t['consensus_errors_threshold_exposed']:>3}"
              f"/{t['consensus_errors_threshold']}  "
              f"consensus-error(rank) {t['consensus_errors_rank_exposed']:>3}"
              f"/{t['consensus_errors_rank']}  "
              f"top40 {t['top40_exposed']:>3}/{t['top40_marked']}", flush=True)


# --- driver ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("q1", "ablate", "q2", "q3"))
    ap.add_argument("--draw", default=None, choices=tuple(DRAWS))
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== R21-H179Q {args.stage}  {time.strftime('%F %T')} ===", flush=True)
    if args.stage == "ablate":
        if not args.draw:
            raise SystemExit("--stage ablate needs --draw (one checkpoint per process)")
        stage_ablate(args.draw)
    elif args.stage == "q1":
        stage_q1()
    elif args.stage == "q2":
        stage_q2()
    else:
        stage_q3()
    print(f"=== R21-H179Q {args.stage} DONE ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
