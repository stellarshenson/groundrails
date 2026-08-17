"""R21-H179 STAGE 1 - blind-arena PER-ITEM scoring over the six flagship draws.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"Round 21 - arena failure-mode autopsy / R21-H179 BLIND-ARENA ERROR AUTOPSY".
MEASUREMENT ONLY: nothing here trains, tunes, selects or adjudicates.

The campaign banks only per-SUBSET aggregates for the blind arena. This stage
closes that gap: it re-runs the shipped windowed decomposed-min read over all
ten RAGBench subsets for each of the six banked flagship endpoints and banks the
PER-ITEM scores the arena AUROC is actually computed over.

    THE AUROC UNIT IS THE RESPONSE. `R16-H142_G1_reads.main` scores each H92
    sentence as max over the item's 1,500/750 windows, takes the MIN over the
    item's sentences to get one score per response, and calls
    `M59.auc_and_f1(y, resp)` with y the response-level `adherence_score`.
    n = 2,264 responses over ten subsets. Per-sentence scores (8,277 rows) are
    banked alongside as the sub-unit the response score is a min over; they are
    NOT the AUROC unit.

THE READ IS THE BANKED ONE, REUSED UNCHANGED. `R16-H142_G1_reads.py` is loaded
and its `main()` is called with only the checkpoint binding and the output path
rebound - the R20-H173 soup-read pattern verbatim. Per-item scores are obtained
by WRAPPING `ARM.score_sets`: the wrapper calls the banked function, keeps its
return value, and passes it through untouched. The read arithmetic is therefore
byte-identical to every banked arena read; no aggregation is re-implemented.

FIDELITY CONTROL, per checkpoint, per subset, before anything is banked:

  * structural fingerprint (n, n_sent, n_pairs) must equal the banked
    `*_windowed_result.json` block exactly
  * the AUROC recomputed from the banked per-item response scores must
    reproduce the banked per-subset AUROC to <= 1e-4

A failure ABORTS the run and banks the failure record instead of the scores -
per-item numbers that do not reproduce the campaign's verdict surface would make
the whole autopsy unreadable.

Stages (each idempotent - an artifact on disk is not rebuilt):

    items       CPU. R21-H179_arena_items.parquet - the item keys and source
                text (RAGBench id, question, response, H92 sentences, the
                retained documents[:8], the sentence-level annotation block)
    score       GPU. R21-H179_arena_scores_<checkpoint>.npz per checkpoint,
                plus R21-H179_arena_scores_fidelity.json
    consensus   CPU. per-draw agreement, consensus errors, rank-loss mass ->
                R21-H179_consensus_errors.{json,parquet}

The `items` stage imports `R12_label_ceiling.py`, which sets
CUDA_VISIBLE_DEVICES="" at import; it therefore runs in its OWN process, before
any GPU work, and never inside the score stage.

Run (detached, GPU0 only, and ONLY once the R20-H174 draw-2 job has left it):
    GPU=0 nohup setsid bash experiments/grounding-semantic/R21-H179_campaign.sh \
        2>&1 | tee logs/R21-H179_arena_scores.log &
"""

import os

# `R16-H142_G1_arm.py` defaults CUDA_VISIBLE_DEVICES to "1" at import, and GPU1
# carries an R20-H174 training draw. The placement is explicit or the run dies.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if "GPU" not in os.environ:
        raise SystemExit(
            "GPU PLACEMENT ABORT: set CUDA_VISIBLE_DEVICES or GPU explicitly "
            "(0 for this arm, empty for the CPU stages) - GPUs 1 and 2 carry "
            "R20-H174 training draws and must not be touched")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import ast
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

ITEMS_PARQUET = HERE / "R21-H179_arena_items.parquet"
FIDELITY_JSON = HERE / "R21-H179_arena_scores_fidelity.json"
CONSENSUS_JSON = HERE / "R21-H179_consensus_errors.json"
CONSENSUS_PARQUET = HERE / "R21-H179_consensus_errors.parquet"

FIDELITY_TOL = 1e-4
TOP_N_PER_SUBSET = 40
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

# The six banked flagship endpoints. `banked` is the per-draw windowed arena
# result this arm's per-item scores must reproduce.
DRAWS = {
    "d1": {"ckpt": "R18-H150-arm-draw1",
           "banked": "R18-H150_arm_draw1_windowed_result.json",
           "label": "H150 draw 1 (seed 1150, monolithic executor)"},
    "d2": {"ckpt": "R18-H150-arm-draw2",
           "banked": "R18-H150_arm_draw2_windowed_result.json",
           "label": "H150 draw 2 (seed 2150, monolithic executor)"},
    "d3": {"ckpt": "R19-H160-arm-draw3",
           "banked": "R19-H160_arm_draw3_windowed_result.json",
           "label": "H160 draw 3 (seed 3150, split-cotangent executor)"},
    "d4": {"ckpt": "R19-H160-arm-draw4",
           "banked": "R19-H160_arm_draw4_windowed_result.json",
           "label": "H160 draw 4 (seed 4150, split-cotangent executor)"},
    "d5": {"ckpt": "R20-H172-arm-draw5",
           "banked": "R20-H172_arm_draw5_windowed_result.json",
           "label": "H172 draw 5 (seed 5150, split-cotangent executor)"},
    "d6": {"ckpt": "R20-H172-arm-draw6",
           "banked": "R20-H172_arm_draw6_windowed_result.json",
           "label": "H172 draw 6 (seed 6150, split-cotangent executor)"},
}

# Faithful-oracle ceiling per subset: `o4_windows_strict_auc` in
# R12_label_ceiling_result.json - the AUROC a PERFECT entailer reaches through
# this read's own machinery (H92 splitter, documents[:8] cap, 1500/750 windows).
# Reported SEPARATELY from error mass, as the registration requires.
CEILING_SOURCE = "R12_label_ceiling_result.json"
CEILING_FIELD = "o4_windows_strict_auc"

H157_SRC = HERE / "R18-H157_finqa_autopsy.py"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def npz_path(tag):
    return HERE / f"R21-H179_arena_scores_{DRAWS[tag]['ckpt']}.npz"


def read_json_path(tag):
    return HERE / f"R21-H179_read_{DRAWS[tag]['ckpt']}_windowed_result.json"


def banked(tag):
    return json.loads((HERE / DRAWS[tag]["banked"]).read_text())


def fidelity_shard(tag):
    return HERE / f"R21-H179_arena_scores_fidelity_{tag}.json"


FIDELITY_CONTROL = ("per-item response scores must reproduce the banked "
                    "per-subset arena AUROC and the banked read fingerprint")


def write_fidelity(path, draws):
    path.write_text(json.dumps(
        {"control": FIDELITY_CONTROL, "tolerance": FIDELITY_TOL, "draws": draws,
         "note": NOTE, "written": time.strftime("%F %T")}, indent=2))


def load_fidelity():
    """The fidelity record, merged over per-draw shards.

    The score stage is fanned out one checkpoint per process when more than one
    card is free; each process writes its OWN shard, so no two processes ever
    write the same artifact. FIDELITY_JSON is the merge, written once by the
    consensus stage (or directly by an unsharded score stage).
    """
    draws = {}
    if FIDELITY_JSON.exists() and FIDELITY_JSON.stat().st_size > 0:
        draws.update(json.loads(FIDELITY_JSON.read_text()).get("draws", {}))
    for t in DRAWS:
        p = fidelity_shard(t)
        if p.exists() and p.stat().st_size > 0:
            draws.update(json.loads(p.read_text()).get("draws", {}))
    return draws


class FidelityAbort(Exception):
    """Per-item scores that do not reproduce the banked arena numbers."""


# --- stage: items (CPU) ------------------------------------------------------------


def stage_items():
    """The item keys and source text, one row per arena response.

    Row order is the read's own: `R8-H77.load_subsets` filter + sample(seed=0).
    `R12_label_ceiling.load_rows` applies the identical filter and sample and
    keeps the annotation columns, so the two align row for row - asserted here
    on the response text of all ten subsets, not just the one H157 checked.
    """
    if ITEMS_PARQUET.exists() and ITEMS_PARQUET.stat().st_size > 0:
        print(f"  SKIP items (on disk: {ITEMS_PARQUET.name})", flush=True)
        return
    H92 = _mod("h92", "R8-H92_decomposed_arena.py")
    ARENA = H92.ARENA
    R12 = _mod("r12", "R12_label_ceiling.py")  # sets CUDA_VISIBLE_DEVICES=""

    subs = ARENA.load_subsets()
    raw = R12.load_rows()
    rows = []
    for sub, (claims, chunks, y) in subs.items():
        r = raw[sub]
        assert list(r["response"]) == list(claims), f"{sub}: row order disagrees"
        assert np.array_equal(r["adherence_score"].cast(pl.Int8).to_numpy(), y), \
            f"{sub}: labels disagree"
        for i, rec in enumerate(r.iter_rows(named=True)):
            sents = H92.sentences(claims[i])
            rows.append({
                "subset": sub,
                "item": i,
                "row_id": str(rec["id"]),
                "label": int(y[i]),
                "question": rec["question"],
                "response": claims[i],
                "n_sent": len(sents),
                "sentences": sents,
                "n_docs": len(chunks[i]),
                "documents": list(chunks[i]),
                "annotation_json": json.dumps({
                    "response_sentences": rec["response_sentences"],
                    "sentence_support_information": rec["sentence_support_information"],
                    "unsupported_response_sentence_keys":
                        rec["unsupported_response_sentence_keys"],
                }, default=str),
            })
    df = pl.DataFrame(rows)
    df.write_parquet(ITEMS_PARQUET)
    print(f"  items -> {ITEMS_PARQUET.name}  {df.shape}  "
          f"({ITEMS_PARQUET.stat().st_size / 1e6:.1f} MB)", flush=True)


# --- stage: score (GPU) ------------------------------------------------------------


_CAPTURE = {}


def _wrap_score_sets(orig):
    """Keep the banked reader's own return value; pass it through untouched."""
    def wrapped(model, tok, flat_s, flat_w, set_index, n_sets, batch=64, tag=""):
        s = orig(model, tok, flat_s, flat_w, set_index, n_sets, batch=batch, tag=tag)
        _CAPTURE[tag.split("/")[-1]] = {
            "s_sent": np.asarray(s, dtype=np.float64),
            "n_sets": int(n_sets), "n_pairs": len(flat_s),
        }
        return s
    return wrapped


def owners_for(H92, claims):
    """sentence -> owning response, the reader's own construction."""
    owner = []
    for i, c in enumerate(claims):
        owner.extend([i] * len(H92.sentences(c)))
    return np.array(owner, dtype=np.int64)


def score_checkpoint(tag, reads, subs, owners, fidelity, out_json):
    """One checkpoint: the banked read, then the per-item bank + fidelity."""
    out_npz = npz_path(tag)
    spec = DRAWS[tag]
    bank = banked(tag)
    if out_npz.exists() and out_npz.stat().st_size > 0:
        print(f"\n--- SKIP {tag} {spec['ckpt']} (on disk: {out_npz.name}) ---",
              flush=True)
        z = np.load(out_npz, allow_pickle=False)
        cap = {s: {"s_sent": z[f"sent__{s}"],
                   "n_sets": int(z[f"sent__{s}"].shape[0]),
                   "n_pairs": int(z[f"npairs__{s}"][0])} for s in subs}
        resp = {s: z[f"resp__{s}"] for s in subs}
    else:
        print(f"\n--- {tag} {spec['ckpt']} windowed arena read  "
              f"{time.strftime('%F %T')} ---", flush=True)
        _CAPTURE.clear()
        reads.ARM.RUNS["twin"]["ckpt"] = spec["ckpt"]
        reads.out_path = lambda run, mode: read_json_path(tag)
        orig = reads.ARM.score_sets
        reads.ARM.score_sets = _wrap_score_sets(orig)
        argv = sys.argv
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        try:
            reads.main()
        finally:
            sys.argv = argv
            reads.ARM.score_sets = orig
        import torch
        torch.cuda.empty_cache()
        cap = dict(_CAPTURE)
        resp = {s: np.array([cap[s]["s_sent"][owners[s] == i].min()
                             for i in range(len(subs[s][2]))])
                for s in subs}

    # --- fidelity control ---------------------------------------------------
    M59 = reads.M59
    per_sub, worst = {}, 0.0
    for s in subs:
        y = subs[s][2]
        b = bank["per_subset"][s]
        fp_got = {"n": len(y), "n_sent": cap[s]["n_sets"], "n_pairs": cap[s]["n_pairs"]}
        fp_want = {"n": b["n"], "n_sent": b["n_sent"], "n_pairs": b["n_pairs"]}
        auc, _, _ = M59.auc_and_f1(y, resp[s])
        d = abs(auc - b["auc"])
        worst = max(worst, d)
        per_sub[s] = {
            "banked_auc": b["auc"], "recomputed_auc": round(float(auc), 6),
            "abs_delta": round(float(d), 8),
            "fingerprint": fp_got, "banked_fingerprint": fp_want,
            "fingerprint_ok": fp_got == fp_want,
            "passes": bool(d <= FIDELITY_TOL and fp_got == fp_want),
        }
        print(f"    {s:12s} banked {b['auc']:.4f}  from-items "
              f"{auc:.6f}  |d| {d:.2e}  "
              f"{'PASS' if per_sub[s]['passes'] else 'FAIL'}", flush=True)

    rec = {
        "checkpoint": spec["ckpt"], "label": spec["label"],
        "banked_source": spec["banked"], "banked_mean": bank["mean"],
        "recomputed_mean": round(float(np.mean(
            [v["recomputed_auc"] for v in per_sub.values()])), 6),
        "tolerance": FIDELITY_TOL, "worst_abs_delta": round(float(worst), 8),
        "per_subset": per_sub,
        "verdict": "PASS" if all(v["passes"] for v in per_sub.values()) else "FAIL",
    }
    fidelity[tag] = rec
    write_fidelity(out_json, fidelity)
    if rec["verdict"] != "PASS":
        raise FidelityAbort(
            f"{tag} ({spec['ckpt']}): worst |delta| {worst:.2e} > {FIDELITY_TOL} "
            "or fingerprint mismatch - per-item scores are NOT the quantity the "
            "campaign's verdicts rest on and are not banked")

    if not (out_npz.exists() and out_npz.stat().st_size > 0):
        payload = {}
        for s in subs:
            payload[f"resp__{s}"] = resp[s].astype(np.float64)
            payload[f"sent__{s}"] = cap[s]["s_sent"].astype(np.float64)
            payload[f"owner__{s}"] = owners[s]
            payload[f"y__{s}"] = subs[s][2].astype(np.int64)
            payload[f"npairs__{s}"] = np.array([cap[s]["n_pairs"]], dtype=np.int64)
        np.savez_compressed(out_npz, **payload)
        print(f"  per-item scores -> {out_npz.name} "
              f"({out_npz.stat().st_size / 1e6:.1f} MB)  "
              f"mean {rec['recomputed_mean']:.5f} (banked {bank['mean']:.5f})",
              flush=True)
    return rec


def stage_score(only=None):
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"GPU PLACEMENT ABORT: {torch.cuda.device_count()} visible devices - "
            "bind exactly one card; the other two carry R20-H174 training draws")
    dev = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})  free "
          f"{free / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    if free < 8e9:
        raise SystemExit(
            f"CARD-BUSY ABORT: only {free / 1e9:.1f} GB free on the bound card - "
            "it is not idle, and this arm must not slow a training draw")

    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    H92 = reads.H92
    subs = reads.ARENA.load_subsets()
    owners = {s: owners_for(H92, subs[s][0]) for s in subs}
    print(f"arena: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} "
          f"responses, {sum(len(o) for o in owners.values())} sentences", flush=True)

    tags = [only] if only else list(DRAWS)
    out_json = fidelity_shard(only) if only else FIDELITY_JSON
    fidelity = {} if only else load_fidelity()
    for tag in tags:
        score_checkpoint(tag, reads, subs, owners, fidelity, out_json)
    print(f"\nfidelity -> {out_json.name}  "
          f"{ {t: fidelity[t]['verdict'] for t in tags} }", flush=True)


# --- stage: consensus (CPU) --------------------------------------------------------


def _verbatim(name):
    """Lift a function from R18-H157_finqa_autopsy.py VERBATIM, without paying
    that module's import chain. The source segment and its digest are banked so
    the reuse is auditable."""
    src = H157_SRC.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            from sklearn.metrics import f1_score
            ns = {"np": np, "f1_score": f1_score}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(H157_SRC), "exec"), ns)
            return ns[name], seg
    raise SystemExit(f"{name} not found in {H157_SRC.name}")


def stage_consensus():
    from sklearn.metrics import roc_auc_score

    rank_loss, rl_src = _verbatim("rank_loss")
    op_threshold, thr_src = _verbatim("op_threshold")
    prov = {
        "source_file": H157_SRC.name,
        "rank_loss_sha256": hashlib.sha256(rl_src.encode()).hexdigest()[:16],
        "op_threshold_sha256": hashlib.sha256(thr_src.encode()).hexdigest()[:16],
        "rank_loss_source": rl_src,
        "op_threshold_source": thr_src,
        "reuse": ("R18-H157's rank-loss and operating-threshold formulations "
                  "lifted verbatim from the banked autopsy, not re-invented"),
    }

    items = pl.read_parquet(ITEMS_PARQUET)
    zs = {t: np.load(npz_path(t), allow_pickle=False) for t in DRAWS}
    fid_draws = load_fidelity()
    write_fidelity(FIDELITY_JSON, fid_draws)
    ceil_raw = json.loads((HERE / CEILING_SOURCE).read_text())["per_subset"]

    subsets = sorted({k.split("__")[1] for k in zs["d1"].files if k.startswith("y__")})
    tags = list(DRAWS)

    rows, per_subset, headroom = [], {}, {}
    for sub in subsets:
        y = zs["d1"][f"y__{sub}"]
        for t in tags:
            assert np.array_equal(zs[t][f"y__{sub}"], y), f"{sub}/{t}: labels differ"
        n = len(y)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        pairs = n_pos * n_neg

        s = {t: zs[t][f"resp__{sub}"] for t in tags}
        auc = {t: float(roc_auc_score(y, s[t])) for t in tags}
        rl = {t: rank_loss(y, s[t]) for t in tags}
        # H157's rank_loss is the per-item SHARE of the misordered pairs,
        # normalised so every item's shares sum to 1 (0.5 on each side by
        # construction). The AUROC deficit each item carries is therefore
        # rl_i * 2 * (1 - AUC): summed over one side it is exactly (1 - AUC),
        # which is the standard 1/(n_pos*n_neg)-per-mis-ranked-pair decomposition.
        dc = {t: rl[t] * 2.0 * (1.0 - auc[t]) for t in tags}
        for t in tags:
            for m in (y == 1, y == 0):
                assert abs(dc[t][m].sum() - (1.0 - auc[t])) < 1e-6, "deficit mismatch"

        # raw misordered-pair count per item (ties 0.5), for reporting
        raw = {t: dc[t] * pairs for t in tags}

        # (1) threshold errors - the H157 precedent, in-sample macro-F1 optimum
        thr = {t: op_threshold(y, s[t]) for t in tags}
        err_thr = {t: ((s[t] >= thr[t]).astype(int) != y) for t in tags}
        n_wrong_thr = np.sum([err_thr[t] for t in tags], axis=0)

        # (2) threshold-free auxiliary - the item sits on the wrong side of the
        # opposite class's median: a positive outranked by > half the negatives,
        # a negative outranking > half the positives
        err_rank = {t: np.where(y == 1, raw[t] > 0.5 * n_neg, raw[t] > 0.5 * n_pos)
                    for t in tags}
        n_wrong_rank = np.sum([err_rank[t] for t in tags], axis=0)

        rl_mean = np.mean([rl[t] for t in tags], axis=0)
        dc_mean = np.mean([dc[t] for t in tags], axis=0)
        auc_mean = float(np.mean([auc[t] for t in tags]))
        deficit_mean = 1.0 - auc_mean

        order = np.argsort(-dc_mean)
        rank_in_sub = np.empty(n, dtype=np.int64)
        rank_in_sub[order] = np.arange(1, n + 1)
        top = rank_in_sub <= TOP_N_PER_SUBSET

        sub_items = items.filter(pl.col("subset") == sub).sort("item")
        assert len(sub_items) == n, f"{sub}: items parquet length {len(sub_items)} != {n}"
        row_id = sub_items["row_id"].to_list()
        for i in range(n):
            r = {"subset": sub, "item": i, "row_id": row_id[i], "label": int(y[i]),
                 "side": "fn" if y[i] == 1 else "fp",
                 "rank_loss_mean": float(rl_mean[i]),
                 "deficit_contrib_mean": float(dc_mean[i]),
                 "side_mass_share_mean": float(dc_mean[i] / max(deficit_mean, 1e-12)),
                 "n_draws_wrong_threshold": int(n_wrong_thr[i]),
                 "n_draws_wrong_rank": int(n_wrong_rank[i]),
                 "consensus_error_threshold": bool(n_wrong_thr[i] == len(tags)),
                 "consensus_error_rank": bool(n_wrong_rank[i] == len(tags)),
                 "rank_in_subset": int(rank_in_sub[i]),
                 "top40": bool(top[i])}
            for t in tags:
                r[f"score_{t}"] = float(s[t][i])
                r[f"rank_loss_{t}"] = float(rl[t][i])
                r[f"deficit_contrib_{t}"] = float(dc[t][i])
                r[f"wrong_threshold_{t}"] = bool(err_thr[t][i])
            rows.append(r)

        def _mass(mask, _dc=dc_mean):
            return float(_dc[mask].sum())

        def _side_top(mask, k=TOP_N_PER_SUBSET, _dc=dc_mean, _n=n):
            """The side's OWN top-k items by mean deficit contribution."""
            idx = np.where(mask)[0]
            idx = idx[np.argsort(-_dc[idx])][:k]
            sel = np.zeros(_n, dtype=bool)
            sel[idx] = True
            return sel, len(idx)

        pos, neg = y == 1, y == 0
        cons_t = n_wrong_thr == len(tags)
        cons_r = n_wrong_rank == len(tags)
        per_subset[sub] = {
            "n": n, "n_positive": n_pos, "n_negative": n_neg, "n_pairs": pairs,
            "auc_per_draw": {t: round(auc[t], 4) for t in tags},
            "auc_k6_mean": round(auc_mean, 5),
            "deficit_k6_mean": round(deficit_mean, 5),
            "draw_agreement_threshold": {
                "definition": ("error = prediction != label at each draw's "
                               "in-sample macro-F1-optimal threshold "
                               "(R18-H157 op_threshold, verbatim)"),
                "histogram_n_draws_wrong": {
                    str(k): int((n_wrong_thr == k).sum()) for k in range(len(tags) + 1)},
                "mean_draws_wrong": round(float(n_wrong_thr.mean()), 4),
                "consensus_errors": int(cons_t.sum()),
                "consensus_fn": int((cons_t & pos).sum()),
                "consensus_fp": int((cons_t & neg).sum()),
                "consensus_mass_share_of_deficit": round(
                    _mass(cons_t) / max(2 * deficit_mean, 1e-12), 4),
            },
            "draw_agreement_rank": {
                "definition": ("threshold-free: a positive outranked by more "
                               "than half the negatives, or a negative "
                               "outranking more than half the positives"),
                "histogram_n_draws_wrong": {
                    str(k): int((n_wrong_rank == k).sum()) for k in range(len(tags) + 1)},
                "mean_draws_wrong": round(float(n_wrong_rank.mean()), 4),
                "consensus_errors": int(cons_r.sum()),
                "consensus_fn": int((cons_r & pos).sum()),
                "consensus_fp": int((cons_r & neg).sum()),
                "consensus_mass_share_of_deficit": round(
                    _mass(cons_r) / max(2 * deficit_mean, 1e-12), 4),
            },
            "rank_loss_mass": {
                "formulation": ("per-item AUROC-deficit contribution: each "
                                "mis-ranked (positive, negative) pair carries "
                                "1/(n_pos*n_neg), split equally in the sense "
                                "that each side's contributions sum to the FULL "
                                "deficit (1 - AUC) - the same pair is counted "
                                "once on each side"),
                "deficit_total_each_side": round(deficit_mean, 5),
                "sides_carry_equal_total_mass_by_construction": (
                    "each mis-ranked pair is counted once on the positive side "
                    "and once on the negative side, so fn_side.mass == "
                    "fp_side.mass == (1 - AUC) ALWAYS. The direction "
                    "information is in the per-item distribution - item counts "
                    "and concentration - never in the two side totals"),
                "fn_side": {
                    "n_items": n_pos,
                    "mass": round(_mass(pos), 5),
                    "n_items_carrying_mass": int((dc_mean[pos] > 0).sum()),
                    "own_top40_n": _side_top(pos)[1],
                    "own_top40_share_of_side_mass": round(
                        _mass(_side_top(pos)[0]) / max(deficit_mean, 1e-12), 4),
                },
                "fp_side": {
                    "n_items": n_neg,
                    "mass": round(_mass(neg), 5),
                    "n_items_carrying_mass": int((dc_mean[neg] > 0).sum()),
                    "own_top40_n": _side_top(neg)[1],
                    "own_top40_share_of_side_mass": round(
                        _mass(_side_top(neg)[0]) / max(deficit_mean, 1e-12), 4),
                },
                "top40": {
                    "n_marked": int(top.sum()),
                    "share_of_total_mass": round(
                        _mass(top) / max(2 * deficit_mean, 1e-12), 4),
                    "share_of_fn_side_mass": round(
                        _mass(top & pos) / max(deficit_mean, 1e-12), 4),
                    "share_of_fp_side_mass": round(
                        _mass(top & neg) / max(deficit_mean, 1e-12), 4),
                    "n_fn": int((top & pos).sum()), "n_fp": int((top & neg).sum()),
                    "n_consensus_threshold": int((top & cons_t).sum()),
                },
            },
        }

        ceil = ceil_raw[sub][CEILING_FIELD]
        headroom[sub] = {
            "faithful_oracle_ceiling": ceil,
            "ceiling_source": f"{CEILING_SOURCE}:{CEILING_FIELD}",
            "auc_k6_mean": round(auc_mean, 5),
            "headroom_to_ceiling": round(ceil - auc_mean, 5),
            "deficit_to_1.0": round(deficit_mean, 5),
        }

    df = pl.DataFrame(rows)
    df.write_parquet(CONSENSUS_PARQUET)

    overall_t = {str(k): int((df["n_draws_wrong_threshold"] == k).sum())
                 for k in range(len(tags) + 1)}
    overall_r = {str(k): int((df["n_draws_wrong_rank"] == k).sum())
                 for k in range(len(tags) + 1)}

    payload = {
        "arm": "R21-H179 blind-arena error autopsy, stage 1 (measurement)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, "
                         "block 'R21-H179 BLIND-ARENA ERROR AUTOPSY'"),
        "licence": ("MEASUREMENT ONLY - no bar, no promotion, no kill, no "
                    "adjudication; stages 2 and 3 (annotation, clustering, "
                    "mechanism) are owned by other agents"),
        "read": ("shipped windowed decomposed-min arena read "
                 "(R16-H142_G1_reads.py, reused unchanged), AUROC unit = the "
                 "RESPONSE (min over H92 sentences of max over 1500/750 windows)"),
        "draws": {t: DRAWS[t] for t in tags},
        "fidelity_control": {t: {
            "verdict": fid_draws[t]["verdict"],
            "worst_abs_delta": fid_draws[t]["worst_abs_delta"],
            "banked_mean": fid_draws[t]["banked_mean"],
            "recomputed_mean": fid_draws[t]["recomputed_mean"]} for t in tags},
        "rank_loss_provenance": prov,
        "totals": {
            "n_items": len(df),
            "n_subsets": len(subsets),
            "draw_agreement_threshold_histogram": overall_t,
            "draw_agreement_rank_histogram": overall_r,
            "consensus_errors_threshold": int(df["consensus_error_threshold"].sum()),
            "consensus_errors_rank": int(df["consensus_error_rank"].sum()),
            "top40_marked": int(df["top40"].sum()),
        },
        "per_subset": per_subset,
        "headroom_reported_separately": {
            "why": ("the registration requires error mass and headroom to the "
                    "faithful-oracle ceiling to be reported SEPARATELY so the "
                    "two are not confused"),
            "per_subset": headroom,
        },
        "artifacts": {
            "items": ITEMS_PARQUET.name,
            "per_item_scores": [npz_path(t).name for t in tags],
            "fidelity": FIDELITY_JSON.name,
            "consensus_parquet": CONSENSUS_PARQUET.name,
        },
        "note": NOTE,
        "written": time.strftime("%F %T"),
    }
    CONSENSUS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nconsensus -> {CONSENSUS_JSON.name}  and  {CONSENSUS_PARQUET.name} "
          f"{df.shape}", flush=True)
    for sub in subsets:
        p = per_subset[sub]
        a = p["draw_agreement_threshold"]
        print(f"  {sub:12s} n={p['n']:>4} auc {p['auc_k6_mean']:.4f} "
              f"deficit {p['deficit_k6_mean']:.4f}  consensus "
              f"{a['consensus_errors']:>3} (fn {a['consensus_fn']:>3} / fp "
              f"{a['consensus_fp']:>3})  top40 mass "
              f"{p['rank_loss_mass']['top40']['share_of_total_mass']:.3f}  "
              f"headroom {headroom[sub]['headroom_to_ceiling']:+.4f}", flush=True)


# --- driver ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("items", "score", "consensus"))
    ap.add_argument("--draw", default=None, choices=tuple(DRAWS),
                    help="score one checkpoint only")
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== R21-H179 arena scores ({args.stage})  {time.strftime('%F %T')} ===",
          flush=True)
    try:
        if args.stage == "items":
            stage_items()
        elif args.stage == "score":
            stage_score(args.draw)
        else:
            stage_consensus()
    except FidelityAbort as exc:
        print(f"\n=== FIDELITY ABORT ===\n{exc}\n=== R21-H179 ABORTED ===",
              flush=True)
        raise SystemExit(2) from exc
    print(f"=== R21-H179 {args.stage} DONE ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
