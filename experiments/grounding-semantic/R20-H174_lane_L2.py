"""R20-H174 LANE L2 `attr_pool` - source_select, build + verify, CPU, no GPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H174
HAGRID/EMANUAL PORTFOLIO ARM": "L2 attr_pool (~20-30k rows, BM25-distractor
construction over MiniCheck + VitaminC, document-disjoint, ISOLATED from the
H159 lanes that caused the collapse)".

THE DEFECT THE LANE TEACHES AGAINST
-----------------------------------
hagrid's AUROC FALLS as passages are added to the pool: vacuous-excluded, one
document reads 0.86 / 0.81, two to three 0.69 / 0.63, four to eight 0.51 / 0.61,
and the k-doc curve on the deep-pool stratum descends 0.6216 -> 0.5096 as k goes
1 -> 8 (R19-H162_hagrid_mechanisms.json, `kdoc_curve_h150d1`).  The model credits
the best topically-adjacent passage instead of the supporting one.  No banked
lane presents a CHOICE among competing passages - every training row is one
claim against one document - so the skill has never been supplied.  The R19-H159
enriched checkpoint, which did supply something like it, lifted the deep-pool
cell +0.152 and the whole subset 0.6423 -> 0.7074: the existence proof.

CONSTRUCTION - two families, both minimal pairs over a pooled presentation
-------------------------------------------------------------------------
Every row's `chunk` is a POOL: one passage per document, joined, 4 to 8 deep.
The mix loader reads lane chunks untruncated and windows them 1,500 / 750, so a
pool of this size enters training as a real multi-window bag and MIL has to pick
the window that carries the support - the geometry the arena's deep-pool stratum
actually has and the training mix never had.

  truth_removed       the registered source_select teacher.  label 1 is the
                      claim against a pool CONTAINING its true passage; label 0
                      is the SAME claim against the SAME distractors with the
                      true passage REPLACED by one more distractor, so support
                      is absent by construction and the claim side is
                      byte-identical between the legs
  unsupported_claim   label 1 and label 0 are two claims about the SAME document
                      - one supported, one not - against a BYTE-IDENTICAL pool,
                      so the pool cannot be read off the label

Distractors are BM25-retrieved from OTHER documents of the same corpus, which
makes them topically adjacent rather than random - the failure mode is credit
given to a near-miss passage, and a random distractor would not exercise it.  A
candidate is rejected if the claim's content-token containment in it reaches
CONTAIN_MAX, the guard against a "distractor" that happens to support.

ISOLATION (registration clause)
-------------------------------
Sources are MiniCheck (MIT) and VitaminC train (CC-BY-SA-3.0) ONLY.  FAVA,
PubHealth, FinDVer and AttributionBench - the other four H159 lanes, diagnosed
in brief B as the near-copy collateral that cost that arm finqa / tatqa /
delucionqa - are not read here at all.

DOCUMENT DISJOINTNESS
---------------------
A document may serve as truth at most TRUTH_CAP times and as a pool member at
most DIST_CAP times, and never as a distractor inside a pair whose truth it is.
Within a pair the distractor set is identical by design: that IS the contrast.

CONTAMINATION
-------------
MiniCheck and VitaminC both hold GREEN on the R14-H136 8-gram wall, but the
POOLED text is new, so `R20-H174_lane_census.py` re-runs the wall on the built
lane before the arm may spend a card.

Run:  uv run python experiments/grounding-semantic/R20-H174_lane_L2.py
"""

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import polars as pl
from rank_bm25 import BM25Okapi

_spec = _ilu.spec_from_file_location("h174common", Path(__file__).parent / "R20-H174_lane_common.py")
C = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(C)

HERE = Path(__file__).parent
OUT = HERE / "R20-H174_lane_L2.parquet"
MANIFEST = HERE / "R20-H174_lane_L2_manifest.json"

SEED = 2174
TAG = "attr_pool"

# 12,000 pairs = 24,000 rows, inside the registered 20-30k band.
TARGET = {
    ("minicheck", "truth_removed"): 4_000,
    ("minicheck", "unsupported_claim"): 2_000,
    ("vitaminc", "truth_removed"): 4_000,
    ("vitaminc", "unsupported_claim"): 2_000,
}

POOL_MIN, POOL_MAX = 3, 7        # distractors; pool depth 4-8 documents
BM25_TOPK = 40                   # candidate window the distractors are drawn from
CONTAIN_MAX = 0.75               # a candidate this close to the claim is not a distractor
TRUTH_CAP, DIST_CAP = 2, 12
PASSAGE_MAX = C.PASSAGE_MAX_CHARS
SEP = "\n\n"


# --------------------------------------------------------------------------- #
# corpora -> (passages, claim rows)
# --------------------------------------------------------------------------- #
def load_minicheck():
    df = C.minicheck().filter(pl.col("doc").str.len_chars() <= PASSAGE_MAX)
    passages = dict(df.select(["doc_id", "doc"]).unique().iter_rows())
    rows = [{"claim": c, "doc_id": d, "group_key": d, "label": int(y),
             "source": "minicheck"}
            for c, d, y in df.select(["claim", "doc_id", "label"]).iter_rows()]
    return passages, rows


def load_vitaminc(rng):
    df = C.vitaminc("train")
    pages = collections.defaultdict(list)
    for page, ev in df.select(["page", "evidence"]).unique().iter_rows():
        pages[page].append(ev)
    canonical = C.vitaminc_passages(df, rng)          # page -> distractor passage
    # the pooling unit is the PAGE, but the support unit is the EVIDENCE
    # SENTENCE, so `unsupported_claim` pairs are grouped on the sentence: both
    # legs then argue about the same proposition, not merely the same page.
    rows = [{"claim": c, "evidence": e, "page": p, "label": int(y),
             "doc_id": p, "group_key": e, "source": "vitaminc"}
            for c, e, p, y in df.select(["claim", "evidence", "page", "label"]).iter_rows()]
    return canonical, rows, pages


# --------------------------------------------------------------------------- #
# BM25 over the corpus's own passages
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, passages, label):
        self.ids = sorted(passages)
        self.texts = [passages[i] for i in self.ids]
        t0 = time.time()
        self.bm25 = BM25Okapi([C.tokens(t) for t in self.texts])
        print(f"  BM25 index {label}: {len(self.ids)} passages "
              f"in {time.time() - t0:.1f}s", flush=True)
        self.pos = {d: i for i, d in enumerate(self.ids)}

    def candidates(self, claim, exclude, k=BM25_TOPK):
        s = self.bm25.get_scores(C.tokens(claim))
        order = np.argsort(-s)[: k + len(exclude) + 4]
        return [self.ids[i] for i in order if self.ids[i] not in exclude][:k]


# --------------------------------------------------------------------------- #
# pool assembly
# --------------------------------------------------------------------------- #
def pick_distractors(claims, cands, passages, need, dist_used, rng):
    """Topically-close, non-supporting, under the reuse cap.

    The guard runs against BOTH claims of the pair: on an `unsupported_claim`
    pair the label-0 claim is a different string, and a pool member that
    happened to support IT would mislabel the row."""
    out = []
    for did in cands:
        if len(out) >= need:
            break
        if dist_used[did] >= DIST_CAP:
            continue
        if max(C.containment(c, passages[did]) for c in claims) >= CONTAIN_MAX:
            continue
        out.append(did)
    if len(out) < need:
        return None
    rng.shuffle(out)
    return out


def pool_text(items, rng):
    order = list(items)
    rng.shuffle(order)
    return SEP.join(t for _d, t in order), [d for d, _t in order]


# --------------------------------------------------------------------------- #
def already_built():
    """Idempotence: a lane whose parquet and manifest are on disk and whose own
    verify block passed is not rebuilt.  `--force` overrides."""
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()):
        return False
    try:
        man = json.loads(MANIFEST.read_text())
        rows = pl.read_parquet(OUT).height
    except Exception:
        return False
    if man.get("verify", {}).get("all_bars_pass") and rows == man.get("rows"):
        print(f"{OUT.name}: {rows} rows already built and passing - skipping "
              f"(pass --force to rebuild)", flush=True)
        return True
    return False


def main():
    if already_built():
        return
    rng = random.Random(SEED)
    print(f"=== R20-H174 lane L2 ({TAG}) seed {SEED}", flush=True)

    mc_pass, mc_rows = load_minicheck()
    print(f"minicheck: {len(mc_pass)} passages <= {PASSAGE_MAX} chars, "
          f"{len(mc_rows)} claims", flush=True)
    vc_pass, vc_rows, vc_pages = load_vitaminc(rng)
    print(f"vitaminc: {len(vc_pass)} page passages, {len(vc_rows)} claims", flush=True)

    retr = {"minicheck": Retriever(mc_pass, "minicheck"),
            "vitaminc": Retriever(vc_pass, "vitaminc")}

    rows, pid = [], 0
    truth_used, dist_used = collections.Counter(), collections.Counter()
    stats = collections.Counter()

    for source in ("minicheck", "vitaminc"):
        base = mc_rows if source == "minicheck" else vc_rows
        passages = mc_pass if source == "minicheck" else vc_pass
        by_doc = collections.defaultdict(lambda: {0: [], 1: []})
        for r in base:
            by_doc[r["group_key"]][r["label"]].append(r)
        keys = sorted(by_doc)
        rng.shuffle(keys)

        # ---- family `unsupported_claim` first: it needs documents carrying BOTH
        # a supported and an unsupported claim, the scarcer supply.
        want_b = TARGET[(source, "unsupported_claim")]
        want_a = TARGET[(source, "truth_removed")]
        got_b = got_a = 0
        for did in keys:
            if got_b >= want_b:
                break
            d = by_doc[did]
            if not d[0] or not d[1]:
                continue
            pos_r, neg_r = rng.choice(d[1]), rng.choice(d[0])
            if truth_used[pos_r["doc_id"]] >= TRUTH_CAP:
                continue
            built = build_pair(source, pos_r, neg_r, "unsupported_claim", passages,
                               vc_pages, retr[source], truth_used, dist_used, rng, pid)
            if built is None:
                stats[f"{source}_b_skipped"] += 1
                continue
            rows += built
            pid += 1
            got_b += 1
            if got_b % 500 == 0:
                print(f"  {source} unsupported_claim {got_b}/{want_b}", flush=True)

        # ---- family `truth_removed`
        for did in keys:
            if got_a >= want_a:
                break
            d = by_doc[did]
            if not d[1]:
                continue
            pos_r = rng.choice(d[1])
            if truth_used[pos_r["doc_id"]] >= TRUTH_CAP:
                continue
            built = build_pair(source, pos_r, None, "truth_removed", passages,
                               vc_pages, retr[source], truth_used, dist_used, rng, pid)
            if built is None:
                stats[f"{source}_a_skipped"] += 1
                continue
            rows += built
            pid += 1
            got_a += 1
            if got_a % 500 == 0:
                print(f"  {source} truth_removed {got_a}/{want_a}", flush=True)
        print(f"  {source}: truth_removed {got_a}/{want_a}, "
              f"unsupported_claim {got_b}/{want_b}", flush=True)

    df = C.dedupe(pl.DataFrame(rows, infer_schema_length=None))
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}", flush=True)

    res = verify(df, rng)
    man = build_manifest(df, res, truth_used, dist_used, stats)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "label_balance", "families", "source_rows",
                       "pool", "document_disjointness", "window_census", "verify")},
                     indent=2), flush=True)
    ok = res["all_bars_pass"]
    print(f"=== R20-H174 LANE L2 {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


def build_pair(source, pos_r, neg_r, family, passages, vc_pages, retr,
               truth_used, dist_used, rng, pid):
    """Two rows sharing one distractor set."""
    did = pos_r["doc_id"]
    if source == "vitaminc":
        true_text = C.vitaminc_passage_for(vc_pages[did], pos_r["evidence"], rng)
    else:
        true_text = passages[did]
    if len(true_text) < 120:
        return None
    # an `unsupported_claim` negative must stay unsupported by the WHOLE true
    # passage, not merely by the sentence it was labelled against
    if neg_r is not None:
        if neg_r["claim"].strip() == pos_r["claim"].strip():
            return None          # the corpus labels the same string both ways
        if C.containment(neg_r["claim"], true_text) >= CONTAIN_MAX:
            return None

    need = rng.randint(POOL_MIN, POOL_MAX)
    cands = retr.candidates(pos_r["claim"], exclude={did})
    claims = [pos_r["claim"]] + ([neg_r["claim"]] if neg_r is not None else [])
    picked = pick_distractors(claims, cands, passages,
                              need + 1, dist_used, rng)
    if picked is None:
        return None
    dist, spare = picked[:need], picked[need]

    items_pos = [(did, true_text)] + [(d, passages[d]) for d in dist]
    if family == "truth_removed":
        items_neg = [(spare, passages[spare])] + [(d, passages[d]) for d in dist]
        pos_claim = neg_claim = pos_r["claim"]
    else:
        items_neg = items_pos
        pos_claim, neg_claim = pos_r["claim"], neg_r["claim"]

    pos_chunk, pos_order = pool_text(items_pos, rng)
    if family == "truth_removed":
        neg_chunk, neg_order = pool_text(items_neg, rng)
    else:
        neg_chunk, neg_order = pos_chunk, pos_order

    truth_used[did] += 1
    for d in dist:
        dist_used[d] += 1
    if family == "truth_removed":
        dist_used[spare] += 1

    base = {"tag": TAG, "neg_family": family, "source": source, "doc_id": did,
            "pool_depth": len(items_pos), "distractors": len(dist),
            "swap_doc_id": spare if family == "truth_removed" else None}
    return [
        dict(pair_id=pid, label=1, claim=pos_claim, chunk=pos_chunk,
             truth_in_pool=True, pool_doc_ids=pos_order, **base),
        dict(pair_id=pid, label=0, claim=neg_claim, chunk=neg_chunk,
             truth_in_pool=(family == "unsupported_claim"),
             pool_doc_ids=neg_order, **base),
    ]


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    out = {}
    out["pair_integrity"] = C.pair_integrity(df)

    # --- the pooled presentation is what the lane claims it is
    errs = []
    for pid, sub in df.group_by("pair_id"):
        s = sub.sort("label", descending=True)
        pos, neg = s.row(0, named=True), s.row(1, named=True)
        why = None
        if pos["neg_family"] == "truth_removed":
            if pos["doc_id"] not in pos["pool_doc_ids"]:
                why = "truth absent from the label-1 pool"
            elif pos["doc_id"] in neg["pool_doc_ids"]:
                why = "truth present in the label-0 pool"
            elif pos["claim"] != neg["claim"]:
                why = "claim differs across a truth_removed pair"
            elif len(pos["pool_doc_ids"]) != len(neg["pool_doc_ids"]):
                why = "pool depth differs across the legs"
        else:
            if pos["chunk"] != neg["chunk"]:
                why = "pool text differs across an unsupported_claim pair"
            elif pos["claim"] == neg["claim"]:
                why = "claims identical across an unsupported_claim pair"
            elif pos["doc_id"] not in pos["pool_doc_ids"]:
                why = "truth absent from the shared pool"
        if why:
            errs.append({"pair_id": int(pid[0]), "why": why})
    out["pool_composition_audit"] = {
        "pairs": int(df["pair_id"].n_unique()), "errors": len(errs),
        "bar": "0 errors", "pass": not errs, "examples": errs[:5]}

    # --- no pool member other than the truth may support the claim
    neg = df.filter(pl.col("label") == 0)
    worst, over = 0.0, collections.Counter()
    for r in neg.iter_rows(named=True):
        for part in r["chunk"].split(SEP):
            c = C.containment(r["claim"], part)
            worst = max(worst, c)
            if c >= CONTAIN_MAX:
                over[r["neg_family"]] += 1
    out["pool_non_support"] = {
        "negative_rows": int(neg.height),
        "max_claim_containment_in_any_pool_member": round(worst, 4),
        "rows_over_guard": dict(over),
        "guard": CONTAIN_MAX,
        "bar": "0 label-0 rows carry a pool member reaching the guard - no "
               "passage in a negative's pool may look like support",
        "pass": sum(over.values()) == 0}

    out["surface_parity"] = C.surface_parity(
        df, report_only=("claim_chunk_containment",))

    probe, score = C.claim_only_probe(df["claim"].to_list(), df["label"].to_list(),
                                      df["doc_id"].to_list(), rng)
    wp = C.within_pair_accuracy(df, score, by="neg_family")
    worst_wp = max(v["acc"] for v in wp.values())
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": "5-fold document-disjoint, out of fold, liblinear tol 1e-7"}
    out["within_pair_claim_only_accuracy"] = {
        "per_family": wp, "worst": round(worst_wp, 4), "bar": "< 0.60",
        "pass": bool(worst_wp < 0.60),
        "note": "truth_removed is 0.5 by construction - the claim is identical "
                "on both legs; the informative cell is unsupported_claim"}

    out["all_bars_pass"] = all(
        out[k]["pass"] for k in
        ("pair_integrity", "pool_composition_audit", "pool_non_support",
         "surface_parity", "claim_only_tfidf_auroc",
         "within_pair_claim_only_accuracy"))
    return out


def build_manifest(df, res, truth_used, dist_used, stats):
    y = df["label"].to_list()
    depth = df["pool_depth"].to_numpy()
    truth_docs, dist_docs = set(truth_used), set(dist_used)
    return dict(
        experiment="R20-H174 lane L2 - source_select / attribution pool (attr_pool)",
        registration="docs/experiments/semantic-grounding-experiments.md, "
                     "block 'R20-H174 HAGRID/EMANUAL PORTFOLIO ARM'",
        tag=TAG,
        dann_group=TAG,
        mix_loader="drop-in for R18-H150_arm_run.make_build_mix - columns "
                   "claim / chunk / label / pair_id / neg_family; chunk is read "
                   "UNTRUNCATED and windowed 1500/750 by the loader, which is "
                   "what turns a pool into a multi-window bag",
        seed=SEED,
        rows=df.height,
        pairs=int(df["pair_id"].n_unique()),
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y)),
                       "positive_share": round(sum(y) / len(y), 4)},
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        source_rows={k: v for k, v in df.group_by("source").len().iter_rows()},
        family_by_source={f"{a}:{b}": n for a, b, n in
                          df.group_by(["source", "neg_family"]).len().iter_rows()},
        sources={k: C.SOURCES[k] for k in ("minicheck", "vitaminc")},
        isolation="FAVA / PubHealth / FinDVer / AttributionBench are NOT read - "
                  "the registration isolates this lane from the four H159 lanes "
                  "diagnosed as the table-collapse collateral",
        pool=dict(distractor_min=POOL_MIN, distractor_max=POOL_MAX,
                  bm25_topk=BM25_TOPK, containment_guard=CONTAIN_MAX,
                  passage_max_chars=PASSAGE_MAX, separator=repr(SEP),
                  mean_depth=round(float(depth.mean()), 3),
                  depth_distribution={str(k): v for k, v in
                                      df.group_by("pool_depth").len().iter_rows()}),
        document_disjointness=dict(
            truth_cap=TRUTH_CAP, distractor_cap=DIST_CAP,
            documents_as_truth=len(truth_docs),
            documents_as_pool_member=len(dist_docs),
            documents_in_both_roles=len(truth_docs & dist_docs),
            note="a document may appear as truth in one pair and as a distractor "
                 "in another - never inside the pair whose truth it is, which is "
                 "enforced by the retriever's exclude set and re-checked by "
                 "pool_composition_audit",
            max_truth_uses=max(truth_used.values()) if truth_used else 0,
            max_distractor_uses=max(dist_used.values()) if dist_used else 0),
        build_skips=dict(stats),
        char_stats=dict(claim=C.char_stats(df["claim"].to_list()),
                        chunk=C.char_stats(df["chunk"].to_list())),
        diversity=dict(distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique()),
                       distinct_truth_documents=int(df["doc_id"].n_unique())),
        window_census=C.window_census(df["chunk"].to_list()),
        verify=res)


if __name__ == "__main__":
    main()
