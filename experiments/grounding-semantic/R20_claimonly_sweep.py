"""R20 - the claim-only shortcut sweep: how far does the HaluEval leak extend?

The dataset-contract pass found HaluEval's label 95.19% predictable from the
CLAIM TEXT ALONE (`contract/halueval_contract_report.json`, converged claim-only
probe AUROC 0.9519). HaluEval's negative leg is ChatGPT-generated, so the
generated text carries a style signature the label rides on. This script asks
two questions the contract pass could not:

  Part 1  how far does the shortcut extend across the mix - a claim-only probe
          per DANN group of the assembled 721,210-row flagship mix
  Part 2  does it TRANSFER to the blind arena - the halueval-fitted probe and
          the whole-mix-fitted probe, scored on the 10 RAGBench subsets
  Part 3  the same two probes on gold_full, the held-out in-domain surface

The instrument is deterministic and torch-free: TF-IDF char_wb(2,5) + word(1,2)
n-grams of the CLAIM ALONE, into liblinear logistic regression at tol 1e-7
(never default lbfgs - R17-H144 finding ii). Per member the primary read is a
fixed-seed STRATIFIED 70/30 split; a document-disjoint 70/30 split (rows sharing
an evidence chunk cannot straddle the split) is carried beside it because a
random split lets a HaluEval pair's two legs land on opposite sides.

Reading bars (for reading, NOT for deciding): claim-only < 0.55 clean,
>= 0.55 leak, >= 0.80 severe.

CPU ONLY - no GPU is touched. Nothing is adjudicated here.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20_claimonly_sweep.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU ONLY - training draws own the cards
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import polars as pl
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20_claimonly_sweep.json"

SEED = 0
TEST_FRAC = 0.30
BARS = {"clean_below": 0.55, "leak_at_or_above": 0.55, "severe_at_or_above": 0.80}

# Coordinator correction (stage `amend`): the leak test is TWO-SIDED. An AUROC of
# 0.155 is not clean - it is a strong signal with the sign flipped, worth 0.845
# to an inverted classifier. Every verdict is restated on
# leak_strength = |AUROC - 0.5|, with the equivalent one-sided AUROC beside it.
BANDS = ((0.05, "clean"), (0.15, "mild"), (0.30, "leak"), (1.01, "severe"))
# Members whose pair key is the CLAIM (or a lane pair_id over near-identical
# claims): the evidence-disjoint split does NOT separate their twins, so an extra
# pair-disjoint split is carried for them as an executor-added diagnostic.
CLAIM_PAIR_KEY = {
    "vitaminc": "claim string - VitaminC's contrastive twins share the claim and revise the evidence",
    "quant_misbind": "pair_id from R17-H146_lane.parquet - minimal pairs over near-identical claims",
    "quant_scale_unit": "pair_id from R18-H150_scaleunit_lane.parquet - minimal pairs",
}

# The H150 flagship mix (R18-H150_arm_run.py LANES / EXPECTED_*), re-derived.
LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540),
)
EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 721_210
EXPECTED_GROUPS = (
    "halueval", "psiloqa", "quant_misbind", "quant_scale_unit", "ragtruth_cn",
    "ragtruth_de", "ragtruth_en", "ragtruth_es", "ragtruth_fr", "ragtruth_hu",
    "ragtruth_it", "ragtruth_pl", "tabfact", "vitaminc",
)

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- data ----------------------------------------------------------------------


def build_mix():
    """The assembled flagship mix's CLAIM side, with the DANN group the loader
    assigns. `public_train` is called with its own chunk truncation in place:
    truncation touches only the evidence, so the claim strings, the labels, the
    group tags and the row set are byte-identical to the untruncated read the
    trainer uses (`R16-H142_G1_arm.untruncated_evidence`)."""
    h108 = _mod("h108", "R10-H108_lane.py")
    claims, chunks, y, tags = h108.public_train()
    if len(y) != EXPECTED_CLEAN_ROWS:
        raise SystemExit(f"CENSUS ABORT: clean mix {len(y)} rows (want {EXPECTED_CLEAN_ROWS})")
    log(f"clean mix {len(y)} rows")

    for fname, group, n_rows in LANES:
        df = pl.read_parquet(HERE / fname)
        if len(df) != n_rows:
            raise SystemExit(f"LANE ABORT ({group}): {len(df)} rows (want {n_rows})")
        claims += df["claim"].to_list()
        chunks += df["chunk"].to_list()
        y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * len(df)
        log(f"lane {group}: {len(df)} rows")

    names = tuple(sorted(set(tags)))
    if names != EXPECTED_GROUPS:
        raise SystemExit(f"GROUP-MAP ABORT: {names} != {EXPECTED_GROUPS}")
    if len(y) != EXPECTED_MIX_ROWS:
        raise SystemExit(f"CENSUS ABORT: mix {len(y)} rows (want {EXPECTED_MIX_ROWS})")

    # Evidence identity, for the document-disjoint split. The evidence text is
    # then dropped - this sweep never looks at it again.
    doc = [hashlib.blake2b(k[:4000].encode("utf-8"), digest_size=8).hexdigest() for k in chunks]
    del chunks
    log(f"mix assembled: {len(y)} rows, {len(names)} groups, {len(set(doc))} distinct evidence keys")
    return claims, y.astype(int), tags, doc


# --- instrument -----------------------------------------------------------------


def fit_probe(texts, labels, tag=""):
    """TF-IDF char_wb(2,5) + word(1,2) of the CLAIM ALONE -> liblinear LR."""
    t = time.time()
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                         max_features=300_000, sublinear_tf=True)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                         max_features=300_000, sublinear_tf=True)
    x = hstack([vc.fit_transform(texts), vw.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
    clf.fit(x, labels)
    log(f"  fit {tag}: {x.shape[0]} rows x {x.shape[1]} features, {time.time() - t:.1f}s")
    return (vc, vw, clf)


def apply_probe(probe, texts):
    vc, vw, clf = probe
    x = hstack([vc.transform(texts), vw.transform(texts)]).tocsr()
    return clf.decision_function(x)


def split_stratified(y, seed=SEED, test_frac=TEST_FRAC):
    """Fixed-seed stratified 70/30 split - the registered primary read."""
    rng = np.random.default_rng(seed)
    te = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        te.append(idx[: int(round(test_frac * idx.size))])
    test = np.sort(np.concatenate(te))
    mask = np.ones(y.size, dtype=bool)
    mask[test] = False
    return np.flatnonzero(mask), test


def split_doc_disjoint(doc, seed=SEED, test_frac=TEST_FRAC):
    """70/30 by evidence key: rows sharing an evidence chunk stay on one side,
    so a pair's positive and negative legs cannot straddle the split."""
    keys = sorted(set(doc))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    order = {k: i for i, k in enumerate(keys)}
    rank = np.array([order[d] for d in doc])
    # walk keys in shuffled order until the row budget for test is met
    counts = np.bincount(rank, minlength=len(keys))
    cum = np.cumsum(counts)
    cut = int(np.searchsorted(cum, test_frac * len(doc)))
    test = np.flatnonzero(rank <= cut)
    mask = np.ones(len(doc), dtype=bool)
    mask[test] = False
    return np.flatnonzero(mask), test


def split_auroc(texts, y, tr, te, tag=""):
    if te.size == 0 or tr.size == 0 or len(set(y[te])) < 2 or len(set(y[tr])) < 2:
        return None
    probe = fit_probe([texts[i] for i in tr], y[tr], tag=tag)
    s = apply_probe(probe, [texts[i] for i in te])
    return float(roc_auc_score(y[te], s))


def surface_auroc(texts, y):
    if len(set(y)) < 2:
        return None, None
    chars = np.array([float(len(t)) for t in texts])
    toks = np.array([float(len(_WORD.findall(t))) for t in texts])
    return float(roc_auc_score(y, chars)), float(roc_auc_score(y, toks))


# --- parts ----------------------------------------------------------------------


def part1(claims, y, tags, doc, payload):
    log("=== PART 1 - per-member claim-only probe ===")
    members = {}
    tag_arr = np.array(tags)
    for g in EXPECTED_GROUPS:
        idx = np.flatnonzero(tag_arr == g)
        txt = [claims[i] for i in idx]
        yy = y[idx]
        dd = [doc[i] for i in idx]
        log(f"{g}: {idx.size} rows, pos {yy.mean():.4f}")
        char_auc, tok_auc = surface_auroc(txt, yy)
        tr, te = split_stratified(yy)
        a_strat = split_auroc(txt, yy, tr, te, tag=f"{g}/stratified")
        tr2, te2 = split_doc_disjoint(dd)
        a_doc = split_auroc(txt, yy, tr2, te2, tag=f"{g}/doc-disjoint")
        row = {
            "rows": int(idx.size),
            "label_balance_positive": round(float(yy.mean()), 4),
            "claim_only_auroc": None if a_strat is None else round(a_strat, 4),
            "claim_only_auroc_doc_disjoint": None if a_doc is None else round(a_doc, 4),
            "claim_char_length_auroc": None if char_auc is None else round(char_auc, 4),
            "claim_token_count_auroc": None if tok_auc is None else round(tok_auc, 4),
            "n_evidence_keys": int(len(set(dd))),
            "doc_disjoint_test_rows": int(te2.size),
            "verdict_read": None,
        }
        a = a_strat if a_strat is not None else 0.0
        row["verdict_read"] = ("severe" if a >= BARS["severe_at_or_above"]
                               else "leak" if a >= BARS["leak_at_or_above"] else "clean")
        members[g] = row
        log(f"  {g:18s} claim-only {row['claim_only_auroc']}  "
            f"doc-disjoint {row['claim_only_auroc_doc_disjoint']}  "
            f"charlen {row['claim_char_length_auroc']}  toks {row['claim_token_count_auroc']}  "
            f"-> {row['verdict_read']}")
        payload["part1_per_member"] = members
        OUT.write_text(json.dumps(payload, indent=2))
    return members


def part2(probes, payload):
    log("=== PART 2 - transfer to the blind arena ===")
    arena = _mod("arena", "R8-H77_unseen_arena.py")
    subs = arena.load_subsets()
    n = sum(len(v[2]) for v in subs.values())
    log(f"arena: {len(subs)} subsets, {n} responses")

    block = {"subsets": len(subs), "responses": int(n), "probes": {}}
    for name, probe in probes.items():
        per_sub, pooled_s, pooled_y = {}, [], []
        for sub, (resp, _chunks, yy) in subs.items():
            s = apply_probe(probe, resp)
            per_sub[sub] = {"n": int(len(yy)),
                            "label_balance_positive": round(float(yy.mean()), 4),
                            "claim_only_auroc": round(float(roc_auc_score(yy, s)), 4)}
            pooled_s.append(s)
            pooled_y.append(yy)
            log(f"  {name:14s} {sub:12s} n={len(yy):>4} claim-only {per_sub[sub]['claim_only_auroc']:.4f}")
        ps, py = np.concatenate(pooled_s), np.concatenate(pooled_y)
        block["probes"][name] = {
            "per_subset": per_sub,
            "arena_auroc_pooled": round(float(roc_auc_score(py, ps)), 4),
            "arena_auroc_subset_mean": round(
                float(np.mean([r["claim_only_auroc"] for r in per_sub.values()])), 4),
        }
        log(f"  {name}: pooled {block['probes'][name]['arena_auroc_pooled']:.4f}  "
            f"subset-mean {block['probes'][name]['arena_auroc_subset_mean']:.4f}")
        payload["part2_arena_transfer"] = block
        OUT.write_text(json.dumps(payload, indent=2))
    return block


def part3(probes, payload):
    log("=== PART 3 - the same probes on gold_full ===")
    h108 = _mod("h108", "R10-H108_lane.py")
    claims, _chunk_lists, y = h108.gold_full()
    log(f"gold_full: {len(claims)} claims, pos {y.mean():.4f}")
    char_auc, tok_auc = surface_auroc(claims, y)
    block = {
        "rows": int(len(claims)),
        "label_balance_positive": round(float(y.mean()), 4),
        "claim_char_length_auroc": round(char_auc, 4),
        "claim_token_count_auroc": round(tok_auc, 4),
        "probes": {},
    }
    for name, probe in probes.items():
        s = apply_probe(probe, claims)
        auc = float(roc_auc_score(y, s))
        block["probes"][name] = {"claim_only_auroc": round(auc, 4)}
        log(f"  {name:14s} gold_full claim-only {auc:.4f}")
    payload["part3_gold_full"] = block
    OUT.write_text(json.dumps(payload, indent=2))
    return block


# --- amend: two-sided verdicts, split selection, negative control -----------------


def band(strength):
    return next(name for edge, name in BANDS if strength < edge)


def two_sided(auc):
    """Restate an AUROC as a two-sided leak reading."""
    if auc is None:
        return None
    s = abs(auc - 0.5)
    return {
        "auroc": round(float(auc), 4),
        "leak_strength": round(float(s), 4),
        "equivalent_one_sided_auroc": round(0.5 + float(s), 4),
        "band": band(s),
        "direction": "aligned" if auc >= 0.5 else "inverted",
    }


def pair_census(y, doc, claim_keys):
    """Does this member have pair structure? Measured, not assumed: among evidence
    keys (and among repeated claim strings) carrying more than one row, the share
    that carry BOTH labels. A high share means a random split puts one twin in
    train and the other in test, which is what drives AUROC below 0.5."""
    out = {}
    for name, keys in (("evidence", doc), ("claim_string", claim_keys)):
        seen = {}
        for k, lab in zip(keys, y, strict=True):
            seen.setdefault(k, set()).add(int(lab))
        multi = [v for v in seen.values() if len(v) > 1]
        out[f"{name}_keys"] = len(seen)
        out[f"{name}_keys_with_both_labels"] = len(multi)
        out[f"{name}_keys_with_both_labels_share"] = round(len(multi) / max(len(seen), 1), 4)
    out["pair_structure"] = bool(
        out["evidence_keys_with_both_labels_share"] >= 0.5
        or out["claim_string_keys_with_both_labels_share"] >= 0.5
    )
    return out


def lane_pair_ids():
    ids = {}
    for fname, group, _n in LANES:
        ids[group] = pl.read_parquet(HERE / fname)["pair_id"].cast(pl.Utf8).to_list()
    return ids


def amend():
    """Stage 2 - the coordinator's corrections, applied to the banked numbers.

    1. every verdict restated two-sided on |AUROC - 0.5|
    2. the verdict-bearing split chosen from MEASURED pair structure: where a
       member's twins would straddle a random split, the disjoint split decides
       and the stratified number is carried as a diagnostic
    3. a label-shuffled negative control on the worst and the cleanest member
    No probe from stage 1 is re-fitted; only the new controls are fitted."""
    payload = json.loads(OUT.read_text())
    claims, y, tags, doc = build_mix()
    tag_arr = np.array(tags)
    lane_ids = lane_pair_ids()
    members = payload["part1_per_member"]

    log("=== AMEND 1/3 - pair census + pair-disjoint diagnostic ===")
    shuffled_inputs = {}
    for g in EXPECTED_GROUPS:
        idx = np.flatnonzero(tag_arr == g)
        txt = [claims[i] for i in idx]
        yy = y[idx]
        dd = [doc[i] for i in idx]
        ck = [hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest() for t in txt]
        cen = pair_census(yy, dd, ck)
        members[g]["pair_census"] = cen
        shuffled_inputs[g] = (txt, yy, dd)

        if g in CLAIM_PAIR_KEY:
            keys = lane_ids[g] if g in lane_ids else ck
            tr, te = split_doc_disjoint(keys)
            a = split_auroc(txt, yy, tr, te, tag=f"{g}/pair-disjoint")
            members[g]["claim_only_auroc_pair_disjoint"] = None if a is None else round(a, 4)
            members[g]["pair_disjoint_key"] = CLAIM_PAIR_KEY[g]
            log(f"  {g:18s} pair-disjoint {members[g]['claim_only_auroc_pair_disjoint']}")

    log("=== AMEND 2/3 - two-sided verdicts and split selection ===")
    for g, row in members.items():
        paired = row["pair_census"]["pair_structure"] or g in CLAIM_PAIR_KEY
        if "verdict_read_v1_one_sided" not in row:  # idempotent: keep the audit trail
            row["verdict_read_v1_one_sided"] = row.pop("verdict_read")
        row["reading"] = {
            "stratified": two_sided(row["claim_only_auroc"]),
            "doc_disjoint": two_sided(row["claim_only_auroc_doc_disjoint"]),
        }
        if "claim_only_auroc_pair_disjoint" in row:
            row["reading"]["pair_disjoint"] = two_sided(row["claim_only_auroc_pair_disjoint"])
        row["has_pair_structure"] = bool(paired)
        row["verdict_bearing_split"] = "doc_disjoint" if paired else "stratified"
        row["verdict_bearing_reason"] = (
            "pair structure measured in this member (a positive and its negative share a key), "
            "so under the stratified split one twin sits in train and the other in test: the "
            "classifier confidently predicts the twin's label and is wrong every time, which "
            "drives AUROC below 0.5 and measures the SPLIT, not the member. The "
            "evidence-disjoint split decides; the stratified number is a diagnostic."
            if paired else
            "no pair structure measured - both splits are informative and are reported; the "
            "registered stratified split decides."
        )
        vb = row["reading"][row["verdict_bearing_split"]]
        row["leak_strength"] = vb["leak_strength"]
        row["equivalent_one_sided_auroc"] = vb["equivalent_one_sided_auroc"]
        row["verdict_read"] = vb["band"]
        row["direction"] = vb["direction"]
        bands = {k: v["band"] for k, v in row["reading"].items() if v}
        row["splits_agree_on_band"] = len(set(bands.values())) == 1
        if "pair_disjoint" in row["reading"] and row["reading"]["pair_disjoint"]:
            row["pair_disjoint_note"] = (
                "executor-added diagnostic: the evidence-disjoint split does not separate THIS "
                "member's twins (its pair key is the claim, not the evidence), so the "
                "pair-disjoint number is the leak-safe one and is reported beside the "
                "coordinator-designated verdict-bearing doc-disjoint number.")
        log(f"  {g:18s} {row['verdict_bearing_split']:12s} auroc {vb['auroc']:.4f} "
            f"strength {vb['leak_strength']:.4f} -> {vb['band']} ({vb['direction']})")

    log("=== AMEND 3/3 - label-shuffled negative control ===")
    ranked = sorted(members.items(), key=lambda kv: -kv[1]["leak_strength"])
    picks = {"worst": ranked[0][0], "cleanest": ranked[-1][0]}
    control = {}
    for role, g in picks.items():
        txt, yy, dd = shuffled_inputs[g]
        rng = np.random.default_rng(20)
        ysh = yy.copy()
        rng.shuffle(ysh)
        split = members[g]["verdict_bearing_split"]
        tr, te = (split_doc_disjoint(dd) if split == "doc_disjoint" else split_stratified(ysh))
        a = split_auroc(txt, ysh, tr, te, tag=f"{g}/label-shuffled")
        control[role] = {
            "member": g, "split": split, "rows": int(len(txt)),
            "observed_leak_strength": members[g]["leak_strength"],
            "shuffled": two_sided(a),
        }
        log(f"  {role} {g}: shuffled auroc {a:.4f} strength {abs(a - 0.5):.4f} "
            f"(observed {members[g]['leak_strength']})")
    control["reading"] = (
        "the negative control: with the labels permuted the same instrument on the same feature "
        "space must read leak_strength ~0. A non-zero reading here would mean the probe "
        "manufactures signal from the feature space and no observed number is evidence.")
    payload["negative_control_label_shuffled"] = control

    log("=== AMEND - two-sided reading of the transfer numbers ===")
    for name, b in payload["part2_arena_transfer"]["probes"].items():
        for sub, r in b["per_subset"].items():
            r["reading"] = two_sided(r["claim_only_auroc"])
        b["reading_pooled"] = two_sided(b["arena_auroc_pooled"])
        b["subset_mean_leak_strength"] = round(
            float(np.mean([r["reading"]["leak_strength"] for r in b["per_subset"].values()])), 4)
        b["subsets_at_or_above_mild"] = sorted(
            s for s, r in b["per_subset"].items() if r["reading"]["leak_strength"] >= 0.05)
        b["which_number_is_comparable"] = (
            "arena_auroc_subset_mean. The arena's positive rate runs from 0.468 (expertqa) to "
            "0.944 (tatqa), so arena_auroc_pooled rewards a probe merely for ranking one subset "
            "above another and overstates within-subset transfer; the campaign's arena mean is "
            "itself a mean of per-subset AUROCs, so the subset-mean is the like-for-like number "
            "and the pooled figure is carried as a diagnostic.")
        log(f"  arena {name}: pooled {b['reading_pooled']['auroc']:.4f} strength "
            f"{b['reading_pooled']['leak_strength']:.4f} -> {b['reading_pooled']['band']}; "
            f"subset-mean strength {b['subset_mean_leak_strength']:.4f}")
    for name, r in payload["part3_gold_full"]["probes"].items():
        r["reading"] = two_sided(r["claim_only_auroc"])
        log(f"  gold_full {name}: {r['reading']['auroc']:.4f} strength "
            f"{r['reading']['leak_strength']:.4f} -> {r['reading']['band']}")
    payload["part3_gold_full"]["claim_char_length_reading"] = two_sided(
        payload["part3_gold_full"]["claim_char_length_auroc"])
    payload["part3_gold_full"]["claim_token_count_reading"] = two_sided(
        payload["part3_gold_full"]["claim_token_count_auroc"])

    payload["verdict_rule"] = {
        "amended_by": "coordinator correction, applied in stage `amend`",
        "leak_strength": "|AUROC - 0.5| - the test is TWO-SIDED; an inverted AUROC is signal, "
                         "not cleanliness (0.155 inverts to 0.845)",
        "equivalent_one_sided_auroc": "0.5 + leak_strength, so members are comparable",
        "bands": {"clean": "< 0.05", "mild": "0.05 - 0.15", "leak": "0.15 - 0.30",
                  "severe": ">= 0.30"},
        "split_selection": ("verdict_bearing_split is doc_disjoint for any member with MEASURED "
                            "pair structure (pair_census.pair_structure) and stratified "
                            "otherwise; every member carries both numbers and the reason it "
                            "was chosen, so a reader can see which number decided the verdict"),
        "superseded": ("verdict_read_v1_one_sided - the first pass's one-sided rule "
                       "(clean < 0.55, leak >= 0.55, severe >= 0.80), kept for audit"),
    }
    payload["note"] = "Numbers recorded, not adjudicated - the coordinator adjudicates."
    OUT.write_text(json.dumps(payload, indent=2))
    log(f"amended results -> {OUT}")

    log("=== SUMMARY (worst-first, verdict-bearing leak strength) ===")
    for g, r in sorted(members.items(), key=lambda kv: -kv[1]["leak_strength"]):
        log(f"  {g:18s} n={r['rows']:>7} split={r['verdict_bearing_split']:12s} "
            f"auroc {r['reading'][r['verdict_bearing_split']]['auroc']:.4f} "
            f"strength {r['leak_strength']:.4f} one-sided {r['equivalent_one_sided_auroc']:.4f} "
            f"{r['verdict_read']} ({r['direction']})")


def main():
    log("R20 claim-only sweep - CPU only, no GPU touched")
    payload = {
        "experiment": "R20 claim-only shortcut sweep",
        "question": ("how far does the HaluEval claim-only shortcut extend across the "
                     "assembled mix, and does it transfer to the blind arena?"),
        "instrument": ("claim-only probe: TF-IDF char_wb(2,5) + word(1,2) n-grams of the "
                       "CLAIM STRING ALONE (evidence never seen), hstacked, into "
                       "LogisticRegression(solver=liblinear, C=4.0, tol=1e-7, max_iter=3000); "
                       "per-member read is a fixed-seed (0) stratified 70/30 split, AUROC on "
                       "the held-out 30%. A document-disjoint 70/30 split (rows sharing an "
                       "evidence chunk kept on one side) is carried beside it because a random "
                       "split lets a pair's two legs straddle the split."),
        "loaders": {
            "mix": ("R10-H108_lane.public_train() + R17-H146_lane.parquet (quant_misbind, "
                    "30,000) + R18-H150_scaleunit_lane.parquet (quant_scale_unit, 5,540) - the "
                    "R18-H150_arm_run.py flagship assembly, 721,210 rows, 14 DANN groups. "
                    "public_train is called with its own evidence truncation in place: "
                    "R16-H142_G1_arm.untruncated_evidence lifts the cut on the EVIDENCE only, "
                    "so claims, labels, tags and the row set are identical either way and this "
                    "sweep reads no evidence."),
            "arena": ("R8-H77_unseen_arena.load_subsets() - the banked RAGBench loader used by "
                      "R8-H92_decomposed_arena.py and R16-H142_G1_reads.py; archive "
                      "data/external/datasets/dataset-ragbench.zip, 250/subset seed 0"),
            "gold_full": "R10-H108_lane.gold_full()",
            "member_identity": "the DANN group tag public_train / the lane assembly assigns",
        },
        "reading_bars": BARS,
        "surface_channels": ("claim_char_length_auroc and claim_token_count_auroc are raw AUROC "
                             "of the label against the unfitted statistic (token regex "
                             "[^\\W\\d_]+|\\d+, unicode); 0.50 is no signal in either direction"),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    OUT.write_text(json.dumps(payload, indent=2))

    claims, y, tags, doc = build_mix()
    part1(claims, y, tags, doc, payload)

    log("=== fitting the two transfer probes ===")
    hidx = np.flatnonzero(np.array(tags) == "halueval")
    probe_hal = fit_probe([claims[i] for i in hidx], y[hidx], tag="halueval/100%")
    probe_mix = fit_probe(claims, y, tag="wholemix/100%")
    probes = {"fit_on_halueval": probe_hal, "fit_on_whole_mix": probe_mix}
    payload["transfer_probes"] = {
        "fit_on_halueval": {"fit_rows": int(hidx.size), "fit_on": "100% of the halueval member"},
        "fit_on_whole_mix": {"fit_rows": int(len(claims)),
                             "fit_on": "100% of the assembled 721,210-row mix"},
    }
    OUT.write_text(json.dumps(payload, indent=2))

    part2(probes, payload)
    part3(probes, payload)

    payload["interpretation"] = (
        "An arena claim-only AUROC near 0.50 means the shortcut does NOT transfer and the "
        "exposure is contained inside the member. Materially above 0.50 means the mix teaches "
        "a claim-shape prior that the arena rewards, and the flagship's 0.71218 arena mean is "
        "partly a shortcut."
    )
    OUT.write_text(json.dumps(payload, indent=2))

    log("=== SUMMARY (worst-first per-member claim-only AUROC) ===")
    rows = sorted(payload["part1_per_member"].items(),
                  key=lambda kv: -(kv[1]["claim_only_auroc"] or 0))
    for g, r in rows:
        log(f"  {g:18s} n={r['rows']:>7} pos={r['label_balance_positive']:.3f} "
            f"claim-only {r['claim_only_auroc']}  doc-disjoint "
            f"{r['claim_only_auroc_doc_disjoint']}  {r['verdict_read']}")
    for name in ("fit_on_halueval", "fit_on_whole_mix"):
        b = payload["part2_arena_transfer"]["probes"][name]
        log(f"  arena {name:18s} pooled {b['arena_auroc_pooled']}  "
            f"subset-mean {b['arena_auroc_subset_mean']}")
        log(f"  gold_full {name:14s} {payload['part3_gold_full']['probes'][name]['claim_only_auroc']}")
    log(f"results -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sweep", choices=("sweep", "amend"),
                    help="sweep = parts 1-3; amend = the coordinator's two-sided verdict rule, "
                         "split selection and label-shuffled negative control, applied to the "
                         "banked JSON without re-fitting any stage-1 probe")
    a = ap.parse_args()
    (main if a.stage == "sweep" else amend)()
