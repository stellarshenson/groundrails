"""R20-H175b - MIX-DISJOINT rebuild of the question-relevance held-out eval. CPU only.

WHY THIS EXISTS
---------------
The banked eval `R20-H175b_qlane_eval.parquet` (1,001 pairs / 487 passages) was
verified document-disjoint from the CONTRAST LANE only.  It was never checked
against the TRAINING MIX.  449 of its 487 passages (92.2%) are present in the
mix through the 61,712 PsiloQA rows `R10-H108_lane.public_train()` carries, so a
model can score that eval by recalling which question went with which answer over
a passage it trained on, without learning question relevance at all.  Both
registered floors (0.5000 question-blind, 0.5816 surface probe) are blind to
that: a memoriser still needs the question.

WHAT THIS BUILDS
----------------
The identical construction on passages that are ABSENT FROM THE ASSEMBLED MIX.
The supply premise the rebuild started from - "PsiloQA validation and test are
untouched by every mix path and are therefore clean by construction" - is FALSE
and is measured here rather than assumed: PsiloQA's splits are cut per question,
not per document, so 5,368 of 5,687 held-out passages are byte-identical to a
train passage that the mix carries.  The operative criterion is therefore
membership of the ASSEMBLED MIX, not split membership, and every candidate
passage is tested against the mix directly.

The clean residue is small and the ceiling is reported, never traded away.

REUSE
-----
Every construction rule, guard and bar comes from `R20-H175b_qlane.py`, imported
and called - derangement over a passage's distinct questions, overlap-matched
selection, answer-Jaccard <= 0.50, question-Jaccard <= 0.80, claim >= 15 chars
and >= 3 tokens, claim-to-passage containment >= 0.50, the surface-parity trim
and the whole `verify()` suite.  Only the SUPPLY changes.

Run:  uv run python experiments/grounding-semantic/R20-H175b_qlane_eval_clean.py
"""

import os

# CPU ONLY - every card is training.  `R10-H108_lane` imports torch and would
# otherwise default CUDA_VISIBLE_DEVICES to "1"; it uses `setdefault`, so this
# assignment wins and no device is ever enumerated.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import collections
import importlib.util as _ilu
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod("h175bqlane", HERE / "R20-H175b_qlane.py")      # the banked builder
C = B.C                                                   # banked lane instruments

OUT = HERE / "R20-H175b_qlane_eval_clean.parquet"
MANIFEST = HERE / "R20-H175b_qlane_eval_clean_manifest.json"
REPORT = HERE / "R20-H175b_qlane_eval_clean_report.json"
OLD_EVAL = HERE / "R20-H175b_qlane_eval.parquet"
LANE = HERE / "R20-H175b_qlane.parquet"

SEED_CLEAN = 3175          # fresh seed - the contaminated build used 1175 / 2175
PAIR_TARGET = 10_000       # deliberately unreachable: volume is whatever is clean

# "balanced" - greedy worst-deviation set selection over the whole clean pool.
# "prefix"   - the registered ascending-mismatch prefix rule, kept runnable so
#              the superseded 16-pair build stays reproducible.
SELECTOR = "prefix" if "--prefix" in sys.argv else "balanced"
SPLITS = ("train", "validation", "test")

# the three lanes the R20-H175b arm folds into the mix, in arm order
ARM_LANES = ("R17-H146_lane.parquet", "R18-H150_scaleunit_lane.parquet",
             "R20-H175b_qlane.parquet")


def norm(s):
    """Whitespace-collapsed, case-folded form - catches a passage that entered
    the mix through a path that re-wrapped it."""
    return " ".join(s.split()).casefold()


# --------------------------------------------------------------------------- #
# the assembled mix - every chunk the R20-H175b arm trains on
# --------------------------------------------------------------------------- #
def assemble_mix():
    """The arm's mix through the banked loader: `public_train()` with the
    evidence cut lifted (the H150/H175b twin protocol reads chunks UNTRUNCATED
    and windows them), plus all three lanes' chunks.  Returns the raw chunk set,
    the truncated-to-`chunk_max_chars` set, their normalised forms, and the
    PsiloQA (passage, question) -> llm_answer map the memorisation probe needs."""
    H108 = _mod("h108lane", HERE / "R10-H108_lane.py")
    M59 = H108.M59
    chunk_max = M59.CFG.chunk_max_chars
    print(f"mix: chunk_max_chars = {chunk_max}", flush=True)

    original = M59.CFG.chunk_max_chars
    M59.CFG.chunk_max_chars = 10**9          # `untruncated_evidence`, inlined
    try:
        claims, chunks, y, tags = H108.public_train()
    finally:
        M59.CFG.chunk_max_chars = original
    print(f"mix: clean public {len(y)} rows over {len(set(tags))} groups", flush=True)
    per_group = collections.Counter(tags)

    raw = set(chunks)
    del claims, chunks, y, tags

    for fname in ARM_LANES:
        p = HERE / fname
        if not p.exists():
            raise SystemExit(f"MIX ABORT: lane {fname} absent")
        d = pl.read_parquet(p)
        raw |= set(d["chunk"].to_list())
        per_group[fname] = d.height
        print(f"mix: lane {fname} {d.height} rows", flush=True)

    trunc = {c[:chunk_max] for c in raw}
    nraw = {norm(c) for c in raw}
    ntrunc = {norm(c) for c in trunc}
    print(f"mix: {len(raw)} distinct raw chunks, {len(trunc)} truncated", flush=True)
    return {"raw": raw, "trunc": trunc, "nraw": nraw, "ntrunc": ntrunc,
            "chunk_max": chunk_max, "rows_per_group": dict(per_group)}


def psiloqa_mix_rows():
    """The exact PsiloQA rows `public_train()` admits - the memorisation surface."""
    z = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    d = pl.read_parquet(io.BytesIO(
        z.read(next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    d = d.filter((pl.col("wiki_passage").str.len_chars() > 50)
                 & (pl.col("llm_answer").str.len_chars() > 10))
    return d


# --------------------------------------------------------------------------- #
# supply
# --------------------------------------------------------------------------- #
def clean_triples(mix):
    """Admitted (passage, question, claim) triples over passages the mix does not
    contain, pooled over ALL PsiloQA splits.

    Pooling is deliberate.  The splits are cut per question, so a passage the mix
    never saw can carry one question in `validation` and another in `test`; the
    unit the builder needs is a passage with >= 2 questions, and refusing to pool
    would discard clean questions for no cleanliness gain.  The cleanliness
    criterion is membership of the assembled mix, which is enforced here on every
    candidate regardless of which split contributed it."""
    z = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    parts = []
    for sp in SPLITS:
        d = pl.read_parquet(io.BytesIO(z.read(f"s-nlp__PsiloQA__{sp}.parquet")))
        parts.append(d.select(["wiki_title", "wiki_passage", "question",
                               "golden_answer", "lang"])
                     .with_columns(pl.lit(sp).alias("split")))
    d = pl.concat(parts).unique(subset=["wiki_passage", "question"],
                                keep="first", maintain_order=True)
    pooled = d.height
    pooled_pass = d["wiki_passage"].n_unique()

    passages = d["wiki_passage"].unique().to_list()
    dirty = {p for p in passages
             if p in mix["raw"] or p in mix["trunc"]
             or p[:mix["chunk_max"]] in mix["raw"] or p[:mix["chunk_max"]] in mix["trunc"]
             or norm(p) in mix["nraw"] or norm(p) in mix["ntrunc"]
             or norm(p)[:mix["chunk_max"]] in mix["ntrunc"]}
    d = d.filter(~pl.col("wiki_passage").is_in(list(dirty)))
    print(f"supply: pooled {pooled} triples / {pooled_pass} passages -> "
          f"{d.height} triples / {d['wiki_passage'].n_unique()} passages absent "
          f"from the mix ({len(dirty)} passages rejected as present)", flush=True)

    d = d.filter(pl.col("golden_answer").str.len_chars() >= B.CLAIM_MIN_CHARS)
    keep = [len(B.tok(a)) >= B.CLAIM_MIN_TOKENS
            and B.containment(a, p) >= B.GROUNDING_MIN
            for a, p in zip(d["golden_answer"].to_list(), d["wiki_passage"].to_list())]
    d = d.filter(pl.Series(keep))
    n2 = d.group_by("wiki_passage").len().filter(pl.col("len") >= 2).height
    print(f"supply: {d.height} admitted triples over {d['wiki_passage'].n_unique()} "
          f"passages, {n2} passages carrying >= 2 admitted questions", flush=True)
    return d, {"pooled_triples": pooled, "pooled_passages": pooled_pass,
               "passages_rejected_as_in_mix": len(dirty),
               "passages_absent_from_mix": pooled_pass - len(dirty),
               "admitted_triples": d.height,
               "admitted_passages": int(d["wiki_passage"].n_unique()),
               "passages_with_two_or_more_questions": n2,
               "split_contributions": {k: v for k, v in
                                       d.group_by("split").len().iter_rows()}}


# --------------------------------------------------------------------------- #
# disjointness, measured on the built artifact
# --------------------------------------------------------------------------- #
def mix_disjointness(df, mix):
    ev = sorted(set(df["chunk"].to_list()))
    cut = mix["chunk_max"]
    forms = {
        "eval_raw_in_mix_raw": sum(1 for p in ev if p in mix["raw"]),
        "eval_raw_in_mix_truncated": sum(1 for p in ev if p in mix["trunc"]),
        "eval_truncated_in_mix_raw": sum(1 for p in ev if p[:cut] in mix["raw"]),
        "eval_truncated_in_mix_truncated": sum(1 for p in ev if p[:cut] in mix["trunc"]),
        "eval_normalised_in_mix_normalised_raw":
            sum(1 for p in ev if norm(p) in mix["nraw"]),
        "eval_normalised_in_mix_normalised_truncated":
            sum(1 for p in ev if norm(p) in mix["ntrunc"]),
    }
    total = sum(forms.values())
    return {"eval_passages": len(ev), "found_in_mix_by_form": forms,
            "eval_passages_found_in_the_mix": total,
            "mix_distinct_raw_chunks": len(mix["raw"]),
            "mix_distinct_truncated_chunks": len(mix["trunc"]),
            "mix_rows_per_group": mix["rows_per_group"],
            "method": "exact string membership of the assembled mix chunk set - "
                      "public_train() with the evidence cut lifted, plus all "
                      f"three arm lanes - in raw, truncated-to-{cut} and "
                      "whitespace-collapsed case-folded forms",
            "bar": "0", "pass": total == 0}


def lane_disjointness(df):
    lane = pl.read_parquet(LANE)
    ev = set(df["chunk"].to_list())
    docs = set(df["doc_id"].to_list())
    shared_c = ev & set(lane["chunk"].to_list())
    shared_d = docs & set(lane["doc_id"].to_list())
    shared_q = set(df["question"].to_list()) & set(lane["question"].to_list())
    return {"lane_parquet": LANE.name, "lane_rows": lane.height,
            "shared_passages": len(shared_c), "shared_documents": len(shared_d),
            "shared_questions": len(shared_q),
            "bar": "0 shared passages", "pass": len(shared_c) == 0}


# --------------------------------------------------------------------------- #
# the memorisation feature the adversarial reviewer built
# --------------------------------------------------------------------------- #
# Memoised tokenisation - `channel_table` is called ~1,500 times by the balanced
# selector and re-tokenises the same long wiki passages every time.  Pure cache,
# no behaviour change: `B.jaccard`, `B.containment` and `B.channel_table` all
# resolve `tok` through this module's globals.
_TOKS = {}
_TOK_ORIG = B.tok


def _tok_cached(text):
    v = _TOKS.get(text)
    if v is None:
        v = _TOK_ORIG(text)
        _TOKS[text] = v
    return v


B.tok = _tok_cached


def subset_frame(units, chosen):
    """The finished artifact for a candidate passage set - emitted, then
    de-duplicated, exactly as the build path does it."""
    rows, uid, pid = [], 0, 0
    for i in sorted(chosen):
        new, pid = B.emit_unit(units[i], uid, pid)
        rows.extend(new)
        uid += 1
    return B.dedupe(pl.DataFrame(rows))


def worst_channel_deviation(df):
    """The registered surface-parity statistic, on this frame's OWN text - its
    own IDF and after its own de-duplication.  The selector optimises exactly the
    number the finished artifact is measured on."""
    ch = B.channel_table(df)
    y = df["label"].to_list()
    return max(abs(C.auroc(y, v) - 0.5) for v in ch.values())


def assemble_balanced(units):
    """Greedy worst-deviation balancing over the whole clean pool.

    NO BAR MOVES.  Surface parity stays 0.05, the trim margin stays
    `B.SURFACE_MARGIN` (0.04), every leak bar and the census stay as registered.
    What changes is the SELECTOR.  The registered rule admits passages in
    ascending mismatch order and keeps a PREFIX; that was chosen when the
    candidate pool held thousands of passages, where a prefix can afford to be
    crude.  On a 54-passage pool it is provably suboptimal - it can only cut
    where the fixed order happens to leave the channels balanced, and it reaches
    8 passages / 16 pairs where the pool supports far more.

    This selector chooses the SET.  It grows one passage at a time, always taking
    the passage that leaves the worst barred channel closest to 0.5, and keeps
    the LARGEST set reached that still sits inside the 0.04 trim margin.  The
    matching that picks each passage's (subset, derangement) is untouched - this
    only decides which passages are admitted."""
    pairs_of = {i: len(u["subset"]) for i, u in enumerate(units)}
    chosen, remaining = set(), set(range(len(units)))
    best, trace = None, []
    while remaining:
        scored = []
        for i in sorted(remaining):
            d = worst_channel_deviation(subset_frame(units, chosen | {i}))
            scored.append(((round(d, 9), -pairs_of[i], i), i, d))
        key, pick, dev = min(scored)
        chosen.add(pick)
        remaining.discard(pick)
        npairs = sum(pairs_of[u] for u in chosen)
        trace.append({"passages": len(chosen), "pairs": npairs,
                      "worst_deviation": round(float(dev), 4)})
        if len(chosen) >= 2 and dev <= B.SURFACE_MARGIN:
            best = (set(chosen), float(dev), npairs)
    if best is None:
        return None, {"selector": "greedy worst-deviation balancing",
                      "admissible": False,
                      "reason": "no set of two or more clean passages sits "
                                f"inside the {B.SURFACE_MARGIN} trim margin"}
    sel, dev, npairs = best
    df = subset_frame(units, sel)
    sel_report = {
        "selector": "greedy worst-deviation balancing over the whole clean pool",
        "bars_unchanged": True,
        "criterion": f"every barred channel within {B.SURFACE_MARGIN} of 0.5, "
                     "recomputed on the candidate subset's own text (own IDF, "
                     "after de-duplication)",
        "registered_surface_bar": B.SURFACE_BAR,
        "trim_margin": B.SURFACE_MARGIN,
        "pool_passages": len(units),
        "pool_pairs": sum(pairs_of.values()),
        "passages_kept": len(sel),
        "pairs_kept": int(df["pair_id"].n_unique()),
        "worst_deviation_kept": round(dev, 4),
        "superseded_prefix_rule": {"passages": 8, "pairs": 16,
                                   "note": "the registered ascending-mismatch "
                                           "prefix on this same pool"},
        "growth_trace": trace,
    }
    print(f"  [balanced] {sel_report['pairs_kept']} pairs over {len(sel)} "
          f"passages, worst channel deviation {dev:.4f}", flush=True)
    return df, sel_report


def channel_ceiling(units):
    """What the clean pool can and cannot deliver, measured exhaustively.

    `B.assemble` binary-searches the trim prefix, which assumes the predicate is
    monotone in prefix length.  At this scale it is not, so the same criterion is
    also applied by a full linear scan - and the registered surface bar (0.05) is
    scanned alongside the builder's stricter trim margin (0.04), so the report
    separates 'the builder's trim' from 'the bar itself'."""
    rows, uid, pid = [], 0, 0
    for u in units:
        new, pid = B.emit_unit(u, uid, pid)
        rows.extend(new)
        uid += 1
    df = pl.DataFrame(rows)
    ch = {k: np.asarray(v, dtype=float) for k, v in B.channel_table(df).items()}
    y = np.asarray(df["label"].to_list())
    ui = np.asarray(df["unit_id"].to_list())
    full = {k: round(float(C.auroc(y, v)), 4) for k, v in ch.items()}
    best = {}
    for name, margin in (("trim_margin_0.04", B.SURFACE_MARGIN),
                         ("registered_bar_0.05", B.SURFACE_BAR)):
        hit = None
        for cut in range(2, uid + 1):
            m = ui < cut
            dev = max(abs(C.auroc(y[m], v[m]) - 0.5) for v in ch.values())
            if dev <= margin:
                hit = {"passages": cut,
                       "pairs": int(df.filter(pl.col("unit_id") < cut)
                                    ["pair_id"].n_unique()),
                       "worst_deviation": round(float(dev), 4)}
        best[name] = hit
    return {"clean_pool_passages": uid, "clean_pool_pairs": pid,
            "per_channel_auroc_over_the_whole_clean_pool": full,
            "worst_channel_over_the_whole_clean_pool":
                round(max(abs(v - 0.5) for v in full.values()), 4),
            "largest_prefix_passing_exhaustive_scan": best,
            "reading": "the five question-x-claim lexical channels are what the "
                       "overlap matching exists to flatten; on this pool it has "
                       "no freedom to flatten them, and buying surface parity "
                       "costs almost the entire pool"}


def run_census():
    """The R14-H136 wall on the built artifact - 8-gram, Jaccard >= 0.3,
    bidirectional, spike control, against all ten walled arena corpora.

    `R20-H175b_lane_census.py` is imported and its `census_one` called directly
    rather than its `main()`, so the banked lane manifests are not rewritten as a
    side effect of censusing a new artifact."""
    CEN = _mod("h175bcensus", HERE / "R20-H175b_lane_census.py")
    arena_texts, _ = CEN.G.load_arena()
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    status = CEN.census_one("qlane_eval_clean", arena_texts)
    return CEN.census_summary("qlane_eval_clean") or {"status": status}


def language_coverage(df):
    """What the clean supply costs in language breadth, against the banked eval."""
    new = {k: v for k, v in sorted(df.group_by("lang").len().iter_rows(),
                                   key=lambda kv: -kv[1])}
    old = {}
    if OLD_EVAL.exists():
        old = {k: v for k, v in sorted(
            pl.read_parquet(OLD_EVAL).group_by("lang").len().iter_rows(),
            key=lambda kv: -kv[1])}
    return {"clean_eval_languages": len(new), "clean_eval_rows_per_language": new,
            "banked_eval_languages": len(old),
            "banked_eval_rows_per_language": old,
            "languages_lost": sorted(set(old) - set(new))}


def auroc_se(n_pos, n_neg, auc=0.80):
    """Hanley-McNeil standard error - what the achieved size can resolve."""
    if not n_pos or not n_neg:
        return None
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return round(float(var ** 0.5), 4)


def memorisation_feature(df, pmix, label):
    """AUROC of pure recall, with no question-relevance channel at all.

    For each eval row the mix is asked: what `llm_answer` did you pair with THIS
    leg's question over THIS passage?  Overlap of that answer with the eval claim
    is scored against the label.  A model that memorised the mix can compute this
    feature; a model that learned question relevance does not need it.  On an eval
    whose passages the mix does not contain, the lookup is empty and the feature
    does not exist - which is the whole point of the rebuild."""
    by_pq = collections.defaultdict(list)
    by_q = collections.defaultdict(list)
    for p, q, a in zip(pmix["wiki_passage"].to_list(), pmix["question"].to_list(),
                       pmix["llm_answer"].to_list()):
        by_pq[(p, q)].append(a)
        by_q[q].append(a)

    variants = {
        "jaccard": lambda c, a: B.jaccard(c, a),
        "claim_into_answer_containment": lambda c, a: B.containment(c, a),
        "answer_into_claim_containment": lambda c, a: B.containment(a, c),
        "shared_token_count": lambda c, a: float(
            len(set(B.tok(c)) & set(B.tok(a)))),
    }
    y = np.asarray(df["label"].to_list())
    out = {}
    for key, table in (("passage_and_question", by_pq), ("question_only", by_q)):
        lookup = [table.get((p, q) if key == "passage_and_question" else q, [])
                  for p, q in zip(df["chunk"].to_list(), df["question"].to_list())]
        covered = sum(1 for v in lookup if v)
        block = {"rows": df.height, "rows_with_a_mix_answer": covered,
                 "coverage": round(covered / df.height, 4) if df.height else 0.0}
        if covered == 0:
            block["auroc"] = None
            block["note"] = ("the mix pairs no answer with any (passage, question) "
                             "of this eval - the feature does not exist here")
        else:
            for vname, fn in variants.items():
                s = np.array([max((fn(c, a) for a in v), default=0.0)
                              for c, v in zip(df["claim"].to_list(), lookup)])
                block[vname] = round(float(C.auroc(y, s)), 4)
            block["auroc"] = max(block[v] for v in variants)
            block["strongest_variant"] = max(variants, key=lambda v: block[v])
        out[key] = block
    print(f"  memorisation [{label}]: {json.dumps(out)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"=== R20-H175b CLEAN EVAL REBUILD - mix-disjoint, seed {SEED_CLEAN}, "
          f"CPU only", flush=True)

    mix = assemble_mix()
    pmix = psiloqa_mix_rows()
    print(f"mix: psiloqa segment {pmix.height} rows over "
          f"{pmix['wiki_passage'].n_unique()} passages", flush=True)

    triples, supply = clean_triples(mix)
    if supply["passages_with_two_or_more_questions"] < 2:
        print("=== BLOCKED: no clean passage carries two admitted questions",
              flush=True)
        REPORT.write_text(json.dumps({"status": "BLOCKED", "supply": supply}, indent=2))
        raise SystemExit(2)

    idf, n_idf = B.build_idf(triples)
    units = B.passage_units(triples, idf, random.Random(SEED_CLEAN))
    avail = sum(len(u["subset"]) for u in units)
    print(f"candidate passages: {len(units)} (pairs available {avail})", flush=True)

    ceiling = channel_ceiling(units)
    print("ceiling: " + json.dumps(ceiling, indent=1), flush=True)

    if SELECTOR == "balanced":
        df, trim = assemble_balanced(units)
        if df is None:
            print("=== BLOCKED: no admissible balanced set", flush=True)
            REPORT.write_text(json.dumps({"status": "BLOCKED", "supply": supply,
                                          "selection": trim}, indent=2))
            raise SystemExit(2)
    else:
        df, trim = B.assemble(units, PAIR_TARGET, "eval_clean")
        df = B.dedupe(df)
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs / "
          f"{df['chunk'].n_unique()} passages -> {OUT.name}", flush=True)

    res = B.verify(df, random.Random(SEED_CLEAN))
    mixdis = mix_disjointness(df, mix)
    lanedis = lane_disjointness(df)
    print(f"mix disjointness: {json.dumps(mixdis['found_in_mix_by_form'])} "
          f"(pass={mixdis['pass']})", flush=True)
    print(f"lane disjointness: {json.dumps({k: lanedis[k] for k in ('shared_passages', 'shared_documents', 'shared_questions')})}",
          flush=True)

    mem_new = memorisation_feature(df, pmix, "clean eval")
    mem_old = None
    if OLD_EVAL.exists():
        mem_old = memorisation_feature(pl.read_parquet(OLD_EVAL), pmix,
                                       "banked contaminated eval")

    man = dict(
        experiment="R20-H175b - MIX-DISJOINT rebuild of the question-relevance "
                   "held-out eval (qrel_contrast)",
        supersedes=OLD_EVAL.name,
        reason="the banked eval was verified document-disjoint from the contrast "
               "lane only; 449 of its 487 passages are present in the training "
               "mix through the 61,712 PsiloQA rows public_train() carries, so a "
               "memorising model can score it without learning question relevance",
        registration="docs/experiments/semantic-grounding-experiments.md, block "
                     "'R20-H175b QUESTION CONDITIONING (measurement only)', "
                     "STAGE 0 - construction, guards and bars unchanged",
        tag=B.TAG, dann_group=B.TAG, parquet=OUT.name,
        seed_note="fresh seed 3175; the banked build used 1175 (train lane) and "
                  "2175 (eval)",
        builder="R20-H175b_qlane.py, imported and called - supply is the only "
                "thing this script changes",
        supply=dict(
            source="s-nlp/PsiloQA, all three splits pooled, deduplicated on "
                   "(wiki_passage, question)",
            cleanliness_criterion="the passage must be absent from the ASSEMBLED "
                                  "MIX (raw, truncated and normalised forms), "
                                  "which is measured, not inferred from split "
                                  "membership",
            split_premise_refuted="PsiloQA's splits are cut per question, not per "
                                  "document: 5,368 of 5,687 validation+test "
                                  "passages are byte-identical to a train passage "
                                  "the mix carries, so 'validation and test are "
                                  "clean by construction' is false",
            **supply),
        requires_question_channel=True,
        loader_warning=B_LOADER_WARNING,
        mix_disjointness=mixdis,
        lane_disjointness=lanedis,
        clean_supply_ceiling=ceiling,
        memorisation_feature=dict(
            construction="max token overlap between the eval claim and any "
                         "llm_answer the mix pairs with that leg's question over "
                         "that same passage; AUROC against the label, no "
                         "relevance channel involved",
            clean_eval=mem_new, banked_contaminated_eval=mem_old),
        **B.block(df, res, "eval_clean", SEED_CLEAN, PAIR_TARGET, trim))
    census = run_census()
    man["census"] = census
    MANIFEST.write_text(json.dumps(man, indent=2))

    bars_ok = bool(res["all_bars_pass"] and mixdis["pass"] and lanedis["pass"])
    npairs = int(df["pair_id"].n_unique())
    se = auroc_se(npairs, npairs)
    # The gate is a held-out AUROC >= 0.80 read against a 0.5816 surface-probe
    # floor.  Whether the artifact can carry that gate is a question about its
    # RESOLUTION, reported as a measured standard error - not a bar invented here.
    status = "BUILT" if bars_ok else "BARS FAILED"
    if bars_ok and se is not None and se > (0.80 - 0.5816) / 4:
        status = "BLOCKED - CLEAN SUPPLY TOO SMALL TO CARRY THE GATE"
    summary = {
        "status": status,
        "rows": man["rows"], "pairs": man["pairs"], "passages": man["passages"],
        "documents": man["documents"], "seed": SEED_CLEAN,
        "pairs_available_before_trim": avail,
        "selection": trim,
        "selector": SELECTOR,
        "gate_resolution": {
            "auroc_standard_error_at_this_size": se,
            "gate": 0.80,
            "banked_surface_probe_floor_from_the_contaminated_eval": 0.5816,
            "clean_eval_own_composite_probe":
                res["lexical_interaction_probe"]["value"],
            "gate_margin_in_standard_errors_over_the_banked_floor":
                round((0.80 - 0.5816) / se, 2) if se else None,
            "ci95_halfwidth_on_a_read_of_0.80":
                round(1.96 * se, 4) if se else None,
            "ci95_on_a_read_of_0.80":
                [round(0.80 - 1.96 * se, 4), round(0.80 + 1.96 * se, 4)]
                if se else None,
            "note": "Hanley-McNeil at AUROC 0.80 with the achieved pair count on "
                    "each label; the 0.5816 floor was calibrated on the "
                    "CONTAMINATED eval and does not transfer - the clean eval's "
                    "own composite lexical-interaction probe is reported beside "
                    "it"},
        "language_coverage": language_coverage(df),
        "clean_supply_ceiling": ceiling,
        "supply": supply,
        "mix_disjointness": mixdis,
        "lane_disjointness": lanedis,
        "census": census,
        "verify": res,
        "memorisation_feature": {"clean_eval": mem_new,
                                 "banked_contaminated_eval": mem_old},
        "seconds": round(time.time() - t0, 1),
    }
    REPORT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"=== R20-H175b CLEAN EVAL {status} ===", flush=True)
    raise SystemExit(0 if status == "BUILT" else 1)


B_LOADER_WARNING = (
    "THIS EVAL IS ONLY VALID UNDER A QUESTION-CONDITIONED PRESENTATION. Both rows "
    "of a pair carry the same claim and the same chunk and differ ONLY in "
    "`question`; read without composing the question the two legs are identical "
    "inputs with opposite labels and the AUROC is 0.5 by construction.")


if __name__ == "__main__":
    main()
