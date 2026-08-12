"""R17-H148 stage 1 - held-out PROCEDURAL-REGISTER probe (~1,000 minimal pairs).

Built from the staged R14-H136 procedural corpora (army-tm technical bulletins /
lubrication orders / MWOs, FAA AMT handbooks) via `R17-H148_extract.py`.

Every pair is a BARE minimal pair over one enumerated procedural list:

  misbound_step   positive cites the step NUMBER the list gives for a step's text;
                  the twin cites a different, real step number of the same list.
                  Only the numeral moves; the cited step's own text is dissimilar
                  (token Jaccard < 0.25) so the misbinding is groundable
  misbound_value  positive cites a numeral the claimed step states; the twin cites
                  a numeral that is real, printed in the chunk, but belongs to a
                  DIFFERENT step and is absent from the claimed step

Both legs' asserted values are present in the evidence, so presence is 1.0 by
construction and the defect is decidable from the chunk alone.

Discipline carried from R17-H146: converged liblinear claim-only probe at
tol 1e-7 over direction-stratified document-disjoint folds (bar < 0.55),
within-pair claim-only accuracy (bar < 0.60), a mechanical re-derivation audit
(bar 0 errors), digit-surface channels reported, both directions balanced 50/50
inside each family.

RESERVATION: the probe consumes every procedural block staged on disk today.
The lane build MUST exclude the doc_ids and chunk hashes recorded in the
manifest; the army-tm crawl (135 / 1,766 fetched) supplies the lane's documents.

CPU only.  Run with the conda interpreter (PyMuPDF/-built blocks, sklearn):
  /opt/conda/bin/python experiments/grounding-semantic/R17-H148_probe.py
"""
import collections
import hashlib
import json
import pathlib
import random
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
BLOCKS = HERE / "R17-H148_blocks.parquet"
OUT = HERE / "R17-H148_probe.parquet"
MANIFEST = HERE / "R17-H148_probe_manifest.json"

SEED = 1148
N_PAIRS = 1_000
STEP_SHARE = 0.60
BLOCK_CAP = 6
OVERBUILD = 2.2            # supply for the mirror-balance pass, which discards
MAX_STEP_DIST = 2          # a misbound step number is at most 2 away from the true one
CHUNK_MAX = 1500
N_FOLDS = 5
AUDIT_N = 100
JACCARD_MAX = 0.25

NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")
WORD = re.compile(r"[a-z0-9]+")

STEP_TEMPLATES = [
    "The procedure gives step {n} as: {a}",
    "Item {n} in the list states: {a}",
    "Step {n} of the listed procedure is: {a}",
    "According to the list, item {n} is: {a}",
    "The document records the following as step {n}: {a}",
    "In the numbered list, entry {n} reads: {a}",
]
VALUE_TEMPLATES = [
    "Step {n} of the listed procedure specifies {v}.",
    "The value given at step {n} is {v}.",
    "According to the list, item {n} gives {v}.",
    "Item {n} in the procedure states {v}.",
    "The document records {v} at step {n}.",
    "In the numbered list, entry {n} carries the value {v}.",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def canon_set(s):
    p = set()
    for m in NUM_FREE.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def toks(s):
    return set(WORD.findall(s.lower()))


def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / max(len(ta | tb), 1)


def as_num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, OverflowError):
        return None


def trailing_zeros(s):
    s = s.replace(",", "").rstrip(".")
    return float(len(s) - len(s.rstrip("0"))) if s.rstrip("0") != "" else 0.0


def digits(s):
    return float(sum(ch.isdigit() for ch in s))


def leading_digit(s):
    for ch in s.lstrip("+-"):
        if ch.isdigit():
            return float(ch)
    return 0.0


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y), np.asarray(s, dtype=float)))


def serialize(heading, nums, texts):
    """Chunk = heading + numbered items, capped at CHUNK_MAX; returns kept indices."""
    head = (heading or "").strip()
    out = [head] if head else []
    kept, size = [], len(head) + (1 if head else 0)
    for i, (n, t) in enumerate(zip(nums, texts)):
        line = f"{n}. {t}"
        if size + len(line) + 1 > CHUNK_MAX:
            break
        out.append(line)
        size += len(line) + 1
        kept.append(i)
    return "\n".join(out), kept


def chash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# builders - one candidate pair per call, or None
# --------------------------------------------------------------------------- #
def build_step(blk, rng, direction):
    nums, texts, chunk = blk["nums"], blk["texts"], blk["chunk"]
    order = list(range(len(nums)))
    rng.shuffle(order)
    for i in order:
        cands = [j for j in order
                 if j != i
                 and ((nums[j] > nums[i]) if direction == "up" else (nums[j] < nums[i]))
                 and abs(nums[j] - nums[i]) <= MAX_STEP_DIST
                 and jaccard(texts[i], texts[j]) < JACCARD_MAX]
        if not cands:
            continue
        j = cands[rng.randrange(len(cands))]
        action = texts[i].rstrip()
        if action not in chunk:
            continue
        return {"neg_family": "misbound_step", "direction": direction,
                "chunk": chunk, "action": action, "value": "",
                "item_index": i, "cited_item_index": j,
                "correct_key": str(nums[i]), "cited_key": str(nums[j]),
                "correct_value": str(nums[i]), "cited_value": str(nums[j])}
    return None


def build_value(blk, rng, direction):
    nums, texts, chunk = blk["nums"], blk["texts"], blk["chunk"]
    order = list(range(len(nums)))
    rng.shuffle(order)
    for i in order:
        own = [v for v in NUM_FREE.findall(texts[i]) if as_num(v) is not None]
        if not own:
            continue
        rng.shuffle(own)
        for vi in own:
            fi = as_num(vi)
            ci = canon_set(texts[i])
            for j in order:
                if j == i:
                    continue
                for vj in NUM_FREE.findall(texts[j]):
                    fj = as_num(vj)
                    if fj is None or vj == vi or fj == fi:
                        continue
                    if (fj > fi) != (direction == "up"):
                        continue
                    if canon_set(vj) & ci:            # cited value also readable in the claimed step
                        continue
                    if vi not in chunk or vj not in chunk:
                        continue
                    return {"neg_family": "misbound_value", "direction": direction,
                            "chunk": chunk, "action": "", "value": vi,
                            "item_index": i, "cited_item_index": j,
                            "correct_key": str(nums[i]), "cited_key": str(nums[i]),
                            "correct_value": vi, "cited_value": vj}
    return None


BUILDERS = {"misbound_step": build_step, "misbound_value": build_value}


def render(fam, ti, n, a, v):
    if fam == "misbound_step":
        return STEP_TEMPLATES[ti % len(STEP_TEMPLATES)].format(n=n, a=a)
    return VALUE_TEMPLATES[ti % len(VALUE_TEMPLATES)].format(n=n, v=v)


def emit(spec, blk, pid):
    fam = spec["neg_family"]
    ti = pid
    base = {k: spec[k] for k in ("chunk", "neg_family", "direction", "item_index",
                                 "cited_item_index", "correct_value", "cited_value")}
    pos_asserted, neg_asserted = spec["correct_value"], spec["cited_value"]
    base.update(doc_id=blk["doc_id"], corpus=blk["corpus"], block_id=blk["block_id"],
                template_id=ti % len(STEP_TEMPLATES), heading=blk["heading"],
                n_items=len(blk["nums"]))
    if fam == "misbound_step":
        pos = render(fam, ti, spec["correct_key"], spec["action"], "")
        neg = render(fam, ti, spec["cited_key"], spec["action"], "")
    else:
        pos = render(fam, ti, spec["correct_key"], "", spec["correct_value"])
        neg = render(fam, ti, spec["correct_key"], "", spec["cited_value"])
    parity = len(toks(pos) ^ toks(neg)) <= 2
    return [dict(pair_id=pid, label=1, claim=pos, surface_parity=parity,
                 asserted_value=pos_asserted, **base),
            dict(pair_id=pid, label=0, claim=neg, surface_parity=parity,
                 asserted_value=neg_asserted, **base)]


def mirror_balance(rows, rng):
    """Exact mirror on the misbound_step numeral: for every kept (a -> b) pair a
    (b -> a) pair is kept too, so the marginal distribution of the asserted step
    number is IDENTICAL on the positive and negative legs and every per-numeral
    feature is chance by construction, not by luck."""
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    step, value = [], []
    for pid, rr in by_pair.items():
        (step if rr[0]["neg_family"] == "misbound_step" else value).append((pid, rr))
    keyed = collections.defaultdict(list)
    for pid, rr in step:
        pos = next(x for x in rr if x["label"] == 1)
        keyed[(int(pos["correct_value"]), int(pos["cited_value"]))].append((pid, rr))
    kept, dropped = [], 0
    for (a, b), items in keyed.items():
        if a > b:
            continue
        other = keyed.get((b, a), [])
        k = min(len(items), len(other))
        dropped += (len(items) - k) + (len(other) - k)
        for lst in (items, other):
            rng.shuffle(lst)
            kept.extend(lst[:k])
    return kept, value, dropped


def trim(pairs, n, rng, mirrored=False):
    """Sample down to n pairs; mirrored families are trimmed in (a,b)/(b,a) units."""
    if len(pairs) <= n:
        return pairs
    if not mirrored:
        rng.shuffle(pairs)
        return pairs[:n]
    keyed = collections.defaultdict(list)
    for pid, rr in pairs:
        pos = next(x for x in rr if x["label"] == 1)
        keyed[(int(pos["correct_value"]), int(pos["cited_value"]))].append((pid, rr))
    out, keys = [], sorted({tuple(sorted(k)) for k in keyed})
    rng.shuffle(keys)
    quota = {k: 0 for k in keys}
    while sum(quota.values()) * 2 < n:
        progressed = False
        for k in keys:
            a, b = k
            avail = min(len(keyed[(a, b)]), len(keyed[(b, a)]))
            if quota[k] < avail and sum(quota.values()) * 2 < n:
                quota[k] += 1
                progressed = True
        if not progressed:
            break
    for k, q in quota.items():
        a, b = k
        for side in ((a, b), (b, a)):
            lst = keyed[side]
            rng.shuffle(lst)
            out.extend(lst[:q])
    return out


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng, by_block):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    claims, labels = df["claim"].to_list(), df["label"].to_list()

    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("doc_id")
               .agg((pl.col("neg_family") + ":" + pl.col("direction")).first()).iter_rows()}
    strata = collections.defaultdict(list)
    for d in sorted(doc_key):
        strata[doc_key[d]].append(d)
    fold_of, i = {}, 0
    for k in sorted(strata):
        ds = strata[k]
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % N_FOLDS
            i += 1
    folds = np.array([fold_of[d] for d in df["doc_id"].to_list()])
    score = np.zeros(len(df))
    idx = np.arange(len(df))
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        if not len(te_i):
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[j] for j in tr_i])
        Xte = vec.transform([claims[j] for j in te_i])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[j] for j in tr_i])
        score[te_i] = clf.decision_function(Xte)
    probe = auroc(labels, score)
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "direction-stratified, liblinear tol 1e-7",
        "documents": len(fold_of), "rows": len(df)}

    scored = df.select(["pair_id", "label", "neg_family"]).with_columns(pl.Series("score", score))
    fam_acc, worst, worst_two = {}, 0.0, 0.0
    for fam, sub in scored.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        if not len(piv):
            continue
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam[0]] = {"acc": round(acc, 4), "pairs": len(piv)}
        worst = max(worst, acc)
        worst_two = max(worst_two, abs(acc - 0.5))
    out["within_pair_claim_only_accuracy"] = {
        "per_family": fam_acc, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60),
        "worst_two_sided_deviation_report_only": round(worst_two, 4)}

    cited_ok = [v in c for v, c in zip(df["asserted_value"], df["chunk"])]
    corr_ok = [v in c for v, c in zip(df["correct_value"], df["chunk"])]
    out["value_presence"] = {
        "cited_verbatim_rate": round(float(np.mean(cited_ok)), 6),
        "correct_verbatim_rate": round(float(np.mean(corr_ok)), 6),
        "bar": "1.0 for both", "pass": bool(all(cited_ok) and all(corr_ok)),
        "note": "every asserted numeral is printed in the chunk; P(0 | absent) undefined"}

    tz = [trailing_zeros(v) for v in df["asserted_value"]]
    dc = [digits(v) for v in df["asserted_value"]]
    per_tz, per_dc = {}, {}
    for fam, sub in df.select(["neg_family", "label", "asserted_value"]).group_by("neg_family"):
        y = sub["label"].to_list()
        per_tz[fam[0]] = round(auroc(y, [trailing_zeros(v) for v in sub["asserted_value"]]), 4)
        per_dc[fam[0]] = round(auroc(y, [digits(v) for v in sub["asserted_value"]]), 4)
    a_tz, a_dc = auroc(labels, tz), auroc(labels, dc)
    dev_tz = max([abs(a_tz - .5)] + [abs(v - .5) for v in per_tz.values()])
    dev_dc = max([abs(a_dc - .5)] + [abs(v - .5) for v in per_dc.values()])
    out["trailing_zero_auroc"] = {"pooled": round(a_tz, 4), "per_family": per_tz,
                                  "bar": "in [0.45, 0.55]", "pass": bool(dev_tz <= 0.05)}
    out["digit_count_auroc"] = {"pooled": round(a_dc, 4), "per_family": per_dc,
                                "bar": "in [0.45, 0.55]", "pass": bool(dev_dc <= 0.05)}

    # --- mechanical audit: re-derive each sampled negative from its source block
    neg = df.filter(pl.col("label") == 0)
    samp = neg.sample(n=min(AUDIT_N, len(neg)), seed=SEED)
    errs = []
    for r in samp.iter_rows(named=True):
        b = by_block.get(r["block_id"])
        i, j = r["item_index"], r["cited_item_index"]
        why = None
        if b is None:
            why = "block not found"
        elif not (0 <= i < len(b["nums"]) and 0 <= j < len(b["nums"])):
            why = "item index out of range"
        elif i == j and r["neg_family"] == "misbound_step":
            why = "cited step equals the claimed step"
        elif r["cited_value"] == r["correct_value"]:
            why = "cited value equals the correct value"
        elif r["cited_value"] not in r["chunk"] or r["correct_value"] not in r["chunk"]:
            why = "an asserted value is not readable in the chunk"
        elif r["neg_family"] == "misbound_step":
            if str(b["nums"][i]) != r["correct_value"]:
                why = "correct step number is not the list's number for the claimed item"
            elif str(b["nums"][j]) != r["cited_value"]:
                why = "cited step number is not a real step number of the list"
            elif jaccard(b["texts"][i], b["texts"][j]) >= JACCARD_MAX:
                why = "cited step's text is not dissimilar from the claimed step"
            elif b["texts"][i] not in r["chunk"]:
                why = "claimed step text is not readable in the chunk"
        else:
            if r["correct_value"] not in b["texts"][i]:
                why = "correct value is not stated by the claimed step"
            elif r["cited_value"] not in b["texts"][j]:
                why = "cited value is not stated by the recorded source step"
            elif canon_set(r["cited_value"]) & canon_set(b["texts"][i]):
                why = "cited value is also readable in the claimed step"
            elif i == j:
                why = "cited value comes from the claimed step itself"
        if why:
            errs.append({"pair_id": r["pair_id"], "block_id": r["block_id"], "why": why})
    out["misbind_rederivation_audit"] = {"sampled": len(samp), "errors": len(errs),
                                         "bar": "0 errors", "pass": len(errs) == 0,
                                         "examples": errs[:5]}

    out["report_only"] = {
        "claim_char_length_auroc": round(auroc(labels, [len(c) for c in claims]), 4),
        "leading_digit_auroc": round(auroc(labels, [leading_digit(v) for v in df["asserted_value"]]), 4),
        "decimal_presence_auroc": round(auroc(labels, [1.0 if "." in v else 0.0
                                                       for v in df["asserted_value"]]), 4),
        "value_magnitude_auroc": round(auroc(labels, [as_num(v) or 0.0
                                                      for v in df["asserted_value"]]), 4),
        "surface_parity_rate": round(float(df["surface_parity"].cast(pl.Float64).mean()), 4),
    }
    out["all_bars_pass"] = all(out[k]["pass"] for k in
                               ("claim_only_tfidf_auroc", "within_pair_claim_only_accuracy",
                                "value_presence", "trailing_zero_auroc", "digit_count_auroc",
                                "misbind_rederivation_audit"))
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    raw = pl.read_parquet(BLOCKS)
    blocks = []
    for k, r in enumerate(raw.iter_rows(named=True)):
        chunk, kept = serialize(r["heading"], r["item_numbers"], r["item_texts"])
        if len(kept) < 3:
            continue
        blocks.append({"block_id": f"{r['doc_id']}#p{r['page']}#{k}", "doc_id": r["doc_id"],
                       "corpus": r["corpus"], "heading": r["heading"], "chunk": chunk,
                       "nums": [r["item_numbers"][i] for i in kept],
                       "texts": [r["item_texts"][i] for i in kept]})
    print(f"{raw.height} extracted blocks -> {len(blocks)} serializable "
          f"({len({b['doc_id'] for b in blocks})} documents)", flush=True)
    by_block = {b["block_id"]: b for b in blocks}

    n_step, n_value = int(round(N_PAIRS * STEP_SHARE)), int(round(N_PAIRS * (1 - STEP_SHARE)))
    target = {("misbound_step", "up"): int(round(n_step * OVERBUILD / 2)),
              ("misbound_step", "down"): int(round(n_step * OVERBUILD / 2)),
              ("misbound_value", "up"): int(round(n_value * OVERBUILD / 2)),
              ("misbound_value", "down"): int(round(n_value * OVERBUILD / 2))}
    built = collections.Counter()
    rows, per_block, seen, pid = [], collections.Counter(), set(), 0

    n_target = sum(target.values())
    for cap in range(1, BLOCK_CAP + 1):
        if pid >= n_target:
            break
        order = list(range(len(blocks)))
        rng.shuffle(order)
        for bi in order:
            if pid >= n_target:
                break
            b = blocks[bi]
            while per_block[b["block_id"]] < cap and pid < n_target:
                spec = None
                for fam, direction in sorted(target, key=lambda c: built[c] - target[c]):
                    if built[(fam, direction)] >= target[(fam, direction)]:
                        continue
                    spec = BUILDERS[fam](b, rng, direction)
                    if spec is not None:
                        break
                if spec is None:
                    break
                sig = (spec["chunk"], spec["neg_family"], spec["correct_key"],
                       spec["correct_value"], spec["cited_value"])
                if sig in seen:
                    break
                seen.add(sig)
                rows.extend(emit(spec, b, pid))
                built[(spec["neg_family"], spec["direction"])] += 1
                per_block[b["block_id"]] += 1
                pid += 1
        print(f"  cap {cap}: {pid} pairs "
              f"{ {f'{a}:{d}': n for (a, d), n in sorted(built.items())} }", flush=True)

    step_pairs, value_pairs, mirror_dropped = mirror_balance(rows, rng)
    step_pairs = trim(step_pairs, n_step, rng, mirrored=True)
    value_pairs = trim(value_pairs, n_value, rng)
    print(f"  mirror balance: dropped {mirror_dropped} unmirrored step pairs; "
          f"kept {len(step_pairs)} step + {len(value_pairs)} value", flush=True)
    rows = [r for _, rr in step_pairs + value_pairs for r in rr]

    df = pl.DataFrame(rows).unique(subset=["claim", "chunk", "label"],
                                   keep="first", maintain_order=True)
    keep = df.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"]
    df = df.filter(pl.col("pair_id").is_in(keep)).sort(["pair_id", "label"],
                                                       descending=[False, True])
    df.write_parquet(OUT)
    n_pairs = df["pair_id"].n_unique()
    print(f"\n{df.height} rows / {n_pairs} pairs over {df['doc_id'].n_unique()} documents, "
          f"{df['block_id'].n_unique()} blocks", flush=True)

    res = verify(df, rng, by_block)
    man = dict(
        seed=SEED, rows=df.height, pairs=n_pairs,
        documents=df["doc_id"].n_unique(), blocks=df["block_id"].n_unique(),
        pairs_per_block=round(n_pairs / max(df["block_id"].n_unique(), 1), 3),
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        directions={f"{a}:{b}": n for a, b, n in
                    df.group_by(["neg_family", "direction"]).len().iter_rows()},
        corpora={k: v for k, v in df.group_by("corpus").len().iter_rows()},
        diversity=dict(distinct_claims=df["claim"].n_unique(),
                       distinct_chunks=df["chunk"].n_unique(),
                       templates={str(k): v for k, v in
                                  df.group_by("template_id").len().iter_rows()}),
        construction=dict(
            mirror_balanced_family="misbound_step",
            mirror_dropped_pairs=mirror_dropped,
            max_step_distance=MAX_STEP_DIST,
            block_cap=BLOCK_CAP,
            jaccard_max=JACCARD_MAX),
        reservation=dict(
            rule="the probe consumes every procedural block staged on disk on "
                 "2026-08-11; the H148 lane build MUST exclude these doc_ids "
                 "(army-tm) and these chunk hashes (faa-amt)",
            army_tm_doc_ids=sorted(df.filter(pl.col("corpus") == "army-tm")["doc_id"].unique()),
            faa_doc_ids=sorted(df.filter(pl.col("corpus") == "faa-amt")["doc_id"].unique()),
            chunk_sha1_16=sorted({chash(c) for c in df["chunk"].unique()}),
            caveat="faa-amt holds only 3 documents, so PDF-level disjointness is "
                   "impossible there; the FAA reservation is chunk-level"),
        verify=res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "blocks", "families", "directions",
                       "corpora", "diversity", "verify")}, indent=2), flush=True)
    ok = res["all_bars_pass"]
    print(f"=== R17-H148 PROBE {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
