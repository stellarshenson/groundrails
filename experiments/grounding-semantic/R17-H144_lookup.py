"""R17-H144 amendment A1 - the verbatim-lookup family (CPU, minutes).

Cycle 1 cleared the pooled bar (0.8171) and missed the conjunctive control clause
(0.8696 < 0.90) because the SFT corpus was 100% derivation pairs: verifying a
value that is simply STATED in the table was an untrained family, and the student
conflated neighbouring rows. This builds that missing family.

Every table, corpus loader, serializer, surface-parity test and content-based
eval-set exclusion is IMPORTED from `R17-H144_pairs.py` rather than reimplemented,
so the disjointness bar and the leak discipline are the banked ones.

Per pair, one positive and one negative over the SAME table, column and row:

  positive          the claim states the table's own value for the row it names
  N_absent          the value is replaced by one that appears nowhere in the
                    table - the eval controls' own negative construction, but
                    surface-parity matched where constructible (same leading
                    digit, digit count, trailing zeros, decimal, sign), so the
                    family cannot be won by reading the numeral's shape
  N_adjacent_misbind  the claim asserts the value of the row ADJACENT to the one
                    it names in the serialized chunk - the exact failure the
                    cycle-1 eval diagnosed. The figure is real and present; only
                    the binding is wrong

Every row carries a deterministic template trace in the teacher's own
`<think>...</think>` shape - no teacher inference is needed, the reasoning for a
lookup is mechanical. Three phrasings per class, rotated by pair index, so the
student learns the operation rather than one string.

Leak bar: claim-only TF-IDF probe (char_wb 2-5, liblinear at tol 1e-7 - the
converged configuration, never default lbfgs) below 0.55 AUROC on a
document-disjoint split.

Run:  python experiments/grounding-semantic/R17-H144_lookup.py
"""

import collections
import importlib.util
import math
import os
import json
import pathlib
import random

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R17-H144_lookup.parquet"
MANIFEST = HERE / "R17-H144_lookup_manifest.json"

# amendment A2: the same generator, a different seed, on documents disjoint from
# BOTH the SFT corpus and the banked eval set - the held-out control-family
# validation split the checkpoint selector reads. Same lineage by construction,
# because it is the same code.
VALSPLIT = os.environ.get("H144_VALSPLIT") == "1"
SEED = 11442 if VALSPLIT else 11441
N_PAIRS = 900 if VALSPLIT else 3_500      # 1,800 rows / 7,000 rows
DOC_CAP = 2
BODY_ROWS = 6
if VALSPLIT:
    OUT = HERE / "R17-H144_ctrlval.parquet"
    MANIFEST = HERE / "R17-H144_ctrlval_manifest.json"
SFT_CORPUS = [HERE / "R17-H144_pairs.parquet", HERE / "R17-H144_lookup.parquet"]


def _pairs_module():
    spec = importlib.util.spec_from_file_location("h144pairs", HERE / "R17-H144_pairs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _pairs_module()


# --------------------------------------------------------------------------- #
# deterministic traces - the reasoning a lookup actually needs
# --------------------------------------------------------------------------- #
POS = [
    "Locating the asserted figure: the claim says the {col} of {key} is {a}. "
    "Reading the row for {key} in the table, its {col} is {t}. The asserted "
    "figure matches the table's own value, and it is read off the row the claim "
    "names. The claim is supported.",
    "The claim asserts {a} as the {col} of {key}. I find the {key} row and read "
    "its {col}: {t}. Same figure, same row - the binding and the value both "
    "agree with the table.",
    "Checking {key} against the table. The table gives {t} for its {col}; the "
    "claim asserts {a}. These are the same value, taken from the row the claim "
    "identifies, so the claim is grounded in the evidence.",
]
ABSENT = [
    "Locating the asserted figure: the claim says the {col} of {key} is {a}. "
    "Reading the row for {key} in the table, its {col} is {t}, not {a}. Nor does "
    "{a} appear anywhere else in the table. The asserted value is absent.",
    "The claim asserts {a} as the {col} of {key}. The {key} row gives {t} "
    "instead. Scanning the rest of the table, no cell holds {a} at all - the "
    "figure is not in the evidence.",
    "Checking {key} against the table. Its {col} is {t}; the claim asserts {a}. "
    "The two disagree, and {a} is nowhere in the table, so the claim states a "
    "value the evidence does not contain.",
]
MISBIND = [
    "Locating the asserted figure: the claim says the {col} of {key} is {a}. "
    "Reading the row for {key}, its {col} is {t}, not {a}. The figure {a} does "
    "appear in the table - it is the {col} of {other}, the adjacent row. The "
    "value has been read off the wrong row, so the binding is wrong.",
    "The claim asserts {a} as the {col} of {key}. The {key} row gives {t}. {a} "
    "belongs to {other}, the neighbouring row, not to {key}. The value is real "
    "but bound to the wrong subject.",
    "Checking {key} against the table. Its {col} is {t}; the claim asserts {a}. "
    "{a} is present in the evidence, but as the {col} of {other} - a different "
    "row. The claim attaches the right number to the wrong row.",
]


def trace(kind, i, col, key, asserted, true_v, other=None):
    bank = {"pos": POS, "absent": ABSENT, "misbind": MISBIND}[kind]
    return bank[i % len(bank)].format(col=col, key=key, a=asserted, t=true_v,
                                      other=other or "")


# --------------------------------------------------------------------------- #
NGRAM_TOL = 0.5           # the banked N7 slack, on this family's own reference
_REF = None


def cell_reference(tables):
    """Character 2/3-gram counts over the TABLE CELL values this family asserts.

    The banked `P.naturalness` yardstick is built over the v2 lane's DERIVED
    results; a lookup family's positives are raw table cells - years, ranks,
    populations - a different distribution entirely, so the reference is rebuilt
    over cells or the tolerance means nothing here."""
    global _REF
    if _REF is None:
        c = collections.Counter()
        for t in tables[:6000]:
            for r in t["body"]:
                for cell in r:
                    v = cell.strip()
                    if P.as_num(v) is None:
                        continue
                    for n in (2, 3):
                        for i in range(len(v) - n + 1):
                            g = v[i:i + n]
                            if all(ch.isdigit() or ch == "." for ch in g):
                                c[g] += 1
        _REF = c
    return _REF


def naturalness(s):
    c = cell_reference(None)
    t = 0.0
    for n in (2, 3):
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            if all(ch.isdigit() or ch == "." for ch in g):
                t += math.log(c[g] + 1.0)
    return t


def absent_value(true_s, banned, rng, up):
    """A numeral absent from the table, surface-matched AND distribution-matched.

    Parity alone is not enough, measured: substituting an inner digit of a real
    cell keeps every surface statistic yet lands outside the value distribution
    ("2013" -> "2113" is not a year), and a character n-gram model reads exactly
    that - the within-pair claim-only probe sat at 0.8975 under parity alone.
    Candidates are adjacent digit TRANSPOSITIONS (digit multiset preserved) plus
    inner single-digit substitutions; of those that survive parity, absence and
    the requested error direction, take the one whose n-gram log-frequency is
    closest to the positive's, and reject the row when even the closest sits
    more than NGRAM_TOL away. The eval controls' random-integer fallback is not
    used - a candidate row that admits no such substitute is skipped.
    """
    base = naturalness(true_s)
    body = list(true_s)
    idx = [k for k, ch in enumerate(body) if ch.isdigit()]
    trailing = {k for k in idx if body[k] == "0" and all(
        (not c.isdigit()) or c == "0" for c in body[k + 1:])}
    cands = set()
    for i in range(len(true_s) - 1):          # adjacent transposition
        if true_s[i].isdigit() and true_s[i + 1].isdigit() and true_s[i] != true_s[i + 1]:
            t = list(true_s)
            t[i], t[i + 1] = t[i + 1], t[i]
            cands.add("".join(t))
    for k in idx[1:]:                          # inner substitution
        if k in trailing:
            continue
        for d in "0123456789":
            if d != body[k]:
                cands.add("".join(body[:k] + [d] + body[k + 1:]))
    best = None
    for cand in sorted(cands):
        if P.as_num(cand) is None or not P.surface_ok(true_s, cand):
            continue
        if P.canon_set(cand) & banned:
            continue
        try:
            if (float(cand.replace(",", "")) > float(true_s.replace(",", ""))) != up:
                continue
        except ValueError:
            continue
        d = abs(naturalness(cand) - base)
        if best is None or d < best[1]:
            best = (cand, d)
    if best is None or best[1] > NGRAM_TOL:
        return None, False
    return best[0], True


def build_pairs(tab, rng, pid, want_misbind, up):
    """One lookup pair over `tab`; returns (rows, used_misbind) or (None, None)."""
    lab_ci = tab["lab_ci"]
    cols = P.numeric_columns(tab["hdr"], tab["body"], lab_ci)
    if not cols:
        return None, None
    ci, vals = cols[rng.randrange(len(cols))]
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    if not P.good_header(col):
        return None, None

    labs = [r[lab_ci].strip() for r in tab["body"]]
    dup = {k for k, n in collections.Counter(labs).items() if n > 1}

    rows_with_val = [ri for ri, _ in vals]
    keep = set(rng.sample(rows_with_val, min(len(rows_with_val), BODY_ROWS)))
    others = [r for r in range(len(tab["body"])) if r not in keep]
    rng.shuffle(others)
    keep |= set(others[: BODY_ROWS - len(keep)])
    ordered = sorted(keep)
    body = [tab["body"][r] for r in ordered]
    chunk = P.serialize(tab["form"], tab["caption"], tab["hdr"], body, lab_ci, (ci,))
    table_nums = P.present_numbers(" ".join(
        [tab["caption"]] + tab["hdr"] + [c for r in tab["body"] for c in r]))
    chunk_nums = P.present_numbers(chunk)
    banned = table_nums | chunk_nums

    # candidate target rows: value present verbatim in the chunk, key unique
    cands = []
    for k, ri in enumerate(ordered):
        key = tab["body"][ri][lab_ci].strip()
        v = tab["body"][ri][ci].strip()
        if not key or key in dup or not v or P.as_num(v) is None:
            continue
        if v not in chunk or key not in chunk:
            continue
        cands.append((k, ri, key, v))
    rng.shuffle(cands)

    for k, ri, key, v in cands:
        claim_pos = f"The {col} of {key} is {v}."
        if claim_pos not in chunk and tab["form"] == "row_prose":
            continue                      # positive must be verbatim where it can be
        if want_misbind:
            # neighbours AS SERIALIZED - the conflation the student actually makes
            nb = [j for j in (k - 1, k + 1) if 0 <= j < len(ordered)]
            rng.shuffle(nb)
            best = None
            for j in nb:
                rj = ordered[j]
                ok = tab["body"][rj][lab_ci].strip()
                ov = tab["body"][rj][ci].strip()
                if not ok or not ov or ov == v or P.as_num(ov) is None or ov not in chunk:
                    continue
                cand = (ok, ov, P.surface_ok(v, ov))
                if cand[2]:
                    best = cand
                    break
                best = best or cand
            # On the A2 validation split the residual table pool is small and
            # atypical (only tables matching neither the eval set nor the SFT
            # corpus survive), and a best-effort parity preference leaves the
            # misbind half readable from the claim alone at 0.624. There parity
            # is REQUIRED, not preferred - a row whose neighbours all fail it is
            # skipped. The banked training family is untouched.
            if best is None or (VALSPLIT and not best[2]):
                continue
            other, a, parity = best
            fam, kind = "lookup:adjacent_misbind", "misbind"
        else:
            # parity needs an inner digit to move that is neither the leading
            # digit nor part of a trailing-zero run
            if P.digits(v) < 3:
                continue
            a, parity = absent_value(v, banned, rng, up)
            if a is None:
                continue
            other, fam, kind = None, "lookup:absent", "absent"

        claim_neg = f"The {col} of {key} is {a}."
        base = dict(chunk=chunk, doc_id=tab["doc_id"], source=tab["source"],
                    neg_family=fam, tag="lookup", claim_form="lookup",
                    serial_form=tab["form"], column=col, key_a=key,
                    key_b=other or "", v_pos=v, v_neg=a, surface_parity=parity)
        return [
            dict(pair_id=pid, label=1, claim=claim_pos, verdict="GROUNDED",
                 think=trace("pos", pid, col, key, v, v), **base),
            dict(pair_id=pid, label=0, claim=claim_neg, verdict="UNGROUNDED",
                 think=trace(kind, pid, col, key, a, v, other), **base),
        ], want_misbind
    return None, None


# --------------------------------------------------------------------------- #
def leak_probe(df, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    docs = sorted(df["doc_id"].unique().to_list())
    rng.shuffle(docs)
    cut = int(0.7 * len(docs))
    tr = df.filter(pl.col("doc_id").is_in(set(docs[:cut])))
    te = df.filter(pl.col("doc_id").is_in(set(docs[cut:])))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                          max_features=300_000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["claim"].to_list())
    clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
    clf.fit(Xtr, tr["label"].to_list())
    s = clf.decision_function(vec.transform(te["claim"].to_list()))
    pooled = P.auroc(te["label"].to_list(), s)

    per_fam, worst = {}, 0.0
    scored = te.select(["pair_id", "label", "neg_family"]).with_columns(pl.Series("score", s))
    for fam, sub in scored.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        if not len(piv):
            continue
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        per_fam[fam[0]] = {"acc": round(acc, 4), "pairs": len(piv)}
        worst = max(worst, acc)
    return (dict(claim_only_tfidf_auroc=round(float(pooled), 4), bar="< 0.55",
                 **{"pass": bool(pooled < 0.55)}, train_docs=cut,
                 test_docs=len(docs) - cut, test_rows=len(te)),
            dict(per_family=per_fam, worst=round(worst, 4), bar="< 0.60",
                 **{"pass": bool(worst < 0.60)}))


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)
    n_eval_prints, n_sft_prints, n_sft_ids = len(prints), 0, 0
    if VALSPLIT:
        # disjointness from the TRAINING corpus too, enforced on content as well
        # as on ids: FEVEROUS ids are order-unstable across rebuilds, so an id
        # test alone cannot carry the bar
        for src in SFT_CORPUS:
            d = pl.read_parquet(src, columns=["chunk", "doc_id"]).unique(subset=["chunk"])
            excluded_ids |= set(d["doc_id"].to_list())
            for c in d["chunk"].to_list():
                fp = frozenset(P.present_numbers(c))
                if len(fp) >= 4:
                    prints.append((c.lower(), fp))
        n_sft_prints = len(prints) - n_eval_prints
        n_sft_ids = len(excluded_ids)
        print(f"SFT corpus: +{n_sft_prints} content fingerprints, "
              f"{n_sft_ids} excluded doc_ids total", flush=True)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = P.tabfact_tables() + P.feverous_tables()
    drop_idx = P.excluded_tables(raw, prints)
    tables, dropped = [], 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None or not P.numeric_columns(t["hdr"], t["body"], lab):
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval content; "
          f"{dropped} dropped; {len(tables)} admitted", flush=True)

    forms = list(P.FORM_WEIGHTS)
    w = np.array([P.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]

    order = list(range(len(tables)))
    rng.shuffle(order)
    rows, per_doc, pid, n_mis, n_abs = [], collections.Counter(), 0, 0, 0
    dir_count = collections.Counter()
    cell_reference(tables)
    for ti in order:
        if pid >= N_PAIRS:
            break
        t = tables[ti]
        for _ in range(DOC_CAP):
            if pid >= N_PAIRS or per_doc[t["doc_id"]] >= DOC_CAP:
                break
            want_mis = n_mis <= n_abs            # 50/50 across the negatives
            up = dir_count["up"] <= dir_count["down"]
            got, used = build_pairs(t, rng, pid, want_mis, up)
            if got is None:
                break
            rows.extend(got)
            per_doc[t["doc_id"]] += 1
            pid += 1
            n_mis += used
            n_abs += not used
            if not used:
                dir_count["up" if float(got[1]["claim"].rsplit(' is ', 1)[1][:-1].replace(',', '')) > float(got[0]["claim"].rsplit(' is ', 1)[1][:-1].replace(',', '')) else "down"] += 1

    df = pl.DataFrame(rows)
    df.write_parquet(OUT)
    print(f"\n{df.height} rows / {pid} pairs over {df['doc_id'].n_unique()} documents",
          flush=True)
    print(df.group_by("neg_family").len().sort("neg_family"), flush=True)
    print(df.group_by("serial_form").len().sort("len", descending=True), flush=True)

    probe, within = leak_probe(df, rng)
    parity = float(df.filter(pl.col("label") == 0)["surface_parity"].mean())
    man = dict(seed=SEED, rows=df.height, pairs=pid,
               documents=df["doc_id"].n_unique(),
               families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
               serial_forms={k: v for k, v in df.group_by("serial_form").len().iter_rows()},
               sources={k: v for k, v in df.group_by("source").len().iter_rows()},
               surface_parity_rate=round(parity, 4),
               valsplit=VALSPLIT,
               eval_disjointness=dict(excluded_doc_ids=len(excluded_ids),
                                      eval_content_fingerprints=n_eval_prints,
                                      sft_content_fingerprints=n_sft_prints,
                                      content_matched_tables=len(drop_idx),
                                      admitted_tables=len(tables),
                                      shared_documents=0),
               claim_only_probe=probe, within_pair_claim_only=within)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "surface_parity_rate",
                       "claim_only_probe", "within_pair_claim_only")}, indent=2), flush=True)
    ok = probe["pass"] and within["pass"]
    print(f"=== LOOKUP FAMILY {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
