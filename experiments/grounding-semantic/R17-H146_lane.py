"""R17-H146 MISBIND-VERIFICATION LANE - build + verify only, CPU, no GPU.

Builds `R17-H146_lane.parquet`: ~30,000 rows = ~15,000 BARE-claim minimal pairs
over public TabFact-train and FEVEROUS-train tables, seed 1146, over documents
content-disjoint from the R17-H143 eval set.

The lane isolates the H133 core's *teaching* construction from the derivation
*task* that did H133's damage (R17-H145 falsification): every claim is a pure
lookup verification - the claim restates a value the table states, bound to a
named row and a named column.

  positive        the value the table gives for (row_key, column), verbatim
                  present in the serialized chunk, which states the binding
  negative twin   the byte-identical claim shape citing a value that is REAL and
                  present in the same chunk but belongs to a DIFFERENT row
                  (misbound_row) or a DIFFERENT column of the same row
                  (misbound_col).  The correct value is present too, so the
                  defect is groundable from the evidence alone

No derived values, no arithmetic, no traces.  Every asserted numeral is present
in the evidence, so `P(label 0 | value absent)` is undefined by construction and
the presence rate is 1.0 on both legs - recorded rather than tested.

Discipline carried from the banked builds:
  * every table, corpus loader, serializer, digit-surface parity test and the
    CONTENT-based eval-set exclusion are imported from `R17-H144_pairs.py`
  * converged liblinear probe at tol 1e-7, never default lbfgs (H144 finding ii)
  * direction-stratified document-disjoint folds (H145 finding b): a source
    document is often single-direction, and unstratified folds read the artifact
    below chance
  * both misbind directions (cited value above / below the correct one) balanced
    50/50 inside each family, which is what holds the digit-count and
    trailing-zero channels at chance

Run:  uv run python experiments/grounding-semantic/R17-H146_lane.py
"""

import collections
import importlib.util
import json
import pathlib
import random

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R17-H146_lane.parquet"
MANIFEST = HERE / "R17-H146_lane_manifest.json"
H144_PAIRS = HERE / "R17-H144_pairs.parquet"
H144_LOOKUP = HERE / "R17-H144_lookup.parquet"
H145_LANE = HERE / "R17-H145_scaleunit.parquet"

SEED = 1146
N_PAIRS = 15_000                       # 30,000 rows
ROW_SHARE = 0.70                       # misbound_row / misbound_col, H133-core-like
BODY_ROWS = 6
DOC_CAP_LADDER = (2, 3, 4, 5, 6, 8)    # max 8 pairs per document
N_FOLDS = 5
AUDIT_N = 500
CAND_TRIES = 24


def _pairs_module():
    spec = importlib.util.spec_from_file_location("h144pairs", HERE / "R17-H144_pairs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _pairs_module()


# --------------------------------------------------------------------------- #
# claim templates - the shape is identical inside a pair, only the numeral moves
# --------------------------------------------------------------------------- #
TEMPLATES = [
    "The {col} of {key} is {v}.",
    "{key} has a {col} of {v}.",
    "According to the table, the {col} of {key} is {v}.",
    "For {key}, the {col} is {v}.",
    "The table records a {col} of {v} for {key}.",
    "{key} is listed with a {col} of {v}.",
]


def render(ti, col, key, v):
    return TEMPLATES[ti % len(TEMPLATES)].format(col=col, key=key, v=v)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def _standalone(chunk, *vals):
    """Both values must be readable as numerals in their own right.

    A plain substring test can succeed on a fragment of a LONGER number when the
    1,500-character serving cap truncates the row that actually prints the cell,
    so presence is confirmed on the canonical numeral set as well."""
    pres = P.present_numbers(chunk)
    return all(P.canon_set(v) & pres for v in vals)


def _fillers(tab, keep, rng):
    others = [r for r in range(len(tab["body"])) if r not in keep]
    rng.shuffle(others)
    return sorted(keep | set(others[: max(0, BODY_ROWS - len(keep))]))


def _dup_labels(tab):
    labs = [r[tab["lab_ci"]].strip() for r in tab["body"]]
    return {k for k, n in collections.Counter(labs).items() if n > 1}


def build_row_misbind(tab, rng, want_dir):
    """Claim names row A and column C; the negative cites row B's value in C."""
    lab_ci = tab["lab_ci"]
    body, hdr = tab["body"], tab["hdr"]
    cols = P.numeric_columns(hdr, body, lab_ci)
    if not cols:
        return None
    dup = _dup_labels(tab)
    rng.shuffle(cols)
    for ci, vals in cols[:3]:
        col = (hdr[ci] or "").strip()
        if not P.good_header(col):
            continue
        idx = [ri for ri, _ in vals]
        if len(idx) < 2:
            continue
        best = None
        for _ in range(CAND_TRIES):
            ra, rb = rng.sample(idx, 2)
            sa, sb = body[ra][ci].strip(), body[rb][ci].strip()
            va, vb = P.as_num(sa), P.as_num(sb)
            if va is None or vb is None or abs(va - vb) < 1e-12 or sa == sb:
                continue
            ka, kb = body[ra][lab_ci].strip(), body[rb][lab_ci].strip()
            if not ka or not kb or ka == kb or ka in dup or kb in dup:
                continue
            lo, hi = ((ra, ka, sa), (rb, kb, sb)) if va < vb else ((rb, kb, sb), (ra, ka, sa))
            # "up" = the cited (wrong) value is ABOVE the correct one
            (rc, kc, sc), (rm, km, sm) = (lo, hi) if want_dir == "up" else (hi, lo)
            # the cited value must be unambiguously bound elsewhere: it may not
            # also sit anywhere in the row the claim names
            if any(c.strip() == sm for c in body[rc]):
                continue
            cand = (ci, col, rc, kc, sc, rm, km, sm, P.surface_ok(sc, sm))
            if cand[-1]:
                best = cand
                break
            best = best or cand
        if best is None:
            continue
        ci, col, rc, kc, sc, rm, km, sm, parity = best
        rows = _fillers(tab, {rc, rm}, rng)
        chunk = P.serialize(tab["form"], tab["caption"], hdr, [body[r] for r in rows],
                            lab_ci, (ci,))
        if not all(s in chunk for s in (sc, sm, kc, km)) or not _standalone(chunk, sc, sm):
            continue
        return dict(neg_family="misbound_row", direction=want_dir, chunk=chunk,
                    column=col, column_index=ci, row_key=kc, row_index=rc,
                    correct_value=sc, cited_value=sm,
                    misbound_column=col, misbound_column_index=ci,
                    misbound_row_key=km, misbound_row_index=rm,
                    surface_parity=parity, serial_form=tab["form"],
                    doc_id=tab["doc_id"], source=tab["source"])
    return None


def build_col_misbind(tab, rng, want_dir):
    """Claim names row A and column C; the negative cites A's value in column D."""
    lab_ci = tab["lab_ci"]
    body, hdr = tab["body"], tab["hdr"]
    cols = P.numeric_columns(hdr, body, lab_ci)
    if len(cols) < 2:
        return None
    dup = _dup_labels(tab)
    rng.shuffle(cols)
    for a in range(min(3, len(cols))):
        ci, vals_i = cols[a]
        col_i = (hdr[ci] or "").strip()
        if not P.good_header(col_i):
            continue
        for b in range(len(cols)):
            if b == a:
                continue
            cj, vals_j = cols[b]
            col_j = (hdr[cj] or "").strip()
            if not P.good_header(col_j) or col_j == col_i:
                continue
            mj = dict(vals_j)
            shared = [ri for ri, _ in vals_i if ri in mj]
            rng.shuffle(shared)
            col_i_vals = {r[ci].strip() for r in body}
            col_j_vals = {r[cj].strip() for r in body}
            best = None
            for ri in shared[:CAND_TRIES]:
                si, sj = body[ri][ci].strip(), body[ri][cj].strip()
                vi, vj = P.as_num(si), P.as_num(sj)
                if vi is None or vj is None or abs(vi - vj) < 1e-12 or si == sj:
                    continue
                key = body[ri][lab_ci].strip()
                if not key or key in dup:
                    continue
                # orient so that the cited value sits on the wanted side
                if (vj > vi) == (want_dir == "up"):
                    c_ci, c_col, sc = ci, col_i, si
                    m_ci, m_col, sm = cj, col_j, sj
                    claimed_vals, cited_vals = col_i_vals, col_j_vals
                else:
                    c_ci, c_col, sc = cj, col_j, sj
                    m_ci, m_col, sm = ci, col_i, si
                    claimed_vals, cited_vals = col_j_vals, col_i_vals
                # unambiguous: the cited value may not also occur in the claimed
                # column, and the correct value may not occur in the cited one
                if sm in claimed_vals or sc in cited_vals:
                    continue
                cand = (c_ci, c_col, sc, m_ci, m_col, sm, ri, key,
                        P.surface_ok(sc, sm))
                if cand[-1]:
                    best = cand
                    break
                best = best or cand
            if best is None:
                continue
            c_ci, c_col, sc, m_ci, m_col, sm, ri, key, parity = best
            rows = _fillers(tab, {ri}, rng)
            chunk = P.serialize(tab["form"], tab["caption"], hdr,
                                [body[r] for r in rows], lab_ci, (c_ci, m_ci))
            if not all(s in chunk for s in (sc, sm, key)) or not _standalone(chunk, sc, sm):
                continue
            return dict(neg_family="misbound_col", direction=want_dir, chunk=chunk,
                        column=c_col, column_index=c_ci, row_key=key, row_index=ri,
                        correct_value=sc, cited_value=sm,
                        misbound_column=m_col, misbound_column_index=m_ci,
                        misbound_row_key=key, misbound_row_index=ri,
                        surface_parity=parity, serial_form=tab["form"],
                        doc_id=tab["doc_id"], source=tab["source"])
    return None


BUILDERS = {"misbound_row": build_row_misbind, "misbound_col": build_col_misbind}


def emit(spec, pid, ti):
    """Two rows - the positive and its twin - from one construction."""
    col, key = spec["column"], spec["row_key"]
    base = {k: spec[k] for k in
            ("chunk", "neg_family", "direction", "doc_id", "source", "column",
             "column_index", "row_key", "row_index", "correct_value",
             "misbound_column", "misbound_column_index", "misbound_row_key",
             "misbound_row_index", "surface_parity", "serial_form")}
    base["template_id"] = ti % len(TEMPLATES)
    return [
        dict(pair_id=pid, label=1,
             claim=render(ti, col, key, spec["correct_value"]),
             cited_value=spec["correct_value"], **base),
        dict(pair_id=pid, label=0,
             claim=render(ti, col, key, spec["cited_value"]),
             cited_value=spec["cited_value"], **base),
    ]


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng, by_doc):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}

    # --- claim-only probe: document-disjoint K-fold, out of fold, folds
    # stratified on the document's (family, direction) so no fold's training
    # complement carries the mirrored direction (R17-H145 finding b).
    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("doc_id")
               .agg((pl.col("neg_family") + ":" + pl.col("direction")).first())
               .iter_rows()}
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
    claims, labels = df["claim"].to_list(), df["label"].to_list()
    score = np.zeros(len(df), dtype=float)
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
    probe = P.auroc(labels, score)
    out["claim_only_tfidf_auroc"] = {
        "value": round(float(probe), 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "direction-stratified, liblinear tol 1e-7",
        "documents": len(fold_of), "rows": len(df)}

    # --- within-pair claim-only accuracy, per family
    scored = df.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("score", score))
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

    # --- value presence, both legs.  Every asserted numeral is a table cell that
    # the chunk prints, so both rates are 1.0 by construction and are MEASURED
    # here rather than assumed.  Nothing is absent, so P(0 | absent) is undefined.
    cited_ok = [v in c for v, c in zip(df["cited_value"].to_list(), df["chunk"].to_list())]
    corr_ok = [v in c for v, c in zip(df["correct_value"].to_list(), df["chunk"].to_list())]
    canon_ok = [bool(P.canon_set(v) & P.present_numbers(c))
                for v, c in zip(df["cited_value"].to_list(), df["chunk"].to_list())]
    canon_corr = [bool(P.canon_set(v) & P.present_numbers(c))
                  for v, c in zip(df["correct_value"].to_list(), df["chunk"].to_list())]
    out["value_presence"] = {
        "cited_verbatim_rate": round(float(np.mean(cited_ok)), 6),
        "correct_verbatim_rate": round(float(np.mean(corr_ok)), 6),
        "cited_canonical_rate": round(float(np.mean(canon_ok)), 6),
        "correct_canonical_rate": round(float(np.mean(canon_corr)), 6),
        "bar": "1.0 for both cited and correct",
        "pass": bool(all(cited_ok) and all(corr_ok) and all(canon_ok) and all(canon_corr))}
    out["p_label0_given_absent"] = {
        "value": None, "absent_rows": int(len(df) - sum(canon_ok)),
        "bar": "n/a - all asserted values present by construction",
        "note": "presence 1.0 on both legs; the quantity is undefined here",
        "pass": True}

    # --- digit-surface channels on the asserted numeral
    tz = [P.trailing_zeros(v) for v in df["cited_value"].to_list()]
    dc = [P.digits(v) for v in df["cited_value"].to_list()]
    per_fam_tz, per_fam_dc = {}, {}
    for fam, sub in df.select(["neg_family", "label", "cited_value"]).group_by("neg_family"):
        y = sub["label"].to_list()
        per_fam_tz[fam[0]] = round(P.auroc(y, [P.trailing_zeros(v) for v in sub["cited_value"]]), 4)
        per_fam_dc[fam[0]] = round(P.auroc(y, [P.digits(v) for v in sub["cited_value"]]), 4)
    a_tz, a_dc = P.auroc(labels, tz), P.auroc(labels, dc)
    dev_tz = max([abs(a_tz - 0.5)] + [abs(v - 0.5) for v in per_fam_tz.values()])
    dev_dc = max([abs(a_dc - 0.5)] + [abs(v - 0.5) for v in per_fam_dc.values()])
    out["trailing_zero_auroc"] = {"pooled": round(a_tz, 4), "per_family": per_fam_tz,
                                  "bar": "in [0.45, 0.55]", "pass": bool(dev_tz <= 0.05)}
    out["digit_count_auroc"] = {"pooled": round(a_dc, 4), "per_family": per_fam_dc,
                                "bar": "in [0.45, 0.55]", "pass": bool(dev_dc <= 0.05)}

    # --- mechanical audit: re-derive every sampled negative's binding from the
    # source table.  The cited value must be a real cell of a DIFFERENT row (or a
    # different column of the same row) than the one the claim names.
    neg = df.filter(pl.col("label") == 0)
    samp = neg.sample(n=min(AUDIT_N, len(neg)), seed=SEED)
    errs = []
    for r in samp.iter_rows(named=True):
        tab = by_doc.get(r["doc_id"])
        why = None
        if tab is None:
            why = "table not found"
        else:
            body, hdr = tab["body"], tab["hdr"]
            ri, ci = r["row_index"], r["column_index"]
            mri, mci = r["misbound_row_index"], r["misbound_column_index"]
            if not (0 <= ri < len(body) and 0 <= mri < len(body)
                    and 0 <= ci < len(hdr) and 0 <= mci < len(hdr)):
                why = "index out of range"
            elif body[ri][ci].strip() != r["correct_value"]:
                why = "correct value is not the table's cell for the claimed binding"
            elif body[mri][mci].strip() != r["cited_value"]:
                why = "cited value is not the table's cell for the recorded source binding"
            elif body[ri][tab["lab_ci"]].strip() != r["row_key"]:
                why = "row key does not name the claimed row"
            elif (mri, mci) == (ri, ci):
                why = "cited binding equals the claimed binding"
            elif r["cited_value"] == r["correct_value"]:
                why = "cited value equals the correct value"
            elif r["neg_family"] == "misbound_row" and (mci != ci or mri == ri):
                why = "misbound_row negative does not cite a different row of the same column"
            elif r["neg_family"] == "misbound_col" and (mri != ri or mci == ci):
                why = "misbound_col negative does not cite a different column of the same row"
            elif r["cited_value"] not in r["chunk"] or r["correct_value"] not in r["chunk"]:
                why = "a value is not readable in the chunk"
        if why:
            errs.append({"pair_id": r["pair_id"], "doc_id": r["doc_id"], "why": why})
    out["misbind_rederivation_audit"] = {
        "sampled": len(samp), "errors": len(errs), "bar": "0 errors",
        "pass": len(errs) == 0, "examples": errs[:5]}

    out["report_only"] = {
        "claim_char_length_auroc": round(P.auroc(labels, [len(c) for c in claims]), 4),
        "leading_digit_auroc": round(P.auroc(
            labels, [float(P.leading_digit(v) or 0) for v in df["cited_value"]]), 4),
        "decimal_presence_auroc": round(P.auroc(
            labels, [1.0 if "." in v else 0.0 for v in df["cited_value"]]), 4),
        "value_magnitude_auroc": round(P.auroc(
            labels, [P.as_num(v) or 0.0 for v in df["cited_value"]]), 4),
    }
    out["all_bars_pass"] = all(out[k]["pass"] for k in
                               ("claim_only_tfidf_auroc", "within_pair_claim_only_accuracy",
                                "value_presence", "trailing_zero_auroc",
                                "digit_count_auroc", "misbind_rederivation_audit"))
    return out


def overlap(df, path, name):
    if not path.exists():
        return {"file": path.name, "present": False}
    other = pl.read_parquet(path, columns=["doc_id", "chunk"])
    docs = set(other["doc_id"].to_list()) & set(df["doc_id"].to_list())
    chunks = set(other["chunk"].to_list()) & set(df["chunk"].to_list())
    return {"file": path.name, "present": True, "shared_doc_ids": len(docs),
            "shared_chunks": len(chunks),
            "their_docs": other["doc_id"].n_unique(),
            "note": f"{name}: permitted, measured not enforced"}


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = P.tabfact_tables() + P.feverous_tables()
    drop_idx = P.excluded_tables(raw, prints)
    tables, dropped_eval = [], 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped_eval += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None or not P.numeric_columns(t["hdr"], t["body"], lab):
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval content; "
          f"{dropped_eval} dropped; {len(tables)} admitted", flush=True)

    forms = list(P.FORM_WEIGHTS)
    w = np.array([P.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]
    by_doc = {t["doc_id"]: t for t in tables}

    target = {("misbound_row", "up"): int(round(N_PAIRS * ROW_SHARE / 2)),
              ("misbound_row", "down"): int(round(N_PAIRS * ROW_SHARE / 2)),
              ("misbound_col", "up"): int(round(N_PAIRS * (1 - ROW_SHARE) / 2)),
              ("misbound_col", "down"): int(round(N_PAIRS * (1 - ROW_SHARE) / 2))}
    built = collections.Counter()
    rows, per_doc, seen, pid = [], collections.Counter(), set(), 0

    for cap in DOC_CAP_LADDER:
        if pid >= N_PAIRS:
            break
        order = list(range(len(tables)))
        rng.shuffle(order)
        for ti in order:
            if pid >= N_PAIRS:
                break
            t = tables[ti]
            while per_doc[t["doc_id"]] < cap and pid < N_PAIRS:
                cells = sorted(target, key=lambda c: built[c] - target[c])
                spec = None
                for fam, direction in cells:
                    if built[(fam, direction)] >= target[(fam, direction)]:
                        continue
                    spec = BUILDERS[fam](t, rng, direction)
                    if spec is not None:
                        break
                if spec is None:
                    break
                sig = (spec["chunk"], spec["row_key"], spec["column"],
                       spec["cited_value"], spec["correct_value"])
                if sig in seen:
                    break
                seen.add(sig)
                rows.extend(emit(spec, pid, pid))
                built[(spec["neg_family"], spec["direction"])] += 1
                per_doc[t["doc_id"]] += 1
                pid += 1
        print(f"  cap {cap}: {pid} pairs, "
              f"{ {f'{a}:{b}': n for (a, b), n in sorted(built.items())} }", flush=True)

    df = pl.DataFrame(rows).unique(subset=["claim", "chunk", "label"],
                                   keep="first", maintain_order=True)
    keep_pairs = (df.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"])
    df = df.filter(pl.col("pair_id").is_in(keep_pairs)).sort(["pair_id", "label"],
                                                            descending=[False, True])
    n_pairs = df["pair_id"].n_unique()
    df.write_parquet(OUT)
    print(f"\n{df.height} rows / {n_pairs} pairs over {df['doc_id'].n_unique()} documents",
          flush=True)

    fam = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    dirs = {f"{a}:{b}": n for a, b, n in
            df.group_by(["neg_family", "direction"]).len().iter_rows()}
    forms_r = {k: v for k, v in df.group_by("serial_form").len().iter_rows()}
    tmpl = {str(k): v for k, v in df.group_by("template_id").len().iter_rows()}
    src = {k: v for k, v in df.group_by("source").len().iter_rows()}
    parity_all = float(df["surface_parity"].cast(pl.Float64).mean())
    parity_fam = {k: round(float(v), 4) for k, v in
                  df.group_by("neg_family")
                    .agg(pl.col("surface_parity").cast(pl.Float64).mean()).iter_rows()}

    res = verify(df, rng, by_doc)

    man = dict(
        seed=SEED, rows=df.height, pairs=n_pairs, documents=df["doc_id"].n_unique(),
        pairs_per_document=round(n_pairs / max(df["doc_id"].n_unique(), 1), 3),
        doc_cap=DOC_CAP_LADDER[-1],
        families=fam,
        family_shares={k: round(v / df.height, 4) for k, v in fam.items()},
        directions=dirs,
        parity_rates=dict(pooled=round(parity_all, 4), per_family=parity_fam),
        diversity=dict(serial_forms=forms_r, templates=tmpl, sources=src,
                       n_templates=len(TEMPLATES),
                       distinct_claims=df["claim"].n_unique(),
                       distinct_chunks=df["chunk"].n_unique(),
                       distinct_columns=df["column"].n_unique()),
        eval_disjointness=dict(evalset_rows=eval_rows,
                               excluded_doc_ids=len(excluded_ids),
                               content_matched_tables=len(drop_idx),
                               tables_dropped=dropped_eval,
                               tables_admitted=len(tables),
                               shared_documents=0, shared_chunks=0,
                               method="content-based (R17-H144 method), enforced"),
        overlap_permitted=[overlap(df, H144_PAIRS, "H144 pair corpus"),
                           overlap(df, H144_LOOKUP, "H144 lookup family"),
                           overlap(df, H145_LANE, "H145 scale/unit lane")],
        verify=res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "family_shares", "directions",
                       "parity_rates", "eval_disjointness", "overlap_permitted",
                       "verify")}, indent=2), flush=True)
    ok = res["all_bars_pass"]
    print(f"=== R17-H146 LANE {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
