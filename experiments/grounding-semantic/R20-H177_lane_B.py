"""R20-H177 LANE B `num_compare` - compare/direction verification, CPU, no GPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H177
NUMERIC-VERIFICATION PORTFOLIO ARM": "Lane B compare/direction (~25-30k pairs:
claims asserting greater/less/highest/increase/decrease between two values both
verbatim in evidence; negative twin flips ONLY the relation/direction word;
TabFact/FEVEROUS tables + EDGAR-restricted MD&A prose, full H146 leak
discipline)".

THE QUESTION THE LANE ASKS
-------------------------
Gate 1 put 31.2% of finqa's derivation-class rank-loss mass on errors decidable
by ORDERING two values the evidence already prints (`R20_gate_1.json`; items 100,
31, 168, 189, 215).  Ordering is the one derivation subtype requiring zero
computation, so H-B asks whether the install law the misbind family established -
nothing installs at sub-block scale (H145 bind_col 0.5721), the same construction
installs at core scale (H146 bind_col 0.9555) - also holds for relation and
direction words, which have only ever ridden at 1,500 sub-block pairs.

CONSTRUCTION - the twin differs by ONE WORD and nothing else
-----------------------------------------------------------
Every pair is two claims over the SAME evidence chunk, byte-identical except for
the relation or direction word:

  label 1   the ordering the evidence supports          "... is greater than ..."
  label 0   the same sentence with the word flipped     "... is less than ..."

Both operands are printed in the evidence and both are printed in the claim, so
the negative is decidable by reading and ordering two present values - there is
no arithmetic anywhere in the lane, which is the damage carrier the H145/H146
dissociation identified and which this construction excludes by design.

WHY THE LEAK SUITE IS TIGHT BY CONSTRUCTION, AND MEASURED ANYWAY
----------------------------------------------------------------
Because the twin shares its operands, every numeric channel a claim-side model
could read - digit count, magnitude, trailing zeros, leading digit - is IDENTICAL
across the pair, so the only claim-side signal available is the relation word
itself.  That word is drawn from explicit target cells in equal numbers per
family, so each of `greater/less`, `higher/lower`, `highest/lowest` and
`increased/decreased` appears exactly as often on the supported leg as on the
corrupted one.  The verify block measures all of it rather than asserting it.

FAMILIES
--------
  cmp_order    TabFact - two rows of one numeric column, ordering of their cells
  cmp_extreme  TabFact - the highest / lowest value among the rows serialised
  cmp_amount   EDGAR   - ordering of two labelled amounts in one MD&A passage
  cmp_trend    EDGAR   - a stated from-X-to-Y movement, direction word flipped

Sources: TabFact TRAIN (public, banked) and the admitted EDGAR-restricted slice.
FinQA and TAT-QA source corpora are WALLED and never opened; HotpotQA train and
HoVer are never opened; FEVEROUS is not admitted (see `feverous_available()`).

Run:  uv run python experiments/grounding-semantic/R20-H177_lane_B.py [--force]
"""

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import random
import sys

import numpy as np
import polars as pl

HERE = Path(__file__).parent


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h177common", HERE / "R20-H177_lane_common.py")
P = _mod("h144pairs", HERE / "R17-H144_pairs.py")

OUT = HERE / "R20-H177_lane_B.parquet"
EVAL_OUT = HERE / "R20-H177_eval_B.parquet"
MANIFEST = HERE / "R20-H177_lane_B_manifest.json"

SEED = 1177
SEED_EVAL = 2177
TAG = "num_compare"
BODY_ROWS = 6
DOC_CAP_LADDER = (2, 3, 4, 5, 6, 8)
EDGAR_CAP_LADDER = (2, 3, 4)        # pairs per EDGAR chunk
CAND_TRIES = 24
N_FOLDS = 5

TABFACT_FAMILIES = ("cmp_order", "cmp_extreme")
EDGAR_FAMILIES = ("cmp_amount", "cmp_trend")

# (family, true relation word) -> target pairs.  Each word's antonym twin gets an
# equal cell, which is what makes the 50/50 direction balance exact rather than
# approximate.
TARGETS = {
    ("cmp_order", "greater"): 1688, ("cmp_order", "less"): 1688,
    ("cmp_order", "higher"): 1687, ("cmp_order", "lower"): 1687,
    ("cmp_extreme", "highest"): 1500, ("cmp_extreme", "lowest"): 1500,
    ("cmp_amount", "greater"): 1250, ("cmp_amount", "less"): 1250,
    ("cmp_amount", "higher"): 1250, ("cmp_amount", "lower"): 1250,
    ("cmp_trend", "increased"): 125, ("cmp_trend", "decreased"): 125,
}
EVAL_TARGETS = {
    ("cmp_order", "greater"): 113, ("cmp_order", "less"): 113,
    ("cmp_order", "higher"): 112, ("cmp_order", "lower"): 112,
    ("cmp_extreme", "highest"): 100, ("cmp_extreme", "lowest"): 100,
    ("cmp_amount", "greater"): 84, ("cmp_amount", "less"): 84,
    ("cmp_amount", "higher"): 83, ("cmp_amount", "lower"): 83,
    ("cmp_trend", "increased"): 8, ("cmp_trend", "decreased"): 8,
}

FLIP = {"greater": "less", "less": "greater", "higher": "lower",
        "lower": "higher", "highest": "lowest", "lowest": "highest",
        "increased": "decreased", "decreased": "increased"}
PAIR_OF = {"greater": 0, "less": 0, "higher": 1, "lower": 1,
           "highest": 0, "lowest": 0, "increased": 0, "decreased": 0}


# --------------------------------------------------------------------------- #
# claim templates - the shape is fixed inside a pair, only the relation word moves
# --------------------------------------------------------------------------- #
TEMPLATES = {
    # greater / less
    ("cmp_order", 0): [
        "The {col} of {ka} ({va}) is {w} than the {col} of {kb} ({vb}).",
        "According to the table, the {col} of {ka}, {va}, is {w} than that of "
        "{kb}, {vb}.",
    ],
    # higher / lower
    ("cmp_order", 1): [
        "{ka} has a {w} {col} ({va}) than {kb} ({vb}).",
        "The {col} of {ka} ({va}) is {w} than the {col} of {kb} ({vb}).",
    ],
    ("cmp_extreme", 0): [
        "Of the {col} values shown, {ka} has the {w}, at {va}.",
        "{ka} records the {w} {col} among the rows listed, at {va}.",
    ],
    ("cmp_amount", 0): [
        "The {ka} of {va} is {w} than the {kb} of {vb}.",
        "The filing reports the {ka} of {va}, which is {w} than the {kb} of {vb}.",
    ],
    ("cmp_amount", 1): [
        "{ka} at {va} is {w} than {kb} at {vb}.",
        "The {ka} of {va} is {w} than the {kb} of {vb}.",
    ],
    ("cmp_trend", 0): [
        "The {ka} {w} from {va} to {vb}.",
        "According to the filing, the {ka} {w} from {va} to {vb}.",
    ],
}


def render(family, ti, word, **kw):
    """The relation word is the ONLY slot that changes inside a pair.

    A template is bound to its antonym pair - `greater/less` and `higher/lower`
    do not take the same frame - so the flip cannot change the grammar of the
    sentence and leave a shape the model could read instead of the relation."""
    return TEMPLATES[(family, PAIR_OF[word])][ti].format(w=word, **kw)


# --------------------------------------------------------------------------- #
# TabFact families
# --------------------------------------------------------------------------- #
def _fillers(tab, keep, rng):
    others = [r for r in range(len(tab["body"])) if r not in keep]
    rng.shuffle(others)
    return sorted(keep | set(others[: max(0, BODY_ROWS - len(keep))]))


def _dup_labels(tab):
    labs = [r[tab["lab_ci"]].strip() for r in tab["body"]]
    return {k for k, n in collections.Counter(labs).items() if n > 1}


def _standalone(chunk, *vals):
    pres = P.present_numbers(chunk)
    return all(P.canon_set(v) & pres for v in vals)


def build_cmp_order(tab, rng, word):
    """Two rows of one numeric column; the claim orders their printed cells."""
    lab_ci, body, hdr = tab["lab_ci"], tab["body"], tab["hdr"]
    cols = P.numeric_columns(hdr, body, lab_ci)
    if not cols:
        return None
    dup = _dup_labels(tab)
    rng.shuffle(cols)
    bigger = word in ("greater", "higher")
    for ci, vals in cols[:3]:
        col = (hdr[ci] or "").strip()
        if not P.good_header(col):
            continue
        idx = [ri for ri, _ in vals]
        if len(idx) < 2:
            continue
        for _ in range(CAND_TRIES):
            ra, rb = rng.sample(idx, 2)
            sa, sb = body[ra][ci].strip(), body[rb][ci].strip()
            va, vb = P.as_num(sa), P.as_num(sb)
            if va is None or vb is None or abs(va - vb) < 1e-12 or sa == sb:
                continue
            ka, kb = body[ra][lab_ci].strip(), body[rb][lab_ci].strip()
            if not ka or not kb or ka == kb or ka in dup or kb in dup:
                continue
            # name the row whose value makes the wanted word TRUE first
            if (va > vb) != bigger:
                ra, rb, sa, sb, ka, kb = rb, ra, sb, sa, kb, ka
            rows = _fillers(tab, {ra, rb}, rng)
            chunk = P.serialize(tab["form"], tab["caption"], hdr,
                                [body[r] for r in rows], lab_ci, (ci,))
            if not all(s in chunk for s in (sa, sb, ka, kb)):
                continue
            if not _standalone(chunk, sa, sb):
                continue
            return dict(chunk=chunk, ka=ka, va=sa, kb=kb, vb=sb, col=col,
                        doc_id=tab["doc_id"], source="tabfact",
                        serial_form=tab["form"],
                        asserted_values=f"{sa}|~|{sb}")
    return None


def build_cmp_extreme(tab, rng, word):
    """The extreme value among the rows the chunk actually serialises.

    The claim is scoped to "the rows listed" and the chunk is built from exactly
    those rows, so the superlative is decidable from the evidence alone.  A
    unique maximum AND a unique minimum are required, so the flipped word is
    false rather than merely unsupported."""
    lab_ci, body, hdr = tab["lab_ci"], tab["body"], tab["hdr"]
    cols = P.numeric_columns(hdr, body, lab_ci)
    if not cols:
        return None
    dup = _dup_labels(tab)
    rng.shuffle(cols)
    want_max = word == "highest"
    for ci, vals in cols[:3]:
        col = (hdr[ci] or "").strip()
        if not P.good_header(col):
            continue
        idx = [ri for ri, _ in vals]
        if len(idx) < 3:
            continue
        for _ in range(CAND_TRIES):
            rows = sorted(rng.sample(idx, min(BODY_ROWS, len(idx))))
            cells = [(r, P.as_num(body[r][ci].strip())) for r in rows]
            if any(v is None for _, v in cells):
                continue
            hi = max(v for _, v in cells)
            lo = min(v for _, v in cells)
            if hi == lo:
                continue
            if sum(v == hi for _, v in cells) != 1 or sum(v == lo for _, v in cells) != 1:
                continue
            keys = [body[r][lab_ci].strip() for r in rows]
            if len(set(keys)) != len(keys) or any((not k) or k in dup for k in keys):
                continue
            pick = next(r for r, v in cells if v == (hi if want_max else lo))
            key, val = body[pick][lab_ci].strip(), body[pick][ci].strip()
            chunk = P.serialize(tab["form"], tab["caption"], hdr,
                                [body[r] for r in rows], lab_ci, (ci,))
            surfaces = [body[r][ci].strip() for r in rows]
            if not all(s in chunk for s in surfaces + keys):
                continue
            if not _standalone(chunk, *surfaces):
                continue
            return dict(chunk=chunk, ka=key, va=val, kb="", vb="", col=col,
                        doc_id=tab["doc_id"], source="tabfact",
                        serial_form=tab["form"],
                        asserted_values=val)
    return None


# --------------------------------------------------------------------------- #
# EDGAR families
# --------------------------------------------------------------------------- #
def build_cmp_amount(row, rng, word):
    """Ordering of two labelled amounts printed in one MD&A passage."""
    chunk = row["chunk"]
    binds = list(row["binds"])
    rng.shuffle(binds)
    bigger = word in ("greater", "higher")
    for i in range(len(binds)):
        for j in range(len(binds)):
            if i == j:
                continue
            (r1, v1), (r2, v2) = binds[i], binds[j]
            if v1 == v2 or r1.lower() in r2.lower() or r2.lower() in r1.lower():
                continue
            a, b = C.parse_amount(v1), C.parse_amount(v2)
            if a is None or b is None or a == b:
                continue
            if (a > b) != bigger:
                continue
            if C.occurrences(chunk, v1) < 1 or C.occurrences(chunk, v2) < 1:
                continue
            return dict(chunk=chunk, ka=r1, va=v1, kb=r2, vb=v2, col="",
                        doc_id=row["doc_id"], source="edgar", serial_form="prose",
                        asserted_values=f"{v1}|~|{v2}")
    return None


def build_cmp_trend(row, rng, word):
    """A movement the filing states in words, with both endpoints printed."""
    chunk = row["chunk"]
    cands = [t for t in row["trends"] if t[1] == word]
    rng.shuffle(cands)
    for role, direction, a, b, _sent in cands:
        if C.occurrences(chunk, a) < 1 or C.occurrences(chunk, b) < 1:
            continue
        return dict(chunk=chunk, ka=role, va=a, kb="", vb=b, col="",
                    doc_id=row["doc_id"], source="edgar", serial_form="prose",
                    asserted_values=f"{a}|~|{b}")
    return None


BUILDERS = {"cmp_order": build_cmp_order, "cmp_extreme": build_cmp_extreme,
            "cmp_amount": build_cmp_amount, "cmp_trend": build_cmp_trend}


def emit(family, spec, word, ti, pid):
    base = {k: spec[k] for k in ("chunk", "doc_id", "source", "serial_form",
                                 "asserted_values")}
    base.update(neg_family=family, tag=TAG, true_word=word, flip_word=FLIP[word],
                template_id=ti, word_pair=PAIR_OF[word],
                subject=spec["ka"], counterpart=spec["kb"], column=spec["col"])
    kw = dict(col=spec["col"], ka=spec["ka"], va=spec["va"], kb=spec["kb"],
              vb=spec["vb"])
    return [
        dict(pair_id=pid, label=1, claim=render(family, ti, word, **kw), **base),
        dict(pair_id=pid, label=0, claim=render(family, ti, FLIP[word], **kw), **base),
    ]


# --------------------------------------------------------------------------- #
# supply
# --------------------------------------------------------------------------- #
def tabfact_pool():
    """TabFact TRAIN tables with a label column and a usable numeric column.

    The R17-H143 eval documents are excluded by the banked CONTENT-based method,
    not by id alone - the same exclusion `R17-H146_lane.py` runs."""
    excluded_ids, prints, eval_rows, _unmatched = P.evalset_documents()
    raw = P.tabfact_tables()
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
    return tables, {"candidate_tables": len(raw), "carry_eval_content": len(drop_idx),
                    "dropped": dropped, "admitted": len(tables),
                    "evalset_rows": eval_rows,
                    "method": "R17-H144 content-based exclusion, enforced"}


def assign_forms(tables, np_rng):
    forms = list(P.FORM_WEIGHTS)
    w = np.array([P.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]


def edgar_items(edgar_rows, split):
    """Extract the EDGAR prose bindings ONCE per chunk.

    The cell-filling loop revisits a chunk several times as the deficit ordering
    rotates, and re-running the role and trend regexes each visit is the whole
    cost of the build, so the extraction is done once here."""
    out = []
    rows = edgar_rows.select(["doc_id", "chunk"]).iter_rows(named=True)
    for i, r in enumerate(rows):
        chunk = r["chunk"]
        binds = [(role, val) for role, val, _ in
                 C.unique_role_bindings(chunk).values()
                 if C.occurrences(chunk, val) == 1]
        trends = C.trend_statements(chunk) if (
            C.word_attested(chunk, "increased")
            and C.word_attested(chunk, "decreased")) else []
        if binds or trends:
            out.append({"doc_id": r["doc_id"], "chunk": chunk, "binds": binds,
                        "trends": trends})
        if (i + 1) % 10000 == 0:
            print(f"  [{split}] edgar extraction {i + 1} chunks", flush=True)
    print(f"  [{split}] edgar items with usable bindings: {len(out)}", flush=True)
    return out


def assemble(split, seed, targets, tables, ed_items):
    """Fill the (family, word) target cells from the split's own documents."""
    rng = random.Random(seed)
    built = collections.Counter()
    rows, per_doc, seen, pid = [], collections.Counter(), set(), 0

    tf_cells = [c for c in targets if c[0] in TABFACT_FAMILIES]
    ed_cells = [c for c in targets if c[0] in EDGAR_FAMILIES]

    def try_cells(cells, source_item, cap_key, cap):
        nonlocal pid
        while per_doc[cap_key] < cap:
            order = sorted(cells, key=lambda c: built[c] - targets[c])
            spec, cell = None, None
            for c in order:
                if built[c] >= targets[c]:
                    continue
                cand = BUILDERS[c[0]](source_item, rng, c[1])
                # the relation word and its flip must be EQUALLY readable in the
                # evidence, or the negative is separable by lexical novelty
                if cand is not None and (
                        C.word_attested(cand["chunk"], c[1])
                        != C.word_attested(cand["chunk"], FLIP[c[1]])):
                    cand = None
                if cand is not None:
                    spec, cell = cand, c
                    break
            if spec is None:
                break
            ti = rng.randrange(len(TEMPLATES[(cell[0], PAIR_OF[cell[1]])]))
            sig = (spec["chunk"], spec["ka"], spec["kb"], spec["va"], spec["vb"],
                   cell[0])
            if sig in seen:
                break
            seen.add(sig)
            rows.extend(emit(cell[0], spec, cell[1], ti, pid))
            built[cell] += 1
            per_doc[cap_key] += 1
            pid += 1

    for cap in DOC_CAP_LADDER:
        if all(built[c] >= targets[c] for c in tf_cells):
            break
        order = list(range(len(tables)))
        rng.shuffle(order)
        for ti in order:
            if all(built[c] >= targets[c] for c in tf_cells):
                break
            t = tables[ti]
            try_cells(tf_cells, t, t["doc_id"], cap)
        print(f"  [{split}] tabfact cap {cap}: "
              f"{ {f'{a}:{b}': n for (a, b), n in sorted(built.items())} }",
              flush=True)

    for cap in EDGAR_CAP_LADDER:
        if all(built[c] >= targets[c] for c in ed_cells):
            break
        order = list(range(len(ed_items)))
        rng.shuffle(order)
        for i in order:
            if all(built[c] >= targets[c] for c in ed_cells):
                break
            item = ed_items[i]
            try_cells(ed_cells, item, f"edgar:{i}", cap)
        print(f"  [{split}] edgar cap {cap}: "
              f"{ {f'{a}:{b}': n for (a, b), n in sorted(built.items())} }",
              flush=True)

    df = C.dedupe(pl.DataFrame(rows))
    return df, {f"{a}:{b}": n for (a, b), n in sorted(built.items())}


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    out = {}
    out["pair_integrity"] = C.pair_integrity(df)

    strata = [f"{f}:{w}" for f, w in
              zip(df["neg_family"].to_list(), df["true_word"].to_list())]
    probe, score, ndocs = C.claim_only_probe_stratified(
        df["claim"].to_list(), df["label"].to_list(), df["doc_id"].to_list(),
        strata, rng, n_folds=N_FOLDS)
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "relation-stratified, liblinear tol 1e-7",
        "documents": ndocs, "rows": int(df.height)}

    wp = C.within_pair_accuracy(df, score, by="neg_family")
    worst = max(v["acc"] for v in wp.values())
    out["within_pair_claim_only_accuracy"] = {
        "per_family": wp, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    out["surface_parity"] = C.surface_parity(df)
    out["direction_word_balance"] = C.flip_word_balance(df)
    out["attestation_symmetry"] = C.attestation_audit(
        df, require_attested=("cmp_trend",))

    # every asserted amount is printed in the evidence.  Two instruments,
    # because the two register halves need different ones: EDGAR amounts carry a
    # currency anchor and are checked verbatim; TabFact cells are bare numerals
    # and are checked with the banked canonical-numeral detector, which is what
    # survives the 1,500-char serving cut.
    out["value_presence_edgar"] = C.verbatim_value_audit(df, EDGAR_FAMILIES)
    tf = df.filter(pl.col("neg_family").is_in(list(TABFACT_FAMILIES)))
    bad = [r["pair_id"] for r in tf.iter_rows(named=True)
           if not _standalone(r["chunk"], *[v for v in
                                            r["asserted_values"].split("|~|") if v])]
    out["value_presence_tabfact"] = {
        "rows_checked": int(tf.height), "rows_with_absent_value": len(bad),
        "instrument": "R17-H144 canon_set / present_numbers (banked)",
        "bar": "0 - every asserted cell is readable in the serialised chunk",
        "pass": not bad, "examples": bad[:5]}

    # the twin shares its operands, so every numeric channel is identical inside
    # a pair by construction - measured, not assumed
    def numchan(fn):
        vals = [fn([v for v in a.split("|~|") if v]) for a in
                df["asserted_values"].to_list()]
        return round(C.auroc(df["label"].to_list(), vals), 4)

    out["operand_channel_parity"] = {
        "digit_count_auroc": numchan(lambda vs: float(sum(P.digits(v) for v in vs))),
        "trailing_zero_auroc": numchan(
            lambda vs: float(sum(P.trailing_zeros(v) for v in vs))),
        "leading_digit_auroc": numchan(
            lambda vs: float(P.leading_digit(vs[0]) or 0) if vs else 0.0),
        "bar": "0.5 exactly - the twin asserts the SAME operands",
        "pass": True}
    out["operand_channel_parity"]["pass"] = all(
        abs(out["operand_channel_parity"][k] - 0.5) < 1e-9
        for k in ("digit_count_auroc", "trailing_zero_auroc", "leading_digit_auroc"))

    out["all_bars_pass"] = all(
        out[k]["pass"] for k in
        ("pair_integrity", "claim_only_tfidf_auroc",
         "within_pair_claim_only_accuracy", "surface_parity",
         "direction_word_balance", "attestation_symmetry", "value_presence_edgar",
         "value_presence_tabfact", "operand_channel_parity"))
    return out


# --------------------------------------------------------------------------- #
def block(df, res, cells, split, seed, tf_note, extra=None):
    y = df["label"].to_list()
    man = dict(
        rows=df.height, pairs=int(df["pair_id"].n_unique()),
        documents=int(df["doc_id"].n_unique()), seed=seed, split=split,
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y))},
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        family_shares={k: round(v / df.height, 4) for k, v in
                       df.group_by("neg_family").len().iter_rows()},
        cells_filled=cells,
        cell_targets={f"{a}:{b}": n for (a, b), n in
                      sorted((TARGETS if split == "train" else EVAL_TARGETS).items())},
        source_rows={k: v for k, v in df.group_by("source").len().iter_rows()},
        serial_forms={k: v for k, v in df.group_by("serial_form").len().iter_rows()},
        templates={str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
        diversity=dict(distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique()),
                       distinct_subjects=int(df["subject"].n_unique())),
        char_stats=dict(claim=C.char_stats(df["claim"].to_list()),
                        chunk=C.char_stats(df["chunk"].to_list())),
        window_census=C.window_census(df["chunk"].to_list()),
        tabfact_admission=tf_note,
        verify=res)
    man.update(extra or {})
    return man


def already_built():
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()
                                     and EVAL_OUT.exists()):
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
    print(f"=== R20-H177 lane B ({TAG}) seed {SEED}", flush=True)
    np_rng = np.random.default_rng(SEED)
    tables, tf_note = tabfact_pool()
    assign_forms(tables, np_rng)
    tf_train = [t for t in tables if not C.is_eval_doc(t["doc_id"])]
    tf_eval = [t for t in tables if C.is_eval_doc(t["doc_id"])]
    ed_train_raw, ed_eval_raw = C.edgar("train"), C.edgar("eval")
    print(f"supply: tabfact {len(tf_train)} train / {len(tf_eval)} eval tables; "
          f"edgar {ed_train_raw.height} train / {ed_eval_raw.height} eval chunks",
          flush=True)
    ed_train = edgar_items(ed_train_raw, "train")
    ed_eval = edgar_items(ed_eval_raw, "eval")

    df, cells = assemble("train", SEED, TARGETS, tf_train, ed_train)
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}",
          flush=True)

    ev, ev_cells = assemble("eval", SEED_EVAL, EVAL_TARGETS, tf_eval, ed_eval)
    ev.write_parquet(EVAL_OUT)
    print(f"{ev.height} rows / {ev['pair_id'].n_unique()} pairs -> {EVAL_OUT.name}",
          flush=True)

    rng = random.Random(SEED)
    res = verify(df, rng)
    ev_res = verify(ev, random.Random(SEED_EVAL))

    shared_docs = set(df["doc_id"].to_list()) & set(ev["doc_id"].to_list())
    shared_chunks = set(df["chunk"].to_list()) & set(ev["chunk"].to_list())

    man = block(df, res, cells, "train", SEED, tf_note, extra=dict(
        experiment="R20-H177 lane B - compare/direction verification (num_compare)",
        registration="docs/experiments/semantic-grounding-experiments.md, block "
                     "'R20-H177 NUMERIC-VERIFICATION PORTFOLIO ARM'",
        tag=TAG, dann_group=TAG,
        mix_loader="drop-in for R18-H150_arm_run.make_build_mix - columns claim / "
                   "chunk / label / pair_id / neg_family; chunk is read "
                   "UNTRUNCATED and windowed 1500/750 by the loader",
        sources=C.SOURCES, walled_never_opened=C.WALLED_NEVER_OPENED,
        feverous=C.feverous_available(),
        parquet=OUT.name,
        held_out_eval=dict(
            parquet=EVAL_OUT.name,
            purpose="the H177 PRIMARY mechanism gate for lane B - same generator, "
                    "different seed, documents excluded from the training lane; "
                    "no model is read on it here (stage 0 is CPU-only)",
            **block(ev, ev_res, ev_cells, "eval", SEED_EVAL, tf_note)),
        document_disjointness=dict(
            rule=f"blake2b(doc_id) % 1000 < {C.EVAL_DOC_PERMILLE} -> eval only",
            shared_documents=len(shared_docs), shared_chunks=len(shared_chunks),
            bar="0 shared documents", pass_=len(shared_docs) == 0)))
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "families", "family_shares",
                       "cells_filled", "source_rows", "verify")}, indent=2),
          flush=True)
    print(json.dumps({"eval_rows": man["held_out_eval"]["rows"],
                      "eval_pairs": man["held_out_eval"]["pairs"],
                      "eval_families": man["held_out_eval"]["families"],
                      "eval_all_bars_pass":
                          man["held_out_eval"]["verify"]["all_bars_pass"],
                      "shared_documents": len(shared_docs)}, indent=2), flush=True)
    ok = (res["all_bars_pass"] and ev_res["all_bars_pass"] and not shared_docs)
    print(f"=== R20-H177 LANE B {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
