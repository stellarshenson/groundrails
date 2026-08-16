"""R20-H177 LANE C `num_rolebind` - operand-role / sign / period misbind, CPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H177
NUMERIC-VERIFICATION PORTFOLIO ARM": "Lane C role/sign/period misbind (~25-30k
pairs over EDGAR prose: role-swap, sign/direction-swap on stated changes,
period-swap; all values verbatim-present)".

THE QUESTION THE LANE ASKS
--------------------------
Gate 2 traced 54.9% of finqa's false-positive rank-loss mass to mislabeled
operand role, sign or period rather than to wrong computation
(`R20_gate_2.json`: 114 the \\$74.9M repurchase used as the decrease, 71 a
per-share figure claimed as the total, 189 a sign flip, 36/229 the right
arithmetic on the wrong year).  None of those needs arithmetic to detect - each
is a BINDING failure over values the passage already prints.  The graduated
misbind construction installed exactly that channel for tables at core scale
(H146 bind_col 0.9555, all holds green); this lane extends it from table
row/column to prose role, sign and period.

CONSTRUCTION - a word-level corruption of a TRUE statement
----------------------------------------------------------
Every pair is two claims over the SAME EDGAR passage, identical except for one
word-level element:

  role_swap    the amount is reattached to a different labelled quantity that
               the same passage binds to a DIFFERENT amount
  sign_swap    the direction the passage states in words is flipped
               ("an increase of \\$X" <-> "a decrease of \\$X")
  period_swap  the amount is rebound to another year the passage names

THREE PROPERTIES MAKE THE NEGATIVE GROUNDABLE RATHER THAN GUESSABLE
-------------------------------------------------------------------
  1. every asserted amount is printed in the evidence VERBATIM - the lane never
     asks whether a number is right, only what it is bound to
  2. every asserted amount occurs EXACTLY ONCE in its passage, so the corrupted
     binding is not attested somewhere else and the label is correct
  3. the corrupted element is itself ATTESTED in the passage - the other role,
     the opposite direction word, the other year all appear in the evidence.
     Without this the negative would be separable by lexical novelty, which is
     not the skill the lane exists to install

BALANCE BY CONSTRUCTION
-----------------------
role_swap and period_swap are emitted in COUPLES over the same passage: one pair
asserts the first binding and corrupts it toward the second, its twin pair
asserts the second and corrupts it toward the first, both on the same template.
Every role phrase and every year therefore appears exactly as often on the
supported leg as on the corrupted one, and claim length balances exactly.
sign_swap is balanced by equal target cells for increase-true and decrease-true.

Source: the admitted EDGAR-restricted slice ONLY.  FinQA and TAT-QA source
corpora are WALLED and never opened; HotpotQA train and HoVer are never opened.

Run:  uv run python experiments/grounding-semantic/R20-H177_lane_C.py [--force]
"""

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import random
import sys

import polars as pl

HERE = Path(__file__).parent


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h177common", HERE / "R20-H177_lane_common.py")

OUT = HERE / "R20-H177_lane_C.parquet"
EVAL_OUT = HERE / "R20-H177_eval_C.parquet"
MANIFEST = HERE / "R20-H177_lane_C_manifest.json"

SEED = 1178
SEED_EVAL = 2178
TAG = "num_rolebind"
CHUNK_CAP_LADDER = (2, 4, 6)
N_FOLDS = 5

FAMILIES = ("role_swap", "sign_swap", "period_swap")

# (family, cell) -> target PAIRS.  role_swap and period_swap fill in couples of
# two pairs, so their targets are even.
TARGETS = {
    ("role_swap", "couple"): 5000,
    ("sign_swap", "increased"): 2500, ("sign_swap", "decreased"): 2500,
    ("period_swap", "couple"): 5000,
}
EVAL_TARGETS = {
    ("role_swap", "couple"): 334,
    ("sign_swap", "increased"): 166, ("sign_swap", "decreased"): 166,
    ("period_swap", "couple"): 334,
}

TEMPLATES = {
    "role_swap": [
        "The filing reports {role} of {v}.",
        "{role} of {v} is reported in the passage.",
    ],
    # verb forms, not "an increase / a decrease" - a nominal frame would move
    # the ARTICLE as well as the direction word and the twin would differ in two
    # places, which the registration's "ONLY the direction word" clause forbids
    "sign_swap": [
        "The filing reports that {role} {w} by {v}.",
        "{role} {w} by {v}, according to the filing.",
    ],
    "period_swap": [
        "In {y}, {role} of {v} is reported.",
        "The filing reports {role} of {v} for {y}.",
    ],
}
SIGN_FLIP = {"increased": "decreased", "decreased": "increased"}


# --------------------------------------------------------------------------- #
# extraction, once per chunk
# --------------------------------------------------------------------------- #
def edgar_items(edgar_rows, split):
    """Per-chunk role, change and period bindings, extracted once.

    A binding is kept only when its amount is printed EXACTLY ONCE in the chunk,
    which is what makes a corrupted binding unsupported rather than merely
    unstated somewhere else in the same passage."""
    out = []
    for i, r in enumerate(edgar_rows.select(["doc_id", "chunk"]).iter_rows(named=True)):
        chunk = r["chunk"]
        roles = [(role, val) for role, val, _ in
                 C.unique_role_bindings(chunk).values()
                 if C.occurrences(chunk, val) == 1]
        both_dirs = (C.word_attested(chunk, "increase")
                     and C.word_attested(chunk, "decrease"))
        changes = ([(d, role, val) for d, role, val, _s in C.change_statements(chunk)
                    if C.occurrences(chunk, val) == 1] if both_dirs else [])
        years = [(y, role, val, sent) for y, role, val, sent in C.year_bindings(chunk)
                 if C.occurrences(chunk, val) == 1]
        if roles or changes or years:
            out.append({"doc_id": r["doc_id"], "chunk": chunk, "roles": roles,
                        "changes": changes, "years": years})
        if (i + 1) % 10000 == 0:
            print(f"  [{split}] extraction {i + 1} chunks", flush=True)
    print(f"  [{split}] chunks with usable bindings: {len(out)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# family builders - each returns a LIST of pair specs (a couple, or one pair)
# --------------------------------------------------------------------------- #
def build_role_swap(item, rng, _cell):
    """Two labelled quantities in one passage, each asserted and each corrupted
    toward the other.  The couple is what makes the role balance exact."""
    chunk = item["chunk"]
    roles = list(item["roles"])
    rng.shuffle(roles)
    for i in range(len(roles)):
        for j in range(len(roles)):
            if i == j:
                continue
            (r1, v1), (r2, v2) = roles[i], roles[j]
            if v1 == v2 or r1.lower() in r2.lower() or r2.lower() in r1.lower():
                continue
            if not (C.word_attested(chunk, r1) and C.word_attested(chunk, r2)):
                continue
            ti = rng.randrange(len(TEMPLATES["role_swap"]))
            return [
                dict(true_word=r1, flip_word=r2, template_id=ti, subject=r1,
                     counterpart=r2, asserted_values=v1,
                     pos=TEMPLATES["role_swap"][ti].format(role=r1, v=v1),
                     neg=TEMPLATES["role_swap"][ti].format(role=r2, v=v1)),
                dict(true_word=r2, flip_word=r1, template_id=ti, subject=r2,
                     counterpart=r1, asserted_values=v2,
                     pos=TEMPLATES["role_swap"][ti].format(role=r2, v=v2),
                     neg=TEMPLATES["role_swap"][ti].format(role=r1, v=v2)),
            ]
    return None


def build_sign_swap(item, rng, cell):
    """A change the passage states in WORDS beside its magnitude - the H157 sign
    exemplars (189, 31, 168).  Flipping the word contradicts the passage and
    needs no computation to detect."""
    chunk = item["chunk"]
    cands = [c for c in item["changes"] if c[0] + "d" == cell]
    rng.shuffle(cands)
    for stem, role, val in cands:
        direction = stem + "d"
        flip = SIGN_FLIP[direction]
        if not (C.word_attested(chunk, direction) and C.word_attested(chunk, flip)):
            continue
        ti = rng.randrange(len(TEMPLATES["sign_swap"]))
        return [dict(true_word=direction, flip_word=flip, template_id=ti,
                     subject=role, counterpart=role, asserted_values=val,
                     pos=TEMPLATES["sign_swap"][ti].format(w=direction, v=val,
                                                           role=role),
                     neg=TEMPLATES["sign_swap"][ti].format(w=flip, v=val,
                                                           role=role))]
    return None


def build_period_swap(item, rng, _cell):
    """The H157 lever-3 freebie - the right value on the wrong year (items 36,
    229).  The corrupting year is one the passage itself names, and it may not
    appear in the sentence that carries the binding, so the true period is
    unambiguous in the evidence."""
    years = list(item["years"])
    rng.shuffle(years)
    for i in range(len(years)):
        for j in range(len(years)):
            if i == j:
                continue
            (y1, r1, v1, s1), (y2, r2, v2, s2) = years[i], years[j]
            if y1 == y2 or v1 == v2:
                continue
            if y2 in s1 or y1 in s2:
                continue
            ti = rng.randrange(len(TEMPLATES["period_swap"]))
            return [
                dict(true_word=y1, flip_word=y2, template_id=ti, subject=r1,
                     counterpart=y2, asserted_values=v1,
                     pos=TEMPLATES["period_swap"][ti].format(y=y1, role=r1, v=v1),
                     neg=TEMPLATES["period_swap"][ti].format(y=y2, role=r1, v=v1)),
                dict(true_word=y2, flip_word=y1, template_id=ti, subject=r2,
                     counterpart=y1, asserted_values=v2,
                     pos=TEMPLATES["period_swap"][ti].format(y=y2, role=r2, v=v2),
                     neg=TEMPLATES["period_swap"][ti].format(y=y1, role=r2, v=v2)),
            ]
    return None


BUILDERS = {"role_swap": build_role_swap, "sign_swap": build_sign_swap,
            "period_swap": build_period_swap}


def emit(family, item, spec, pid):
    base = dict(chunk=item["chunk"], doc_id=item["doc_id"], source="edgar",
                tag=TAG, neg_family=family, true_word=spec["true_word"],
                flip_word=spec["flip_word"], template_id=spec["template_id"],
                subject=spec["subject"], counterpart=spec["counterpart"],
                asserted_values=spec["asserted_values"])
    return [dict(pair_id=pid, label=1, claim=spec["pos"], **base),
            dict(pair_id=pid, label=0, claim=spec["neg"], **base)]


def assemble(split, seed, targets, items):
    rng = random.Random(seed)
    built = collections.Counter()
    rows, per_chunk, seen, pid = [], collections.Counter(), set(), 0
    cells = list(targets)

    for cap in CHUNK_CAP_LADDER:
        if all(built[c] >= targets[c] for c in cells):
            break
        order = list(range(len(items)))
        rng.shuffle(order)
        for k in order:
            if all(built[c] >= targets[c] for c in cells):
                break
            item = items[k]
            while per_chunk[k] < cap:
                order_cells = sorted(cells, key=lambda c: built[c] - targets[c])
                specs, cell = None, None
                for c in order_cells:
                    if built[c] >= targets[c]:
                        continue
                    cand = BUILDERS[c[0]](item, rng, c[1])
                    if cand is not None:
                        specs, cell = cand, c
                        break
                if specs is None:
                    break
                # a couple must survive de-duplication WHOLE.  If any of its four
                # rows repeats a row already emitted, `dedupe` would drop that row,
                # orphan its twin and break the exact element balance the couple
                # exists to give - so the couple is rejected before it is emitted.
                sig = [(item["chunk"], s[k]) for s in specs for k in ("pos", "neg")]
                if any(x in seen for x in sig):
                    break
                seen.update(sig)
                for s in specs:
                    rows.extend(emit(cell[0], item, s, pid))
                    pid += 1
                built[cell] += len(specs)
                per_chunk[k] += 1
        print(f"  [{split}] cap {cap}: "
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
                   "element-stratified, liblinear tol 1e-7",
        "documents": ndocs, "rows": int(df.height)}

    wp = C.within_pair_accuracy(df, score, by="neg_family")
    worst = max(v["acc"] for v in wp.values())
    out["within_pair_claim_only_accuracy"] = {
        "per_family": wp, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    out["surface_parity"] = C.surface_parity(df)
    out["swapped_element_balance"] = C.flip_word_balance(df)
    out["attestation_symmetry"] = C.attestation_audit(df, require_attested=FAMILIES)
    out["value_presence"] = C.verbatim_value_audit(df)
    out["value_uniqueness"] = C.value_uniqueness_audit(df, FAMILIES)

    # the corrupted element must genuinely differ from the true one
    same = [r["pair_id"] for r in df.filter(pl.col("label") == 0).iter_rows(named=True)
            if r["true_word"].lower() == r["flip_word"].lower()]
    identical = [r["pair_id"] for r in df.group_by("pair_id").agg(
        pl.col("claim").n_unique().alias("n")).filter(pl.col("n") < 2).iter_rows(
            named=True)]
    out["corruption_is_real"] = {
        "pairs_with_equal_elements": len(same),
        "pairs_with_identical_claims": len(identical),
        "bar": "0 on both", "pass": not (same or identical),
        "examples": (same + identical)[:5]}

    out["all_bars_pass"] = all(
        out[k]["pass"] for k in
        ("pair_integrity", "claim_only_tfidf_auroc",
         "within_pair_claim_only_accuracy", "surface_parity",
         "swapped_element_balance", "attestation_symmetry", "value_presence",
         "value_uniqueness", "corruption_is_real"))
    return out


# --------------------------------------------------------------------------- #
def block(df, res, cells, split, seed, targets):
    y = df["label"].to_list()
    fams = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    return dict(
        rows=df.height, pairs=int(df["pair_id"].n_unique()),
        documents=int(df["doc_id"].n_unique()),
        chunks=int(df["chunk"].n_unique()), seed=seed, split=split,
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y))},
        families=fams,
        family_shares={k: round(v / df.height, 4) for k, v in fams.items()},
        families_below_15pct=[k for k, v in fams.items() if v / df.height < 0.15],
        cells_filled=cells,
        cell_targets={f"{a}:{b}": n for (a, b), n in sorted(targets.items())},
        templates={str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
        diversity=dict(distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique()),
                       distinct_subjects=int(df["subject"].n_unique()),
                       distinct_swapped_elements=int(df["flip_word"].n_unique())),
        char_stats=dict(claim=C.char_stats(df["claim"].to_list()),
                        chunk=C.char_stats(df["chunk"].to_list())),
        window_census=C.window_census(df["chunk"].to_list()),
        verify=res)


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
    print(f"=== R20-H177 lane C ({TAG}) seed {SEED}", flush=True)
    ed_train, ed_eval = C.edgar("train"), C.edgar("eval")
    print(f"supply: edgar {ed_train.height} train / {ed_eval.height} eval chunks "
          f"over {ed_train['doc_id'].n_unique()} / {ed_eval['doc_id'].n_unique()} "
          f"filings", flush=True)
    items = edgar_items(ed_train, "train")
    items_eval = edgar_items(ed_eval, "eval")

    df, cells = assemble("train", SEED, TARGETS, items)
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}",
          flush=True)

    ev, ev_cells = assemble("eval", SEED_EVAL, EVAL_TARGETS, items_eval)
    ev.write_parquet(EVAL_OUT)
    print(f"{ev.height} rows / {ev['pair_id'].n_unique()} pairs -> {EVAL_OUT.name}",
          flush=True)

    res = verify(df, random.Random(SEED))
    ev_res = verify(ev, random.Random(SEED_EVAL))
    shared_docs = set(df["doc_id"].to_list()) & set(ev["doc_id"].to_list())
    shared_chunks = set(df["chunk"].to_list()) & set(ev["chunk"].to_list())

    man = dict(
        experiment="R20-H177 lane C - operand-role / sign / period misbind "
                   "(num_rolebind)",
        registration="docs/experiments/semantic-grounding-experiments.md, block "
                     "'R20-H177 NUMERIC-VERIFICATION PORTFOLIO ARM'",
        tag=TAG, dann_group=TAG,
        mix_loader="drop-in for R18-H150_arm_run.make_build_mix - columns claim / "
                   "chunk / label / pair_id / neg_family; chunk is read "
                   "UNTRUNCATED and windowed 1500/750 by the loader",
        parquet=OUT.name,
        sources={"edgar": C.SOURCES["edgar"]},
        walled_never_opened=C.WALLED_NEVER_OPENED,
        **block(df, res, cells, "train", SEED, TARGETS))
    man["held_out_eval"] = dict(
        parquet=EVAL_OUT.name,
        purpose="the H177 PRIMARY mechanism gate for lane C - same generator, "
                "different seed, filings excluded from the training lane; no "
                "model is read on it here (stage 0 is CPU-only)",
        **block(ev, ev_res, ev_cells, "eval", SEED_EVAL, EVAL_TARGETS))
    man["document_disjointness"] = dict(
        rule=f"blake2b(doc_id) % 1000 < {C.EVAL_DOC_PERMILLE} -> eval only",
        shared_documents=len(shared_docs), shared_chunks=len(shared_chunks),
        bar="0 shared documents", pass_=len(shared_docs) == 0)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "families", "family_shares",
                       "families_below_15pct", "cells_filled", "verify")},
                     indent=2), flush=True)
    print(json.dumps({"eval_rows": man["held_out_eval"]["rows"],
                      "eval_pairs": man["held_out_eval"]["pairs"],
                      "eval_families": man["held_out_eval"]["families"],
                      "eval_all_bars_pass":
                          man["held_out_eval"]["verify"]["all_bars_pass"],
                      "shared_documents": len(shared_docs)}, indent=2), flush=True)
    ok = res["all_bars_pass"] and ev_res["all_bars_pass"] and not shared_docs
    print(f"=== R20-H177 LANE C {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
