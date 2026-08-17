"""Follow-up detail behind the `tabfact` clause verdicts. CPU only, Polars only.

Five measurements, each closing a question the first pass raised rather than
letting it be asserted:

  D1  stem-collision CONTENT - TabFact writes one Wikipedia table under both a
      `1-` and a `2-` csv id. Train/validation/test are disjoint on the id
      STRING; this measures whether the colliding stems are the same DOCUMENT.
  D2  the anti-gaming probe sets - their construction drops every table_id
      present in TabFact train, an exact-id rule. This measures the content
      behind their surviving stem collisions with the member.
  D3  C6 quota control - the leave-one-out table-label feature is re-measured on
      labels PERMUTED WITHIN each table. A permutation preserves each table's
      label quota and destroys everything else, so a reproduced AUROC proves the
      feature carries the quota and nothing more.
  D4  the 51 claims carrying both labels - same table or different tables.
  D5  the R20-H177_eval_B leak, attributed: which rows, pairs, families and
      serialisation forms sit on the passages and documents the member carries.

Out: tabfact_detail.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member.parquet"
OUT = HERE / "tabfact_detail.json"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    return " ".join(s.split()).casefold()


def stem(t):
    return t[2:] if len(t) > 2 and t[0] in "12" and t[1] == "-" else t


def build_chunk(cap, tbl):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")


def split_tables():
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    out = {}
    for s in ("train", "validation", "test"):
        n = next(x for x in z.namelist() if x.endswith(f"__{s}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n))).unique(subset=["table_id"], keep="first")
        out[s] = {t: build_chunk(c, b) for t, c, b in
                  zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                      d["table_text"].to_list(), strict=True)}
    return out


def compare_stems(a_tabs, b_tabs, Q):
    """b's tables against a's, matched on stem. Returns the content verdicts."""
    a_by_stem = collections.defaultdict(list)
    for t, txt in a_tabs.items():
        a_by_stem[stem(t)].append(txt)
    rows, jac = [], []
    exact = nrm = 0
    for t, txt in b_tabs.items():
        cand = a_by_stem.get(stem(t))
        if not cand:
            continue
        best = max(cand, key=lambda c: Q.jaccard(c, txt))
        j = Q.jaccard(best, txt)
        jac.append(j)
        e = best == txt
        n = norm(best) == norm(txt)
        exact += e
        nrm += n
        rows.append({"id": t, "exact": bool(e), "normalised_equal": bool(n),
                     "token_jaccard": round(j, 4)})
    j = np.array(jac) if jac else np.zeros(1)
    return {
        "stem_colliding_tables": len(rows),
        "byte_identical": exact,
        "normalised_identical": nrm,
        "token_jaccard": {"mean": round(float(j.mean()), 4),
                          "median": round(float(np.median(j)), 4),
                          "max": round(float(j.max()), 4),
                          "share_ge_0.9": round(float((j >= 0.9).mean()), 4),
                          "share_ge_0.5": round(float((j >= 0.5).mean()), 4)},
        "examples": sorted(rows, key=lambda r: -r["token_jaccard"])[:5],
    }


def main():
    t0 = time.time()
    Q = _mod("qlane", SEM / "R20-H175b_qlane.py")
    df = pl.read_parquet(MEMBER)
    res = {"member": "tabfact"}

    tabs = split_tables()
    member_tabs = {t: c for t, c in zip(df["table_id"].to_list(),
                                        df["chunk_untrunc"].to_list(), strict=True)}

    # ---- D1 ---------------------------------------------------------------
    res["D1_stem_collision_content_vs_member"] = {
        "question": "train/validation/test share 0 table_id STRINGS. Are the "
                    "stem-colliding tables the same document?",
        "validation": compare_stems(member_tabs, tabs["validation"], Q),
        "test": compare_stems(member_tabs, tabs["test"], Q),
    }
    print("D1 done", flush=True)

    # ---- D2 ---------------------------------------------------------------
    heldout_tabs = {**tabs["validation"], **tabs["test"]}
    ag = {}
    for path in sorted(SEM.glob("*antigaming_set.parquet")):
        d = pl.read_parquet(path)
        if "table_id" not in d.columns:
            continue
        ids = set(d["table_id"].to_list())
        sub = {t: heldout_tabs[t] for t in ids if t in heldout_tabs}
        ag[path.name] = {
            "distinct_table_id": len(ids),
            "resolved_in_tabfact_heldout_splits": len(sub),
            "exact_table_id_in_member": len(ids & set(member_tabs)),
            **compare_stems(member_tabs, sub, Q),
        }
    res["D2_antigaming_probe_sets"] = {
        "construction": "R14-H133_antigaming - TabFact test+validation with every "
                        "table_id present in TabFact train removed (an EXACT id rule)",
        "per_file": ag,
    }
    print("D2 done", flush=True)

    # ---- D3 ---------------------------------------------------------------
    y = df["label"].to_numpy()
    tids = df["table_id"].to_list()
    by_tab = collections.defaultdict(list)
    for i, t in enumerate(tids):
        by_tab[t].append(i)

    def loo_auroc(labels):
        v = np.empty(len(labels))
        for idxs in by_tab.values():
            s = sum(labels[i] for i in idxs)
            k = len(idxs)
            for i in idxs:
                v[i] = (s - labels[i]) / (k - 1) if k > 1 else 0.5
        return float(roc_auc_score(labels.astype(int), v))

    rng = np.random.default_rng(0)
    perm_aurocs = []
    for _ in range(5):
        yp = y.copy()
        for idxs in by_tab.values():
            arr = yp[idxs]
            rng.shuffle(arr)
            yp[idxs] = arr
        perm_aurocs.append(loo_auroc(yp))
    res["D3_C6_quota_control"] = {
        "observed_loo_auroc": round(loo_auroc(y), 4),
        "within_table_label_permutation_auroc": [round(a, 4) for a in perm_aurocs],
        "permutation_mean": round(float(np.mean(perm_aurocs)), 4),
        "reading": "a within-table permutation preserves each table's label QUOTA "
                   "and destroys every association between a statement and its "
                   "label. If the permuted AUROC reproduces the observed one, the "
                   "leave-one-out feature carries the quota alone - the "
                   "hypergeometric anti-correlation of sampling without "
                   "replacement - and no statement-level association",
    }
    print("D3 done", flush=True)

    # ---- D4 ---------------------------------------------------------------
    lab_by_claim = collections.defaultdict(set)
    tab_by_claim = collections.defaultdict(set)
    for c, v, t in zip(df["claim"].to_list(), y, tids, strict=True):
        lab_by_claim[c].add(float(v))
        tab_by_claim[c].add(t)
    both = [c for c, s in lab_by_claim.items() if len(s) > 1]
    same_tab = sum(1 for c in both if len(tab_by_claim[c]) == 1)
    res["D4_claims_carrying_both_labels"] = {
        "claims": len(both),
        "on_the_SAME_table": same_tab,
        "on_different_tables": len(both) - same_tab,
        "rows_involved": int(sum(1 for c in df["claim"].to_list() if c in set(both))),
        "reading": "a claim on different tables carrying different labels is "
                   "correct behaviour; on the SAME table it is an annotation "
                   "contradiction the head cannot satisfy",
        "examples": [{"claim": c, "tables": sorted(tab_by_claim[c])} for c in both[:5]],
    }
    print("D4 done", flush=True)

    # ---- D5 ---------------------------------------------------------------
    ev = pl.read_parquet(SEM / "R20-H177_eval_B.parquet")
    mem_norm = {norm(c) for c in df["chunk_untrunc"].to_list()}
    mem_stems = {stem(t) for t in tids}
    ev = ev.with_columns([
        pl.col("chunk").map_elements(lambda c: norm(c) in mem_norm,
                                     return_dtype=pl.Boolean).alias("passage_in_member"),
        pl.col("doc_id").map_elements(
            lambda d: d.startswith("tabfact:") and stem(d[len("tabfact:"):]) in mem_stems,
            return_dtype=pl.Boolean).alias("document_in_member"),
    ])
    p = ev.filter(pl.col("passage_in_member"))
    dcm = ev.filter(pl.col("document_in_member"))
    res["D5_eval_B_leak_attributed"] = {
        "eval_rows": ev.height, "eval_pairs": ev["pair_id"].n_unique(),
        "distinct_passages": ev["chunk"].n_unique(),
        "passage_channel": {
            "passages_normalised_identical_to_member_evidence": p["chunk"].n_unique(),
            "share_of_distinct_passages": round(
                p["chunk"].n_unique() / ev["chunk"].n_unique(), 4),
            "rows": p.height, "pairs": p["pair_id"].n_unique(),
            "by_source_and_serial_form": [
                dict(zip(("source", "serial_form", "rows"), r, strict=True))
                for r in p.group_by(["source", "serial_form"]).len().iter_rows()],
            "by_neg_family": dict(p.group_by("neg_family").len().iter_rows()) if p.height else {},
        },
        "document_channel": {
            "rows": dcm.height, "pairs": dcm["pair_id"].n_unique(),
            "documents": dcm["doc_id"].n_unique(),
            "share_of_eval_rows": round(dcm.height / ev.height, 4),
            "by_source_and_serial_form": [
                dict(zip(("source", "serial_form", "rows"), r, strict=True))
                for r in dcm.group_by(["source", "serial_form"]).len().iter_rows()],
            "note": "every TabFact document eval_B draws is a document the member "
                    "carries; the passage channel only catches the ONE "
                    "serialisation form the member also uses",
        },
    }
    print("D5 done", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name} ({res['elapsed_s']}s)", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("D1", "D3", "D4"))},
                     indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
