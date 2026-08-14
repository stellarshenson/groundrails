"""R19 supply wave - pair-format the gate-GREEN corpora into lane parquets.

Registered in docs/experiments/semantic-dataset-enhancements.md section "R19
supply wave".  SUPPLY ONLY: nothing here enters a training mix; a later
registered hypothesis decides that.

Per corpus (gated GREEN by R19_supply_gates.py - a RED or missing gate refuses
the build):
  * `R19_<name>_lane.parquet` - one row per (claim, evidence) pair under the
    campaign's lane conventions: pair_id, claim, chunk (evidence), label,
    doc_id, source, tag, plus per-corpus provenance columns
  * `R19_<name>_lane_manifest.json` - counts, label and source distributions,
    provenance references
  * `R19_<name>_lane_verify.json` - the R14-H136-convention verify block:
    integrity checks (label domain, no empty text, zero duplicates), a
    claim-only TF-IDF baseline (5-fold document-disjoint, liblinear tol 1e-7,
    the H150 discipline) as a MEASUREMENT, and disjointness stats against the
    banked R17-H143 eval set (content fingerprints)

Label mappings (registered coordinator rulings):
  fava              any error span in the tagged completion -> 0, clean -> 1
  pubhealth         true -> 1; false/unproven/mixture -> 0 (support, not truth)
  minicheck         shipped label (1 supported / 0 not)
  factscore         S -> 1, NS -> 0 per atomic fact; IR excluded (off-topic)
  findver           entailed -> 1, refuted -> 0; subset tags retained
  attributionbench  attributable -> 1, not attributable -> 0

Run:  uv run python experiments/grounding-semantic/R19_supply_lanes.py [name ...]
"""

import collections
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
EVALSET = HERE / "R17-H143_evalset.parquet"

_spec = importlib.util.spec_from_file_location("provgate", HERE / "provenance_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

N_FOLDS = 5
SEED = 19


def zip_parquets(name):
    z = zipfile.ZipFile(DATA / f"dataset-{name}.zip")
    out = {}
    for m in z.namelist():
        if m.endswith(".parquet"):
            out[m[:-len(".parquet")].split("__")[-1]] = pl.read_parquet(io.BytesIO(z.read(m)))
    return out


def fingerprint(text):
    return hashlib.blake2b(G.normalize(text).encode(), digest_size=16).hexdigest()


# --------------------------------------------------------------------------- #
# fava
# --------------------------------------------------------------------------- #
FAVA_PREFIX = "Read the following references:\n"
FAVA_SEP = ("\nPlease identify all the errors in the following passage using the "
            "references provided and suggest edits:\nText: ")
FAVA_TAGS = ("entity", "relation", "contradictory", "subjective", "unverifiable", "invented")
FAVA_EDIT = re.compile(r"<(entity|relation)>(.*?)</\1>", re.DOTALL)
# LLM-written completions frequently leave the wrapper unclosed
# (`<relation><mark>X</mark><delete>Y</delete>.` with no `</relation>`) - the
# fallback captures the same span without requiring the closer
FAVA_EDIT_OPEN = re.compile(
    r"<(entity|relation)>\s*<mark>(.*?)</mark>\s*<delete>(.*?)</delete>", re.DOTALL)
FAVA_WRAP = re.compile(
    r"<(contradictory|subjective|unverifiable|invented)>(.*?)</\1>", re.DOTALL)
FAVA_MARK = re.compile(r"<mark>(.*?)</mark>", re.DOTALL)
FAVA_DEL = re.compile(r"<delete>(.*?)</delete>", re.DOTALL)
FAVA_ANYTAG = re.compile(r"</?(" + "|".join(FAVA_TAGS) + r"|mark|delete)\s*>")
# the label rule: ANY open error tag, closed or not, marks the row hallucinated
FAVA_ERR_OPEN = re.compile(r"<(" + "|".join(FAVA_TAGS) + r")[>\s]")


def fava_spans(completion):
    """Extract error spans from a tagged completion, and reconstruct the
    original pre-edit passage (delete branches kept, mark branches dropped).

    Tag semantics verified against the paper (arXiv 2401.06855, Appendix C
    prompt): <mark> carries the CORRECT text matching the reference, <delete>
    carries the error text present in the original passage."""
    spans = []

    def rec_edit(m):
        body = m.group(2)
        mark = FAVA_MARK.search(body)
        dele = FAVA_DEL.search(body)
        spans.append({"type": m.group(1),
                      "mark": mark.group(1).strip() if mark else None,
                      "delete": dele.group(1).strip() if dele else None})
        return dele.group(1) if dele else ""  # mark-only edit = inserted text

    def rec_open(m):
        spans.append({"type": m.group(1), "mark": m.group(2).strip(),
                      "delete": m.group(3).strip()})
        return m.group(3)

    def rec_wrap(m):
        spans.append({"type": m.group(1), "mark": None, "delete": m.group(2).strip()})
        return m.group(2)

    recon = FAVA_EDIT_OPEN.sub(rec_open, FAVA_EDIT.sub(rec_edit, completion))
    recon = FAVA_WRAP.sub(rec_wrap, recon)
    residual = bool(FAVA_ANYTAG.search(recon))
    recon = FAVA_ANYTAG.sub("", recon)
    return spans, recon, residual


def build_fava():
    df = zip_parquets("fava")["train"]
    rows, orphan_markup, recon_match = [], 0, 0
    for r in df.iter_rows(named=True):
        p = r["prompt"]
        if not (p.startswith(FAVA_PREFIX) and FAVA_SEP in p):
            continue
        refs = p[len(FAVA_PREFIX):p.index(FAVA_SEP)]
        claim = p[p.index(FAVA_SEP) + len(FAVA_SEP):]
        comp = r["completion"]
        spans, recon, residual = fava_spans(comp)
        orphan_markup += residual
        if recon.strip() == claim.strip():
            recon_match += 1
        rows.append({
            "claim": claim.strip(), "chunk": refs.strip(),
            "label": 0 if FAVA_ERR_OPEN.search(comp) else 1,
            "doc_id": hashlib.blake2b(p.encode(), digest_size=8).hexdigest(),
            "source": "train",
            "n_spans": len(spans),
            "error_types": json.dumps(sorted({s["type"] for s in spans})),
            "spans_json": json.dumps(spans),
            "completion_tagged": comp,
            "reconstruction_matches_draft": recon.strip() == claim.strip(),
        })
    df_out = pl.DataFrame(rows)
    stats = {"rows_in": df.height,
             "rows_with_orphan_tag_markup": orphan_markup,
             "orphan_markup_note": "LLM-written completions carry unclosed or "
                 "wrapper-less tags (e.g. a bare <delete>early</delete>); the "
                 "label rule counts any error-tag open, so labels are unaffected "
                 "- only span completeness is",
             "reconstruction_match_rate": round(recon_match / max(len(rows), 1), 6),
             "reconstruction_note": "share of rows where the tag-resolved "
                 "completion reproduces the prompt's draft byte-for-byte; the "
                 "claim is the prompt's draft regardless, so a miss costs "
                 "nothing - it flags annotation/draft drift on that row"}
    return df_out, stats


# --------------------------------------------------------------------------- #
# pubhealth
# --------------------------------------------------------------------------- #
def build_pubhealth():
    parts = zip_parquets("pubhealth")
    rows, dropped_empty, dropped_stray = [], 0, 0
    for split, df in parts.items():
        for r in df.iter_rows(named=True):
            text = (r.get("main_text") or "").strip()
            verdict = (r.get("label") or "").strip()
            if not text:
                dropped_empty += 1
                continue
            if verdict not in ("true", "false", "unproven", "mixture"):
                dropped_stray += 1
                continue
            rows.append({
                "claim": (r.get("claim") or "").strip(), "chunk": text,
                "label": 1 if verdict == "true" else 0,
                "doc_id": f"pubhealth_{r.get('claim_id')}",
                "source": split,
                "verdict": verdict,
                "explanation": r.get("explanation") or "",
                "fact_checkers": r.get("fact_checkers") or "",
                "subjects": r.get("subjects") or "",
                "date_published": r.get("date_published") or "",
            })
    stats = {"rows_in": sum(df.height for df in parts.values()),
             "dropped_empty_main_text": dropped_empty,
             "dropped_stray_label": dropped_stray}
    return pl.DataFrame(rows), stats


# --------------------------------------------------------------------------- #
# minicheck
# --------------------------------------------------------------------------- #
def build_minicheck():
    parts = zip_parquets("minicheck")
    rows = []
    for split, df in parts.items():
        for r in df.iter_rows(named=True):
            rows.append({
                "claim": r["claim"].strip(), "chunk": r["doc"].strip(),
                "label": int(r["label"]),
                "doc_id": hashlib.blake2b(r["doc"].encode(), digest_size=8).hexdigest(),
                "source": split,
            })
    return pl.DataFrame(rows), {"rows_in": sum(df.height for df in parts.values())}


# --------------------------------------------------------------------------- #
# factscore
# --------------------------------------------------------------------------- #
def build_factscore():
    tree = DATA / "factscore"
    index = json.loads((tree / "evidence" / "_index.json").read_text())
    rows, ir_dropped, missing_ev = [], 0, 0
    for model in ("InstructGPT", "ChatGPT", "PerplexityAI"):
        for ln in open(tree / "data" / "labeled" / f"{model}.jsonl"):
            r = json.loads(ln)
            topic = r["topic"]
            ev = index.get(topic)
            if not ev:
                missing_ev += 1
                continue
            chunk = (tree / "evidence" / ev["file"]).read_text()
            for ann in r.get("annotations") or []:
                for f in ann.get("human-atomic-facts") or []:
                    lab = f.get("label")
                    if lab == "IR":
                        ir_dropped += 1
                        continue
                    if lab not in ("S", "NS"):
                        continue
                    rows.append({
                        "claim": f["text"].strip(), "chunk": chunk,
                        "label": 1 if lab == "S" else 0,
                        "doc_id": topic, "source": model,
                        "sentence": ann.get("text", ""),
                        "sentence_is_relevant": bool(ann.get("is-relevant")),
                        "wiki_oldid": str(ev["oldid"]),
                        "wiki_rev_timestamp": ev["timestamp"],
                    })
    stats = {"ir_facts_dropped": ir_dropped,
             "biographies_missing_evidence": missing_ev}
    return pl.DataFrame(rows), stats


# --------------------------------------------------------------------------- #
# findver
# --------------------------------------------------------------------------- #
def build_findver():
    tree = DATA / "findver"
    rows, missing_ctx = [], 0
    docs = {}
    for split in ("test", "testmini"):
        for r in json.loads((tree / "data" / f"{split}.json").read_text()):
            rep = r["report"]
            if rep not in docs:
                doc = json.loads((tree / "financial_reports" / rep).read_text())
                docs[rep] = {c["id"]: c for c in doc["context"]}
            ctx = docs[rep]
            parts = []
            ok = True
            for i in r["relevant_context"]:
                if i in ctx:
                    c = ctx[i]
                    parts.append(f"[context {i} | {c.get('type', 'text')}]\n{c['context']}")
                else:
                    ok = False
            if not parts:
                missing_ctx += 1
                continue
            rows.append({
                "claim": r["statement"].strip(), "chunk": "\n\n".join(parts),
                "label": 1 if r["entailment_label"] else 0,
                "doc_id": rep, "source": split,
                "example_id": r["example_id"], "subset": r["subset"],
                "explanation": r.get("explanation", ""),
                "context_ids": json.dumps(r["relevant_context"]),
                "all_contexts_resolved": ok,
            })
    stats = {"claims_missing_all_contexts": missing_ctx}
    return pl.DataFrame(rows), stats


# --------------------------------------------------------------------------- #
# attributionbench
# --------------------------------------------------------------------------- #
def build_attributionbench():
    parts = zip_parquets("attributionbench")
    rows = []
    for split, df in parts.items():
        for r in df.iter_rows(named=True):
            refs = [x for x in (r.get("references") or []) if x]
            claim = (r.get("claim") or "").strip()
            if not refs or not claim:
                continue
            chunk = "\n\n".join(refs)
            rows.append({
                "claim": claim, "chunk": chunk,
                "label": 1 if r["attribution_label"] == "attributable" else 0,
                "doc_id": hashlib.blake2b(chunk.encode(), digest_size=8).hexdigest(),
                "source": split,
                "src_dataset": r["src_dataset"],
                "question": r.get("question") or "",
                "ab_id": r.get("id") or "",
                "n_references": len(refs),
            })
    return pl.DataFrame(rows), {"rows_in": sum(df.height for df in parts.values())}


BUILDERS = {
    "fava": build_fava,
    "pubhealth": build_pubhealth,
    "minicheck": build_minicheck,
    "factscore": build_factscore,
    "findver": build_findver,
    "attributionbench": build_attributionbench,
}


# --------------------------------------------------------------------------- #
# verify (R14-H136 conventions: claim-only baseline, integrity, disjointness)
# --------------------------------------------------------------------------- #
def auroc(labels, scores):
    """Rank-based AUROC, O(n log n) - safe at 30k+ rows."""
    order = np.argsort(np.asarray(scores), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    y = np.asarray(labels)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def verify(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    out = {}
    labels = df["label"].to_list()
    claims = df["claim"].to_list()

    # integrity - must pass by construction
    dup = df.height - df.unique(subset=["claim", "chunk", "label"]).height
    out["integrity"] = {
        "labels_in_01": set(labels) <= {0, 1},
        "empty_claims": int(sum(not c for c in claims)),
        "empty_chunks": int(sum(not c for c in df["chunk"].to_list())),
        "duplicate_claim_chunk_label_rows": int(dup),
        "pass": bool(set(labels) <= {0, 1} and dup == 0
                     and all(claims) and all(df["chunk"].to_list()))}

    # claim-only baseline - a MEASUREMENT on supply lanes (the < 0.55 bar is
    # registered for synthetic minimal-pair lanes; real corpora legitimately
    # carry claim-side signal).  5-fold document-disjoint, liblinear tol 1e-7.
    groups = np.array(df["doc_id"].to_list())
    y = np.array(labels)
    score = np.zeros(len(df))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(claims, y, groups):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform(claims[j] for j in tr)
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, y[tr])
        score[te] = clf.decision_function(vec.transform(claims[j] for j in te))
    probe = float(auroc(y, score))
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55 on synthetic minimal-pair lanes",
        "pass": bool(probe < 0.55),
        "applicability": "measurement only for supply lanes - real corpora carry "
                         "legitimate claim-side signal; banking is decided by the "
                         "licence + contamination gates",
        "scoring": f"{N_FOLDS}-fold document-disjoint (GroupKFold on doc_id), "
                   "liblinear tol 1e-7",
        "documents": int(df["doc_id"].n_unique()), "rows": df.height}

    # disjointness vs the banked R17-H143 eval set (content fingerprints)
    ev = {"evalset_present": EVALSET.exists()}
    if EVALSET.exists():
        ev_prints = {fingerprint(c)
                     for c in pl.read_parquet(EVALSET, columns=["chunk"])["chunk"].to_list()}
        cand_prints = {fingerprint(c) for c in df["chunk"].to_list()}
        ev["evalset_shared_content_fingerprints"] = len(ev_prints & cand_prints)
    out["disjointness"] = {
        **ev,
        "distinct_claims": int(df["claim"].n_unique()),
        "distinct_chunks": int(df["chunk"].n_unique()),
        "distinct_documents": int(df["doc_id"].n_unique()),
        "pass": not ev.get("evalset_shared_content_fingerprints")}
    out["all_integrity_pass"] = out["integrity"]["pass"] and out["disjointness"]["pass"]
    return out


def build_one(name):
    gate_json = HERE / f"R19_{name}_gate.json"
    if not gate_json.exists():
        print(f"=== {name}: no gate JSON - run R19_supply_gates.py first; refused",
              flush=True)
        return "NO-GATE"
    status = json.loads(gate_json.read_text())["status"]
    if status != "GREEN":
        print(f"=== {name}: gate {status} - QUARANTINED, no lane built", flush=True)
        return "QUARANTINED"

    print(f"=== {name}: gate GREEN - building lane", flush=True)
    df, stats = BUILDERS[name]()
    df = df.unique(subset=["claim", "chunk", "label"], keep="first", maintain_order=True)
    df = df.with_row_index("pair_id").with_columns(
        pl.col("pair_id").cast(pl.Int64), pl.lit(name).alias("tag"))
    cols = ["pair_id", "claim", "chunk", "label", "doc_id", "source", "tag"]
    df = df.select(cols + [c for c in df.columns if c not in cols])

    out = HERE / f"R19_{name}_lane.parquet"
    df.write_parquet(out)
    res = verify(df)

    label_ct = {str(k): v for k, v in df.group_by("label").len().iter_rows()}
    src_ct = {str(k): v for k, v in df.group_by("source").len().iter_rows()}
    manifest = {
        "experiment": f"R19 supply wave lane - {name} (SUPPLY ONLY; no training "
                      "use without a registered hypothesis)",
        "registration": "docs/experiments/semantic-dataset-enhancements.md, "
                        "section 'R19 supply wave' (2026-08-13)",
        "rows": df.height, "documents": int(df["doc_id"].n_unique()),
        "label_distribution": label_ct, "source_distribution": src_ct,
        "tag": name,
        "provenance_columns": [c for c in df.columns
                               if c not in ("pair_id", "claim", "chunk", "label",
                                            "doc_id", "source", "tag")],
        "build_stats": stats,
        "gate": f"R19_{name}_gate.json (status GREEN)",
        "fetch": (f"data/external/datasets/dataset-{name}.zip"
                  if (DATA / f"dataset-{name}.zip").exists()
                  else f"data/external/datasets/{name}/ (tree)"),
        "verify": res,
    }
    (HERE / f"R19_{name}_lane_manifest.json").write_text(json.dumps(manifest, indent=2))
    (HERE / f"R19_{name}_lane_verify.json").write_text(json.dumps(
        {"rows": df.height, "verify": res}, indent=2))
    print(f"  {df.height} rows, labels {label_ct}, sources {src_ct}", flush=True)
    print(f"  claim-only AUROC {res['claim_only_tfidf_auroc']['value']} "
          f"(measurement), integrity {res['all_integrity_pass']}", flush=True)
    print(f"=== {name} LANE BANKED -> {out.name}", flush=True)
    return "BANKED"


def main():
    names = sys.argv[1:] or list(BUILDERS)
    summary = {}
    for name in names:
        try:
            summary[name] = build_one(name)
        except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
            summary[name] = f"DEFECT: {type(e).__name__}: {str(e)[:200]}"
            print(f"=== {name} LANE DEFECT: {summary[name]}", flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
