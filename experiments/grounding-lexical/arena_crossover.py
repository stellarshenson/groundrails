"""R14 - lexical arena crossover: the SHIPPED lexical tier measured blind on
RAGBench and RAGTruth EN.

Registered pre-measurement in docs/experiments/lexical-grounding-experiments.md
(round 14). Every lexical number in that log sits on private gold v2, VitaminC
or articles; the semantic track competes on the public RAGBench arena. This
script runs the shipped lexical pipeline - regex `extract_claims` segmentation
into per-claim `ground()` verdicts under the UNMODIFIED bundled config
(R12 synthetic-retrained HIGH manifold, single global cut 0.50) - eval-only over
the identical frozen gate protocol the semantic reads use:

  RAGBench   same zip, same filter (adherence non-null, response > 20 chars,
             documents non-empty, >= 40 items with both classes), same
             seed-0 sample of <= 250 per subset, same 8-chunk cap
  RAGTruth   the `R7-H60.load_english` eval slice verbatim (600 items, seed 0)

Per item the response is segmented by the shipped extractor; each extracted
claim is grounded against the item's documents; the claim score is the shipped
`lex_p` (`m.verdict_probability`, falling back to `m.agreement_score` when the
manifold did not run - the joint.py formula). Item score = MIN over claim
lex_p (the decomposed-min analog); an item yielding ZERO extracted claims
scores 1.0 (supported pass - a gate that never sees a claim cannot flag it;
the R13 convention). No fitting, no threshold search: AUROC is threshold-free
and the 0.50 cut is pre-registered.

Discipline: CPU only (torch-free stack, no CUDA vars set), Polars for data,
eval-only under the contamination wall, per-subset checkpoints so an
interruption loses nothing. Multiprocessing over items is order-stable
(executor.map preserves input order; every per-item score is deterministic).
One fork pool serves the whole run; the parent keeps its own memory free of
the MT/SaT thread pools (sanity A runs in a subprocess) so forked workers
never inherit a lock owned by a dead thread.

Run:  uv run python experiments/grounding-lexical/arena_crossover.py
"""

import os

# Offline determinism: never let a missing argos pair reach the network; a
# confidently non-English claim with no installed bridge is blocked instead
# (counted per subset; ~never fires on these English corpora).
os.environ.setdefault("GROUNDRAILS_ARGOS_AUTO_INSTALL", "0")

import concurrent.futures as cf
import hashlib
import io
import json
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
ARCHIVE = ROOT / "data" / "external" / "datasets" / "dataset-ragbench.zip"
RAGTRUTH_ZIP = ROOT / "data" / "external" / "datasets" / "dataset-ragtruth.zip"
SEM = ROOT / "experiments" / "grounding-semantic"
CONFIG_YAML = ROOT / "src" / "groundrails" / "config_document_processing.yaml"
CKPT_DIR = HERE / "arena_crossover_ckpt"
OUT_ARENA = HERE / "arena_crossover_ragbench.json"
OUT_RT = HERE / "arena_crossover_ragtruth.json"

MAX_CHUNKS = 8
N_PER_SUBSET = 250
N_RT = 600
CUT = 0.50  # the pre-registered global cut shipped in config (R12)
WORKERS = 24

# Per-process shipped-pipeline state (initializer / parent both boot here).
_CFG = None


def _boot():
    """Mark the runtime ready and load the shipped document-processing config.

    `load_config_file` on the repo-root groundrails.json is what the CLI does
    at startup; the config loader resolves to the bundled yaml (verified: no
    project-local or user-level override exists)."""
    global _CFG
    if _CFG is not None:
        return
    from groundrails import settings
    from groundrails.config import load_document_processing_config

    settings.load_config_file(str(ROOT / "groundrails.json"))
    _CFG = load_document_processing_config()


def _worker_boot():
    _boot()


# --- sanity bar A: prove the wired path IS the golden-tested shipped path -----
#
# Runs in a SUBPROCESS, deliberately: the golden fixture's cross-lingual items
# pull the MT stack (ctranslate2 / SaT thread pools) into the process, and a
# parent that has ever held those threads cannot safely fork pool workers (the
# children inherit locks owned by dead threads and deadlock). Keeping the
# parent's own memory thread-clean is what makes the fork pool below safe.

_SANITY_SNIPPET = r"""
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
from groundrails import settings
settings.load_config_file(str(root / "groundrails.json"))
spec = importlib.util.spec_from_file_location(
    "golden_test", root / "tests" / "test_equivalence_golden.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
expected = json.loads(mod.GOLDEN.read_text(encoding="utf-8"))
actual = mod._snapshot()
max_abs, n_float, mism = 0.0, 0, []
for k, rec in actual.items():
    for f, v in rec.items():
        e = expected[k][f]
        if isinstance(v, float) and isinstance(e, float):
            n_float += 1
            max_abs = max(max_abs, abs(v - e))
        if v != e:
            mism.append((k, f, e, v))
print("RESULT_JSON:" + json.dumps({
    "items": len(actual), "float_fields_compared": n_float,
    "max_abs_diff": max_abs, "mismatches": mism[:5], "passed": not mism,
}))
"""


def sanity_golden():
    """Re-run the golden fixture through the shipped call path in a clean
    subprocess and diff against tests/data/grounding_golden.json. Uses the test
    module's own FIXTURE / FIELDS / _snapshot so the proof is the test's."""
    proc = subprocess.run(
        [sys.executable, "-c", _SANITY_SNIPPET, str(ROOT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    if proc.returncode != 0:
        raise SystemExit(f"SANITY A subprocess failed:\n{proc.stderr[-2000:]}")
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT_JSON:"))
    out = json.loads(line[len("RESULT_JSON:"):])
    out["method"] = (
        "tests/test_equivalence_golden.py _snapshot() vs tests/data/grounding_golden.json "
        "(subprocess)"
    )
    return out


# --- data: frozen arena gate protocol, byte-identical item sets ---------------


def load_subsets():
    """Verbatim copy of R8-H77_unseen_arena.load_subsets() minus torch."""
    z = zipfile.ZipFile(ARCHIVE)
    out = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
        out[sub] = (
            df["response"].to_list(),
            [d[:MAX_CHUNKS] for d in df["documents"].to_list()],
            df["adherence_score"].cast(pl.Int8).to_numpy(),
        )
    return out


def load_ragtruth_en():
    """Verbatim copy of R7-H60_multilingual_parallel.load_english() minus torch."""
    z = zipfile.ZipFile(RAGTRUTH_ZIP)
    n = next(x for x in z.namelist() if x.endswith("__test.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n)))
    df = df.with_columns(
        (
            (pl.col("hallucination_labels_processed").struct.field("evident_conflict") == 0)
            & (pl.col("hallucination_labels_processed").struct.field("baseless_info") == 0)
        )
        .cast(pl.Int8)
        .alias("label")
    )
    df = df.filter(
        (pl.col("context").str.len_chars() > 50) & (pl.col("output").str.len_chars() > 20)
    )
    df = df.sample(min(N_RT, len(df)), seed=0)
    return df["output"].to_list(), df["context"].to_list(), df["label"].to_numpy()


# --- per-item shipped-pipeline scoring ----------------------------------------


def score_item(arg):
    """One item through the shipped pipeline: extract_claims -> ground per
    claim -> lex_p; item score = min over claim lex_p (1.0 when no claim is
    scored). Returns the per-item record; never raises."""
    from groundrails.extract import extract_claims
    from groundrails.grounding import UnsupportedLanguageError, ground

    idx, response, docs = arg
    rec = {
        "idx": idx,
        "n_claims": 0,
        "n_blocked": 0,
        "n_errors": 0,
        "first_error": None,
        "score": 1.0,
        "mean_score": 1.0,
    }
    try:
        claims = extract_claims(response)
    except Exception as exc:  # extraction itself failed: unsupported pass + defect
        rec["n_errors"] = 1
        rec["first_error"] = f"extract: {type(exc).__name__}: {exc}"[:300]
        return rec
    rec["n_claims"] = len(claims)
    ps = []
    for c in claims:
        try:
            m = ground(c.claim, docs, config=_CFG)
            ps.append(m.verdict_probability if m.verdict_probability >= 0 else m.agreement_score)
        except UnsupportedLanguageError:
            rec["n_blocked"] += 1  # the tier refuses this claim - it cannot flag it
        except Exception as exc:  # noqa: BLE001 - keep the run alive, count the defect
            rec["n_errors"] += 1
            if rec["first_error"] is None:
                rec["first_error"] = f"ground: {type(exc).__name__}: {exc}"[:300]
    if ps:
        rec["score"] = float(min(ps))
        rec["mean_score"] = float(sum(ps) / len(ps))
    return rec


# --- metrics (threshold-free AUROC + frozen-cut rates) -------------------------


def metrics(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= CUT).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auroc": round(float(roc_auc_score(y, s)), 4),
        "f1_macro_at_050": round(float(f1_score(y, pred, average="macro")), 4),
        "tpr_at_050": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "tnr_at_050": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "confusion_at_050": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


# --- subset driver with checkpointing -----------------------------------------


def run_subset(name, items, pool, force=False):
    """Score one subset's items (order-stable), checkpoint, return the block."""
    ckpt = CKPT_DIR / f"{name}.json"
    if ckpt.exists() and not force:
        return json.loads(ckpt.read_text(encoding="utf-8"))
    t0 = time.time()
    claims, docs_lists, y = items
    args = [(i, c, d) for i, (c, d) in enumerate(zip(claims, docs_lists, strict=True))]
    try:
        recs = list(pool.map(score_item, args, chunksize=1))
        recs.sort(key=lambda r: r["idx"])
        scores = [r["score"] for r in recs]
        mean_scores = [r["mean_score"] for r in recs]
        n_scored_items = sum(1 for r in recs if r["n_claims"] - r["n_blocked"] - r["n_errors"] > 0)
        m_min = metrics(y, scores)
        block = {
            "subset": name,
            "n_items": len(y),
            "pos_rate": round(float(np.mean(y)), 4),
            "claims_per_item_mean": round(float(np.mean([r["n_claims"] for r in recs])), 3),
            "claims_per_item_max": int(max(r["n_claims"] for r in recs)),
            "zero_claim_rate": round(1.0 - n_scored_items / len(recs), 4),
            "blocked_claims": int(sum(r["n_blocked"] for r in recs)),
            "error_claims": int(sum(r["n_errors"] for r in recs)),
            "first_error": next((r["first_error"] for r in recs if r["first_error"]), None),
            **{f"min_{k}": v for k, v in m_min.items()},
            "mean_claim_diagnostic_auroc": metrics(y, mean_scores)["auroc"],
            "runtime_s": round(time.time() - t0, 1),
            "items": [
                {
                    "idx": r["idx"],
                    "label": int(y[r["idx"]]),
                    "score": round(r["score"], 6),
                    "mean_score": round(r["mean_score"], 6),
                    "n_claims": r["n_claims"],
                    "n_blocked": r["n_blocked"],
                    "n_errors": r["n_errors"],
                }
                for r in recs
            ],
        }
    except Exception as exc:  # noqa: BLE001 - one subset must not kill the run
        block = {
            "subset": name,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": round(time.time() - t0, 1),
        }
    ckpt.write_text(json.dumps(block, indent=2), encoding="utf-8")
    return block


def _lean(block):
    """The per-subset summary without the per-item rows."""
    return {k: v for k, v in block.items() if k != "items"}


# --- semantic references for the two-tier table --------------------------------


def semantic_refs():
    def per_subset(fname):
        d = json.loads((SEM / fname).read_text(encoding="utf-8"))
        return {s: r["auc"] for s, r in d["per_subset"].items()}, d.get("mean")

    twin_d1, m_t1 = per_subset("R16-H142_G1_twin_windowed_result.json")
    twin_d2, m_t2 = per_subset("R16-H142_T_draw2_windowed_result.json")
    h150_d1, m_h = per_subset("R18-H150_arm_draw1_windowed_result.json")
    lettuce = json.loads((SEM / "R18-H150_arm_draw1_windowed_result.json").read_text())[
        "per_subset"
    ]
    lettuce = {s: r["lettuce_auc"] for s, r in lettuce.items()}
    return {
        "lettucedetect_v2": (lettuce, 0.6461),
        "twin_d1": (twin_d1, m_t1),
        "twin_d2": (twin_d2, m_t2),
        "h150_d1": (h150_d1, m_h),
    }


def comparison_table(lex):
    refs = semantic_refs()
    subs = sorted(lex)
    header = "| subset | lexical (R14) | lettucedetect-v2 | twin d1 | twin d2 | H150 d1 |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for s in subs:
        row = [s, f"{lex[s]:.4f}"]
        for key in ("lettucedetect_v2", "twin_d1", "twin_d2", "h150_d1"):
            row.append(f"{refs[key][0][s]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    means = [f"{float(np.mean([lex[s] for s in subs])):.4f}"]
    for key in ("lettucedetect_v2", "twin_d1", "twin_d2", "h150_d1"):
        means.append(f"{refs[key][1]:.4f}" if refs[key][1] is not None else "-")
    lines.append("| **mean** | **" + "** | **".join(means) + "** |")
    return "\n".join(lines)


# --- main ----------------------------------------------------------------------


def main():
    t_all = time.time()
    _boot()
    CKPT_DIR.mkdir(exist_ok=True)

    print("[sanity A] golden-fixture equivalence through the shipped call path", flush=True)
    sa = sanity_golden()
    print(
        f"  items={sa['items']} float_fields={sa['float_fields_compared']} "
        f"max_abs_diff={sa['max_abs_diff']} passed={sa['passed']}",
        flush=True,
    )
    if not sa["passed"]:
        raise SystemExit(f"SANITY A FAILED - wiring is not the shipped path: {sa['mismatches']}")

    subs = load_subsets()
    print(
        f"[arena] {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} items",
        flush=True,
    )
    # One fork pool for the whole run, created from a thread-clean parent
    # (sanity A ran in a subprocess; nothing here has loaded the MT/SaT stack).
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=ctx, initializer=_worker_boot
    ) as pool:
        arena = {}
        for name, items in subs.items():
            block = run_subset(f"ragbench__{name}", items, pool)
            arena[name] = block
            if "error" in block:
                print(f"  {name:12s} ERROR {block['error']}", flush=True)
            else:
                print(
                    f"  {name:12s} n={block['n_items']:4d} pos={block['pos_rate']:.3f} "
                    f"claims/item={block['claims_per_item_mean']:.2f} "
                    f"zero={block['zero_claim_rate']:.3f} auc={block['min_auroc']:.4f} "
                    f"f1@0.5={block['min_f1_macro_at_050']:.4f} ({block['runtime_s']:.0f}s)",
                    flush=True,
                )

        ok = {s: b for s, b in arena.items() if "error" not in b}
        mean_auroc = float(np.mean([b["min_auroc"] for b in ok.values()]))
        print(
            f"[arena] 10-subset equal-weight mean AUROC (min read): {mean_auroc:.4f}",
            flush=True,
        )

        print("[ragtruth_en] loading the R7-H60 English eval slice", flush=True)
        cl, ctxs, y_rt = load_ragtruth_en()
        rt_block = run_subset("ragtruth_en", (cl, [[c] for c in ctxs], y_rt), pool)
        if "error" not in rt_block:
            print(
                f"  ragtruth_en n={rt_block['n_items']} pos={rt_block['pos_rate']:.3f} "
                f"auc={rt_block['min_auroc']:.4f} f1@0.5={rt_block['min_f1_macro_at_050']:.4f}",
                flush=True,
            )

    config_sha = hashlib.sha256(CONFIG_YAML.read_bytes()).hexdigest()
    provenance = {
        "config_path": str(CONFIG_YAML),
        "config_sha256": config_sha,
        "pipeline_entry": (
            "groundrails.extract.extract_claims(response) -> per claim "
            "groundrails.grounding.ground(claim, documents, config=cfg) -> "
            "lex_p = m.verdict_probability if m.verdict_probability >= 0 else m.agreement_score"
        ),
        "extractor": "regex extract_claims (English verb/copula gate) - the shipped CLI extractor",
        "frozen_cut": CUT,
        "item_score": "min over claim lex_p; zero-claimed item scores 1.0 (R13 convention)",
        "blocked_claim_policy": "UnsupportedLanguageError claims are skipped; an item with no scored claim scores 1.0",
        "env": {"GROUNDRAILS_ARGOS_AUTO_INSTALL": "0", "cuda_vars_set": False},
        "sanity_A": sa,
        "sanity_B": {
            "counts_match": all(
                b["n_items"] == e
                for b, e in zip(
                    (arena[s] for s in sorted(arena)),
                    (245, 184, 132, 203, 250, 250, 250, 250, 250, 250),
                    strict=True,
                )
                if "error" not in b
            ),
            "note": "expected census covidqa 245, delucionqa 184, emanual 132, expertqa 203, "
            "finqa/hagrid/hotpotqa/pubmedqa/tatqa/techqa 250 (banked arena read)",
        },
    }

    lex_aucs = {s: b["min_auroc"] for s, b in ok.items()}
    table = comparison_table(lex_aucs)
    print("\n" + table, flush=True)

    OUT_ARENA.write_text(
        json.dumps(
            {
                "experiment": "R14 lexical arena crossover - RAGBench blind read",
                "mean_auroc_min_read": round(mean_auroc, 5),
                "mean_auroc_mean_claim_diagnostic": round(
                    float(np.mean([b["mean_claim_diagnostic_auroc"] for b in ok.values()])), 5
                ),
                "n_subsets": len(ok),
                "n_items": sum(b["n_items"] for b in ok.values()),
                "per_subset": {s: _lean(b) for s, b in sorted(arena.items())},
                "comparison_vs_semantic": {
                    "lettucedetect_v2_mean": 0.6461,
                    "twin_d1_mean": semantic_refs()["twin_d1"][1],
                    "twin_d2_mean": semantic_refs()["twin_d2"][1],
                    "h150_d1_mean": semantic_refs()["h150_d1"][1],
                },
                "comparison_table_markdown": table,
                "provenance": provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    OUT_RT.write_text(
        json.dumps(
            {
                "experiment": "R14 lexical arena crossover - RAGTruth EN eval slice",
                "caveat": (
                    "asymmetry: the semantic track TRAINS on ragtruth_en, so this slice is a "
                    "clean read for the LEXICAL tier only; the lexical manifold's fit data is "
                    "gold v2 + VitaminC + synthetic negatives, never RAGTruth"
                ),
                **_lean(rt_block),
                "provenance": provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nresults -> {OUT_ARENA}\n        -> {OUT_RT}", flush=True)
    print(f"[done] total runtime {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("=== R14 LEXICAL ARENA CROSSOVER DONE ===", flush=True)


if __name__ == "__main__":
    main()
