"""R19 supply wave - per-corpus contamination gates, CPU only, no GPU.

Registered in docs/experiments/semantic-dataset-enhancements.md section "R19
supply wave".  Every corpus the fetcher (`scripts/fetch_grounding_datasets.py`)
landed is checked here before banking:

  1. LICENCE SIDECAR - the tracked `data/external/datasets/dataset-<name>.md`
     sidecar exists and records the licence tag re-read from the source at pull
     time (not the recon's say-so)
  2. RESTRICTION RE-VERIFICATION - each corpus's admission clause is re-derived
     from the data, never trusted from the fetcher's self-report: AttributionBench
     must contain ZERO ExpertQA/HAGRID rows (the carve-out), FinDVer reports must
     all be 2024 10-K/10-Q filings served from sec.gov (the EDGAR-population
     precedent), PubHealth's empty-evidence and stray-label rates are measured,
     MiniCheck's labels must be binary, FAVA's prompt structure must parse,
     FActScore's evidence coverage is measured
  3. R14-H136 PROVENANCE GATE - the registered instrument
     (`provenance_gate.py`, ruling 2 form: 8-gram, Jaccard >= 0.3, bidirectional)
     against ALL TEN walled arena corpora, KILL at > 2% max fraction, with the
     spike control first (arena units injected into the candidate side must all
     be detected).  The arena reference n-grams are the ONLY thing read from
     RAGBench - never item content

Gate units are the corpus's EVIDENCE texts (references / documents / articles /
context passages), deduplicated - contamination is a document-overlap property.

Any corpus over bar quarantines: the gate JSON says RED and the lane builder
refuses it.  A corpus that cannot be read at all is recorded as a defect and
the wave continues.

Run:  uv run python experiments/grounding-semantic/R19_supply_gates.py [name ...]
"""

import collections
import importlib.util
import io
import json
import pathlib
import re
import sys
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

GATE_N = 8
GATE_JACCARD = 0.3
GATE_KILL = 0.02

_spec = importlib.util.spec_from_file_location("provgate", HERE / "provenance_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

# MiniCheck seed provenance, extracted verbatim from Appendix D of ACL
# 2024.emnlp-main.499 (fetched 2026-08-13, /tmp probe copy): the seeds are NOT
# walled RAGBench source corpora.  The 8-gram instrument is the binding check.
MINICHECK_SEEDS = {
    "identified": True,
    "paper": "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents "
             "(EMNLP 2024 main, 2024.emnlp-main.499), Appendix D 'Synthetic Data Details'",
    "c2d_seeds": "~400 claims from Wikipedia that have cited web articles "
                 "(Kamoi et al., 2023; Petroni et al., 2023)",
    "d2c_seeds": "~300 Google News articles scraped since November 2023 across "
                 "science, politics, world, entertainment, business, technology",
    "walled_seed_present": False,
    "note": "neither seed population is one of the ten walled RAGBench source "
            "corpora; the 8-gram instrument runs regardless and is binding",
}


def zip_parquets(name):
    """split -> polars frame for every parquet inside dataset-<name>.zip."""
    z = zipfile.ZipFile(DATA / f"dataset-{name}.zip")
    out = {}
    for m in z.namelist():
        if m.endswith(".parquet"):
            split = m[:-len(".parquet")].split("__")[-1]
            out[split] = pl.read_parquet(io.BytesIO(z.read(m)))
    return out


# --------------------------------------------------------------------------- #
# per-corpus readers: (gate_texts, restriction_evidence)
# --------------------------------------------------------------------------- #
FAVA_PREFIX = "Read the following references:\n"
FAVA_SEP = ("\nPlease identify all the errors in the following passage using the "
            "references provided and suggest edits:\nText: ")
FAVA_TAG = re.compile(r"<(entity|relation|contradictory|subjective|unverifiable|invented)[>\s]")


def read_fava():
    parts = zip_parquets("fava")
    df = parts["train"]
    prompts = df["prompt"].to_list()
    blocks, parsed = [], 0
    for p in prompts:
        if p.startswith(FAVA_PREFIX) and FAVA_SEP in p:
            blocks.append(p[len(FAVA_PREFIX):p.index(FAVA_SEP)])
            parsed += 1
    ev = {"rows": df.height,
          "prompt_structure_parsed": parsed,
          "prompt_structure_rate": round(parsed / max(df.height, 1), 6),
          "completions_with_error_tags": int(sum(bool(FAVA_TAG.search(c))
                                                 for c in df["completion"].to_list())),
          "gold_rows_banked_not_gated": int(parts["gold"].height),
          "pass": parsed / max(df.height, 1) >= 0.95}
    return sorted(set(blocks)), ev


PUBHEALTH_LABELS = {"true", "false", "unproven", "mixture"}


def read_pubhealth():
    parts = zip_parquets("pubhealth")
    df = pl.concat(list(parts.values()), how="diagonal_relaxed")
    texts = df["main_text"].to_list()
    labels = collections.Counter(df["label"].to_list())
    empty = sum(1 for t in texts if not t or not t.strip())
    stray = sum(v for k, v in labels.items() if k not in PUBHEALTH_LABELS)
    ev = {"rows": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "empty_main_text_rows": empty,
          "empty_main_text_rate": round(empty / max(df.height, 1), 6),
          "label_distribution": dict(labels),
          "stray_label_rows": stray,
          "note": "empty-evidence and stray-label rows stay in the archive; the "
                  "lane excludes them and reports the same counts",
          "pass": True}  # rates are the re-verification; no hard bar registered
    return sorted({t for t in texts if t and t.strip()}), ev


def read_minicheck():
    parts = zip_parquets("minicheck")
    df = pl.concat(list(parts.values()))
    labels = sorted(set(df["label"].to_list()))
    ev = {"rows": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "label_values": labels,
          "label_distribution": {str(k): v for k, v in
                                 df.group_by("label").len().iter_rows()},
          "seed_provenance": MINICHECK_SEEDS,
          "pass": set(labels) <= {0, 1}}
    return sorted(set(df["doc"].to_list())), ev


def read_factscore():
    tree = DATA / "factscore"
    counts = json.loads((tree / "_counts.json").read_text())["counts"]
    index = json.loads((tree / "evidence" / "_index.json").read_text())
    texts = sorted({(tree / "evidence" / v["file"]).read_text() for v in index.values()})
    fact_labels = collections.Counter()
    n_rows = 0
    for model in ("InstructGPT", "ChatGPT", "PerplexityAI"):
        for ln in open(tree / "data" / "labeled" / f"{model}.jsonl"):
            r = json.loads(ln)
            n_rows += 1
            for ann in r.get("annotations") or []:
                for f in ann.get("human-atomic-facts") or []:
                    fact_labels[f.get("label", "?")] += 1
    ev = {"biographies": n_rows,
          "atomic_fact_labels": dict(fact_labels),
          "atomic_facts_total": sum(fact_labels.values()),
          "evidence_topics": len(index),
          "evidence_failed": counts.get("_evidence_failed", []),
          "evidence_pin": "last revision on or before 2023-04-01 via the Wikipedia "
                          "API (the corpus's enwiki-20230401 knowledge source)",
          "pass": len(index) >= 0.95 * counts.get("entities", 183)}
    return texts, ev


FINDVER_FILE = re.compile(r"^[A-Z0-9-]+-2024-\d{2}-\d{2}_10-[KQ]\.json$")


def read_findver():
    tree = DATA / "findver"
    claims = []
    for split in ("test", "testmini"):
        for r in json.loads((tree / "data" / f"{split}.json").read_text()):
            r["split"] = split
            claims.append(r)
    reports = sorted(set(c["report"] for c in claims))
    bad_names = [r for r in reports if not FINDVER_FILE.match(r)]
    bad_hosts, texts = [], set()
    subset_ct = collections.Counter(c["subset"] for c in claims)
    label_ct = collections.Counter(str(c["entailment_label"]) for c in claims)
    for rep in reports:
        doc = json.loads((tree / "financial_reports" / rep).read_text())
        if "sec.gov" not in doc.get("url", ""):
            bad_hosts.append(rep)
        ctx = {c["id"]: c["context"] for c in doc["context"]}
        for c in claims:
            if c["report"] == rep:
                for i in c["relevant_context"]:
                    if i in ctx:
                        texts.add(ctx[i])
    ev = {"claims": len(claims),
          "reports_referenced": len(reports),
          "reports_on_disk": len(list((tree / "financial_reports").glob("*.json"))),
          "subset_distribution": dict(subset_ct),
          "entailment_distribution": dict(label_ct),
          "non_2024_10KQ_report_names": bad_names,
          "non_sec_gov_report_urls": bad_hosts,
          "precedent": "EDGAR-population ruling: 2024 documents vs the walled "
                       "pre-2020 FinQA/TAT-QA populations make document overlap "
                       "structurally impossible; the instrument runs anyway",
          "pass": not bad_names and not bad_hosts
                  and set(subset_ct) <= {"ie", "numeric", "knowledge"}}
    return sorted(texts), ev


def read_attributionbench():
    parts = zip_parquets("attributionbench")
    df = pl.concat(list(parts.values()))
    src = collections.Counter(df["src_dataset"].to_list())
    walled_present = {k: v for k, v in src.items() if k in {"ExpertQA", "HAGRID"}}
    texts = set()
    for refs in df["references"].to_list():
        texts.update(r for r in refs if r)
    counts = json.loads(zipfile.ZipFile(DATA / "dataset-attributionbench.zip")
                        .read("_counts.json"))["counts"]
    ev = {"rows_kept": df.height,
          "splits": {k: v.height for k, v in parts.items()},
          "src_dataset_distribution": dict(src),
          "walled_rows_remaining": walled_present,
          "dropped_at_fetch": counts.get("_dropped_walled_by_split"),
          "label_distribution": dict(collections.Counter(df["attribution_label"].to_list())),
          "pass": not walled_present}
    return sorted(texts), ev


READERS = {
    "fava": read_fava,
    "pubhealth": read_pubhealth,
    "minicheck": read_minicheck,
    "factscore": read_factscore,
    "findver": read_findver,
    "attributionbench": read_attributionbench,
}

LICENCE_TOKEN = {  # the tag re-read from the source at pull time
    "fava": "CC-BY-4.0", "pubhealth": "MIT", "minicheck": "MIT",
    "factscore": "MIT", "findver": "MIT", "attributionbench": "Apache-2.0",
}


def check_sidecar(name):
    sc = DATA / f"dataset-{name}.md"
    ev = {"path": str(sc), "present": sc.exists()}
    ok = sc.exists()
    if ok:
        txt = sc.read_text()
        ev["records_licence_token"] = LICENCE_TOKEN[name] in txt
        ev["records_r19_registration"] = "R19 supply wave" in txt
        ok &= ev["records_licence_token"] and ev["records_r19_registration"]
    return ok, ev


def gate_one(name, arena_texts):
    print(f"\n=== {name}", flush=True)
    res = {"corpus": name,
           "instrument": "provenance_gate.py (R14-H136 ruling 2 form: 8-gram, "
                         "Jaccard >= 0.3, bidirectional, KILL > 2%)"}
    ok_side, ev_side = check_sidecar(name)
    res["licence_sidecar"] = {"pass": ok_side, **ev_side}

    texts, ev_restrict = READERS[name]()
    print(f"  gate units (deduped evidence texts): {len(texts)}", flush=True)
    res["restriction_reverification"] = {"pass": ev_restrict.pop("pass"), **ev_restrict}

    spike = G.spike_control(texts[:2000], arena_texts, n=GATE_N,
                            jaccard=GATE_JACCARD, k=10, label=f"{name}_spike")
    print(f"  spike control: {spike}", flush=True)
    gate = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label=f"r19_{name}", arena_texts=arena_texts)
    print(f"  gate verdict {gate['verdict']} at max fraction "
          f"{gate['max_fraction']} (best-Jaccard max "
          f"{gate['candidate_vs_arena'].get('best_jaccard', {}).get('max')})",
          flush=True)
    res["provenance_gate"] = {"pass": gate["verdict"] != "KILL" and spike["passes"],
                              "spike_control": spike, "result": gate}

    res["status"] = ("GREEN" if all(res[k]["pass"] for k in (
        "licence_sidecar", "restriction_reverification", "provenance_gate")) else "RED")
    out = HERE / f"R19_{name}_gate.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  === {name} GATE {res['status']} -> {out.name}", flush=True)
    return res["status"]


def main():
    names = sys.argv[1:] or list(READERS)
    arena_texts, _ = G.load_arena()  # all ten walled subsets; n-grams only
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)
    summary = {}
    for name in names:
        try:
            summary[name] = gate_one(name, arena_texts)
        except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
            summary[name] = f"DEFECT: {type(e).__name__}: {str(e)[:200]}"
            print(f"  === {name} GATE DEFECT: {summary[name]}", flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
