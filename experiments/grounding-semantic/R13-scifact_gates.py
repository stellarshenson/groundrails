"""SciFact pre-build gates (R13 ruling 2) - run regardless of promotion.

Ruling 2: SciFact is admissible CONDITIONAL on its provenance gate - drop any
abstract matching a ragbench pubmedqa, covidqa or expertqa document at 8-gram
Jaccard >= 0.3; KILL the corpus at > 2% of abstracts matching. Executed with
the canonical ruling-4 instrument in its Jaccard variant.

Also reported (free, same pass): the buildable near-miss pair counts implied by
SciFact's SUPPORT / CONTRADICT / NEI structure, and the label-schema mapping
into our (claim, evidence, label) row format.

Run: uv run python experiments/grounding-semantic/R13-scifact_gates.py
"""

import collections
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets" / "scifact" / "data"
OUT = HERE / "R13-scifact_gates_result.json"

N_GRAM = 8
JACCARD = 0.3
KILL = 0.02
SUBSETS = ["pubmedqa", "covidqa", "expertqa"]

_spec = importlib.util.spec_from_file_location("pg", HERE / "provenance_gate.py")
PG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PG)


def jl(name):
    return [json.loads(x) for x in (DATA / f"{name}.jsonl").open()]


def near_miss_counts(claims):
    """SUPPORT / CONTRADICT / NEI structure -> buildable (claim, abstract) rows."""
    support = contradict = nei_rows = 0
    support_rationales = contradict_rationales = 0
    multi_sent_rationales = 0
    labelled = 0
    per_claim = collections.Counter()
    for c in claims:
        ev = c.get("evidence") or {}
        cited = {str(d) for d in c.get("cited_doc_ids", [])}
        if ev:
            labelled += 1
        for rats in ev.values():
            lab = rats[0]["label"]
            per_claim[lab] += 1
            if lab == "SUPPORT":
                support += 1
                support_rationales += len(rats)
            else:
                contradict += 1
                contradict_rationales += len(rats)
            multi_sent_rationales += sum(1 for r in rats if len(r["sentences"]) >= 2)
        nei_rows += len(cited - set(ev.keys()))
    return {
        "claims": len(claims),
        "claims_with_evidence": labelled,
        "claim_abstract_SUPPORT": support,
        "claim_abstract_CONTRADICT": contradict,
        "claim_abstract_NEI": nei_rows,
        "support_rationales": support_rationales,
        "contradict_rationales": contradict_rationales,
        "multi_sentence_rationales": multi_sent_rationales,
        "total_labelled_claim_abstract_rows": support + contradict + nei_rows,
    }


def claim_negation_pairs(claims):
    """SciFact ships original / negated claim twins - the tightest near-miss."""
    by_doc = collections.defaultdict(list)
    for c in claims:
        for d in c.get("cited_doc_ids", []):
            by_doc[d].append(c)
    twins = 0
    for cs in by_doc.values():
        labs = set()
        for c in cs:
            for rats in (c.get("evidence") or {}).values():
                labs.add(rats[0]["label"])
        if {"SUPPORT", "CONTRADICT"} <= labs:
            twins += 1
    return twins


def main():
    corpus = jl("corpus")
    train, dev, test = jl("claims_train"), jl("claims_dev"), jl("claims_test")
    labelled = train + dev

    abstracts = [c["title"] + " " + " ".join(c["abstract"]) for c in corpus]
    arena, _ = PG.load_arena(SUBSETS)
    prov = PG.run_gate(
        abstracts,
        n=N_GRAM,
        arena_texts=arena,
        jaccard=JACCARD,
        kill=KILL,
        label="scifact_abstracts",
    )
    control = PG.spike_control(abstracts, arena, n=N_GRAM, jaccard=JACCARD)
    # ruling 2 adjudicates on the candidate side (fraction of abstracts matching)
    frac = prov["candidate_vs_arena"]["fraction"]
    dropped = prov["candidate_vs_arena"]["units_with_hit"]

    gates = {
        "provenance": {
            "metric": f"abstracts matching a {'/'.join(SUBSETS)} document at {N_GRAM}-gram Jaccard >= {JACCARD}",
            "value": frac,
            "abstracts_dropped": dropped,
            "bar": f"<= {KILL}",
            "verdict": "KILL" if frac > KILL else "PASS",
        },
        "license": {
            "metric": "allenai/scifact LICENSE.md",
            "value": {
                "claims_and_evidence": "CC BY 4.0",
                "abstracts_corpus": "ODC-By 1.0 (S2ORC)",
                "code": "Apache 2.0",
            },
            "bar": "permissive",
            "verdict": "PASS",
            "note": (
                "the HF mirror allenai/scifact carries a license:cc-by-nc-2.0 tag that "
                "contradicts the upstream LICENSE.md; the upstream repository terms are "
                "recorded as authoritative and the data was fetched from the AI2 S3 release"
            ),
        },
    }
    overall = "KILL" if any(g["verdict"] == "KILL" for g in gates.values()) else "PASS"

    res = {
        "hypothesis": "SCIFACT-ABSTRACT-NEARMISS - pre-build gates (R13 ruling 2)",
        "dataset": {
            "source": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "abstracts": len(corpus),
            "claims_train": len(train),
            "claims_dev": len(dev),
            "claims_test_blind": len(test),
            "claims_total": len(train) + len(dev) + len(test),
        },
        "provenance": prov,
        "provenance_positive_control": control,
        "near_miss_structure": near_miss_counts(labelled),
        "claim_twin_abstracts": claim_negation_pairs(labelled),
        "label_schema_mapping": {
            "SUPPORT rationale": "(claim, rationale sentences of the cited abstract, label=1)",
            "CONTRADICT rationale": "(claim, rationale sentences of the cited abstract, label=0) - refuted near-miss",
            "NEI (cited abstract with no evidence entry)": "(claim, full cited abstract, label=0) - topical near-miss",
            "claims_test": "blind - no evidence field, unusable for supervision",
            "note": (
                "evidence is a dict doc_id -> list of rationale objects "
                "{sentences: [int], label: SUPPORT|CONTRADICT}; the abstract itself is a "
                "list of sentences in corpus.jsonl, so rationale indices address it directly"
            ),
        },
        "gates": gates,
        "overall": overall,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"gates": gates, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
