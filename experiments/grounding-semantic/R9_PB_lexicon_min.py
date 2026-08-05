"""R9_PB - precursor P-B: the lexicon-excluded min on the frozen H90 dump.

Registered in round 9 as a pre-condition, not a hypothesis. Tests the one
load-bearing claim the sent-abstain-class refutation left standing: does
dropping discourse-register sentences from the MIN aggregation lift pubmedqa?
Threshold: pubmedqa delta >= +0.03 earns the abstain-class head a training run.

CAVEAT (read before using any number here): the dump carries per-sentence
TRUNCATED-read scores on the H90 checkpoint. H90 is disqualified as a
deliverable under the clean-mix protocol but remains valid for a MECHANISM
test, and the truncated-vs-windowed distinction is immaterial on pubmedqa,
where windowing is an exact no-op (+0.0000 on every recorded draw).

LEXICON PROVENANCE (important): no discourse lexicon exists in the repo. The
failure analysis counted inference markers by manual case reading; nothing was
committed. The lexicon below is RECONSTRUCTED from the prose of
reports/R8_architecture_failure_analysis.md, which names four categories
(inference, hedge, absence, calculation-step) and gives one verbatim exemplar
for each. Terms are tagged by provenance: "report" = appears verbatim in that
document, "category" = the minimal standard marker for a category it names.
Any P-B verdict is therefore lexicon-sensitive - which is what the ORACLE read
below exists to settle.

ORACLE BOUND: alongside the lexicon read, the script computes the AUC when the
single lowest-scoring sentence of every response is dropped from the min. No
sentence-exclusion rule of any kind - lexicon, learned head, or otherwise - can
beat that read, because it drops exactly the sentence the min is reading. If
the oracle fails the +0.03 bar, the exclusion mechanism is refuted for this
subset independently of which lexicon is chosen.

Run:  uv run python experiments/grounding-semantic/R9_PB_lexicon_min.py
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R8_analysis_h90_dump.json"
OUT = HERE / "R9_PB_result.json"

# (term, provenance) - matched case-insensitively as a substring of the sentence
LEXICON = [
    # inference / conclusion - report exemplar: "This suggests..."
    ("this suggests", "report"),
    ("suggests that", "report"),
    ("data suggests", "report"),
    ("these findings", "category"),
    ("given these findings", "category"),
    ("based on the provided information", "category"),
    ("based on the information provided", "category"),
    ("in conclusion", "category"),
    ("overall,", "category"),
    ("therefore,", "category"),
    ("thus,", "category"),
    ("indicates that", "category"),
    ("it appears that", "category"),
    # hedge - report exemplar: "However, more research is needed"
    ("more research is needed", "report"),
    ("further research", "category"),
    ("further evaluation", "category"),
    ("further study", "category"),
    ("further studies", "category"),
    ("however,", "report"),
    ("may still be", "category"),
    ("it is possible that", "category"),
    ("cannot be ruled out", "category"),
    # absence / refusal - report exemplar: "There is no information provided about..."
    ("no information provided", "report"),
    ("is no information", "report"),
    ("not mentioned", "category"),
    ("does not mention", "category"),
    ("not provided in the context", "category"),
    ("not specified", "category"),
    ("no mention of", "category"),
    ("cannot be determined", "category"),
    # calculation narration - report exemplar:
    # "Subtract the beginning balance from the ending balance"
    ("subtract the", "report"),
    ("divide the", "category"),
    ("multiply the", "category"),
    ("to calculate", "category"),
    ("the calculation is", "category"),
    ("we need to", "category"),
]

TERMS = [t for t, _ in LEXICON]
PUBMEDQA_BAR = 0.03


def is_marker(sentence):
    s = sentence.lower()
    return any(t in s for t in TERMS)


def main():
    d = json.loads(DUMP.read_text())
    recs = d["records"]
    subsets = sorted({r["subset"] for r in recs})

    per_subset = {}
    n_sent_total = n_sent_marked = 0
    n_resp_fallback = n_resp_total = 0

    for sub in subsets:
        rs = [r for r in recs if r["subset"] == sub]
        y = np.array([r["label"] for r in rs])
        s_incl, s_excl, s_oracle = [], [], []
        sub_fallback = 0

        for r in rs:
            scores = r["sent_scores"]
            sents = r["sentences"]
            s_incl.append(min(scores))

            keep = [sc for sc, se in zip(scores, sents, strict=True) if not is_marker(se)]
            n_sent_total += len(scores)
            n_sent_marked += len(scores) - len(keep)
            if keep:
                s_excl.append(min(keep))
            else:
                s_excl.append(min(scores))  # all sentences matched -> plain min
                sub_fallback += 1

            # oracle: drop the single lowest sentence (the one the min reads)
            if len(scores) > 1:
                s_oracle.append(min(sorted(scores)[1:]))
            else:
                s_oracle.append(min(scores))

        n_resp_fallback += sub_fallback
        n_resp_total += len(rs)

        auc_i = float(roc_auc_score(y, np.array(s_incl)))
        auc_e = float(roc_auc_score(y, np.array(s_excl)))
        auc_o = float(roc_auc_score(y, np.array(s_oracle)))
        per_subset[sub] = {
            "n": len(rs),
            "auc_included": round(auc_i, 4),
            "auc_excluded": round(auc_e, 4),
            "delta": round(auc_e - auc_i, 4),
            "auc_oracle_drop_argmin": round(auc_o, 4),
            "oracle_delta": round(auc_o - auc_i, 4),
            "fallback_rate": round(sub_fallback / len(rs), 3),
        }

    mean_i = float(np.mean([v["auc_included"] for v in per_subset.values()]))
    mean_e = float(np.mean([v["auc_excluded"] for v in per_subset.values()]))
    mean_o = float(np.mean([v["auc_oracle_drop_argmin"] for v in per_subset.values()]))

    res = {
        "note": "P-B precursor; H90 truncated per-sentence scores; lexicon reconstructed from the failure-analysis prose",
        "lexicon_terms": len(TERMS),
        "lexicon_provenance": {
            "report": sum(1 for _, p in LEXICON if p == "report"),
            "category": sum(1 for _, p in LEXICON if p == "category"),
        },
        "per_subset": per_subset,
        "mean_included": round(mean_i, 4),
        "mean_excluded": round(mean_e, 4),
        "mean_delta": round(mean_e - mean_i, 4),
        "mean_oracle_drop_argmin": round(mean_o, 4),
        "mean_oracle_delta": round(mean_o - mean_i, 4),
        "fallback_rate": round(n_resp_fallback / n_resp_total, 4),
        "lexicon_match_rate_sentences": round(n_sent_marked / n_sent_total, 4),
        "pubmedqa_delta": per_subset["pubmedqa"]["delta"],
        "pubmedqa_oracle_delta": per_subset["pubmedqa"]["oracle_delta"],
        "pubmedqa_bar": PUBMEDQA_BAR,
    }

    w = 96
    print("=" * w)
    print("R9 P-B - lexicon-excluded min vs plain min (frozen H90 dump, adjudication is external)")
    print("=" * w)
    print(f"  lexicon {len(TERMS)} terms  |  sentence match rate {res['lexicon_match_rate_sentences']:.3f}"
          f"  |  all-matched fallback rate {res['fallback_rate']:.3f}")
    print()
    print(f"  {'subset':<14}{'included':>10}{'excluded':>10}{'delta':>10}{'oracle':>10}{'orc-delta':>11}")
    for sub, v in per_subset.items():
        print(f"  {sub:<14}{v['auc_included']:>10.4f}{v['auc_excluded']:>10.4f}"
              f"{v['delta']:>+10.4f}{v['auc_oracle_drop_argmin']:>10.4f}{v['oracle_delta']:>+11.4f}")
    print(f"  {'MEAN':<14}{mean_i:>10.4f}{mean_e:>10.4f}{mean_e - mean_i:>+10.4f}"
          f"{mean_o:>10.4f}{mean_o - mean_i:>+11.4f}")
    print()
    print(f"  pubmedqa lexicon delta {res['pubmedqa_delta']:+.4f}   bar +{PUBMEDQA_BAR:.2f}")
    print(f"  pubmedqa ORACLE delta  {res['pubmedqa_oracle_delta']:+.4f}"
          f"   (upper bound on ANY sentence-exclusion rule)")
    print()
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  results -> {OUT}")


if __name__ == "__main__":
    main()
