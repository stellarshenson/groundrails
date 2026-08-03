"""R8_analysis_failure_stats - CPU pass over the H90 dump.

Computes splitter stats, score-distribution shape, min-vs-mean aggregation AUCs,
lettuce-agreement (label-noise triangulation), and writes ranked worst-FP /
worst-FN case files for manual reading.

Run:  uv run python experiments/grounding-semantic/R8_analysis_failure_stats.py
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R8_analysis_h90_dump.json"
STATS_OUT = HERE / "R8_analysis_failure_stats.json"
CASES_DIR = HERE / "R8_analysis_cases"

FOCUS = ["finqa", "delucionqa", "pubmedqa", "hagrid"]
TOP = 12


def main():
    d = json.loads(DUMP.read_text())
    recs = d["records"]
    subsets = sorted({r["subset"] for r in recs})
    CASES_DIR.mkdir(exist_ok=True)

    stats = {"aucs_recorded": d["aucs"], "per_subset": {}}
    for sub in subsets:
        rs = [r for r in recs if r["subset"] == sub]
        y = np.array([r["label"] for r in rs])
        s_min = np.array([r["score"] for r in rs])
        s_mean = np.array([r["mean_score"] for r in rs])
        lett = np.array([r["lettuce_score"] for r in rs])
        n_sent = np.array([r["n_sent"] for r in rs])
        fallback = np.array([r["fallback_whole"] for r in rs])
        capped = np.array([r["truncated_at_cap"] for r in rs])
        dropped = np.array([r["n_short_dropped"] for r in rs])

        # a response whose min is far below its mean was sunk by one sentence
        sink = s_mean - s_min

        stats["per_subset"][sub] = {
            "n": len(rs),
            "grounded_rate": round(float(y.mean()), 3),
            "auc_min": round(float(roc_auc_score(y, s_min)), 4),
            "auc_mean": round(float(roc_auc_score(y, s_mean)), 4),
            "auc_lettuce": round(float(roc_auc_score(y, lett)), 4),
            "fallback_whole_rate": round(float(fallback.mean()), 3),
            "truncated_at_cap_rate": round(float(capped.mean()), 3),
            "mean_short_dropped": round(float(dropped.mean()), 2),
            "n_sent_median": int(np.median(n_sent)),
            "n_sent_p90": int(np.percentile(n_sent, 90)),
            "score_median_pos": round(float(np.median(s_min[y == 1])), 4),
            "score_median_neg": round(float(np.median(s_min[y == 0])), 4),
            "frac_scores_below_0.1": round(float((s_min < 0.1).mean()), 3),
            "frac_pos_below_0.1": round(float((s_min[y == 1] < 0.1).mean()), 3),
            "sink_median_pos": round(float(np.median(sink[y == 1])), 4),
            # grounded responses killed by ONE sentence: min low, mean high
            "pos_min_low_mean_high": int(((y == 1) & (s_min < 0.2) & (s_mean > 0.6)).sum()),
            # label-noise candidates: BOTH models confidently disagree with label
            "both_high_on_neg": int(((y == 0) & (s_min > 0.7) & (lett > 0.7)).sum()),
            "both_low_on_pos": int(((y == 1) & (s_min < 0.2) & (lett < 0.3)).sum()),
            "n_neg": int((y == 0).sum()),
            "n_pos": int((y == 1).sum()),
        }

        if sub in FOCUS:
            # FP: hallucinated (label 0) scored grounded -> highest scores
            # FN: grounded (label 1) scored hallucinated -> lowest scores
            neg = sorted([r for r in rs if r["label"] == 0], key=lambda r: -r["score"])
            pos = sorted([r for r in rs if r["label"] == 1], key=lambda r: r["score"])
            for kind, cases in (("FP", neg[:TOP]), ("FN", pos[:TOP])):
                lines = []
                for r in cases:
                    lines.append("=" * 100)
                    lines.append(
                        f"[{kind}] {sub} idx={r['idx']} label={r['label']} "
                        f"score(min)={r['score']} mean={r['mean_score']} "
                        f"lettuce={r['lettuce_score']} n_sent={r['n_sent']} "
                        f"n_chunks={r['n_chunks']} resp_chars={r['resp_chars']} "
                        f"fallback={r['fallback_whole']} capped={r['truncated_at_cap']}"
                    )
                    lines.append(f"--- ARGMIN sentence (score {r['score']}):")
                    lines.append(f"    {r['argmin_sentence']}")
                    lines.append("--- all sentences:")
                    for sc, st in zip(r["sent_scores"], r["sentences"], strict=True):
                        lines.append(f"    [{sc:.3f}] {st[:300]}")
                    lines.append("--- RESPONSE:")
                    lines.append(r["response"][:1500])
                    lines.append("--- CHUNKS (truncated):")
                    for j, k in enumerate(r["chunks"]):
                        lines.append(f"  <chunk {j}> {k[:900]}")
                    lines.append("")
                (CASES_DIR / f"{sub}_{kind}.txt").write_text("\n".join(lines))

    STATS_OUT.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["per_subset"], indent=2))
    print(f"-> {STATS_OUT}\n-> case files in {CASES_DIR}")


if __name__ == "__main__":
    main()
