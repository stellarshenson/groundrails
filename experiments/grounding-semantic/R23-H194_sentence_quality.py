"""R23-H194 SENTENCE-LEVEL SCORING QUALITY - MEASUREMENT ONLY, CPU, no bar.

P-D (R23-H193) closed the aggregation question at the window level and handed the
residual to per-pair SCORING QUALITY: the lenient oracle sits 0.1884 above the
strict one, but the best of nine fixed window aggregators recovers 3% of that, so
the information needed to compose support across windows is not in the per-window
scores. R9 handed its residual to the same place (Modes B/C) and R16-H140's
readout was aimed there. Three lines converge, and none of them ever measured it.

The campaign has only ever scored the arena at the RESPONSE level. This measures
the layer underneath: given a single response sentence and the evidence, how well
does the model separate sentences the annotators marked supported from those they
marked unsupported? That is the quantity every aggregation sits on top of.

MEASUREMENT ONLY. No bar, no gate, no promotion, no kill, and deliberately no
pre-registered threshold - because nothing may be selected on this. It is a
diagnostic of where the loss lives, and any lever it motivates needs its own
registration and a non-arena selection surface.

Inputs, both already on disk - nothing is re-scored, so no GPU:
    R23-H193_window_agg_dump.npz   per-(sentence, window) logits for banked
                                   flagship draw 1, with sentence and response
                                   owners; the shipped per-sentence score is the
                                   max over that sentence's windows
    RAGBench annotations           via R12_label_ceiling.load_rows(), which is
                                   R8-H77.load_subsets' filter and seed-0 sample
                                   with the annotation columns retained, so row
                                   order matches the dump exactly

Sentence labels come from `unsupported_response_sentence_keys`, the field R12
established is populated on all ten subsets (`sentence_support_information[]
.fully_supported` is NULL on 8 of 10 and is NOT usable). An H92 sentence is
labelled supported when every annotated sentence it maps to is absent from that
set. Sentences that map to no annotation are DROPPED and counted, never guessed.

Run (CPU, under a minute):
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
    uv run python experiments/grounding-semantic/R23-H194_sentence_quality.py
"""

import importlib.util
import json
import pathlib
import statistics as st

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R23-H193_window_agg_dump.npz"
OUT = HERE / "R23-H194_sentence_quality.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    H92 = _mod("h92", "R8-H92_decomposed_arena.py")
    R12 = _mod("r12", "R12_label_ceiling.py")
    dec = json.loads((HERE / "R22_per_subset_decomposition.json").read_text())["per_subset"]

    d = np.load(DUMP)
    rows_by_sub = R12.load_rows()
    subsets = sorted({k.split("__")[0] for k in d.files})

    per_subset, totals = {}, {"sentences": 0, "unmapped": 0, "supported": 0, "unsupported": 0}
    for sub in subsets:
        lg, sent, owner = d[f"{sub}__logit"], d[f"{sub}__sent"], d[f"{sub}__owner"]
        # shipped per-sentence score: max over that sentence's windows
        s_score = np.full(len(owner), -np.inf, dtype=np.float64)
        np.maximum.at(s_score, sent, lg.astype(np.float64))

        df = rows_by_sub[sub]
        y_sent, keep, unmapped = [], [], 0
        for ri, row in enumerate(df.iter_rows(named=True)):
            unsupported = set(row["unsupported_response_sentence_keys"] or [])
            key_text = {p[0]: p[1] for p in (row["response_sentences"] or []) if len(p) >= 2}
            ssi = row["sentence_support_information"] or []
            keys = [x["response_sentence_key"] for x in ssi] or list(key_text)
            ann_texts = [key_text.get(k, "") for k in keys]
            sents = H92.sentences(row["response"])
            hits = R12.map_sentences(sents, ann_texts)
            sids = np.flatnonzero(owner == ri)
            if len(sids) != len(sents):          # dump and mapping must agree
                raise SystemExit(f"H194 ABORT: {sub} row {ri} has {len(sids)} dumped "
                                 f"sentences but {len(sents)} re-split - order broken")
            for sid, h in zip(sids, hits, strict=True):
                if not h:
                    unmapped += 1
                    continue
                keep.append(int(sid))
                y_sent.append(int(all(keys[j] not in unsupported for j in h)))

        y_sent = np.asarray(y_sent)
        keep = np.asarray(keep, dtype=int)
        n_pos, n_neg = int(y_sent.sum()), int((1 - y_sent).sum())
        auc = (float(roc_auc_score(y_sent, s_score[keep]))
               if n_pos and n_neg else float("nan"))
        per_subset[sub] = {
            "sentences_scored": int(len(owner)),
            "sentences_labelled": int(len(keep)),
            "sentences_unmapped_dropped": int(unmapped),
            "supported": n_pos, "unsupported": n_neg,
            "sentence_auroc": round(auc, 5) if auc == auc else None,
            "response_auroc_k6": dec[sub]["auc_k6_mean"],
            "sentence_minus_response": (round(auc - dec[sub]["auc_k6_mean"], 5)
                                        if auc == auc else None),
        }
        totals["sentences"] += len(owner)
        totals["unmapped"] += unmapped
        totals["supported"] += n_pos
        totals["unsupported"] += n_neg

    ok = [s for s, v in per_subset.items() if v["sentence_auroc"] is not None]
    sent_mean = st.mean(per_subset[s]["sentence_auroc"] for s in ok)
    resp_mean = st.mean(per_subset[s]["response_auroc_k6"] for s in ok)

    res = {
        "arm": "R23-H194 sentence-level scoring quality",
        "licence": ("MEASUREMENT ONLY - no bar, no gate, no promotion, no kill, and no "
                    "pre-registered threshold, because nothing may be selected on it. Any lever "
                    "this motivates needs its own registration and a non-arena selection "
                    "surface."),
        "source": {"scores": "R23-H193_window_agg_dump.npz (banked flagship draw 1, max over "
                             "each sentence's windows - the shipped per-sentence score)",
                   "labels": "unsupported_response_sentence_keys, the field R12 established is "
                             "populated on all ten subsets"},
        "coverage": {**totals,
                     "labelled_share": round(1 - totals["unmapped"] / totals["sentences"], 4)},
        "uniform_mean": {"sentence_auroc": round(sent_mean, 5),
                         "response_auroc_k6": round(resp_mean, 5),
                         "difference": round(sent_mean - resp_mean, 5)},
        "per_subset": per_subset,
        "caveats": [
            "one checkpoint (banked flagship draw 1) - unreplicated across draws",
            "sentences mapping to no annotation are DROPPED, not guessed; the dropped count is "
            "reported per subset and is not small on every subset",
            "the sentence label is annotator support, which is a different and stricter target "
            "than the response label the arena scores",
        ],
    }
    OUT.write_text(json.dumps(res, indent=2))

    print(f"written -> {OUT}\n")
    print(f"{'subset':<12}{'sent AUROC':>12}{'resp AUROC':>12}{'diff':>9}"
          f"{'labelled':>10}{'dropped':>9}{'unsup':>8}")
    for s in sorted(per_subset, key=lambda x: per_subset[x]["sentence_auroc"] or 0):
        v = per_subset[s]
        print(f"{s:<12}{v['sentence_auroc']:>12.4f}{v['response_auroc_k6']:>12.4f}"
              f"{v['sentence_minus_response']:>+9.4f}{v['sentences_labelled']:>10}"
              f"{v['sentences_unmapped_dropped']:>9}{v['unsupported']:>8}")
    print(f"{'MEAN':<12}{sent_mean:>12.4f}{resp_mean:>12.4f}{sent_mean - resp_mean:>+9.4f}")
    print(f"\ncoverage: {totals['sentences'] - totals['unmapped']}/{totals['sentences']} "
          f"sentences carry a label ({res['coverage']['labelled_share']:.1%}); "
          f"{totals['supported']} supported / {totals['unsupported']} unsupported")


if __name__ == "__main__":
    main()
