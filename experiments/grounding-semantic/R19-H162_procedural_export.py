"""R19-H162 - error-record export for manual reading. ANALYSIS ONLY, CPU only.

Companion to `R19-H162_procedural_autopsy.py`. Joins the banked R19-H161 per-pair
logit dump onto the rebuilt sentence / window text and writes a human-readable
dossier of the error items at the in-sample macro-F1-optimal threshold, so the
procedural-register failure mechanisms can be read off real claims and real
evidence rather than inferred from aggregates.

Writes `R19-H162_procedural_errors.parquet` (machine) and
`R19-H162_procedural_errors.txt` (reading copy).

Run:  uv run python experiments/grounding-semantic/R19-H162_procedural_export.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import pathlib

import numpy as np
import polars as pl
from sklearn.metrics import f1_score

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R19-H161_pairs_h150d1.parquet"
OUT_PARQUET = HERE / "R19-H162_procedural_errors.parquet"
OUT_TXT = HERE / "R19-H162_procedural_errors.txt"

WIN_CHARS = 1500


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA


def op_threshold(y, s):
    cand = np.unique(s)
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 400))
    best, best_t = -1.0, float(np.median(s))
    for t in cand:
        f = f1_score(y, (s >= t).astype(int), average="macro", zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def texts(subs, subset):
    claims, chunk_lists, _y = subs[subset]
    sent, win = {}, {}
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wi = 0
        for k in ks:
            for w in ARM.windows(k):
                win[(i, wi)] = w
                wi += 1
        for si, s in enumerate(H92.sentences(c)):
            sent[(i, si)] = s
    return sent, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="emanual,techqa")
    ap.add_argument("--max-items", type=int, default=60)
    args = ap.parse_args()

    subs = ARENA.load_subsets()
    df = pl.read_parquet(DUMP)
    rows, lines = [], []

    for subset in args.subsets.split(","):
        d = df.filter(pl.col("subset") == subset)
        sent_txt, win_txt = texts(subs, subset)

        sent = (
            d.group_by(["item_id", "sent_idx"])
            .agg(
                pl.col("logit").max().alias("s"),
                pl.col("label").first(),
                pl.col("n_win_sent").first(),
                pl.col("logit").arg_max().alias("am"),
            )
            .sort(["item_id", "sent_idx"])
        )
        # argmax provenance straight from the dump's own flag
        am = (
            d.filter(pl.col("is_argmax"))
            .group_by(["item_id", "sent_idx"])
            .agg(
                pl.col("win_idx").first(),
                pl.col("doc_idx").first(),
                pl.col("tok_containment").first(),
                pl.col("num_containment").first(),
            )
        )
        # num_containment is null when the sentence carries no numeral at all;
        # -1.0 marks "no numerals to bind" and keeps it distinct from a real 0.0
        sent = sent.join(am, on=["item_id", "sent_idx"], how="left").drop("am")
        sent = sent.with_columns(pl.col("num_containment").fill_null(-1.0))

        item = (
            sent.group_by("item_id")
            .agg(pl.col("s").min().alias("score"), pl.col("label").first())
            .sort("item_id")
        )
        y = item["label"].to_numpy()
        sv = item["score"].to_numpy()
        thr = op_threshold(y, sv)
        pred = (sv >= thr).astype(int)
        err_ids = item["item_id"].to_numpy()[pred != y]

        lines.append(
            f"\n{'=' * 100}\n### {subset}  threshold {thr:.4f}  errors {len(err_ids)} of {len(y)}\n"
        )

        for iid in err_ids[: args.max_items]:
            srow = item.filter(pl.col("item_id") == iid)
            lab = int(srow["label"][0])
            sc = float(srow["score"][0])
            kind = (
                "FALSE_POSITIVE (unsupported scored high)"
                if lab == 0
                else "FALSE_NEGATIVE (supported scored low)"
            )
            ss = sent.filter(pl.col("item_id") == iid).sort("s")
            sink = ss.row(0, named=True)
            lines.append(f"\n--- {subset} item {iid}  {kind}  item_score {sc:.3f} (thr {thr:.3f})")
            lines.append(
                f"    sinking sentence [{sink['sent_idx']}] score {sink['s']:.3f} "
                f"nwin {sink['n_win_sent']} tok_cont {sink['tok_containment']:.3f} "
                f"num_cont {sink['num_containment']:.3f}"
            )
            lines.append("    RESPONSE SENTENCES (score, text):")
            for r in ss.sort("sent_idx").iter_rows(named=True):
                mark = " <== SINK" if r["sent_idx"] == sink["sent_idx"] else ""
                lines.append(
                    f"      [{r['sent_idx']}] {r['s']:+.3f}{mark}  "
                    f"{sent_txt[(iid, r['sent_idx'])][:400]}"
                )
            wtx = win_txt.get((iid, sink["win_idx"]), "")
            lines.append(
                f"    ARGMAX WINDOW for the sink (doc {sink['doc_idx']}, win {sink['win_idx']}):"
            )
            lines.append(f"      {wtx[:WIN_CHARS]}")

            rows.append(
                {
                    "subset": subset,
                    "item_id": int(iid),
                    "label": lab,
                    "item_score": sc,
                    "threshold": float(thr),
                    "error_kind": "FP" if lab == 0 else "FN",
                    "sink_sent_idx": int(sink["sent_idx"]),
                    "sink_score": float(sink["s"]),
                    "sink_n_win": int(sink["n_win_sent"]),
                    "sink_tok_containment": float(sink["tok_containment"]),
                    "sink_num_containment": float(sink["num_containment"]),
                    "sink_sent_text": sent_txt[(iid, sink["sent_idx"])],
                    "sink_argmax_window": wtx,
                }
            )

    pl.DataFrame(rows).write_parquet(OUT_PARQUET)
    OUT_TXT.write_text("\n".join(lines))
    print(f"wrote {OUT_PARQUET} ({len(rows)} error records) and {OUT_TXT}")


if __name__ == "__main__":
    main()
