"""R14 evidence E4 - item-level forensics on finqa and delucionqa.

ANALYSIS ONLY. Reads the frozen R12-H121 Gate A dump (per-(sentence, window)
scores from the R9-H105 draw-1 clean checkpoint) and reproduces the banked
windowed decomposed-min arena read, then decomposes the AUROC loss on finqa and
delucionqa into per-item discordance contributions.

No quantity produced here may enter a lane's size, thresholds or mix.

Run:  uv run python experiments/grounding-semantic/R14_E4_item_forensics.py
"""

import json
import pathlib

import numpy as np
import polars as pl
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).parent
SCORES = HERE / "R12-H121_gateA_scores.parquet"
OUT = HERE / "R14_E4_item_forensics.json"
DUMP = HERE / "R14_E4_worst_items.txt"


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    df = pl.read_parquet(SCORES)

    sent = df.group_by(["subset", "resp_idx", "sent_idx"]).agg(
        pl.col("score").max().alias("sent_score"),
        pl.col("score").arg_max().alias("arg_win"),
        pl.col("label").first(),
        pl.col("resp_label").first(),
        pl.len().alias("n_win"),
        pl.col("sent_text").first(),
    )
    resp = sent.group_by(["subset", "resp_idx"]).agg(
        pl.col("sent_score").min().alias("resp_score"),
        pl.col("resp_label").first(),
        pl.len().alias("n_sent"),
        (pl.col("label") == 0).sum().alias("n_lab0"),
        (pl.col("label") == 1).sum().alias("n_lab1"),
        (pl.col("label") == -1).sum().alias("n_unlab"),
    )

    out = {"reproduction": {}, "subsets": {}}
    aucs = []
    for sub in sorted(resp["subset"].unique().to_list()):
        g = resp.filter(pl.col("subset") == sub)
        a = round(auroc(g["resp_label"].to_numpy(), g["resp_score"].to_numpy()), 4)
        out["reproduction"][sub] = a
        aucs.append(a)
    out["reproduction"]["mean"] = round(float(np.mean(aucs)), 5)

    lines = []
    for sub in ["finqa", "delucionqa"]:
        g = resp.filter(pl.col("subset") == sub).sort("resp_score")
        y = g["resp_label"].to_numpy()
        s = g["resp_score"].to_numpy()
        idx = g["resp_idx"].to_numpy()
        pos = s[y == 1]  # supported (should score HIGH)
        neg = s[y == 0]  # unsupported (should score LOW)
        n1, n0 = len(pos), len(neg)
        a = auroc(y, s)
        disc = (1 - a) * n0 * n1

        # per-item discordance: supported item loses to how many unsupported
        sup_loss = {}
        for i in range(len(y)):
            if y[i] == 1:
                sup_loss[int(idx[i])] = float(
                    (neg > s[i]).sum() + 0.5 * (neg == s[i]).sum()
                )
        uns_win = {}
        for i in range(len(y)):
            if y[i] == 0:
                uns_win[int(idx[i])] = float(
                    (pos < s[i]).sum() + 0.5 * (pos == s[i]).sum()
                )
        tot_sup = sum(sup_loss.values())
        tot_uns = sum(uns_win.values())

        sup_sorted = sorted(sup_loss.items(), key=lambda kv: -kv[1])
        uns_sorted = sorted(uns_win.items(), key=lambda kv: -kv[1])

        out["subsets"][sub] = {
            "n_resp": len(g),
            "n_supported": n1,
            "n_unsupported": n0,
            "base_rate_supported": round(float(y.mean()), 4),
            "auroc": round(a, 4),
            "discordant_pairs": round(disc, 1),
            "total_pairs": n0 * n1,
            "discordance_from_supported_side": round(tot_sup, 1),
            "discordance_from_unsupported_side": round(tot_uns, 1),
            "top5_supported_share_of_discordance": round(
                sum(v for _, v in sup_sorted[:5]) / max(tot_sup, 1e-9), 4
            ),
            "top5_unsupported_share_of_discordance": round(
                sum(v for _, v in uns_sorted[:5]) / max(tot_uns, 1e-9), 4
            ),
            "n_supported_with_zero_loss": sum(1 for v in sup_loss.values() if v == 0),
            "n_unsupported_with_zero_win": sum(1 for v in uns_win.values() if v == 0),
            "score_mean_supported": round(float(pos.mean()), 4),
            "score_mean_unsupported": round(float(neg.mean()), 4),
            "worst_supported": [(k, v) for k, v in sup_sorted[:20]],
            "worst_unsupported": [(k, v) for k, v in uns_sorted[:20]],
        }

        # verbatim dump: worst items, mixed, ranked by discordance contribution
        picks = [("SUPPORTED-scored-low (false positive)", k, v) for k, v in sup_sorted[:15]]
        picks += [("UNSUPPORTED-scored-high (false negative)", k, v) for k, v in uns_sorted[:15]]
        lines.append(f"\n{'='*100}\nSUBSET {sub}  AUROC {a:.4f}  n_sup {n1}  n_uns {n0}\n{'='*100}")
        for kind, ridx, contrib in picks:
            r = resp.filter((pl.col("subset") == sub) & (pl.col("resp_idx") == ridx)).row(0, named=True)
            ss = sent.filter((pl.col("subset") == sub) & (pl.col("resp_idx") == ridx)).sort("sent_score")
            lines.append(
                f"\n--- [{kind}] resp_idx={ridx} resp_score={r['resp_score']:.4f} "
                f"discordance={contrib:.0f} n_sent={r['n_sent']} "
                f"lab0={r['n_lab0']} lab1={r['n_lab1']} unlab={r['n_unlab']}"
            )
            for row in ss.head(3).iter_rows(named=True):
                w = df.filter(
                    (pl.col("subset") == sub)
                    & (pl.col("resp_idx") == ridx)
                    & (pl.col("sent_idx") == row["sent_idx"])
                ).sort("score", descending=True).row(0, named=True)
                lines.append(
                    f"  sent_idx={row['sent_idx']} sent_label={row['label']} "
                    f"score={row['sent_score']:.4f} n_win={row['n_win']}"
                )
                lines.append(f"    SENT: {row['sent_text']}")
                lines.append(f"    BESTWIN(doc={w['doc_idx']} off={w['char_offset']} len={w['doc_len']}): {w['win_text']}")

    OUT.write_text(json.dumps(out, indent=2))
    DUMP.write_text("\n".join(lines))
    print(json.dumps(out["reproduction"], indent=2))
    for sub in ["finqa", "delucionqa"]:
        d = {k: v for k, v in out["subsets"][sub].items() if not k.startswith("worst_")}
        print(sub, json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
