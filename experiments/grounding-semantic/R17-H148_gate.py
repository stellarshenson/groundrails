"""R17-H148 stage 1 - KILL-GATE read of the banked control on the procedural probe.

Frozen-weights, held-out, arena-free.  Scores `R17-H148_probe.parquet` with the
campaign control `models/R9-H105-mmbert-dann-clean` and writes
`R17-H148_gate_result.json`.

Registered gate: LICENSE the lane only if control AUROC <= 0.65.  The flag is
written; the coordinator adjudicates.

`models/R16-H142-G0` is read as a REFERENCE only.  G0's adapter conditions each
window's logit on the pooled context of the whole window set; a probe pair is a
single (claim, chunk) window, so the context degenerates to the pair's own CLS.
The number is therefore not comparable to G0's banked windowed arena reads.

GPU2 only:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    uv run python experiments/grounding-semantic/R17-H148_gate.py
"""
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np                                                  # noqa: E402
import polars as pl                                                 # noqa: E402

import R15_gate_common as G                                         # noqa: E402

HERE = pathlib.Path(__file__).parent
PROBE = HERE / "R17-H148_probe.parquet"
MANIFEST = HERE / "R17-H148_probe_manifest.json"
CENSUS = HERE / "R17-H148_census.json"
OUT = HERE / "R17-H148_gate_result.json"

CONTROL = "R9-H105-mmbert-dann-clean"
GATE_BAR = 0.65


def reads(df, s):
    y = df["label"].to_numpy()
    out = {"auroc": round(G.auroc(s[y == 1], s[y == 0]), 4),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    piv = (df.select(["pair_id", "label", "neg_family", "corpus"])
             .with_columns(pl.Series("score", s)))
    per = {}
    for keys, sub in piv.group_by("neg_family"):
        yy = sub["label"].to_numpy()
        ss = sub["score"].to_numpy()
        p = sub.pivot(on="label", index="pair_id", values="score",
                      aggregate_function="first").drop_nulls()
        per[keys[0]] = {
            "auroc": round(G.auroc(ss[yy == 1], ss[yy == 0]), 4),
            "within_pair_acc": round(float(((p["1"].to_numpy() > p["0"].to_numpy())
                                            + 0.5 * (p["1"].to_numpy() == p["0"].to_numpy())
                                            ).mean()), 4),
            "pairs": len(p)}
    out["per_family"] = per
    corp = {}
    for keys, sub in piv.group_by("corpus"):
        yy, ss = sub["label"].to_numpy(), sub["score"].to_numpy()
        corp[keys[0]] = {"auroc": round(G.auroc(ss[yy == 1], ss[yy == 0]), 4),
                         "rows": len(sub)}
    out["per_corpus"] = corp
    p = piv.pivot(on="label", index="pair_id", values="score",
                  aggregate_function="first").drop_nulls()
    pos, neg = p["1"].to_numpy(), p["0"].to_numpy()
    out["within_pair_acc"] = round(float(((pos > neg) + 0.5 * (pos == neg)).mean()), 4)
    out["mean_pos"] = round(float(pos.mean()), 4)
    out["mean_neg"] = round(float(neg.mean()), 4)
    # AUROC standard error, Hanley-McNeil - the H147 instrument-precision discipline
    a = out["auroc"]
    n1, n2 = out["n_pos"], out["n_neg"]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    out["auroc_se_hanley_mcneil"] = round(
        float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                       + (n2 - 1) * (q2 - a * a)) / (n1 * n2))), 4)
    return out


def main():
    df = pl.read_parquet(PROBE)
    man = json.loads(MANIFEST.read_text())
    census = json.loads(CENSUS.read_text())
    claims, chunks = df["claim"].to_list(), df["chunk"].to_list()
    print(f"probe: {df.height} rows / {df['pair_id'].n_unique()} pairs", flush=True)

    tok, trunk, head = G.load_ckpt(CONTROL)
    s = G.score(tok, trunk, head, claims, chunks)
    control = reads(df, s)
    print(f"control {CONTROL}: AUROC {control['auroc']} "
          f"(SE {control['auroc_se_hanley_mcneil']})", flush=True)

    ref = None
    try:
        import torch
        from transformers import AutoTokenizer
        import importlib.util
        spec = importlib.util.spec_from_file_location("h147", HERE / "R17-H147_autopsy.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        base = G.MODELS / m.G0_BASE
        scorer = m.AdapterScorer(base, G.MODELS / "R16-H142-G0").cuda().eval()
        gtok = AutoTokenizer.from_pretrained(str(base))
        out = np.zeros(len(claims), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(claims), G.BATCH):
                enc = gtok(claims[i:i + G.BATCH], chunks[i:i + G.BATCH], return_tensors="pt",
                           padding=True, truncation=True, max_length=G.MAX_LEN)
                enc = {k: v.cuda() for k, v in enc.items()}
                cls = scorer.encode(enc)
                out[i:i + G.BATCH] = torch.sigmoid(
                    scorer.pair_logits(cls, cls).float()).cpu().numpy()
        ref = reads(df, out)
        ref["caveat"] = ("single-window probe degenerates G0's set context to the "
                         "pair's own CLS - not comparable to its banked arena reads")
        print(f"reference R16-H142-G0: AUROC {ref['auroc']}", flush=True)
    except Exception as e:                                          # noqa: BLE001
        ref = {"error": f"{type(e).__name__}: {e}"}
        print(f"reference read skipped: {ref['error']}", flush=True)

    lic = bool(control["auroc"] <= GATE_BAR)
    res = {
        "hypothesis": "R17-H148 PROCEDURAL-REGISTER LANE - stage 1 kill-gate",
        "gpu": "GPU2 (RTX 5000 Ada), CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2",
        "sources_census": {
            "army_tm": {"pdfs_on_disk": census["army-tm"]["pdfs"],
                        "crawl_targets": 1766,
                        "crawl_share": round(census["army-tm"]["pdfs"] / 1766, 4),
                        "identifier_families": {"LO": 85, "TB": 43, "MWO": 7,
                                                "TM_operator_manuals": 0},
                        "pages": census["army-tm"]["pages"],
                        "procedural_blocks": census["army-tm"]["blocks_clean"],
                        "docs_with_blocks": census["army-tm"]["docs_with_blocks"],
                        "items": census["army-tm"]["items"],
                        "licence": "public domain (17 U.S.C. 105)"},
            "faa_amt": {"pdfs_on_disk": census["faa-amt"]["pdfs"],
                        "pages": census["faa-amt"]["pages"],
                        "procedural_blocks": census["faa-amt"]["blocks_clean"],
                        "items": census["faa-amt"]["items"],
                        "licence": "public domain (17 U.S.C. 105)"},
            "multidoc2dial": {"available": False,
                              "cached": "loader script only "
                                        "(datasets--IBM--multidoc2dial, 40 KB, no data "
                                        "shards); the `datasets` package is absent from "
                                        "the project venv",
                              "note": "no network fetch attempted - out of scope for "
                                      "stage 1"},
            "total_procedural_blocks": census["total_blocks"],
            "total_documents": census["total_documents"]},
        "probe": {"file": PROBE.name, "manifest": MANIFEST.name,
                  "pairs": man["pairs"], "rows": man["rows"],
                  "documents": man["documents"], "blocks": man["blocks"],
                  "families": man["families"], "corpora": man["corpora"],
                  "construction": man["construction"],
                  "verify": man["verify"]},
        "control_checkpoint": CONTROL,
        "control_auroc": control["auroc"],
        "control_read": control,
        "reference_R16_H142_G0": ref,
        "gate": {"bar": f"control AUROC <= {GATE_BAR}", "license": lic},
        "license": lic,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({"control_auroc": control["auroc"], "license": lic,
                      "per_family": control["per_family"],
                      "per_corpus": control["per_corpus"]}, indent=2), flush=True)
    print(f"=== R17-H148 GATE: control {control['auroc']} -> "
          f"{'LICENSE' if lic else 'KILL'} ===", flush=True)


if __name__ == "__main__":
    main()
