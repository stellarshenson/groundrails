"""R10-H111 stage 1 - dropout-dial generation at scale, referee v3.

Stage 0 FIRED (canonical log, round 10): mbart-large-50 under MC dropout at
p=0.2 yields honest fluent-drift 0.302 with ~zero paraphrase-mislabel. The
recorded amendment binds this stage: the token-level degeneracy gate admits
symbol-soup and char-run junk (LaTeX-heavy scientific seeds worst), so referee
v3 adds a CHAR-LEVEL degeneracy check (max repeated char-run + symbol-density,
thresholds calibrated on the p=0.05 distribution like every other gate) and
scientific seeds are math-stripped BEFORE reconstruction.

Output: R10-H111_stage1_pairs.parquet - admitted DRIFT reconstructions as
label-0 negatives against the seed's own evidence (tag ae_drift_<register>)
plus certified-paraphrase rows as label-1 augmentation (tag ae_para_<register>,
cap 40k balanced). Rounds of full-seed decoding at p=0.2 with fresh torch seeds
repeat until ~80k admitted drift or ROUNDS_MAX exhausted. Checkpoints land
after every referee pass - death loses nothing.

Training draw: HELD - this script generates data only.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R10-H111_stage1.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import io
import json
import pathlib
import random
import re
import zipfile

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT_PARQ = HERE / "R10-H111_stage1_pairs.parquet"
OUT_REPORT = HERE / "R10-H111_stage1_report.md"
OUT_PROGRESS = HERE / "R10-H111_stage1_progress.json"

SEED = 0
P_GEN = 0.2  # calibrated best_p (stage 0b)
CAL_N = 3000  # calibration subsample for referee thresholds
ROUNDS_MAX = 4
DRIFT_TARGET = 80_000
PARA_CAP_PER_REG = 13_333  # ~40k total, balanced
CHUNK_CAP = 1500

spec = importlib.util.spec_from_file_location("s0", HERE / "R10-H111_stage0.py")
S0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S0)
log = S0.log

random.seed(SEED)
np.random.seed(SEED)

MATH_CHARS = re.compile(r"[\\${}^_=]")
SYMBOL_OK = set(" .,;:'\"()%-|/")


def mathy(text):
    """True when a seed is too math-laden to reconstruct meaningfully."""
    if len(MATH_CHARS.findall(text)) > 2:
        return True
    sym = sum(1 for c in text if not (c.isalnum() or c in SYMBOL_OK))
    return sym / max(len(text), 1) > 0.10


def degeneracy_tok(text):
    toks = text.lower().split()
    if len(toks) < 3:
        return 1.0, 1
    grams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    d3 = len(set(grams)) / len(grams)
    run, best = 1, 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return d3, best


def degeneracy_char(text):
    """(max repeated single-char run, symbol-density) - catches '-s---s' soup."""
    best, run = 1, 1
    for a, b in zip(text, text[1:]):
        run = run + 1 if a == b and not a.isspace() else 1
        best = max(best, run)
    sym = sum(1 for c in text if not (c.isalnum() or c in SYMBOL_OK))
    return best, sym / max(len(text), 1)


# ------------------------------------------------------------------- seeds
def load_seeds_scaled():
    """[(register, claim, chunk)] - every seed keeps its own evidence."""
    rows = []

    a = pl.read_parquet(DATA / "R10-H107_pairs.parquet")
    a = a.filter(pl.col("label") == 1).unique(subset=["claim"], keep="first")
    for claim, chunk in zip(a["claim"].to_list(), a["chunk"].to_list()):
        rows.append(("procedural", claim, chunk))

    b = pl.read_parquet(HERE / "R10-H108_pairs.parquet")
    b = b.filter(pl.col("label") == 1).unique(subset=["claim"], keep="first")
    for claim, chunk in zip(b["claim"].to_list(), b["chunk"].to_list()):
        rows.append(("quantitative", claim, chunk))

    # TabFact label-1 positives, evidence built exactly as the clean trainer
    zt = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(
        io.BytesIO(zt.read(next(x for x in zt.namelist() if x.endswith("__train.parquet"))))
    )
    dt = dt.filter((pl.col("label") == 1) & (pl.col("statement").str.len_chars() > 10))
    dt = dt.sample(n=min(17_000, len(dt)), seed=SEED)
    for st, cap, tbl in zip(
        dt["statement"].to_list(), dt["table_caption"].to_list(), dt["table_text"].to_list()
    ):
        chunk = f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_CAP]
        rows.append(("quantitative", st, chunk))

    # scientific: arxiv abstracts (CC0 mirror, stage-0 source), math-stripped
    from datasets import load_dataset

    log("streaming arxiv abstracts for scientific seeds ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    sci = []
    for i, ex in enumerate(ds):
        if len(sci) >= 30_000 or i > 120_000:
            break
        abstract = S0._clean(ex.get("abstract", ""))
        if len(abstract) < 200 or mathy(abstract):
            continue
        taken = 0
        for sent in re.split(r"(?<=[.!?])\s+", abstract):
            w = sent.split()
            if not (12 <= len(w) <= 45) or not sent or not sent[0].isupper():
                continue
            if mathy(sent):
                continue
            sci.append(("scientific", sent, abstract[:CHUNK_CAP]))
            taken += 1
            if taken >= 2:
                break
    rows += sci

    out = []
    for reg, claim, chunk in rows:
        claim = S0._clean(claim)[:350]
        if len(claim) >= 40 and chunk and len(chunk) >= 80:
            out.append((reg, claim, chunk))
    counts = {r: sum(1 for x in out if x[0] == r) for r in ("procedural", "quantitative", "scientific")}
    log(f"seeds: {len(out)} total {counts}")
    return out


# ------------------------------------------------------------------- referee
def calibrate(tok, model, seeds):
    cal = random.Random(SEED).sample(seeds, min(CAL_N, len(seeds)))
    texts = [c for _, c, _ in cal]
    log(f"calibration: reconstructing {len(texts)} seeds at p=0.05 ...")
    torch.manual_seed(SEED + 5)
    recs = S0.reconstruct(tok, model, texts, p=0.05, train_mode=True)
    nll = S0.gpt2_nll(recs)
    deg = [degeneracy_tok(t) for t in recs]
    ch = [degeneracy_char(t) for t in recs]
    th = {
        "nll_max": float(np.percentile(nll, 95)),
        "distinct3_min": float(np.percentile([d for d, _ in deg], 5)),
        "maxrun_max": float(np.percentile([r for _, r in deg], 95)),
        "charrun_max": float(np.percentile([r for r, _ in ch], 95)),
        "symdens_max": float(np.percentile([s for _, s in ch], 95)),
    }
    log(f"referee v3 thresholds (p=0.05 calibration): {json.dumps(th)}")
    return th


def referee_round(texts, recs, th):
    """Classify each (seed, rec): 'drift' | 'paraphrase' | 'noise'; return cls + scores."""
    nll = S0.gpt2_nll(recs)
    fwd_am, fwd_pe = S0.nli_entail(list(zip(texts, recs)))
    bwd_am, bwd_pe = S0.nli_entail(list(zip(recs, texts)))
    out = []
    for i, rec in enumerate(recs):
        d3, mrun = degeneracy_tok(rec)
        crun, sdens = degeneracy_char(rec)
        degen = (
            d3 < th["distinct3_min"]
            or mrun > th["maxrun_max"]
            or crun > th["charrun_max"]
            or sdens > th["symdens_max"]
        )
        fluent = nll[i] <= th["nll_max"]
        exact = rec.lower() == texts[i].lower()
        para = (fwd_am[i] and bwd_am[i]) or exact
        if degen or not fluent:
            cls = "noise"
        elif para:
            cls = "paraphrase" if not exact else "noise"  # exact copies are useless rows
        else:
            cls = "drift"
        out.append(
            {
                "cls": cls,
                "nll": float(nll[i]),
                "distinct3": float(d3),
                "maxrun": int(mrun),
                "charrun": int(crun),
                "symdens": float(sdens),
                "nli_fwd": float(fwd_pe[i]),
                "nli_bwd": float(bwd_pe[i]),
            }
        )
    return out


# ------------------------------------------------------------------- main
def main():
    seeds = load_seeds_scaled()
    tok, model = S0.load_mbart()
    th = calibrate(tok, model, seeds)

    kept = []  # accumulated output rows
    seen = set()  # (seed, rec) dedup
    para_counts = {"procedural": 0, "quantitative": 0, "scientific": 0}
    drift_total = 0
    recon_total = 0

    def checkpoint():
        if kept:
            pl.DataFrame(kept).write_parquet(OUT_PARQ)
        OUT_PROGRESS.write_text(
            json.dumps(
                {
                    "recon_total": recon_total,
                    "drift_total": drift_total,
                    "para_counts": para_counts,
                    "rows": len(kept),
                    "thresholds": th,
                }
            )
        )

    CHUNK = 12_000  # referee in slabs so checkpoints land steadily
    for rnd in range(ROUNDS_MAX):
        if drift_total >= DRIFT_TARGET:
            break
        log(f"=== round {rnd + 1}/{ROUNDS_MAX} - decoding {len(seeds)} seeds at p={P_GEN} ===")
        torch.manual_seed(SEED + 100 + rnd)
        order = list(range(len(seeds)))
        random.Random(SEED + rnd).shuffle(order)
        for lo in range(0, len(order), CHUNK):
            idx = order[lo : lo + CHUNK]
            texts = [seeds[i][1] for i in idx]
            recs = S0.reconstruct(tok, model, texts, p=P_GEN, train_mode=True)
            recon_total += len(recs)
            scored = referee_round(texts, recs, th)
            for j, i in enumerate(idx):
                reg, claim, chunk = seeds[i]
                rec, s = recs[j], scored[j]
                key = hash((claim, rec))
                if key in seen:
                    continue
                if s["cls"] == "drift":
                    seen.add(key)
                    drift_total += 1
                    kept.append(
                        {
                            "claim": rec,
                            "chunk": chunk,
                            "label": 0,
                            "tag": f"ae_drift_{reg}",
                            "seed": claim,
                            "p": P_GEN,
                            **{k: v for k, v in s.items() if k != "cls"},
                        }
                    )
                elif s["cls"] == "paraphrase" and para_counts[reg] < PARA_CAP_PER_REG:
                    seen.add(key)
                    para_counts[reg] += 1
                    kept.append(
                        {
                            "claim": rec,
                            "chunk": chunk,
                            "label": 1,
                            "tag": f"ae_para_{reg}",
                            "seed": claim,
                            "p": P_GEN,
                            **{k: v for k, v in s.items() if k != "cls"},
                        }
                    )
            checkpoint()
            log(
                f"  progress: recons {recon_total}  drift {drift_total}  "
                f"para {sum(para_counts.values())}  rows {len(kept)}"
            )
            if drift_total >= DRIFT_TARGET:
                break

    checkpoint()

    # ----------------------------------------------------------- report + eyeball
    df = pl.DataFrame(kept)
    cells = df.group_by(["tag", "label"]).len().sort("tag")
    log(f"final cells:\n{cells}")
    for tag in sorted(df["tag"].unique().to_list()):
        sub = df.filter(pl.col("tag") == tag)
        for row in sub.sample(n=min(10, len(sub)), seed=SEED).iter_rows(named=True):
            log(f"  [{tag}] seed: {row['seed'][:110]}")
            log(f"  [{tag}] rec : {row['claim'][:110]}")

    drift_df = df.filter(pl.col("label") == 0)
    eye = drift_df.sample(n=min(50, len(drift_df)), seed=SEED)
    border = (
        df.filter(pl.col("label") == 1)
        .with_columns(pl.min_horizontal("nli_fwd", "nli_bwd").alias("minent"))
        .sort("minent")
        .head(25)
    )
    lines = [
        "# R10-H111 stage 1 - generation report",
        "",
        f"Model facebook/mbart-large-50, p={P_GEN}, referee v3 thresholds: {json.dumps(th)}",
        f"Reconstructions consumed: {recon_total}; admitted drift {drift_total}; "
        f"paraphrase augmentation {sum(para_counts.values())} {para_counts}",
        "",
        "## Counts",
        "",
        str(cells),
        "",
        "## Eyeball - 50 admitted drift (main-session precision adjudication)",
        "",
    ]
    for k, row in enumerate(eye.iter_rows(named=True)):
        lines += [
            f"**D{k + 1}** [{row['tag']}] min-entail "
            f"{min(row['nli_fwd'], row['nli_bwd']):.2f}",
            f"- seed: {row['seed']}",
            f"- rec : {row['claim']}",
            "",
        ]
    lines += ["## Borderline paraphrases (25 lowest min-entailment)", ""]
    for k, row in enumerate(border.iter_rows(named=True)):
        lines += [
            f"**B{k + 1}** [{row['tag']}] min-entail {row['minent']:.2f}",
            f"- seed: {row['seed']}",
            f"- rec : {row['claim']}",
            "",
        ]
    OUT_REPORT.write_text("\n".join(lines))
    log(f"report -> {OUT_REPORT}")
    log(f"pairs  -> {OUT_PARQ}  ({len(kept)} rows)")
    log("=== R10-H111 STAGE1 DONE ===")


if __name__ == "__main__":
    main()
