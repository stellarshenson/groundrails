"""R16-H140 G1 stage 1 - cache frozen (sentence, window) pooled embeddings.

The serving read reduces every (sentence, window) pair to sigmoid(task_head(CLS))
and then takes a hard max. This stage caches the CLS vector the head consumes,
alongside that scalar, for:

  * the BLIND R8-H77 arena (read-only; nothing here is trained on)
  * a PUBLIC-ONLY training slice: RAGTruth train (full untruncated context,
    windowed 1500/750, per-sentence labels derived from the released
    hallucination character spans) + the R10-H108 manufactured quantitative
    lane (public FEVEROUS / InfoTabs / SciTab / TabFact-corrupt rows)

No private gold rows and no arena rows enter the training slice.

Numerics are byte-identical to R8-H77.score_student: fp32 trunk, max_length 512,
batch 64, CLS = last_hidden_state[:, 0]. The hard-max reproduction of the banked
read is the cache-fidelity check.

Run (GPU2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run python \
    experiments/grounding-semantic/R16-H140_G1_cache.py --model models/R10-H108-lane-draw1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import importlib.util
import io
import json
import pathlib
import re
import time
import zipfile

import numpy as np
import polars as pl
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
CACHE = HERE / "R16-H140_cache"
CACHE.mkdir(exist_ok=True)

WIN, STRIDE = 1500, 750
MAX_LEN, BATCH = 512, 64
LANE_SAMPLE = 12_000
SEED = 0


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")
G0 = _mod("g0", "R16-H140_G0_census.py")

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def windows(chunk):
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def sentences_with_offsets(text):
    """H8-H92.sentences, with the char offsets needed for span labelling."""
    segs, prev = [], 0
    for m in _SPLIT.finditer(text):
        segs.append((prev, m.start()))
        prev = m.end()
    segs.append((prev, len(text)))
    out = []
    for a, b in segs:
        seg = text[a:b]
        st = seg.strip()
        if len(st) < H92.MIN_SENT_CHARS:
            continue
        off = a + (len(seg) - len(seg.lstrip()))
        out.append((st, off, off + len(st)))
        if len(out) == H92.MAX_SENTS:
            break
    if len(out) < 2:
        return [(text, 0, len(text))]
    return out


# --- the frozen scorer ---------------------------------------------------------


class Scorer:
    def __init__(self, mdir):
        mdir = pathlib.Path(mdir)
        self.tok = AutoTokenizer.from_pretrained(str(mdir))
        self.trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
        st = torch.load(mdir / "dann_student.pt", map_location="cpu", weights_only=False)
        head = nn.Linear(self.trunk.config.hidden_size, 1)
        head.load_state_dict(st["task_head"])
        self.head = head.cuda().eval()
        self.d = self.trunk.config.hidden_size

    @torch.inference_mode()
    def run(self, sents, wins, tag=""):
        n = len(sents)
        emb = np.zeros((n, self.d), dtype=np.float16)
        sc = np.zeros(n, dtype=np.float32)
        t0 = time.time()
        for i in range(0, n, BATCH):
            enc = self.tok(
                sents[i : i + BATCH], wins[i : i + BATCH], return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_LEN,
            )
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = self.trunk(**enc).last_hidden_state[:, 0]
            emb[i : i + BATCH] = cls.half().cpu().numpy()
            sc[i : i + BATCH] = torch.sigmoid(self.head(cls).float().squeeze(-1)).cpu().numpy()
            if (i // BATCH) % 200 == 0:
                done = min(i + BATCH, n)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {tag} {done}/{n} ({rate:.0f} pairs/s)", flush=True)
        return emb, sc


# --- arena ---------------------------------------------------------------------


def build_arena(scorer):
    ARENA = _mod("arena", "R8-H77_unseen_arena.py")
    subs = ARENA.load_subsets()
    meta = {}
    for sub, (claims, chunks, y) in subs.items():
        f = CACHE / f"arena_{sub}.npz"
        if f.exists():
            print(f"  arena {sub}: cached", flush=True)
            meta[sub] = int(np.load(f)["score"].shape[0])
            continue
        flat_s, flat_w, pair_sent = [], [], []
        sent_owner, sent_cross = [], []
        for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
            sl = H92.sentences(c)
            wlist = [w for k in ks for w in windows(k)]
            low_chunks = [k.lower() for k in ks]
            win_lists = [G0.windows(k) for k in ks]
            caches = [{} for _ in ks]
            for s in sl:
                sid = len(sent_owner)
                sent_owner.append(i)
                r = G0.sentence_dispersion(s, low_chunks, win_lists, caches)
                sent_cross.append(bool(r is not None and not r["within_window"]))
                for w in wlist:
                    flat_s.append(s)
                    flat_w.append(w)
                    pair_sent.append(sid)
        emb, sc = scorer.run(flat_s, flat_w, tag=f"arena/{sub}")
        np.savez_compressed(
            f, emb=emb, score=sc,
            pair_sent=np.array(pair_sent, dtype=np.int32),
            sent_owner=np.array(sent_owner, dtype=np.int32),
            sent_cross=np.array(sent_cross, dtype=bool),
            y=np.asarray(y, dtype=np.int8),
        )
        meta[sub] = len(sc)
        print(f"  arena {sub}: {len(sc)} pairs, {len(sent_owner)} sentences -> {f.name}", flush=True)
    return meta


# --- public training slice ------------------------------------------------------


def ragtruth_rows():
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    n = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n))).filter(pl.col("context").str.len_chars() > 50)
    out = []
    for ctx, outp, spans_raw in zip(
        df["context"].to_list(), df["output"].to_list(), df["hallucination_labels"].to_list(),
        strict=True,
    ):
        try:
            spans = [(int(s["start"]), int(s["end"])) for s in json.loads(spans_raw)]
        except Exception:
            spans = []
        sents = sentences_with_offsets(outp)
        wlist = windows(ctx)
        for st, a, b in sents:
            lab = 0.0 if any(a < e and b > s for s, e in spans) else 1.0
            out.append((st, wlist, lab, "ragtruth_en"))
    return out


def lane_rows():
    df = pl.read_parquet(HERE / "R10-H108_pairs.parquet")
    df = df.sample(n=LANE_SAMPLE, seed=SEED, shuffle=True)
    return [
        (c, windows(k), float(lb), f"h108_{tg}")
        for c, k, lb, tg in zip(
            df["claim"].to_list(), df["chunk"].to_list(),
            df["label"].to_list(), df["tag"].to_list(), strict=True,
        )
    ]


def build_train(scorer):
    f = CACHE / "train.npz"
    if f.exists():
        print("  train slice: cached", flush=True)
        return json.loads((CACHE / "train_composition.json").read_text())

    rows = ragtruth_rows() + lane_rows()
    print(f"  train slice: {len(rows)} sentence-sets", flush=True)

    flat_s, flat_w, pair_sent = [], [], []
    labels, sources = [], []
    for sid, (st, wlist, lab, src) in enumerate(rows):
        labels.append(lab)
        sources.append(src)
        for w in wlist:
            flat_s.append(st)
            flat_w.append(w)
            pair_sent.append(sid)

    emb, sc = scorer.run(flat_s, flat_w, tag="train")
    src_ids = sorted(set(sources))
    smap = {s: i for i, s in enumerate(src_ids)}
    np.savez_compressed(
        f, emb=emb, score=sc,
        pair_sent=np.array(pair_sent, dtype=np.int32),
        label=np.array(labels, dtype=np.float32),
        source=np.array([smap[s] for s in sources], dtype=np.int16),
    )
    comp = {
        "source_ids": src_ids,
        "sentence_sets_by_source": {s: int(sum(1 for x in sources if x == s)) for s in src_ids},
        "n_sentence_sets": len(rows),
        "n_pairs": len(sc),
        "positives": float(np.mean(labels)),
        "window_set_size": {
            "mean": float(np.mean([len(w) for _, w, _, _ in rows])),
            "max": int(max(len(w) for _, w, _, _ in rows)),
        },
        "provenance": "PUBLIC ONLY - RAGTruth train + R10-H108 manufactured lane "
                      "(FEVEROUS / InfoTabs / SciTab / TabFact-corrupt). "
                      "No private gold rows, no RAGBench/arena rows.",
    }
    (CACHE / "train_composition.json").write_text(json.dumps(comp, indent=2))
    print(f"  train slice: {len(sc)} pairs -> {f.name}", flush=True)
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "R10-H108-lane-draw1"))
    ap.add_argument("--only", choices=("all", "train", "arena"), default="all",
                    help="split the build across free cards; both halves are independent")
    args = ap.parse_args()
    print(f"=== R16-H140 G1 cache {time.strftime('%F %T')} model={args.model} "
          f"only={args.only} ===", flush=True)
    scorer = Scorer(args.model)
    if args.only in ("all", "train"):
        build_train(scorer)
    if args.only in ("all", "arena"):
        build_arena(scorer)
    print(f"=== G1 CACHE DONE ({args.only}) ===", flush=True)


if __name__ == "__main__":
    main()
