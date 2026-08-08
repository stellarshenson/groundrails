"""R13 read-amendment instrument - per-(sentence, chunk, window) score dump.

Task A of the R13 read-amendment campaign (R13-H124 WINDOW-CONSENSUS,
R13-H125 TOP2-UNION, ANCHOR-TEACHER salvage diagnostic).

The M3 instrument was built for R12-H121 Gate A
(`R12-H121_gateA_dump.py`, stage 1) and already retains the three levels
H124 needs - `doc_idx` (chunk), `win_in_doc` (window index within chunk) and
`win_idx` (flat window index, document order). Its stage-3 reconstruction
reproduces `R9-H105_windowed_result.json` exactly, so the matrix is trusted.
This script is that stage 1 with the checkpoint parameterised: every helper
(`windows`, `load_annotated`, `sentence_labels`) is imported from the Gate A
module rather than re-implemented, so the dumps are byte-identical in
construction across checkpoints.

CONTAMINATION DISCIPLINE (author ruling 4, round 12): sentence-level
annotations are carried for ANALYSIS labels only (the H125 fire-rate
diagnostic splits on the response-level `adherence_score`, which is the read's
own label). No quantity here may enter any lane's size, thresholds or mix.

  R13_dump_h105d1.parquet   == R12-H121_gateA_scores.parquet (symlinked, exists)
  R13_dump_h105d2.parquet   R9-H105-draw2
  R13_dump_h108d1.parquet   R10-H108-lane-draw1
  R13_dump_h108d2.parquet   R10-H108-lane-draw2

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R13_reads_dump.py --tags h108d1,h108d2,h105d2
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GA = _mod("gateA", "R12-H121_gateA_dump.py")

# tag -> banked windowed result whose "model" field resolves the checkpoint
CHECKPOINTS = {
    "h105d1": "R9-H105_windowed_result.json",
    "h105d2": "R9-H105_draw2_windowed_result.json",
    "h108d1": "R10-H108_lane_draw1_windowed_result.json",
    "h108d2": "R10-H108_lane_draw2_windowed_result.json",
}


def model_of(tag):
    return str(ROOT / json.loads((HERE / CHECKPOINTS[tag]).read_text())["model"])


def dump_path(tag):
    return HERE / f"R13_dump_{tag}.parquet"


def dump(tag):
    out = dump_path(tag)
    if out.exists():
        print(f"{tag}: skipped - {out.name} exists", flush=True)
        return
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    H92 = _mod("h92", "R8-H92_decomposed_arena.py")
    model = model_of(tag)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"{tag}: model {model}", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    state = torch.load(
        pathlib.Path(model) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(model) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    task_head = nn.Linear(trunk.config.hidden_size, 1)
    task_head.load_state_dict(state["task_head"])
    task_head = task_head.cuda().eval()

    subs = GA.load_annotated()
    rows = {k: [] for k in (
        "subset", "resp_idx", "sent_idx", "label", "resp_label",
        "win_idx", "doc_idx", "win_in_doc", "n_win_in_doc", "char_offset", "doc_len", "score",
    )}
    sent_texts, win_texts = [], []

    for sub, d in subs.items():
        t0 = time.time()
        flat_s, flat_w, meta = [], [], []
        for i, (resp, docs) in enumerate(zip(d["response"], d["documents"], strict=True)):
            sl = H92.sentences(resp)
            labs = GA.sentence_labels(
                resp, sl, d["response_sentences"][i], d["unsupported"][i], d["ssi"][i]
            )
            wlist = []
            for di, k in enumerate(docs):
                ws = GA.windows(k)
                for wi, (wtext, off) in enumerate(ws):
                    wlist.append((wtext, di, wi, len(ws), off, len(k)))
            for si, s in enumerate(sl):
                for gi, (wtext, di, wi, nw, off, dl) in enumerate(wlist):
                    flat_s.append(s)
                    flat_w.append(wtext)
                    meta.append((i, si, labs[si], int(d["adherence"][i]), gi, di, wi, nw, off, dl))

        s = np.zeros(len(flat_s), dtype=np.float32)
        with torch.inference_mode():
            for j in range(0, len(flat_s), 64):
                enc = tok(
                    flat_s[j : j + 64], flat_w[j : j + 64], return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                )
                enc = {k: v.cuda() for k, v in enc.items()}
                cls = trunk(**enc).last_hidden_state[:, 0]
                s[j : j + 64] = torch.sigmoid(task_head(cls).float().squeeze(-1)).cpu().numpy()

        for (i, si, lab, rl, gi, di, wi, nw, off, dl), sc in zip(meta, s, strict=True):
            rows["subset"].append(sub)
            rows["resp_idx"].append(i)
            rows["sent_idx"].append(si)
            rows["label"].append(lab)
            rows["resp_label"].append(rl)
            rows["win_idx"].append(gi)
            rows["doc_idx"].append(di)
            rows["win_in_doc"].append(wi)
            rows["n_win_in_doc"].append(nw)
            rows["char_offset"].append(off)
            rows["doc_len"].append(dl)
            rows["score"].append(float(sc))
        sent_texts += flat_s
        win_texts += flat_w
        print(f"  {sub:14s} pairs={len(flat_s):>6}  ({time.time() - t0:.0f}s)", flush=True)

    df = pl.DataFrame(rows).with_columns(
        pl.Series("sent_text", sent_texts), pl.Series("win_text", win_texts)
    )
    df.write_parquet(out)
    print(f"{tag} -> {out}  ({len(df)} rows)", flush=True)
    del trunk, task_head
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="h108d1,h108d2,h105d2")
    args = ap.parse_args()
    for tag in args.tags.split(","):
        dump(tag.strip())


if __name__ == "__main__":
    main()
