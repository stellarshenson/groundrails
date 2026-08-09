"""R15 gate wave - shared frozen-read helpers.

Every gate in the wave is a frozen-weights, held-out, arena-free read. This
module carries only what more than one gate needs: checkpoint loading, the
pair scorer, AUROC (sklearn, identical to `R7-H59.auc_and_f1`'s first return),
the TabFact held-out loader and the table/number helpers the banked probes use
byte-identically.

Import AFTER setting CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES in the caller.
"""

import io
import pathlib
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
MODELS = ROOT / "models"

MAX_LEN = 512
BATCH = 64
CHUNK_MAX = 1500

NUM_CELL = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")


# --------------------------------------------------------------------------- #
# numbers and tables - byte-identical to R14_H133_probe / R15_P1_typeprobe
# --------------------------------------------------------------------------- #
def parse(tbl):
    rows = [r.split("#") for r in tbl.replace("\r\n", "\n").strip().split("\n") if r.strip()]
    rows = [[c.strip() for c in r] for r in rows]
    if len(rows) < 4:
        return None, None
    w = len(rows[0])
    return rows[0], [r for r in rows[1:] if len(r) == w]


def as_num(s):
    s = s.strip()
    if not NUM_CELL.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def serialize(cap, hdr, body):
    """The probes' evidence string, rebuilt from a (possibly edited) table."""
    tbl = "\n".join(["#".join(hdr)] + ["#".join(r) for r in body])
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX]


def canon_set(s):
    """P3 / L2 absence detector - canonical numeral forms present in a string."""
    p = set()
    for m in NUM_FREE.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def asserts_absent(claim, evidence):
    """True iff the claim carries a numeral whose canonical form is not in the evidence."""
    c = canon_set(claim)
    return bool(c) and bool(c - canon_set(evidence))


def held_tabfact():
    """TabFact test+validation, table_id-disjoint from the train split every arm trained on."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_ids = set(
        pl.read_parquet(
            io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith("__train.parquet"))))
        )["table_id"].to_list()
    )
    held = pl.concat([
        pl.read_parquet(io.BytesIO(z.read(n)))
        for n in z.namelist()
        if n.endswith("__test.parquet") or n.endswith("__validation.parquet")
    ]).unique(subset=["table_id"], keep="first")
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    return (held["table_caption"].to_list(), held["table_text"].to_list(),
            held["table_id"].to_list())


# --------------------------------------------------------------------------- #
# frozen reads
# --------------------------------------------------------------------------- #
def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    return float(roc_auc_score(y, np.concatenate([np.asarray(pos), np.asarray(neg)])))


def load_ckpt(name):
    """(tokenizer, trunk, task head) for a banked cross-encoder checkpoint."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    d = MODELS / name
    tok = AutoTokenizer.from_pretrained(str(d))
    state = torch.load(d / "dann_student.pt", map_location="cpu", weights_only=False)
    trunk = AutoModel.from_pretrained(str(d / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    return tok, trunk, head.cuda().eval()


def load_trunk(path, tokpath=None):
    """(tokenizer, trunk) for a bare pretrained or fine-tuned trunk - no head."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokpath or path)
    trunk = AutoModel.from_pretrained(path).cuda().eval()
    if hasattr(trunk.config, "reference_compile"):
        trunk.config.reference_compile = False
    return tok, trunk


def score(tok, trunk, head, claims, evs, max_len=MAX_LEN, want_lens=False):
    """Sigmoid task score of (claim, evidence) pairs - the shipped read."""
    import torch

    s = np.zeros(len(claims), dtype=np.float32)
    lens = np.zeros(len(claims), dtype=np.int32) if want_lens else None
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            cb, eb = claims[j:j + BATCH], evs[j:j + BATCH]
            if want_lens:
                lens[j:j + BATCH] = [len(x) for x in tok(cb, eb, truncation=False)["input_ids"]]
            enc = tok(cb, eb, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[j:j + BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    return (s, lens) if want_lens else s


def cls_of(tok, trunk, claims, evs, max_len=256):
    """Frozen trunk [CLS] matrix - the substrate probe's input."""
    import torch

    h = trunk.config.hidden_size
    C = np.zeros((len(claims), h), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j:j + BATCH], evs[j:j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len)
            enc = {k: v.cuda() for k, v in enc.items()}
            C[j:j + BATCH] = trunk(**enc).last_hidden_state[:, 0].float().cpu().numpy()
    return C
