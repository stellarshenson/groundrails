"""R17-H147 - FLOOR-SUBSET AUTOPSY (hagrid + emanual). ANALYSIS ONLY.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R17-H147 FLOOR-SUBSET AUTOPSY (hagrid + emanual)". Nothing here trains, tunes
or selects anything: the two floor subsets of the blind RAGBench arena are read
with FROZEN banked checkpoints and the per-item outcome is decomposed.

Checkpoints (all banked, none re-trained):

    clean_d1   models/R9-H105-mmbert-dann-clean   (the campaign control, draw 1)
    clean_d2   models/R9-H105-draw2               (control, draw 2)
    h108_d1    models/R10-H108-lane-draw1         (the first admitted lane, draw 1)
    h108_d2    models/R10-H108-lane-draw2         (draw 2)
    g0_adapter models/R16-H142-G0                 (H108 draw-1 trunk + the window-
                                                   ensemble adapter side-head)

The read is the PRIMARY windowed decomposed-min convention (R8-H101): each H92
sentence of the response is scored against every 1,500-char window (stride 750)
of every document of `documents[:8]`; MAX over windows, then MIN over sentences.
The adapter checkpoint is read through its own forward path (R16-H142 G0's
`pair_logits` after mean-pooling the window-ensemble context), so its logit
contribution is never dropped.

POSITIVE CONTROL, run before any analysis: all ten (checkpoint, subset) AUROCs
must reproduce their banked JSON values to within 1e-3. The run aborts otherwise.

Measurements produced per (subset, checkpoint):

  1. per-item response score, per-sentence scores, and the document each
     sentence's arg-max window came from                     -> item_scores.parquet
  2. error taxonomy at the subset's macro-F1-optimal operating threshold, plus a
     threshold-free per-item rank-loss decomposition of (1 - AUROC)
  3. evidence dispersion - how many distinct documents the arg-max windows span,
     and whether the annotated support itself spans documents
  4. faithful-oracle headroom - the R16-H140/R12 label-ceiling convention
     (O1 annotation-AND -> O2 splitter -> O3 chunk cap -> O4 windows), recomputed
     per item for these two subsets and cross-checked against the banked
     R12_label_ceiling_result.json
  5. error concentration - every nameable slice's share of the errors, against
     the R15 >= 30% kill-gate convention (share reported WITH prevalence and
     lift; a 30% share at 30% prevalence is not concentration)
  6. cross-checkpoint error-set Jaccard - same items or different items

Run (detached, GPU2 only):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python experiments/grounding-semantic/R17-H147_autopsy.py \
    >> logs/R17-H147_autopsy.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import itertools
import json
import pathlib
import re
import time

import numpy as np
import polars as pl
import torch
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R17-H147_autopsy.json"
OUT_PARQUET = HERE / "R17-H147_item_scores.parquet"

SUBSETS = ("hagrid", "emanual")
WIN, STRIDE = 1500, 750
MAX_LEN, BATCH = 512, 64
CONTROL_TOL = 1e-3


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA
M59 = ARENA.M59

# The banked windowed decomposed-min AUROCs these reads must reproduce.
# clean_d1/clean_d2 : R9-H105_windowed_result.json, R9-H105_draw2_windowed_result.json
# h108_d1/h108_d2   : R10-H108_lane_draw{1,2}_windowed_result.json
# g0_adapter        : R16-H142_G0_result.json (per_subset[*].adapter_auc)
CHECKPOINTS = {
    "clean_d1": {"dir": "R9-H105-mmbert-dann-clean", "kind": "plain",
                 "banked": {"hagrid": 0.6259, "emanual": 0.6883}},
    "clean_d2": {"dir": "R9-H105-draw2", "kind": "plain",
                 "banked": {"hagrid": 0.6420, "emanual": 0.7070}},
    "h108_d1": {"dir": "R10-H108-lane-draw1", "kind": "plain",
                "banked": {"hagrid": 0.6599, "emanual": 0.6719}},
    "h108_d2": {"dir": "R10-H108-lane-draw2", "kind": "plain",
                "banked": {"hagrid": 0.6354, "emanual": 0.6132}},
    "g0_adapter": {"dir": "R16-H142-G0", "kind": "adapter",
                   "banked": {"hagrid": 0.6805, "emanual": 0.6628}},
}
G0_BASE = "R10-H108-lane-draw1"

# R12_label_ceiling_result.json, O4-strict - the faithful-entailer ceiling.
BANKED_ORACLE_STRICT = {"hagrid": 0.7833, "emanual": 0.8160}
BANKED_ORACLE_LENIENT = {"hagrid": 0.9319, "emanual": 0.9643}


# The registered deliverable's second half: the mechanism-candidate table. Every
# number quoted in `evidence` is produced by this script and lives elsewhere in
# the same JSON, so the table can be audited against its own report. The
# recommendations are the executor's diagnosis; the coordinator adjudicates, and
# any lever becomes a SEPARATE registration built from public data with
# pre-registered bars (the H141 discipline - arena statistics diagnose, never tune).
MECHANISM_CANDIDATES = [
    {
        "name": "procedural list-register verification (emanual)",
        "evidence":
            "The list-structured slice is 70 of 132 emanual items (53.0%) and carries "
            "16 of 16 consensus errors - share 1.000, lift 1.89, in-slice error rate "
            "0.229 against 0.000 outside. Its AUROC is at chance on ALL FIVE banked "
            "checkpoints (0.4737 / 0.5169 / 0.5182 / 0.5223 / 0.5371) while the "
            "non-list half reads 0.9016-1.0000. 13 of emanual's 14 ungrounded items "
            "are list-structured. Imperative-step sentences are 154 of 748 scored "
            "sentences (20.6%) and carry a 9.09% annotated-unsupported rate against "
            "4.24% for declaratives.",
        "caveats":
            "The non-list half contains ONE ungrounded item, so its ~1.0 AUROC is a "
            "single-item measurement; the robust half of the contrast is the list "
            "slice's chance reading, which rests on 13 negatives x 57 positives. "
            "Prior art: the DR procedural lane already read emanual 0.8117 (draw 1 "
            "margin) against 0.6556 control but 0.6901 on draw 2 - a 0.12 draw spread "
            "on a subset whose AUROC standard error is 0.0686. An earlier "
            "procedural-manual hypothesis was refuted at review, but on a DIFFERENT "
            "mechanism (truncation, which windowing fixed); this evidence is "
            "register discrimination at fixed geometry.",
        "buildable_from_public_data": True,
        "public_sources": "army-tm (1,766 public-domain US Army operator/maintenance "
                          "manuals), FAA AMT handbooks, multidoc2dial - all already "
                          "staged in data/external/datasets",
        "recommendation": "build",
        "condition": "the registration must be powered on a held-out procedural probe, "
                     "NOT on emanual's arena AUROC (see the measurement-precision defect)",
    },
    {
        "name": "bare-assertion absent-proposition verification (hagrid)",
        "evidence":
            "140 of 250 hagrid items (56.0%) are single-scored-sentence responses; "
            "their AUROC across the five checkpoints is 0.4923 / 0.5381 / 0.5717 / "
            "0.6101 / 0.6331 (mean 0.569) against 0.6207-0.6907 (mean 0.655) on "
            "multi-sentence items. The observed false-positive mode is a lone "
            "declarative naming the right entity and asserting an evidence-absent "
            "proposition, scored 0.82-0.84. Declarative-register within-sentence "
            "AUROC is 0.6024-0.6384 - plain-declarative discrimination is the "
            "binding number, and it is the prose analogue of the H144 lookup gap.",
        "caveats":
            "Only 15 of the 140 single-sentence items are ungrounded, and the "
            "draw-to-draw range on this slice is 0.14 - three times the subset's "
            "own standard error. The slice is a hypothesis generator, not a bar.",
        "buildable_from_public_data": True,
        "public_sources": "FEVER/VitaminC/WICE are already in the mix; the missing "
                          "construction is open-domain absent-proposition negatives at "
                          "document scale (present entity, absent proposition, surface "
                          "parity) - the prose analogue of the H144 verbatim-lookup family",
        "recommendation": "build",
        "condition": "measured on a held-out constructed probe with pre-registered bars",
    },
    {
        "name": "aggregation or sentence-exclusion redesign",
        "evidence":
            "Eight re-aggregations of the SAME banked per-sentence scores. The shipped "
            "min is best or within 0.03 of best on every checkpoint of both subsets: "
            "hagrid clean_d1 min 0.6259 against mean 0.5603, max 0.4512, drop-first "
            "0.5931, drop-single-lowest 0.6151, exclude-discourse 0.6275, "
            "declarative-only 0.6274; emanual clean_d1 min 0.6883 against mean 0.6217, "
            "drop-first 0.6265, drop-single-lowest 0.6308, exclude-imperative 0.6616. "
            "Reproduces the R8 precursor-P-B oracle bound on the two floor subsets "
            "specifically.",
        "caveats": "none - the bound covers the whole rule class, learned rules included",
        "buildable_from_public_data": True,
        "recommendation": "kill",
    },
    {
        "name": "retrieval geometry - window size, stride, evidence dispersion",
        "evidence":
            "Direct answer to the registered dispersion question: failure is NOT "
            "concentrated where evidence spans documents. hagrid "
            "support_spans_multiple_docs carries 13.2% of errors at 26.8% prevalence "
            "(lift 0.49, anti-concentrated); emanual 31.2% at 27.3% (lift 1.15, inside "
            "noise on 16 errors). any_doc_exceeds_window lift 0.53 (hagrid) / 0.46 "
            "(emanual). R12 measures 0.0% of supported sentences with no window "
            "carrying any support, and the oracle ladder shows window geometry costs "
            "nothing at O3 -> O4-lenient.",
        "caveats":
            "The READ-side slice argmax_windows_span_multiple_docs does concentrate "
            "(lift 1.62 hagrid / 1.85 emanual) but is confounded with response length - "
            "it can only exceed 1 when the response has at least two scored sentences.",
        "buildable_from_public_data": True,
        "recommendation": "kill",
    },
    {
        "name": "per-sentence entailment quality (the binding constraint)",
        "evidence":
            "Sentence-level AUROC on annotated scored sentences is 0.6065-0.6646 "
            "(hagrid) and 0.6154-0.6907 (emanual) - equal to the response-level AUROC "
            "of the same checkpoints. Arg-min localisation is 0.636-0.727 on hagrid: "
            "the MIN mostly hands the response score to the genuinely unsupported "
            "sentence. There is therefore no aggregation loss to recover; the floor is "
            "the entailer's per-sentence discrimination on these two registers.",
        "caveats": "This is the framing for candidates 1 and 2, not an independent lever",
        "buildable_from_public_data": True,
        "recommendation": "build",
        "condition": "only through a register-coverage data lane - candidates 1 and 2",
    },
    {
        "name": "emanual as an adjudication instrument (measurement defect)",
        "evidence":
            "emanual carries 14 ungrounded items in 132; the Hanley-McNeil standard "
            "error of its AUROC is 0.0686. The campaign's standing per-subset hold "
            "(no subset below control-pair - 0.06) is INSIDE ONE standard error on "
            "this subset. The H108 pair - identical data and recipe, different seed - "
            "spreads 0.0587 on emanual, and the five banked checkpoints range 0.0938. "
            "hagrid is better but not comfortable: 38 negatives, standard error 0.0446, "
            "H108 pair spread 0.0245.",
        "caveats":
            "emanual's RAGBench test split is 132 rows in total and is already read "
            "whole, so the negative pool cannot be widened from the arena. The only "
            "remedies are re-pricing the clause or reading the register on a "
            "constructed held-out set.",
        "buildable_from_public_data": True,
        "recommendation": "build",
        "condition": "a coordinator re-pricing of the emanual hold clause, or its "
                     "demotion to advisory - not a training arm",
    },
    {
        "name": "window-ensemble adapter (R16-H142 G0) on the floor subsets",
        "evidence":
            "Recorded observationally. hagrid 0.6805 against its own base h108_d1 "
            "0.6599 (+0.0206, 0.46 standard errors) - the best hagrid read in the "
            "campaign; emanual 0.6628 against 0.6719 (-0.0091, 0.13 standard errors). "
            "The hagrid gain sits in the multi-sentence half (0.6282 -> 0.6907); the "
            "adapter does not move emanual's list slice (0.4737 -> 0.5169).",
        "caveats": "both deltas are inside two standard errors of their subsets",
        "buildable_from_public_data": True,
        "recommendation": "kill",
        "condition": "as a floor-subset lever specifically; the adapter's own "
                     "registration (H142 G1) is adjudicated on its own bars",
    },
]


def windows(chunk):
    """R8-H101 - 1,500-char sliding windows at stride 750, final flush to end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


# --- checkpoint loading and per-pair scoring ------------------------------------


class PlainScorer(nn.Module):
    """The incumbent DANN student read path: trunk [CLS] -> task head -> sigmoid.

    Byte-identical in numerics to R8-H77.score_student's DANN branch (fp32 trunk,
    max_length 512, CLS = last_hidden_state[:, 0]); the only difference is that
    this returns the PER-PAIR score instead of the max-aggregated one.
    """

    def __init__(self, ckpt):
        super().__init__()
        st = torch.load(ckpt / "dann_student.pt", map_location="cpu", weights_only=False)
        self.trunk = AutoModel.from_pretrained(str(ckpt / "trunk"))
        self.trunk.config.reference_compile = False
        self.head = nn.Linear(self.trunk.config.hidden_size, 1)
        self.head.load_state_dict(st["task_head"])

    def pair_scores(self, enc, _set_index, _n_sets):
        cls = self.trunk(**enc).last_hidden_state[:, 0]
        return torch.sigmoid(self.head(cls).float().squeeze(-1))


class AdapterScorer(nn.Module):
    """R16-H142 G0's forward path: the ensemble context conditions the per-window
    logit. Two-phase, because the context is pooled over the WHOLE window set
    before the adapter runs."""

    def __init__(self, base_ckpt, adapter_ckpt):
        super().__init__()
        st = torch.load(base_ckpt / "dann_student.pt", map_location="cpu", weights_only=False)
        self.trunk = AutoModel.from_pretrained(str(base_ckpt / "trunk"),
                                               attn_implementation="sdpa")
        self.trunk.config.reference_compile = False
        d = self.trunk.config.hidden_size
        self.score_head = nn.Linear(d, 1)
        self.score_head.load_state_dict(st["task_head"])
        self.h_norm = nn.LayerNorm(d)
        self.ctx_norm = nn.LayerNorm(d)
        self.adapter = nn.Sequential(nn.Linear(2 * d, 512), nn.GELU(), nn.Linear(512, 1))
        ad = torch.load(adapter_ckpt / "adapter.pt", map_location="cpu", weights_only=False)
        _missing, unexpected = self.load_state_dict(ad["state"], strict=False)
        assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"

    def encode(self, enc):
        return self.trunk(**enc).last_hidden_state[:, 0]

    def pair_logits(self, cls, ctx_rows):
        z = torch.cat([self.h_norm(cls), self.ctx_norm(ctx_rows)], dim=-1)
        return self.score_head(cls).squeeze(-1) + self.adapter(z).squeeze(-1)


@torch.inference_mode()
def score_pairs(scorer, tok, kind, flat_s, flat_w, set_index, n_sets, tag=""):
    """Per-(sentence, window) score for every pair. Plain checkpoints score pair
    by pair; the adapter needs its set context, so its CLS vectors are cached
    first and the head runs afterwards."""
    n = len(flat_s)
    t0 = time.time()
    if kind == "plain":
        out = np.zeros(n, dtype=np.float32)
        for i in range(0, n, BATCH):
            enc = tok(flat_s[i : i + BATCH], flat_w[i : i + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            out[i : i + BATCH] = scorer.pair_scores(enc, None, None).cpu().numpy()
        print(f"    {tag} {n} pairs in {time.time() - t0:.0f}s", flush=True)
        return out

    cls_all = torch.zeros(n, scorer.trunk.config.hidden_size, dtype=torch.float32)
    for i in range(0, n, BATCH):
        enc = tok(flat_s[i : i + BATCH], flat_w[i : i + BATCH], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + BATCH] = scorer.encode(enc).float().cpu()
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    cls_gpu = cls_all.cuda()
    summed = torch.zeros(n_sets, cls_gpu.shape[1], device="cuda")
    summed.index_add_(0, si, cls_gpu)
    counts = torch.zeros(n_sets, device="cuda")
    counts.index_add_(0, si, torch.ones_like(si, dtype=torch.float32))
    ctx = summed / counts.clamp(min=1).unsqueeze(-1)
    lg = scorer.pair_logits(cls_gpu, ctx[si])
    print(f"    {tag} {n} pairs in {time.time() - t0:.0f}s", flush=True)
    return torch.sigmoid(lg.float()).cpu().numpy()


def build_flat(claims, chunk_lists):
    """The windowed decomposed read's pair list, with each pair's provenance."""
    flat_s, flat_w, set_index, owner, pair_doc = [], [], [], [], []
    sent_texts, n_windows = [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wlist = [(w, di) for di, k in enumerate(ks) for w in windows(k)]
        n_windows.append(len(wlist))
        for s in H92.sentences(c):
            sid = len(owner)
            owner.append(i)
            sent_texts.append(s)
            for w, di in wlist:
                flat_s.append(s)
                flat_w.append(w)
                set_index.append(sid)
                pair_doc.append(di)
    return (flat_s, flat_w, np.array(set_index), np.array(owner),
            np.array(pair_doc), sent_texts, np.array(n_windows))


# --- item features (checkpoint-independent, CPU) ---------------------------------

_CITE = re.compile(r"\[\s*\d+\s*\]")
_URL = re.compile(r"https?://|www\.")
_LIST = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
# hagrid responses cite their evidence in PROSE ("as stated in context 3"), never
# in brackets - measured 0.000 bracket share on the sampled 250.
_CTXREF = re.compile(r"\bcontext\s*\d", re.I)
# Sentence-initial discourse frames: the class that carries no propositional
# content of its own and therefore has nothing in the evidence to entail it.
_DISCOURSE = re.compile(
    r"^(however|additionally|therefore|moreover|furthermore|in addition|overall|"
    r"in conclusion|in summary|based on|according to|it is (?:important|worth|not)|"
    r"note that|unfortunately|thus|hence|consequently|also[,\s]|the (?:provided |given )?"
    r"context|there is no|the (?:documents|passages) (?:do not|don't))\b", re.I)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to in and or is are was were be been being for on at by with as it its this "
    "that these those you your i we they he she from not no do does did can could should would "
    "will may might have has had if then than there their our but so such about into over under "
    "more most other some any all which who what when where how".split())


# Sentence registers observed in the two floor subsets' responses. Precedence is
# reference -> imperative -> discourse -> declarative.
_REFLINE = re.compile(r"https?://|www\.|^available\s*:|\.pdf\b|\bencyclopedia\b\s*$", re.IGNORECASE)
_IMPERATIVE = re.compile(
    r"^(make sure|makes sure|check|ensure|press|select|go to|tap|click|connect|try|use|"
    r"verify|reboot|restart|turn|set|enter|open|choose|follow|restore|disconnect|adjust|"
    r"install|scroll|move|hold|remove|insert|start|launch|navigate|configure|update|"
    r"activate|deactivate|change|switch|run|download|reset|unplug|plug|place|point|"
    r"allows you to|can be used|provides|repeat|wait|confirm|touch|swipe)\b",
    re.IGNORECASE)


def sentence_register(s):
    if _REFLINE.search(s):
        return "reference_line"
    if _IMPERATIVE.match(s):
        return "imperative_step"
    if _DISCOURSE.match(s):
        return "discourse_frame"
    return "declarative"


def content_words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def lex_overlap(sent, evidence_words):
    cw = content_words(sent)
    return float(len(cw & evidence_words) / len(cw)) if cw else 1.0


def item_features(row, claims_i, chunks_i):
    resp = claims_i
    sents = H92.sentences(resp)
    doc_lens = [len(d) for d in chunks_i]
    nw = sum(len(windows(d)) for d in chunks_i)
    ev = content_words(" ".join(chunks_i))
    ovs = [lex_overlap(s, ev) for s in sents]
    disc = [bool(_DISCOURSE.match(s)) for s in sents]
    return {
        "n_docs": len(chunks_i),
        "evidence_chars": int(sum(doc_lens)),
        "evidence_chars_max_doc": int(max(doc_lens)) if doc_lens else 0,
        "n_windows": int(nw),
        "any_doc_over_window": bool(any(x > WIN for x in doc_lens)),
        "n_sent_h92": len(sents),
        "resp_chars": len(resp),
        "sent_chars_mean": float(np.mean([len(s) for s in sents])),
        "sent_chars_min": int(min(len(s) for s in sents)),
        "question_chars": len(row["question"] or ""),
        "has_citation": bool(_CITE.search(resp)),
        "has_context_ref": bool(_CTXREF.search(resp)),
        "has_url": bool(_URL.search(resp)),
        "has_list": bool(_LIST.search(resp)),
        "n_discourse_sent": int(sum(disc)),
        "frac_discourse_sent": float(np.mean(disc)),
        "lex_overlap_min": float(min(ovs)),
        "lex_overlap_mean": float(np.mean(ovs)),
        "n_sent_low_overlap": int(sum(1 for o in ovs if o < 0.5)),
        "n_ann_sent": len(row["response_sentences"] or []),
        "n_unsupported": len(row["unsupported_response_sentence_keys"] or []),
    }


# --- faithful oracle (R12 label-ceiling convention), per item ---------------------


def oracle_per_item(R12, df):
    """Recompute the R12 O1..O4 ladder for one subset, per item.

    R12_label_ceiling.py blanks CUDA_VISIBLE_DEVICES at import, so the caller
    imports it only after the GPU stage has run.
    """
    recs = []
    for row in df.iter_rows(named=True):
        ssi = row["sentence_support_information"] or []
        unsupported = set(row["unsupported_response_sentence_keys"] or [])
        key_text = {p[0]: p[1] for p in (row["response_sentences"] or []) if len(p) >= 2}
        keys = [d["response_sentence_key"] for d in ssi] or list(key_text)
        supp_keys = {d["response_sentence_key"]:
                     [k for k in (d["supporting_sentence_keys"] or []) if k] for d in ssi}
        ann_texts = [key_text.get(k, "") for k in keys]
        ann_ok = [k not in unsupported for k in keys]

        v1 = 1.0 if not unsupported else 0.0
        hs = R12.sentences(row["response"])
        mapping = R12.map_sentences(hs, ann_texts)
        per_sent = []
        n_unmapped = 0
        for hits in mapping:
            if not hits:
                n_unmapped += 1
                per_sent.append((1.0, []))
                continue
            per_sent.append((1.0 if all(ann_ok[j] for j in hits) else 0.0, hits))
        v2 = min(t for t, _ in per_sent)

        docs_kept = row["documents"][: R12.MAX_CHUNKS]
        smap = {}
        for di, ds in enumerate(row["documents_sentences"]):
            for pair in ds:
                if len(pair) >= 2:
                    smap[pair[0]] = (di, pair[1])
        spans = {di: R12.win_spans(d) for di, d in enumerate(docs_kept)}

        t3, t4s, t4l = [], [], []
        n_supp_keys, n_multidoc, n_over_cap, n_no_window_strict = 0, 0, 0, 0
        supp_docs_union = set()
        for truth, hits in per_sent:
            if truth == 0.0 or not hits:
                t3.append(truth), t4s.append(truth), t4l.append(truth)
                continue
            sk = []
            for j in hits:
                sk.extend(supp_keys.get(keys[j], []))
            if not sk:
                t3.append(1.0), t4s.append(1.0), t4l.append(1.0)
                continue
            n_supp_keys += len(sk)
            by_doc, over_cap, unlocatable = {}, False, 0
            for k in sk:
                if k not in smap:
                    unlocatable += 1
                    continue
                di, txt = smap[k]
                if di >= len(docs_kept):
                    over_cap = True
                    continue
                sp = R12.locate(txt, docs_kept[di])
                if sp is None:
                    unlocatable += 1
                    continue
                by_doc.setdefault(di, []).append(sp)
            supp_docs_union.update(by_doc)
            if over_cap:
                n_over_cap += 1
                t3.append(0.0), t4s.append(0.0), t4l.append(0.0)
                continue
            t3.append(1.0)
            if not by_doc:
                t4s.append(1.0), t4l.append(1.0)
                continue
            if len(by_doc) > 1:
                n_multidoc += 1
            len_ok = any(ws <= lo and hi <= we for di, sps in by_doc.items()
                         for (lo, hi) in sps for (ws, we) in spans.get(di, []))
            str_ok = False
            if len(by_doc) == 1 and unlocatable == 0:
                di, sps = next(iter(by_doc.items()))
                lo, hi = min(a for a, _ in sps), max(b for _, b in sps)
                str_ok = any(ws <= lo and hi <= we for (ws, we) in spans.get(di, []))
            elif unlocatable > 0 and len(by_doc) == 1:
                str_ok = len_ok
            t4s.append(1.0 if str_ok else 0.0)
            t4l.append(1.0 if len_ok else 0.0)
            n_no_window_strict += int(not str_ok)

        recs.append({
            # per-H92-sentence annotated truth, in the read's own sentence order:
            # 1 = every annotated sentence this scored sentence covers is supported,
            # 0 = at least one is not. `mapped` is False where the scored sentence
            # matched no annotated sentence (scored 1 optimistically in the oracle).
            "h92_sent_truth": [t for t, _ in per_sent],
            "h92_sent_mapped": [bool(h) for _, h in per_sent],
            "oracle_o1": v1, "oracle_o2": v2,
            "oracle_o3": min(t3) if t3 else 1.0,
            "oracle_o4_strict": min(t4s) if t4s else 1.0,
            "oracle_o4_lenient": min(t4l) if t4l else 1.0,
            "n_h92_unmapped": n_unmapped,
            "n_supporting_keys": n_supp_keys,
            "n_sent_support_multidoc": n_multidoc,
            "n_sent_support_over_chunkcap": n_over_cap,
            "n_sent_no_single_window": n_no_window_strict,
            "n_support_docs": len(supp_docs_union),
            "support_spans_docs": bool(n_multidoc > 0),
        })
    return pl.DataFrame(recs)


# --- analysis helpers -------------------------------------------------------------


def op_threshold(y, s):
    """Macro-F1-optimal threshold on the subset's own score distribution.

    Stated choice: IN-SAMPLE. The alternative (the campaign's half-split
    `auc_and_f1` threshold) is unusable here - at n=132 the half-split threshold
    moves the error set by tens of items, which would make the taxonomy a
    statement about the split rather than about the subset. Nothing is tuned on
    this threshold; it only labels items correct/error, and every headline is
    reported alongside the threshold-free rank-loss decomposition.
    """
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: f1_score(y, (s >= t).astype(int), average="macro")))


def rank_loss(y, s):
    """Per-item share of the AUROC's misordered pairs (ties count 0.5).

    sum(weights) == 1 exactly when the subset has any discordant pair; the item
    weight is its own misordered-pair count over the total misordered count.
    """
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    per = np.zeros(len(y))
    for i in pos:
        per[i] = np.sum(s[neg] > s[i]) + 0.5 * np.sum(s[neg] == s[i])
    for j in neg:
        per[j] = np.sum(s[pos] < s[j]) + 0.5 * np.sum(s[pos] == s[j])
    total = per[pos].sum()  # == per[neg].sum(); each misordered pair counted once per side
    return per / max(total, 1e-9) / 2.0  # halved so the two sides sum to 1 jointly


def describe_split(name, values, err_mask, binary=False):
    v = np.asarray(values, dtype=float)
    e, c = v[err_mask], v[~err_mask]
    if len(e) < 3 or len(c) < 3:
        return None
    row = {
        "axis": name,
        "errors_mean": round(float(e.mean()), 4),
        "corrects_mean": round(float(c.mean()), 4),
        "errors_median": round(float(np.median(e)), 4),
        "corrects_median": round(float(np.median(c)), 4),
    }
    if binary:
        a, b = int(e.sum()), int(len(e) - e.sum())
        cc, d = int(c.sum()), int(len(c) - c.sum())
        try:
            _, p = stats.fisher_exact([[a, b], [cc, d]])
        except Exception:
            p = float("nan")
        row["test"] = "fisher"
    else:
        try:
            _, p = stats.mannwhitneyu(e, c, alternative="two-sided")
        except Exception:
            p = float("nan")
        row["test"] = "mannwhitney"
    row["p"] = round(float(p), 5)
    # rank-biserial effect size, sign +: errors score HIGHER on this axis
    n1, n2 = len(e), len(c)
    u = stats.mannwhitneyu(e, c, alternative="two-sided").statistic if n1 and n2 else np.nan
    row["effect_rank_biserial"] = round(float(2 * u / (n1 * n2) - 1), 4)
    return row


def slice_table(slices, err_mask, rloss):
    n = len(err_mask)
    n_err = int(err_mask.sum())
    out = []
    for name, mask in slices.items():
        mask = np.asarray(mask, dtype=bool)
        k = int(mask.sum())
        if k == 0 or k == n:
            continue
        in_err = int((mask & err_mask).sum())
        prev = k / n
        share = in_err / max(n_err, 1)
        out.append({
            "slice": name,
            "n": k,
            "prevalence": round(prev, 4),
            "errors_in_slice": in_err,
            "share_of_errors": round(share, 4),
            "lift": round(share / prev, 3) if prev else None,
            "error_rate_in": round(in_err / k, 4),
            "error_rate_out": round((n_err - in_err) / max(n - k, 1), 4),
            "share_of_rank_loss": round(float(rloss[mask].sum()), 4),
        })
    return sorted(out, key=lambda r: -r["share_of_errors"])


def register_table(reg_rows):
    """Per sentence register: how many, how often the annotation calls it
    unsupported, and whether the scorer separates supported from unsupported
    INSIDE that register."""
    out = {}
    for g in sorted({r["register"] for r in reg_rows}):
        rs = [r for r in reg_rows if r["register"] == g]
        m = [r for r in rs if r["mapped"]]
        ys = np.array([r["truth"] for r in m])
        ss = np.array([r["score"] for r in m])
        out[g] = {
            "n_sentences": len(rs),
            "n_mapped": len(m),
            "unsupported_rate": round(float((ys == 0).mean()), 4) if len(ys) else None,
            "mean_score": round(float(np.mean([r["score"] for r in rs])), 4),
            "mean_score_supported": round(float(ss[ys == 1].mean()), 4)
            if len(ys) and (ys == 1).any() else None,
            "mean_score_unsupported": round(float(ss[ys == 0].mean()), 4)
            if len(ys) and (ys == 0).any() else None,
            "within_register_sentence_auc": round(float(roc_auc_score(ys, ss)), 4)
            if len(ys) and len(np.unique(ys)) > 1 and (ys == 0).sum() >= 5 else None,
        }
    return out


def auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUROC.

    Both floor subsets are label-imbalanced, so the AUROC's precision is set by
    the MINORITY class count, not by the item count: this is what says whether a
    per-subset delta between two draws is a result or a coin flip.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def agg_counterfactuals(rows, registers_sub, y):
    """DIAGNOSTIC ONLY - which sentence register is poisoning the MIN.

    Each variant re-aggregates the SAME banked per-sentence scores under a
    different rule and reports the AUROC. These are read-geometry probes in the
    R8 precursor-P-B tradition (an oracle bound on a class of rules), NOT
    candidate aggregations: nothing here may be selected on, since the selection
    would be made on arena statistics (the H141 discipline).
    """
    def build(fn):
        return np.array([fn(np.asarray(r["sent_scores"], dtype=float),
                            registers_sub[r["item"]]) for r in rows])

    def excl(sc, rg, kind):
        keep = [v for v, g in zip(sc, rg, strict=True) if g != kind]
        return min(keep) if keep else float(sc.min())

    variants = {
        "min_all (the shipped rule)": lambda sc, rg: float(sc.min()),
        "mean": lambda sc, rg: float(sc.mean()),
        "max": lambda sc, rg: float(sc.max()),
        "min_drop_first_sentence": lambda sc, rg: float(sc[1:].min()) if len(sc) > 1
        else float(sc[0]),
        "min_drop_single_lowest": lambda sc, rg: float(np.sort(sc)[1]) if len(sc) > 1
        else float(sc[0]),
        "min_excluding_discourse_frames": lambda sc, rg: excl(sc, rg, "discourse_frame"),
        "min_excluding_imperative_steps": lambda sc, rg: excl(sc, rg, "imperative_step"),
        "min_over_declarative_only": lambda sc, rg: (
            min([v for v, g in zip(sc, rg, strict=True) if g == "declarative"] or [sc.min()])),
    }
    out = {}
    for name, fn in variants.items():
        v = build(fn)
        out[name] = round(float(roc_auc_score(y, v)), 4)
    return out


def slice_aucs(y, sv, slices, min_n=30):
    out = {}
    for name, mask in slices.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < min_n or len(np.unique(y[mask])) < 2:
            continue
        out[name] = {"n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                     "auc": round(float(roc_auc_score(y[mask], sv[mask])), 4)}
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else None


# --- main -------------------------------------------------------------------------


def gpu_stage():
    """All checkpoint reads. Returns the long-form per-item score frame."""
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    if "RTX 5000 Ada" not in dev:
        raise SystemExit(f"wrong GPU: {dev} - R17-H147 is pinned to GPU2 (RTX 5000 Ada)")

    subs = ARENA.load_subsets()
    flats, smeta = {}, {}
    for sub in SUBSETS:
        claims, chunks, y = subs[sub]
        flats[sub] = (build_flat(claims, chunks), y, claims, chunks)
        f = flats[sub][0]
        # per-scored-sentence style metadata, so the argmin sentence - the one the
        # MIN aggregation hands the whole response's score to - can be characterised
        ev_words = [content_words(" ".join(ks)) for ks in chunks]
        smeta[sub] = {
            "disc": np.array([bool(_DISCOURSE.match(s)) for s in f[5]]),
            "ov": np.array([lex_overlap(s, ev_words[o]) for s, o in zip(f[5], f[3], strict=True)]),
            "chars": np.array([len(s) for s in f[5]]),
        }
        print(f"{sub}: n={len(y)} sentences={len(f[3])} pairs={len(f[0])}", flush=True)

    records, control = [], {}
    for tag, spec in CHECKPOINTS.items():
        ck = ROOT / "models" / spec["dir"]
        if spec["kind"] == "plain":
            scorer = PlainScorer(ck).cuda().eval()
            tok = AutoTokenizer.from_pretrained(str(ck))
        else:
            base = ROOT / "models" / G0_BASE
            scorer = AdapterScorer(base, ck).cuda().eval()
            tok = AutoTokenizer.from_pretrained(str(base))

        for sub in SUBSETS:
            (flat_s, flat_w, set_index, owner, pair_doc, _st, _nw), y, _, _ = flats[sub]
            p = score_pairs(scorer, tok, spec["kind"], flat_s, flat_w, set_index,
                            len(owner), tag=f"{tag}/{sub}")
            n_sets = len(owner)
            sent_score = np.zeros(n_sets, dtype=np.float64)
            sent_doc = np.zeros(n_sets, dtype=np.int32)
            starts = np.searchsorted(set_index, np.arange(n_sets), side="left")
            ends = np.searchsorted(set_index, np.arange(n_sets), side="right")
            for sid in range(n_sets):
                a, b = starts[sid], ends[sid]
                j = a + int(np.argmax(p[a:b]))
                sent_score[sid] = p[j]
                sent_doc[sid] = pair_doc[j]

            meta = smeta[sub]
            for i in range(len(y)):
                m = owner == i
                sids = np.where(m)[0]
                ss = sent_score[m]
                sd = sent_doc[m]
                k = int(np.argmin(ss))
                gk = int(sids[k])
                records.append({
                    "subset": sub, "checkpoint": tag, "item": i, "label": int(y[i]),
                    "score": float(ss.min()),
                    "max_sent_score": float(ss.max()),
                    "sent_score_spread": float(ss.max() - ss.min()),
                    "min_sent_idx": k,
                    "min_sent_rel_pos": float(k / max(len(ss) - 1, 1)),
                    "min_sent_is_discourse": bool(meta["disc"][gk]),
                    "min_sent_lex_overlap": float(meta["ov"][gk]),
                    "min_sent_chars": int(meta["chars"][gk]),
                    "min_sent_argmax_doc": int(sd[k]),
                    "n_distinct_argmax_docs": len(set(sd.tolist())),
                    "argmax_doc_is_doc0": bool(sd[k] == 0),
                    "sent_scores": [float(x) for x in ss],
                })
            auc = float(roc_auc_score(y, np.array(
                [r["score"] for r in records if r["checkpoint"] == tag and r["subset"] == sub])))
            banked = spec["banked"][sub]
            control[f"{tag}/{sub}"] = {"reproduced": round(auc, 6), "banked": banked,
                                       "abs_delta": round(abs(auc - banked), 6),
                                       "pass": bool(abs(auc - banked) <= CONTROL_TOL)}
            print(f"  CONTROL {tag:11s} {sub:8s} read {auc:.4f}  banked {banked:.4f}  "
                  f"delta {auc - banked:+.5f}  "
                  f"{'PASS' if abs(auc - banked) <= CONTROL_TOL else 'FAIL'}", flush=True)
        del scorer
        torch.cuda.empty_cache()

    return pl.DataFrame(records), control


def main():
    t0 = time.time()
    print(f"=== R17-H147 floor-subset autopsy  {time.strftime('%F %T')} ===", flush=True)

    scores, control = gpu_stage()
    failed = [k for k, v in control.items() if not v["pass"]]
    if failed:
        OUT_JSON.write_text(json.dumps(
            {"aborted": "positive control failed", "positive_control": control}, indent=2))
        raise SystemExit(f"positive control FAILED on {failed} - read path not trusted")
    print("\npositive control: 10/10 banked AUROCs reproduced to <= 1e-3\n", flush=True)

    # --- CPU features + oracle ------------------------------------------------
    subs = ARENA.load_subsets()
    keep = os.environ.get("CUDA_VISIBLE_DEVICES")
    R12 = _mod("r12", "R12_label_ceiling.py")
    if keep is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = keep
    raw_rows = R12.load_rows()

    feats = {}
    for sub in SUBSETS:
        claims, chunks, y = subs[sub]
        df_raw = raw_rows[sub]
        y_raw = df_raw["adherence_score"].cast(pl.Int8).to_numpy()
        assert np.array_equal(y, y_raw), f"{sub}: arena and R12 row order disagree"
        rows = [item_features(r, claims[i], chunks[i])
                for i, r in enumerate(df_raw.iter_rows(named=True))]
        fdf = pl.DataFrame(rows)
        odf = oracle_per_item(R12, df_raw)
        feats[sub] = pl.concat([fdf, odf], how="horizontal").with_columns(
            pl.Series("item", np.arange(len(y))), pl.Series("label", y))
        print(f"{sub}: features built for {len(y)} items", flush=True)

    scores = scores.join(
        pl.concat([feats[s].with_columns(pl.lit(s).alias("subset")) for s in SUBSETS],
                  how="vertical_relaxed").drop("label"),
        on=["subset", "item"], how="left")
    scores.write_parquet(OUT_PARQUET)
    print(f"per-item scores -> {OUT_PARQUET}  ({scores.shape})", flush=True)

    registers = {sub: [[sentence_register(x) for x in H92.sentences(c)]
                       for c in subs[sub][0]] for sub in SUBSETS}

    # --- per-subset analysis ---------------------------------------------------
    report = {"per_subset": {}, "positive_control": control}
    for sub in SUBSETS:
        f = feats[sub]
        y = f["label"].to_numpy()
        n = len(y)
        diag_slices = {
            "single_scored_sentence": f["n_sent_h92"].to_numpy() == 1,
            "multi_sentence": f["n_sent_h92"].to_numpy() >= 2,
            "has_list_structure": f["has_list"].to_numpy(),
            "no_list_structure": ~f["has_list"].to_numpy(),
            "faithful_oracle_reachable (o4_strict == 1)":
                f["oracle_o4_strict"].to_numpy() == 1.0,
            "faithful_oracle_unreachable (o4_strict == 0)":
                f["oracle_o4_strict"].to_numpy() == 0.0,
            "single_doc_evidence": f["n_docs"].to_numpy() == 1,
            "multi_doc_evidence (>= 3)": f["n_docs"].to_numpy() >= 3,
        }
        sub_rep = {
            "n": n, "grounded_rate": round(float(y.mean()), 4),
            "read": "windowed decomposed-min (1500/750, max-over-windows, min-over-sentences)",
            "checkpoints": {},
        }

        # 4. faithful-oracle headroom
        oracle = {}
        for tag in ("oracle_o1", "oracle_o2", "oracle_o3",
                    "oracle_o4_strict", "oracle_o4_lenient"):
            oracle[tag] = round(float(roc_auc_score(y, f[tag].to_numpy())), 4)
        best_read = max(CHECKPOINTS[t]["banked"][sub] for t in CHECKPOINTS)
        sub_rep["oracle"] = {
            **oracle,
            "banked_r12_o4_strict": BANKED_ORACLE_STRICT[sub],
            "banked_r12_o4_lenient": BANKED_ORACLE_LENIENT[sub],
            "reproduces_banked": bool(
                abs(oracle["oracle_o4_strict"] - BANKED_ORACLE_STRICT[sub]) <= 2e-3),
            "best_checkpoint_read": best_read,
            "headroom_to_faithful_ceiling": round(oracle["oracle_o4_strict"] - best_read, 4),
            "headroom_to_lenient_ceiling": round(oracle["oracle_o4_lenient"] - best_read, 4),
        }

        err_sets, thresholds, rlosses = {}, {}, {}
        for tag in CHECKPOINTS:
            s = (scores.filter((pl.col("subset") == sub) & (pl.col("checkpoint") == tag))
                 .sort("item"))
            sv = s["score"].to_numpy()
            thr = op_threshold(y, sv)
            pred = (sv >= thr).astype(int)
            err = pred != y
            rl = rank_loss(y, sv)
            err_sets[tag] = set(np.where(err)[0].tolist())
            thresholds[tag] = thr
            rlosses[tag] = rl

            # Sentence-level diagnostic. Separates two very different failures:
            # a per-sentence entailer with no signal on this register, versus a
            # sound entailer whose MIN hands the response score to the wrong
            # sentence. Scored only on H92 sentences that map to an annotated one.
            ys_s, ss_s, loc_hit, loc_n = [], [], 0, 0
            reg_rows = []
            for r_ in s.iter_rows(named=True):
                tr, mp, sc = r_["h92_sent_truth"], r_["h92_sent_mapped"], r_["sent_scores"]
                regs = registers[sub][r_["item"]]
                for t, m, v, g in zip(tr, mp, sc, regs, strict=True):
                    reg_rows.append({"register": g, "score": float(v),
                                     "truth": int(t) if m else None, "mapped": m})
                    if m:
                        ys_s.append(int(t))
                        ss_s.append(float(v))
                if r_["label"] == 0:
                    bad = [j for j, (t, m) in enumerate(zip(tr, mp, strict=True))
                           if m and t == 0]
                    if bad:
                        loc_n += 1
                        loc_hit += int(int(np.argmin(sc)) in bad)
            ys_s, ss_s = np.array(ys_s), np.array(ss_s)
            sent_block = {
                "n_scored_sentences_mapped": int(len(ys_s)),
                "n_unsupported_scored_sentences": int((ys_s == 0).sum()),
                "sentence_level_auc": (round(float(roc_auc_score(ys_s, ss_s)), 4)
                                       if len(np.unique(ys_s)) > 1 else None),
                "mean_score_supported_sent": round(float(ss_s[ys_s == 1].mean()), 4),
                "mean_score_unsupported_sent": round(float(ss_s[ys_s == 0].mean()), 4)
                if (ys_s == 0).any() else None,
                "argmin_localisation_rate": round(loc_hit / max(loc_n, 1), 4),
                "n_ungrounded_items_with_located_bad_sentence": loc_n,
                "by_sentence_register": register_table(reg_rows),
                "auc_by_item_slice": slice_aucs(y, sv, diag_slices),
                "aggregation_counterfactual_auc": agg_counterfactuals(
                    list(s.iter_rows(named=True)), registers[sub], y),
            }
            auc_v = float(roc_auc_score(y, sv))
            sub_rep["checkpoints"][tag] = {
                "auc_standard_error": round(auc_se(auc_v, int(y.sum()),
                                                   int((y == 0).sum())), 4),
                **sent_block,
                "auc": round(float(roc_auc_score(y, sv)), 4),
                "operating_threshold": round(thr, 4),
                "n_errors": int(err.sum()),
                "n_false_negatives": int(((y == 1) & (pred == 0)).sum()),
                "n_false_positives": int(((y == 0) & (pred == 1)).sum()),
                "margin_errors_mean": round(float(np.mean(np.abs(sv - thr)[err])), 4),
                "margin_corrects_mean": round(float(np.mean(np.abs(sv - thr)[~err])), 4),
                "score_mean_pos": round(float(sv[y == 1].mean()), 4),
                "score_mean_neg": round(float(sv[y == 0].mean()), 4),
            }

        # consensus error set: erred by a majority of the five checkpoints
        cnt = np.zeros(n, dtype=int)
        for tag in CHECKPOINTS:
            cnt[list(err_sets[tag])] += 1
        consensus = cnt >= 3
        aucs = [sub_rep["checkpoints"][t]["auc"] for t in CHECKPOINTS]
        ses = [sub_rep["checkpoints"][t]["auc_standard_error"] for t in CHECKPOINTS]
        sub_rep["measurement_precision"] = {
            "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
            "mean_auc_standard_error": round(float(np.mean(ses)), 4),
            "observed_seed_spread_clean_pair": round(
                abs(sub_rep["checkpoints"]["clean_d1"]["auc"]
                    - sub_rep["checkpoints"]["clean_d2"]["auc"]), 4),
            "observed_seed_spread_h108_pair": round(
                abs(sub_rep["checkpoints"]["h108_d1"]["auc"]
                    - sub_rep["checkpoints"]["h108_d2"]["auc"]), 4),
            "range_over_5_checkpoints": round(max(aucs) - min(aucs), 4),
            "reading": "the AUROC's precision is set by the minority (ungrounded) class; "
                       "a per-subset delta smaller than ~2 standard errors is not a result",
        }
        sub_rep["consensus_errors"] = {
            "definition": "item mis-classified by >= 3 of the 5 banked checkpoints "
                          "at each checkpoint's own macro-F1-optimal threshold",
            "n": int(consensus.sum()),
            "share_of_items": round(float(consensus.mean()), 4),
            "n_errored_by_all_5": int((cnt == 5).sum()),
            "n_errored_by_none": int((cnt == 0).sum()),
        }

        # 6. cross-checkpoint Jaccard
        pairs = {}
        for a, b in itertools.combinations(CHECKPOINTS, 2):
            pairs[f"{a}|{b}"] = jaccard(err_sets[a], err_sets[b])
        vals = [v for v in pairs.values() if v is not None]
        sub_rep["cross_checkpoint"] = {
            "pairwise_error_jaccard": pairs,
            "mean_jaccard": round(float(np.mean(vals)), 4),
            "mean_jaccard_within_seed_pairs": round(float(np.mean(
                [pairs["clean_d1|clean_d2"], pairs["h108_d1|h108_d2"]])), 4),
            "score_rank_correlation": {
                f"{a}|{b}": round(float(stats.spearmanr(
                    scores.filter((pl.col("subset") == sub) & (pl.col("checkpoint") == a))
                    .sort("item")["score"].to_numpy(),
                    scores.filter((pl.col("subset") == sub) & (pl.col("checkpoint") == b))
                    .sort("item")["score"].to_numpy()).statistic), 4)
                for a, b in itertools.combinations(CHECKPOINTS, 2)
            },
        }

        # 2/3. taxonomy + dispersion, on the consensus error set and on clean_d1
        ref = "clean_d1"
        sref = (scores.filter((pl.col("subset") == sub) & (pl.col("checkpoint") == ref))
                .sort("item"))
        axes_num = ["n_windows", "evidence_chars", "evidence_chars_max_doc", "n_docs",
                    "n_sent_h92", "resp_chars", "sent_chars_mean", "sent_chars_min",
                    "question_chars", "n_ann_sent", "n_unsupported", "n_supporting_keys",
                    "n_support_docs", "n_sent_support_multidoc", "n_h92_unmapped",
                    "n_sent_no_single_window", "n_discourse_sent", "frac_discourse_sent",
                    "lex_overlap_min", "lex_overlap_mean", "n_sent_low_overlap"]
        axes_bin = ["has_citation", "has_context_ref", "has_url", "has_list",
                    "any_doc_over_window", "support_spans_docs"]
        axes_read = ["n_distinct_argmax_docs", "sent_score_spread", "max_sent_score",
                     "min_sent_rel_pos", "min_sent_lex_overlap", "min_sent_chars"]
        axes_read_bin = ["min_sent_is_discourse"]

        tax = {}
        for label, mask in (("consensus", consensus),
                            (ref, np.isin(np.arange(n), list(err_sets[ref])))):
            rows = []
            for a in axes_num:
                r = describe_split(a, f[a].to_numpy(), mask)
                if r:
                    rows.append(r)
            for a in axes_bin:
                r = describe_split(a, f[a].cast(pl.Int8).to_numpy(), mask, binary=True)
                if r:
                    rows.append(r)
            for a in axes_read:
                r = describe_split(a, sref[a].to_numpy(), mask)
                if r:
                    rows.append(r)
            for a in axes_read_bin:
                r = describe_split(a, sref[a].cast(pl.Int8).to_numpy(), mask, binary=True)
                if r:
                    rows.append(r)
            r = describe_split("label_is_grounded", y, mask, binary=True)
            if r:
                rows.append(r)
            tax[label] = sorted(rows, key=lambda x: x["p"])
        sub_rep["error_taxonomy"] = tax

        # 5. error concentration
        med_ev = float(np.median(f["evidence_chars"].to_numpy()))
        med_rc = float(np.median(f["resp_chars"].to_numpy()))
        slices = {
            "ungrounded (label=0)": y == 0,
            "grounded (label=1)": y == 1,
            "single_scored_sentence (n_sent_h92 == 1)": f["n_sent_h92"].to_numpy() == 1,
            "multi_sentence (n_sent_h92 >= 3)": f["n_sent_h92"].to_numpy() >= 3,
            "multi_doc_evidence (n_docs >= 3)": f["n_docs"].to_numpy() >= 3,
            "single_doc_evidence (n_docs == 1)": f["n_docs"].to_numpy() == 1,
            "long_evidence (evidence_chars > subset median)":
                f["evidence_chars"].to_numpy() > med_ev,
            "any_doc_exceeds_window (>1500 chars)": f["any_doc_over_window"].to_numpy(),
            "multi_window (n_windows > n_docs)":
                f["n_windows"].to_numpy() > f["n_docs"].to_numpy(),
            "short_response (resp_chars < subset median)": f["resp_chars"].to_numpy() < med_rc,
            "has_citation_markers": f["has_citation"].to_numpy(),
            "has_prose_context_reference (\"...in context 3\")": f["has_context_ref"].to_numpy(),
            "has_list_structure": f["has_list"].to_numpy(),
            "min_sentence_is_discourse_frame (read)":
                sref["min_sent_is_discourse"].to_numpy(),
            "min_sentence_lex_overlap < 0.5 (read)":
                sref["min_sent_lex_overlap"].to_numpy() < 0.5,
            "any_sentence_lex_overlap < 0.5": f["n_sent_low_overlap"].to_numpy() > 0,
            "response_has_a_discourse_frame_sentence": f["n_discourse_sent"].to_numpy() > 0,
            "support_spans_multiple_docs (annotation)": f["support_spans_docs"].to_numpy(),
            "argmax_windows_span_multiple_docs (read)":
                sref["n_distinct_argmax_docs"].to_numpy() > 1,
            "oracle_o4_strict_unreachable_positive":
                (f["oracle_o4_strict"].to_numpy() == 0.0) & (y == 1),
            "oracle_o4_strict == 0 (read cannot express the support)":
                f["oracle_o4_strict"].to_numpy() == 0.0,
            "h92_splitter_dropped_annotated_sentence":
                f["n_h92_unmapped"].to_numpy() > 0,
            "no_supporting_keys_recorded": f["n_supporting_keys"].to_numpy() == 0,
        }
        conc = {}
        for label, mask, rl in (("consensus", consensus, np.mean(
                [rlosses[t] for t in CHECKPOINTS], axis=0)),
                (ref, np.isin(np.arange(n), list(err_sets[ref])), rlosses[ref])):
            tbl = slice_table(slices, mask, rl)
            gate = [r for r in tbl if r["share_of_errors"] >= 0.30 and (r["lift"] or 0) > 1.0]
            conc[label] = {
                "n_errors": int(mask.sum()),
                "gate": "R15 convention: a slice qualifies at share_of_errors >= 0.30; "
                        "lift > 1.0 is required in addition, since a slice covering 30% of "
                        "the subset trivially carries 30% of its errors",
                "qualifying_slices": gate,
                "all_slices": tbl,
            }
        sub_rep["error_concentration"] = conc
        report["per_subset"][sub] = sub_rep

    report["mechanism_candidates"] = MECHANISM_CANDIDATES
    report["meta"] = {
        "experiment": "R17-H147 FLOOR-SUBSET AUTOPSY (hagrid + emanual)",
        "licence": "ANALYSIS ONLY - no training, no lane building, no tuning of any "
                   "kind was performed on these findings; arena statistics inform the "
                   "diagnosis only (the H141 discipline)",
        "checkpoints": {k: str(ROOT / "models" / v["dir"]) for k, v in CHECKPOINTS.items()},
        "gpu": torch.cuda.get_device_name(0),
        "runtime_seconds": round(time.time() - t0, 1),
        "artifacts": {"item_scores": str(OUT_PARQUET), "json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nautopsy -> {OUT_JSON}  ({time.time() - t0:.0f}s)", flush=True)
    print("=== R17-H147 AUTOPSY COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
